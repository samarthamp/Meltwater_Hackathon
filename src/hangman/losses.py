"""Multi-task training objective.

Several supervision signals share one trunk:

``set``
    BCE of the presence head against the teacher's soft posterior (soft
    targets as in Hinton, Vinyals & Dean, 2015), plus an optional hard-label
    term anchoring it to ground truth.

``pos``
    Per-position cross-entropy against the true letter at hidden positions,
    i.e. masked language modelling over characters (Devlin et al., 2019).

``count``
    Occurrence-count buckets ``{0, 1, 2, >=3}`` per letter.

``value``
    Win probability from the current state, regressed on Monte-Carlo returns.
    Used by one-ply lookahead and as the RL baseline.

``llm``
    Optional KL to an LLM teacher's (Qwen3-4B) posterior, used only in the
    distillation stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .models.hangmanformer import HangmanOutput

#: Sentinel written by :mod:`hangman.collect` when a state has no rollout.
VALUE_UNKNOWN = -1.0


@dataclass
class LossBreakdown:
    """Scalar components, kept separate so training curves are diagnosable."""

    total: torch.Tensor
    set_soft: torch.Tensor
    set_hard: torch.Tensor
    pos: torch.Tensor
    count: torch.Tensor
    value: torch.Tensor
    llm: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().item())
            for name in ("total", "set_soft", "set_hard", "pos", "count", "value", "llm")
        }


def _masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE-with-logits restricted to ``valid`` entries, mean over the batch.

    Already-guessed letters are excluded rather than pushed to zero, so the
    model cannot reduce the loss by learning the mask instead of the language.
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * valid.to(loss.dtype)
    per_state = loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(loss.dtype)
    if weight is not None:
        return (per_state * weight).sum() / weight.sum().clamp_min(1e-6)
    return per_state.mean()


def compute_loss(
    out: HangmanOutput,
    batch: dict[str, torch.Tensor],
    cfg: LossConfig,
) -> LossBreakdown:
    """Assemble the multi-task loss for one batch.

    Expected ``batch`` keys
    -----------------------
    ``guessed``      (B, 26) bool  - letters already guessed
    ``target_soft``  (B, 26) float - teacher posterior
    ``target_hard``  (B, 26) float - 1 where the letter occurs in the word
    ``char_target``  (B, SEQ_LEN-1) long - true letter at each *hidden*
                     position, -1 elsewhere; already aligned to the position
                     head's width by :meth:`hangman.dataset.RecordDataset.batch`
    ``count_target`` (B, 26) long  - true occurrence bucket per letter
    ``value_target`` (B,) float    - MC return, or ``VALUE_UNKNOWN``
    ``sample_weight``(B,) float, optional - per-state weight
    ``llm_target``   (B, 26) float, optional - Qwen posterior
    """
    device = out.set_logits.device
    zero = torch.zeros((), device=device)
    valid = ~batch["guessed"]
    weight = batch.get("sample_weight")

    smooth = cfg.label_smoothing
    soft = batch["target_soft"]
    if smooth > 0.0:
        soft = soft * (1.0 - smooth) + 0.5 * smooth
    set_soft = _masked_bce(out.set_logits, soft, valid, weight)

    set_hard = zero
    if cfg.w_set_hard > 0.0:
        hard = batch["target_hard"]
        if smooth > 0.0:
            hard = hard * (1.0 - smooth) + 0.5 * smooth
        set_hard = _masked_bce(out.set_logits, hard, valid, weight)

    pos_loss = zero
    if cfg.w_pos > 0.0 and out.pos_logits is not None:
        # ``char_target`` is -1 at revealed and padded positions; scoring
        # revealed letters would reward copying rather than inference.
        labels = batch["char_target"]
        seq = min(out.pos_logits.shape[1], labels.shape[1])
        logits = out.pos_logits[:, :seq]
        pos_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels[:, :seq].reshape(-1),
            ignore_index=-1,
        )
        if torch.isnan(pos_loss):
            pos_loss = zero

    count_loss = zero
    if cfg.w_count > 0.0 and out.count_logits is not None:
        logits = out.count_logits
        labels = batch["count_target"].clone()
        labels[batch["guessed"]] = -100
        count_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )
        if torch.isnan(count_loss):
            count_loss = zero

    value_loss = zero
    if cfg.w_value > 0.0 and out.value is not None:
        target = batch["value_target"]
        known = target >= 0.0
        if bool(known.any()):
            value_loss = F.binary_cross_entropy_with_logits(
                out.value[known].float(), target[known].float()
            )

    llm_loss = zero
    if cfg.w_llm > 0.0 and "llm_target" in batch:
        temp = cfg.distill_temperature
        student = torch.sigmoid(out.set_logits / temp)
        teacher = batch["llm_target"].clamp(1e-6, 1.0 - 1e-6)
        teacher = torch.sigmoid(torch.logit(teacher) / temp)
        # Bernoulli KL per letter, restricted to unguessed letters.
        kl = teacher * (teacher.log() - student.clamp_min(1e-6).log()) + (
            1.0 - teacher
        ) * ((1.0 - teacher).log() - (1.0 - student).clamp_min(1e-6).log())
        kl = kl * valid.to(kl.dtype)
        llm_loss = (kl.sum(dim=1) / valid.sum(dim=1).clamp_min(1)).mean() * (temp ** 2)

    total = (
        cfg.w_set_soft * set_soft
        + cfg.w_set_hard * set_hard
        + cfg.w_pos * pos_loss
        + cfg.w_count * count_loss
        + cfg.w_value * value_loss
        + cfg.w_llm * llm_loss
    )
    return LossBreakdown(
        total=total,
        set_soft=set_soft,
        set_hard=set_hard,
        pos=pos_loss,
        count=count_loss,
        value=value_loss,
        llm=llm_loss,
    )


@torch.no_grad()
def top1_agreement(out: HangmanOutput, batch: dict[str, torch.Tensor]) -> float:
    """Fraction of states where the model's guess matches the teacher's.

    A cheap training-time proxy for policy quality.
    """
    valid = ~batch["guessed"]
    student = out.set_logits.masked_fill(~valid, float("-inf")).argmax(dim=1)
    teacher = batch["target_soft"].masked_fill(~valid, float("-inf")).argmax(dim=1)
    return float((student == teacher).float().mean().item())


@torch.no_grad()
def hit_rate(out: HangmanOutput, batch: dict[str, torch.Tensor]) -> float:
    """Fraction of states where the model's guess is in the word."""
    valid = ~batch["guessed"]
    student = out.set_logits.masked_fill(~valid, float("-inf")).argmax(dim=1)
    hits = batch["target_hard"].gather(1, student.unsqueeze(1)).squeeze(1)
    return float((hits > 0.5).float().mean().item())
