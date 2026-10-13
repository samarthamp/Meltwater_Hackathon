"""Guess-selection policies.

Greedy ``argmax P(letter is in the word)`` maximises the chance of a hit, not
of finishing the word within six misses.  Two refinements target the gap:
one-ply value lookahead in low-life states, and exact search over the
dictionary-consistent candidate set once it is small (Reams, 2009; Meng &
Lucas, 2024).  ``AutoPolicy`` and ``HybridEnsemblePolicy`` dispatch between
them per state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .config import MAX_WRONG_GUESSES, PolicyConfig
from .models.layers import noisy_or_presence
from .ngram import HOLE
from .state import encode_globals, encode_tokens
from .teacher import BayesTeacher


class ProbabilitySource(Protocol):
    """Anything that scores letters for a batch of states."""

    def presence(
        self,
        board: torch.Tensor,
        lengths: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
    ) -> torch.Tensor:
        ...


@dataclass
class ModelSource:
    """Wraps a trained network as a presence-probability source."""

    model: torch.nn.Module
    #: Blend weight between the presence head and the noisy-OR pooling of the
    #: per-position head; fitted per member by ``scripts/08_ensemble_eval.py``.
    noisy_or_weight: float = 0.5
    amp_dtype: torch.dtype = torch.bfloat16

    @torch.no_grad()
    def _forward(self, board, lengths, present, absent):
        tokens, attn = encode_tokens(board)
        globals_vec = encode_globals(lengths, present, absent)
        device_type = tokens.device.type
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            out = self.model(tokens=tokens, attn_mask=attn, globals=globals_vec)
        return out

    @torch.no_grad()
    def presence(self, board, lengths, present, absent) -> torch.Tensor:
        out = self._forward(board, lengths, present, absent)
        probs = torch.sigmoid(out.set_logits.float())
        if self.noisy_or_weight > 0.0 and out.pos_logits is not None:
            hidden = (board == HOLE)
            seq = min(out.pos_logits.shape[1], hidden.shape[1])
            pooled = noisy_or_presence(out.pos_logits[:, :seq].float(), hidden[:, :seq])
            probs = (1.0 - self.noisy_or_weight) * probs + self.noisy_or_weight * pooled
        return probs.masked_fill(present | absent, 0.0)

    @torch.no_grad()
    def full(self, board, lengths, present, absent):
        """Presence, expected occurrence count and state value together."""
        out = self._forward(board, lengths, present, absent)
        probs = torch.sigmoid(out.set_logits.float())
        if self.noisy_or_weight > 0.0 and out.pos_logits is not None:
            hidden = (board == HOLE)
            seq = min(out.pos_logits.shape[1], hidden.shape[1])
            pooled = noisy_or_presence(out.pos_logits[:, :seq].float(), hidden[:, :seq])
            probs = (1.0 - self.noisy_or_weight) * probs + self.noisy_or_weight * pooled
        probs = probs.masked_fill(present | absent, 0.0)

        expected_count = None
        if out.count_logits is not None:
            buckets = out.count_logits.float().softmax(dim=-1)
            weights = torch.tensor([0.0, 1.0, 2.0, 3.5], device=buckets.device)
            expected_count = (buckets * weights).sum(dim=-1)
        value = torch.sigmoid(out.value.float()) if out.value is not None else None
        return probs, expected_count, value


@dataclass
class EnsembleSource:
    """Logit-space average of several sources, with per-source calibration.

    Averaging logits rather than probabilities keeps confident members from
    being diluted by uncertain ones.  Weights and temperatures are fitted on
    the validation dictionary in :mod:`hangman.calibrate`.
    """

    sources: list[ProbabilitySource]
    weights: torch.Tensor | None = None
    temperatures: torch.Tensor | None = None

    @torch.no_grad()
    def presence(self, board, lengths, present, absent) -> torch.Tensor:
        logits = []
        for i, src in enumerate(self.sources):
            probs = src.presence(board, lengths, present, absent).clamp(1e-6, 1 - 1e-6)
            logit = torch.logit(probs)
            if self.temperatures is not None:
                logit = logit / self.temperatures[i]
            logits.append(logit)
        stacked = torch.stack(logits, dim=0)
        if self.weights is not None:
            w = self.weights.view(-1, 1, 1)
            pooled = (stacked * w).sum(dim=0) / w.sum()
        else:
            pooled = stacked.mean(dim=0)
        return torch.sigmoid(pooled).masked_fill(present | absent, 0.0)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class GreedyPolicy:
    """Pick the most likely unguessed letter."""

    source: ProbabilitySource

    @torch.no_grad()
    def __call__(self, board, lengths, present, absent, lives, active) -> torch.Tensor:
        probs = self.source.presence(board, lengths, present, absent)
        return probs.argmax(dim=1)


@dataclass
class ValueGuidedPolicy:
    """One-ply expectimax over the top-k letters using the value head.

    For each candidate letter ``l``:

        EV(l) = p_l * V(state after a hit) + (1 - p_l) * V(state after a miss)

    The hit successor marks ``l`` as present without revealing positions.
    Each call costs ``2k`` extra forward passes, so lookahead runs only on
    states with at most ``max_lives`` lives left.
    """

    source: ModelSource
    topk: int = 5
    max_lives: int = 2
    #: Optional separate source for the letter probabilities (e.g. a calibrated
    #: ensemble); the value head is always read from ``source``.
    prob_source: ProbabilitySource | None = None
    #: Fall back to greedy when the value head's std across the batch is below
    #: this, as it is before any DAgger round supplies Monte-Carlo returns.
    min_value_spread: float = 0.02

    @torch.no_grad()
    def __call__(self, board, lengths, present, absent, lives, active) -> torch.Tensor:
        probs, _, value = self.source.full(board, lengths, present, absent)
        if self.prob_source is not None:
            probs = self.prob_source.presence(board, lengths, present, absent)
        greedy = probs.argmax(dim=1)

        if value is None or (
            value.numel() > 1 and float(value.std()) < self.min_value_spread
        ):
            return greedy

        eligible = active & (lives <= self.max_lives)
        if not bool(eligible.any()):
            return greedy

        idx = torch.nonzero(eligible, as_tuple=True)[0]
        sub_board = board[idx]
        sub_len = lengths[idx]
        sub_present = present[idx]
        sub_absent = absent[idx]
        sub_probs = probs[idx]
        k = min(self.topk, 26)
        cand = sub_probs.topk(k, dim=1).indices               # (M, k)

        best_ev = torch.full((idx.numel(),), -1.0, device=board.device)
        best_letter = greedy[idx].clone()

        for j in range(k):
            letter = cand[:, j]
            p_hit = sub_probs.gather(1, letter.unsqueeze(1)).squeeze(1)

            hit_present = sub_present.clone()
            hit_present.scatter_(1, letter.unsqueeze(1), True)
            # Approximate the hit successor: the letter joins the present set,
            # but the board is left unrevealed.
            hit_board = sub_board.clone()
            _, _, v_hit = self.source.full(hit_board, sub_len, hit_present, sub_absent)

            miss_absent = sub_absent.clone()
            miss_absent.scatter_(1, letter.unsqueeze(1), True)
            _, _, v_miss = self.source.full(sub_board, sub_len, sub_present, miss_absent)

            if v_hit is None or v_miss is None:
                return greedy
            ev = p_hit * v_hit + (1.0 - p_hit) * v_miss
            better = ev > best_ev
            best_ev = torch.where(better, ev, best_ev)
            best_letter = torch.where(better, letter, best_letter)

        out = greedy.clone()
        out[idx] = best_letter
        return out


@dataclass
class ExactSearchPolicy:
    """Exact information-set re-ranking of the model's top-k candidates.

    Applies where the dictionary-consistent candidate set ``C`` is non-empty
    and at most ``max_candidates``; elsewhere the ``fallback`` policy decides.
    The candidate dictionary does not contain the target words, so the model
    probability stays the primary signal and only its top ``topk`` letters
    are re-scored:

        score(l) = log p_model(l) - lambda * E[|C'| | guess l] / |C|

    where ``E[|C'|]`` is the expected size of the surviving candidate set
    (Reams, 2009; Meng & Lucas, 2024).  Guessing ``l`` partitions ``C`` by
    reveal signature, so ``E[|C'|] = (sum_s n_s^2) / n``.  Two candidates share
    a signature iff their 0/1 indicator rows ``a, b`` for ``l`` satisfy
    ``rowsum(a) + rowsum(b) - 2 * <a, b> == 0``, which reduces the count to a
    batched matmul.
    """

    teacher: BayesTeacher
    fallback: object
    max_candidates: int = 200
    length: int | None = None
    topk: int = 5
    lambda_size: float = 0.35
    source: ProbabilitySource | None = None
    #: Penalty for a letter that no consistent candidate contains.  Off by
    #: default: the candidate dictionary excludes the target words, so this is
    #: weak evidence.
    miss_penalty: float = 0.0

    @torch.no_grad()
    def __call__(self, board, lengths, present, absent, lives, active) -> torch.Tensor:
        base = self.fallback(board, lengths, present, absent, lives, active)
        length = self.length or int(lengths.max().item())
        if length not in self.teacher._chars:
            return base

        trimmed = board[:, :length]
        ok = self.teacher._consistency(length, trimmed, present, absent, tier=0)
        n_cand = ok.sum(dim=1)
        usable = active & (n_cand > 0) & (n_cand <= self.max_candidates)
        if not bool(usable.any()):
            return base

        rows = torch.nonzero(usable, as_tuple=True)[0]
        device = board.device
        chars = self.teacher._chars[length]                       # (N, L)

        # Pad every state's candidate set to a common width so the whole batch
        # can be scored at once.
        width = int(n_cand[rows].max().item())
        order = torch.argsort(ok[rows].to(torch.int8), dim=1, descending=True, stable=True)
        picked = order[:, :width]                                 # (S, width)
        valid = torch.gather(ok[rows], 1, picked)                 # (S, width)
        cand = chars[picked]                                      # (S, width, L)
        counts = n_cand[rows].to(torch.float32)                   # (S,)

        probs = self._probabilities(board, lengths, present, absent, rows)
        k = min(self.topk, 26)
        letters = probs.topk(k, dim=1).indices                    # (S, k)

        best_score = torch.full((rows.numel(),), -float("inf"), device=device)
        best_letter = base[rows].clone()
        valid_f = valid.to(torch.float32)

        for j in range(k):
            letter = letters[:, j]                                # (S,)
            hits = (cand == letter.view(-1, 1, 1)) & valid.unsqueeze(-1)
            indicator = hits.to(torch.float32)                    # (S, width, L)

            occurrences = indicator.sum(dim=2)                    # (S, width)
            p_hit = (occurrences > 0).to(torch.float32).mul(valid_f).sum(dim=1) / counts

            gram = indicator @ indicator.transpose(1, 2)          # (S, width, width)
            rowsum = occurrences.unsqueeze(2) + occurrences.unsqueeze(1)
            same = (rowsum - 2.0 * gram).abs() < 0.5
            pair_mask = valid_f.unsqueeze(2) * valid_f.unsqueeze(1)
            expected_left = (same.to(torch.float32) * pair_mask).sum(dim=(1, 2)) / counts

            model_p = probs.gather(1, letter.unsqueeze(1)).squeeze(1).clamp_min(1e-9)
            score = model_p.log() - self.lambda_size * expected_left / counts
            if self.miss_penalty:
                score = score - self.miss_penalty * (p_hit <= 0.0).to(torch.float32)

            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best_letter = torch.where(better, letter, best_letter)

        out = base.clone()
        out[rows] = best_letter
        return out

    def _probabilities(self, board, lengths, present, absent, rows) -> torch.Tensor:
        """Model presence probabilities for the states being refined."""
        if self.source is not None:
            return self.source.presence(
                board[rows], lengths[rows], present[rows], absent[rows]
            )
        inner = getattr(self.fallback, "source", None)
        if inner is not None:
            return inner.presence(board[rows], lengths[rows], present[rows], absent[rows])
        # No model available: fall back to the teacher's own marginals.
        out = self.teacher.posterior(
            int(lengths.max().item()),
            board[rows][:, : int(lengths.max().item())],
            present[rows],
            absent[rows],
        )
        return out.presence


@dataclass
class AutoPolicy:
    """Dispatches greedy / value-guided / exact search per state.

    In ``auto`` mode, exact search handles states with a small candidate set,
    value lookahead handles low-life states, and the rest play greedily.
    """

    source: ModelSource
    cfg: PolicyConfig
    teacher: BayesTeacher | None = None
    length: int | None = None

    def __post_init__(self) -> None:
        self._greedy = GreedyPolicy(self.source)
        self._value = ValueGuidedPolicy(
            self.source, topk=self.cfg.value_topk, max_lives=self.cfg.value_lookahead_max_lives
        )
        self._exact = (
            ExactSearchPolicy(
                self.teacher,
                fallback=self._value if self.cfg.mode == "auto" else self._greedy,
                max_candidates=self.cfg.exact_max_candidates,
                length=self.length,
                topk=self.cfg.value_topk,
                source=self.source,
            )
            if self.teacher is not None and self.cfg.mode in ("auto", "exact")
            else None
        )

    @torch.no_grad()
    def __call__(self, board, lengths, present, absent, lives, active) -> torch.Tensor:
        if self.cfg.mode == "greedy" or (self._exact is None and self.cfg.mode != "value"):
            return self._greedy(board, lengths, present, absent, lives, active)
        if self.cfg.mode == "value" or self._exact is None:
            return self._value(board, lengths, present, absent, lives, active)
        return self._exact(board, lengths, present, absent, lives, active)


@dataclass
class HybridEnsemblePolicy:
    """Ensemble presence scores with primary-member value lookahead.

    Presence probabilities come from the calibrated ensemble; the value head
    belongs to an individual model, so it is read from ``primary``.  This lets
    the ensemble drive the search policies as well as greedy play.
    """

    ensemble: ProbabilitySource
    primary: ModelSource
    cfg: PolicyConfig
    teacher: BayesTeacher | None = None
    length: int | None = None

    def __post_init__(self) -> None:
        self._greedy = GreedyPolicy(self.ensemble)
        self._value = ValueGuidedPolicy(
            self.primary,
            topk=self.cfg.value_topk,
            max_lives=self.cfg.value_lookahead_max_lives,
            prob_source=self.ensemble,
        )
        inner = self._value if self.cfg.mode == "auto" else self._greedy
        self._exact = (
            ExactSearchPolicy(
                self.teacher,
                fallback=inner,
                max_candidates=self.cfg.exact_max_candidates,
                length=self.length,
                topk=self.cfg.value_topk,
                source=self.ensemble,
            )
            if self.teacher is not None and self.cfg.mode in ("exact", "auto")
            else None
        )

    @torch.no_grad()
    def __call__(self, board, lengths, present, absent, lives, active) -> torch.Tensor:
        if self.cfg.mode == "greedy":
            return self._greedy(board, lengths, present, absent, lives, active)
        if self.cfg.mode == "value" or self._exact is None:
            return self._value(board, lengths, present, absent, lives, active)
        # ExactSearchPolicy already falls through to its inner policy wherever
        # the candidate set is empty or too large, so its output is the answer.
        return self._exact(board, lengths, present, absent, lives, active)


def build_policy(
    source,
    cfg: PolicyConfig,
    teacher: BayesTeacher | None = None,
    length: int | None = None,
    primary: ModelSource | None = None,
):
    """Convenience factory used by evaluation and submission.

    ``source`` may be a single :class:`ModelSource` or an
    :class:`EnsembleSource`.  In the ensemble case ``primary`` supplies the
    auxiliary heads the search policies need.
    """
    if cfg.mode == "greedy":
        return GreedyPolicy(source)
    if isinstance(source, ModelSource):
        return AutoPolicy(source=source, cfg=cfg, teacher=teacher, length=length)
    if primary is None:
        raise ValueError("an ensemble source needs `primary` for the auxiliary heads")
    return HybridEnsemblePolicy(
        ensemble=source, primary=primary, cfg=cfg, teacher=teacher, length=length
    )
