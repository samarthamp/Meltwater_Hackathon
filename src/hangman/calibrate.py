"""Probability calibration and ensemble weighting.

Fits one temperature per source (Guo et al., 2017) and then non-negative
mixture weights over the sources, both by minimising presence NLL on
held-out validation states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .simulator import BatchGame


@dataclass
class CalibrationFit:
    """Fitted per-source temperatures and mixture weights."""

    temperatures: torch.Tensor
    weights: torch.Tensor
    nll_before: float
    nll_after: float

    def summary(self) -> str:
        temps = ", ".join(f"{t:.3f}" for t in self.temperatures.tolist())
        wts = ", ".join(f"{w:.3f}" for w in self.weights.tolist())
        return (
            f"temperatures=[{temps}]  weights=[{wts}]  "
            f"NLL {self.nll_before:.4f} -> {self.nll_after:.4f}"
        )


@torch.no_grad()
def collect_calibration_states(
    words: list[str],
    sources: list,
    policy_factory,
    device: str | torch.device = "cuda",
    max_states: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Play games under ``policy_factory`` and record every source's presence logits.

    Returns ``(logits, labels)`` where ``logits`` is ``(n_sources, S, 26)`` and
    ``labels`` is ``(S, 26)`` in ``{0, 1}``.  Only unguessed letters carry a
    label; guessed positions are marked with ``-1`` and skipped downstream.
    States come from on-policy play.
    """
    from collections import defaultdict

    by_length: dict[int, list[str]] = defaultdict(list)
    for word in words:
        by_length[len(word)].append(word)

    per_source: list[list[torch.Tensor]] = [[] for _ in sources]
    labels: list[torch.Tensor] = []
    total = 0

    for length in sorted(by_length):
        if total >= max_states:
            break
        group = by_length[length]
        game = BatchGame(group, device=device)
        policy = policy_factory(length)
        for _ in range(32):
            active = game.active()
            if not bool(active.any()) or total >= max_states:
                break
            idx = torch.nonzero(active, as_tuple=True)[0]
            board = game.board()[idx]
            lengths = game.lengths[idx]
            present = game.present[idx]
            absent = game.absent[idx]

            for s, source in enumerate(sources):
                probs = source.presence(board, lengths, present, absent)
                per_source[s].append(torch.logit(probs.clamp(1e-6, 1 - 1e-6)).cpu())

            truth = game.truth_presence[idx].to(torch.float32)
            truth = truth.masked_fill(present | absent, -1.0)
            labels.append(truth.cpu())
            total += int(idx.numel())

            guesses = policy(
                board=game.board(), lengths=game.lengths,
                present=game.present, absent=game.absent,
                lives=game.lives(), active=active,
            )
            game.apply(guesses, active)

    stacked = torch.stack([torch.cat(chunks) for chunks in per_source], dim=0)
    return stacked, torch.cat(labels)


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Masked binary NLL over valid (unguessed) letters."""
    valid = labels >= 0
    probs = torch.sigmoid(logits)[valid].clamp(1e-6, 1 - 1e-6)
    target = labels[valid]
    return float(-(target * probs.log() + (1 - target) * (1 - probs).log()).mean())


def fit_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature_grid: np.ndarray | None = None,
    weight_steps: int = 400,
    seed: int = 0,
) -> CalibrationFit:
    """Fit per-source temperatures then non-negative mixture weights.

    Temperatures are grid-searched per source.  The weights are
    softmax-parameterised and fitted with Adam on the NLL of the weighted
    logits.  ``nll_before`` is the NLL of the uniform average of the
    temperature-scaled logits.
    """
    n_sources = logits.shape[0]
    grid = temperature_grid if temperature_grid is not None else np.linspace(0.5, 2.5, 41)

    temperatures = torch.ones(n_sources)
    for s in range(n_sources):
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            value = _nll(logits[s] / float(t), labels)
            if value < best_nll:
                best_t, best_nll = float(t), value
        temperatures[s] = best_t

    scaled = logits / temperatures.view(-1, 1, 1)
    nll_before = _nll(scaled.mean(dim=0), labels)

    raw = torch.zeros(n_sources, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    valid = labels >= 0
    target = labels[valid]
    for _ in range(weight_steps):
        weights = torch.softmax(raw, dim=0).view(-1, 1, 1)
        pooled = (scaled * weights).sum(dim=0)
        probs = torch.sigmoid(pooled)[valid].clamp(1e-6, 1 - 1e-6)
        loss = -(target * probs.log() + (1 - target) * (1 - probs).log()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights = torch.softmax(raw, dim=0)
        nll_after = _nll((scaled * weights.view(-1, 1, 1)).sum(dim=0), labels)

    return CalibrationFit(
        temperatures=temperatures,
        weights=weights.detach(),
        nll_before=nll_before,
        nll_after=nll_after,
    )
