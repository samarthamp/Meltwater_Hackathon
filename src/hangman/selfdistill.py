"""Self-distillation: using the trained student as a teacher for the next one.

Why
---
The Bayes oracle that labelled the first generation is *worse than the model it
trained*. Measured on the states where it matters - oracle-blind and short-word
states - top-1 hit rate:

    Bayes teacher   0.656
    student         0.744

So the best target available is no longer the dictionary posterior. This is the
setting Born-Again Networks describes (Furlanello et al., 2018, *Born-Again
Neural Networks*): retrain from scratch against the previous generation's own
predictions, which carry generalisation the original labels did not.

Blending, and why it is gated
-----------------------------
The student is not uniformly better. Where the consistent candidate set is
large, the exact posterior is a genuinely exact answer and the student is only
approximating it; where the set is empty - 45% of on-policy states - the oracle
says nothing at all and only the student has an opinion. So the blend is gated
on candidate count, reusing the Jelinek-Mercer form already used for the
teacher's own backoff in :mod:`hangman.teacher`:

    lambda = n_cand / (n_cand + kappa)      # weight on the Bayes posterior
    target = lambda * bayes + (1 - lambda) * student

``lambda`` is the weight on the *exact* source, matching the sense of the
teacher's own backoff in :mod:`hangman.teacher`.  So ``kappa -> 0`` is the pure
Bayes posterior and ``kappa -> inf`` is the pure student: a larger gate means
"trust the dictionary less".  The value is chosen by measurement in
``tools/blend_sweep.py``, not by taste.
"""

from __future__ import annotations

import numpy as np
import torch

#: Gate value meaning "ignore the dictionary posterior entirely".
PURE_STUDENT = float("inf")
#: Gate value meaning "ignore the student entirely" - the generation-one target.
PURE_BAYES = 0.0


def blend_weight(n_candidates: torch.Tensor, kappa: float) -> torch.Tensor:
    """Weight on the Bayes posterior, in ``[0, 1]``, per state.

    ``kappa == 0`` puts everything on the Bayes posterior; ``kappa == inf`` puts
    everything on the student.  In between, states with more consistent
    candidates lean on the dictionary and states with fewer lean on the model.
    """
    n = n_candidates.to(torch.float32)
    if kappa <= 0.0:
        return torch.ones_like(n)
    if not np.isfinite(kappa):
        return torch.zeros_like(n)
    return n / (n + kappa)


def blend_targets(
    bayes: torch.Tensor,
    student: torch.Tensor,
    n_candidates: torch.Tensor,
    kappa: float,
    guessed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine an exact posterior with a student's predictions.

    Parameters
    ----------
    bayes, student : (B, 26) float - presence probabilities from each source.
    n_candidates : (B,) - size of the Tier-1 consistent candidate set.
    kappa : gate constant; see module docstring.
    guessed : (B, 26) bool, optional - already-guessed letters, zeroed on output.
    """
    lam = blend_weight(n_candidates, kappa).unsqueeze(1)
    out = lam * bayes.float() + (1.0 - lam) * student.float()
    if guessed is not None:
        out = out.masked_fill(guessed, 0.0)
    return out.clamp(0.0, 1.0)


def top1_hit_rate(
    scores: torch.Tensor, guessed: torch.Tensor, truth: torch.Tensor
) -> float:
    """Fraction of states whose top *unguessed* letter is really in the word.

    The metric used throughout to compare label sources, because it is what a
    greedy policy trained on those labels would actually do.
    """
    pick = scores.masked_fill(guessed, -1e9).argmax(dim=1)
    return float(truth.gather(1, pick.unsqueeze(1)).mean().item())


def describe(kappa: float) -> str:
    """Human-readable name for a gate setting, for logs and tables."""
    if kappa <= 0.0:
        return "pure Bayes"
    if not np.isfinite(kappa):
        return "pure student"
    return f"gated kappa={kappa:g}"
