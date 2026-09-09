"""Three-tier Bayes posterior oracle - the label generator for every model.

Why a teacher at all
--------------------
Training a hangman net on hard 0/1 "is this letter in the word" labels throws
away almost all of the available information.  Given a board, the *exact*
posterior probability that each letter occurs can be computed by enumerating
the dictionary words consistent with the board, and that posterior is a far
richer target: it carries the inter-letter structure that hard labels destroy
(Hinton, Vinyals & Dean, 2015, *Distilling the Knowledge in a Neural Network*).
We use it as a soft target, i.e. we distil an exact solver into a network.

Leave-one-out
-------------
The played word is removed from its own candidate set.  Without this the
teacher can identify the answer outright and would teach the student to expect
information it will never have at test time - train and test dictionaries are
disjoint.  Leave-one-out makes the teacher face exactly the test condition.

The relaxation ladder
---------------------
Measured on this data, **44.4% of on-policy states have no consistent
dictionary word at all**.  A single exact posterior is therefore silent on
nearly half of the states a real game visits.  We recover signal with a ladder
of progressively weaker constraint sets, stopping at the first that yields
candidates:

  Tier 1  positional match  AND  no present letter hidden  AND  no absent letter
  Tier 2  positional match  AND  no absent letter                (drop present)
  Tier 3  positional match                                        (drop both)
  Tier 4  interpolated character n-gram backoff (see ngram.py)

Tiers 1-3 are exact posteriors over successively coarser candidate sets; Tier 4
is a smoothed generative model that is defined everywhere.  Their outputs are
blended by candidate count with a Jelinek-Mercer style weight
``lambda = n / (n + kappa)``, so the exact posterior dominates when it is well
supported and decays smoothly into the backoff when it is not.

Implementation
--------------
Everything is batched on the GPU.  The candidate filter for a batch of ``S``
states against a length bucket of ``N`` words is ``L`` elementwise passes over
an ``(S, N)`` boolean, and the posterior itself is a single ``(S, N) @ (N, 26)``
matmul.  This labels millions of states per minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import TeacherConfig
from .data import LengthBuckets
from .ngram import HOLE, NgramModel

#: Candidate filtering runs through matrix products.  Values are small integers
#: (position matches and letter counts, both <= 29), which fp16 represents
#: exactly, and torch accumulates fp16 matmuls in fp32 - so the "arithmetic"
#: filter is bit-exact with the elementwise one, just far faster.
MATMUL_DTYPE = torch.float16

#: Number of exact tiers in the relaxation ladder (n-gram backoff is separate).
N_EXACT_TIERS = 3


@dataclass
class TeacherOutput:
    """Soft targets for a batch of states.

    Attributes
    ----------
    presence : (B, 26) float32
        Blended posterior probability that each letter occurs in the word.
        Already-guessed letters are zeroed.
    n_candidates : (B,) int64
        Size of the Tier-1 consistent candidate set (0 when it is empty).
    tier : (B,) int8
        Which tier supplied the dominant exact posterior; ``N_EXACT_TIERS``
        means the n-gram backoff carried the state alone.
    """

    presence: torch.Tensor
    n_candidates: torch.Tensor
    tier: torch.Tensor


class BayesTeacher:
    """Batched exact-posterior oracle with an n-gram backoff."""

    def __init__(
        self,
        buckets: LengthBuckets,
        ngram: NgramModel,
        cfg: TeacherConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.cfg = cfg
        self.ngram = ngram
        self.device = torch.device(device)
        self.buckets = buckets

        # Dictionary resident on the accelerator, one entry per word length.
        # ``_onehot[L]`` is (N, L*26): word n has a 1 at ``i * 26 + c`` when its
        # i-th character is ``c``.  ``_counts[L]`` is (N, 26) occurrence counts.
        # Both exist so that candidate filtering is matrix multiplication rather
        # than a loop of elementwise comparisons - see ``_consistency``.
        self._chars: dict[int, torch.Tensor] = {}
        self._presence: dict[int, torch.Tensor] = {}
        self._onehot: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}
        for length in buckets.lengths:
            chars_np = buckets.chars[length].astype(np.int64)
            n_words = chars_np.shape[0]
            self._chars[length] = torch.from_numpy(chars_np).to(self.device)
            self._presence[length] = torch.from_numpy(
                buckets.presence[length]
            ).to(self.device, MATMUL_DTYPE)

            onehot = torch.zeros((n_words, length * 26), dtype=MATMUL_DTYPE, device=self.device)
            flat = self._chars[length] + torch.arange(
                0, length * 26, 26, device=self.device
            ).unsqueeze(0)
            onehot.scatter_(1, flat, 1.0)
            self._onehot[length] = onehot

            counts = torch.zeros((n_words, 26), dtype=MATMUL_DTYPE, device=self.device)
            counts.scatter_add_(
                1, self._chars[length], torch.ones_like(self._chars[length], dtype=MATMUL_DTYPE)
            )
            self._counts[length] = counts

    # -- exact tiers ---------------------------------------------------------

    def _consistency(
        self,
        length: int,
        boards: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
        tier: int,
    ) -> torch.Tensor:
        """``(S, N)`` bool mask of dictionary words consistent with each board.

        The three constraints are expressed as matrix products rather than as a
        loop over positions, which is what makes labelling millions of states
        practical (measured: ~50x faster than the elementwise formulation).

        1. *Positional match.*  With ``Q[s, i*26+c] = 1`` when position ``i`` is
           revealed showing letter ``c``, the product ``onehot @ Q.T`` counts how
           many revealed positions a word matches; a word is positionally
           consistent exactly when that count equals the number of revealed
           positions.
        2. *No surplus guessed letters.*  Positional match already forces a word
           to carry every revealed letter at its revealed positions, so the only
           remaining requirement is that it carries no **extra** occurrence of
           any guessed letter: ``counts @ guessed <= n_revealed``.  A guessed
           letter that is absent from the word has target count zero, so this
           single inequality covers both the present and the absent rule.
        3. *No absent letter at all*, ``counts @ absent == 0``, which is the
           relaxed rung used when the strict set empties.

        ``tier`` selects the rung: 0 applies 1+2, 1 applies 1+3, 2 applies 1.
        """
        onehot = self._onehot[length]                    # (N, L*26)
        counts = self._counts[length]                    # (N, 26)
        n_states = boards.shape[0]

        revealed = boards != HOLE                        # (S, L)
        n_revealed = revealed.sum(dim=1).to(MATMUL_DTYPE)
        query = torch.zeros((n_states, length * 26), dtype=MATMUL_DTYPE, device=self.device)
        offsets = torch.arange(0, length * 26, 26, device=self.device).unsqueeze(0)
        flat = boards.clamp(0, 25) + offsets
        query.scatter_(1, flat, revealed.to(MATMUL_DTYPE))

        matched = query @ onehot.T                       # (S, N)
        ok = matched >= n_revealed.unsqueeze(1) - 0.5

        if tier == 0:
            guessed = (present | absent).to(MATMUL_DTYPE)
            surplus = guessed @ counts.T                 # (S, N)
            ok &= surplus <= n_revealed.unsqueeze(1) + 0.5
        elif tier == 1:
            banned = absent.to(MATMUL_DTYPE) @ counts.T
            ok &= banned < 0.5
        return ok

    def _posterior_from_mask(self, length: int, ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a consistency mask into a letter-presence posterior."""
        counts = (ok.to(MATMUL_DTYPE) @ self._presence[length]).float()   # (S, 26)
        n_cand = ok.sum(dim=1)                                            # (S,)
        prior = self.cfg.presence_prior
        denom = (n_cand.to(torch.float32) + 2.0 * prior).clamp_min(1e-6).unsqueeze(1)
        return (counts + prior) / denom, n_cand

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def posterior(
        self,
        length: int,
        boards: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
        exclude_rows: torch.Tensor | None = None,
        ngram_scores: torch.Tensor | None = None,
    ) -> TeacherOutput:
        """Soft targets for a batch of same-length states.

        Parameters
        ----------
        length : int
            Word length shared by every state in the batch.
        boards : (S, L) int64
            Letter ids, ``HOLE`` where still hidden.
        present, absent : (S, 26) bool
            Guessed letters that did / did not occur.
        exclude_rows : (S,) int64, optional
            Row of the played word inside this length bucket, removed
            leave-one-out.  ``-1`` disables exclusion for that state.
        ngram_scores : (S, 26) float32, optional
            Precomputed Tier-4 evidence; computed on demand when omitted.
        """
        n_states = boards.shape[0]
        blended = torch.zeros((n_states, 26), dtype=torch.float32, device=self.device)
        weight_left = torch.ones(n_states, dtype=torch.float32, device=self.device)
        n_cand_tier1 = torch.zeros(n_states, dtype=torch.long, device=self.device)
        tier_used = torch.full((n_states,), N_EXACT_TIERS, dtype=torch.int8, device=self.device)

        if length in self._chars:
            kappa = self.cfg.backoff_kappa
            for tier in range(min(self.cfg.n_exact_tiers, N_EXACT_TIERS)):
                if weight_left.max() <= 1e-6:
                    break
                ok = self._consistency(length, boards, present, absent, tier)
                if exclude_rows is not None:
                    valid = exclude_rows >= 0
                    if valid.any():
                        rows = torch.nonzero(valid, as_tuple=True)[0]
                        ok[rows, exclude_rows[rows]] = False

                post, n_cand = self._posterior_from_mask(length, ok)
                if tier == 0:
                    n_cand_tier1 = n_cand

                lam = n_cand.to(torch.float32) / (n_cand.to(torch.float32) + kappa)
                take = weight_left * lam
                blended += take.unsqueeze(1) * post
                newly = (take > 1e-6) & (tier_used == N_EXACT_TIERS)
                tier_used[newly] = tier
                weight_left = weight_left - take

        if ngram_scores is None:
            ngram_scores = self._ngram_batch(length, boards, present, absent)
        # ``ngram_scores`` are presence marginals (noisy-OR pooled), i.e. on the
        # same scale as the exact posterior, so they mix without renormalising.
        blended += weight_left.unsqueeze(1) * ngram_scores

        guessed = present | absent
        blended = blended.masked_fill(guessed, 0.0)
        return TeacherOutput(presence=blended, n_candidates=n_cand_tier1, tier=tier_used)

    def _ngram_batch(
        self,
        length: int,
        boards: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
    ) -> torch.Tensor:
        """Tier-4 presence marginals, computed on the host and moved back."""
        boards_np = boards.detach().cpu().numpy().astype(np.int64)
        lengths_np = np.full(boards_np.shape[0], length, dtype=np.int64)
        scores = self.ngram.score_batch(
            boards_np,
            lengths_np,
            present.detach().cpu().numpy(),
            absent.detach().cpu().numpy(),
        )
        return torch.from_numpy(scores).to(self.device, torch.float32)

    @torch.no_grad()
    def greedy_guess(
        self,
        length: int,
        boards: torch.Tensor,
        present: torch.Tensor,
        absent: torch.Tensor,
        exclude_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The teacher's own move: argmax of the blended posterior."""
        out = self.posterior(length, boards, present, absent, exclude_rows)
        return out.presence.argmax(dim=1)
