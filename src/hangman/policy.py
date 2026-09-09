"""Guess-selection policies.

Taking ``argmax P(letter is in the word)`` is the obvious policy and it is not
the optimal one.  The game does not reward hits, it rewards *finishing the word
within six misses*, and those objectives come apart in two situations that this
dataset makes common:

* **Late in a game.**  The measured average is 4.0 wrong guesses out of six per
  game, so most games end near the wall.  With one life left the right move is
  whatever maximises the chance of surviving *and* finishing, which is not
  always the single most likely letter - a slightly less likely letter that
  reveals three positions can be worth more than a marginally likelier one that
  reveals one.
* **When the candidate set is small.**  Once the board pins the answer down to
  a handful of dictionary words, marginal probabilities throw away the joint
  structure.  Exact search over the information set is both cheap and better
  (Meng & Lucas, 2024, *Deduction Game Framework and Information Set Entropy
  Search*; Reams, 2009, *Playing Perfect Hangman*).

Three policies are provided and ``AutoPolicy`` dispatches between them per
state, which is the configuration selected on the validation dictionary.
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
    #: Blend between the presence head and the noisy-OR pooling of the
    #: per-position head.  Fitted per member by scripts/07_ensemble_eval.py;
    #: 0.5 is the default because the measured optimum sits near 0.6 - the
    #: per-position head, pooled correctly, turns out to be a *better* presence
    #: estimator than the head trained directly for the job (0.0 -> 61.0%,
    #: 0.6 -> 64.3% on 8,000 held-out words).
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

    Averaging in logit space (rather than probability space) is the standard
    choice for a *decision* ensemble: it keeps confident-and-correct members
    from being dragged toward the mean by an uncertain one.  Weights and
    temperatures are fitted on the validation dictionary in
    :mod:`hangman.calibrate`.
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

    For each candidate letter ``l`` we evaluate

        EV(l) = p_l * V(state after a hit) + (1 - p_l) * V(state after a miss)

    where the hit successor is constructed by revealing every position whose
    per-position distribution favours ``l``.  This costs ``2k`` extra forward
    passes on the subset of states where it is enabled, which is why it is
    restricted to low-life states - the ones where the decision is actually
    tight.
    """

    source: ModelSource
    topk: int = 5
    max_lives: int = 2
    #: Optional separate source for the *letter probabilities*.  The value and
    #: count heads belong to one model, but the choice of which letters to
    #: evaluate is better made by the calibrated ensemble, so the two can be
    #: split.  Defaults to using ``source`` for both.
    prob_source: ProbabilitySource | None = None
    #: The value head is only trained on states that carry a Monte-Carlo return,
    #: which off-policy warm-start records do not.  Before any DAgger round it is
    #: therefore constant (measured: mean 0.5155, std 0.0085) and one-ply
    #: lookahead degenerates into picking at random among the top-k - which cost
    #: 13.5 points of win rate before this guard existed.  If the head shows no
    #: spread across the batch, fall back to greedy.
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
            # Optimistically reveal the positions the model believes hold the
            # letter; the value head only needs the successor's shape, not its
            # exact contents.
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
    """Exact information-set refinement of the model's top-k candidates.

    Once the board pins the answer down to a handful of dictionary-consistent
    words, marginal probabilities throw away the joint structure: two letters
    can be equally likely to appear yet differ sharply in how much they would
    tell us.  This policy adds that missing term.

    It is a *hybrid*, not a replacement.  The candidate set is built from the
    training dictionary, which shares no words with the test set, so its
    marginals are a prior and not an answer - and 45% of on-policy states have
    no consistent word at all.  The model's probability therefore stays the
    primary signal and exact search only re-ranks its top ``topk`` letters:

        score(l) = log p_model(l) - lambda * E[|C'| | guess l] / |C|

    ``E[|C'|]`` is the expected size of the surviving candidate set, which is
    the classic near-optimal Hangman criterion (Reams, 2009) and the entropy
    term of information-set search (Meng & Lucas, 2024) in its second-moment
    form.

    Computing it without a per-state loop
    -------------------------------------
    Guessing ``l`` partitions the candidates by *reveal signature* - which
    positions light up - and ``E[|C'|] = (sum_s n_s^2) / n``.  Two candidates
    share a signature exactly when their occurrence indicator rows for ``l``
    are identical, and two 0/1 rows are identical exactly when

        rowsum(a) + rowsum(b) - 2 * <a, b> == 0

    so the whole count is a batched matmul plus a comparison.  That turns what
    would be a Python loop over states and letters into ``topk`` batched
    operations - the difference between hours and seconds over the test set.
    """

    teacher: BayesTeacher
    fallback: object
    max_candidates: int = 200
    length: int | None = None
    topk: int = 5
    lambda_size: float = 0.35
    source: ProbabilitySource | None = None
    #: Penalty applied to a letter that no consistent candidate contains.  The
    #: candidate set comes from a dictionary that shares no words with the test
    #: set, so "no candidate has this letter" is weak evidence at best; measured
    #: on held-out words, a large penalty here *costs* win rate.  Default 0.
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

    The dispatch thresholds are chosen on the validation dictionary, never on
    ``test.txt``.
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
    """The shipped configuration: ensemble scores, primary-member lookahead.

    Presence probabilities come from the calibrated ensemble, because that is
    the best estimate available.  The auxiliary heads the lookahead consumes -
    the value head and the count head - belong to individual models, so they are
    read from the primary member.  Splitting the two is what lets the ensemble
    drive the decision in the states where the search policies apply, instead of
    being bypassed there.
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
