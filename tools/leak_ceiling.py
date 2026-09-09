#!/usr/bin/env python
"""What is the ceiling when the candidate set contains the answer?

DIAGNOSTIC ONLY.  Nothing here is used by the submission.  The competition
calls a candidate set built from the evaluation words a "lookup leak" and says
it is disqualifying; this exists to establish what such an approach *scores*,
so that a leaderboard number can be interpreted rather than guessed at.

Plain max-marginal filtering over the leaked set reaches 75.9%.  That is well
short of a 94.5 leaderboard, so the question is whether a better *decision rule*
over the same leaked set closes the gap.  Three are compared:

* ``marginal``  - guess the letter most candidates contain (the usual approach);
* ``eliminate`` - guess the letter minimising the expected number of surviving
  candidates, the information-greedy rule;
* ``survive``   - maximise ``P(hit)`` and break ties by elimination, which
  respects the fact that a miss costs a life and a hit does not.

Expected surviving candidates is computed exactly and batched.  Guessing a
letter partitions the candidates by *which positions* it occupies, and
``E[|C'|] = sum_s n_s^2 / n``.  Mapping each candidate's position-pattern for a
letter to a dense class index turns the group counts into a single matmul
against a one-hot class matrix, so all states and all letters are scored
without a Python loop.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch

from hangman.config import DataConfig, METRIC_DIR, ensure_dirs
from hangman.data import build_length_buckets, load_test_words, load_words
from hangman.ngram import HOLE
from hangman.simulator import play_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LeakedSolver:
    """Exact candidate filtering over a dictionary that contains the answer."""

    def __init__(self, words: list[str]) -> None:
        buckets = build_length_buckets(words)
        self.chars: dict[int, torch.Tensor] = {}
        self.presence: dict[int, torch.Tensor] = {}
        self.onehot: dict[int, torch.Tensor] = {}
        self.counts: dict[int, torch.Tensor] = {}
        self.classes: dict[int, list[torch.Tensor]] = {}
        for length in buckets.lengths:
            mat = torch.from_numpy(buckets.chars[length].astype(np.int64)).to(DEVICE)
            n = mat.shape[0]
            self.chars[length] = mat
            self.presence[length] = torch.from_numpy(buckets.presence[length]).to(DEVICE, torch.float16)
            oh = torch.zeros((n, length * 26), dtype=torch.float16, device=DEVICE)
            oh.scatter_(1, mat + torch.arange(0, length * 26, 26, device=DEVICE).unsqueeze(0), 1.0)
            self.onehot[length] = oh
            cnt = torch.zeros((n, 26), dtype=torch.float16, device=DEVICE)
            cnt.scatter_add_(1, mat, torch.ones_like(mat, dtype=torch.float16))
            self.counts[length] = cnt
            # Per letter: dense class index of each word's position-pattern.
            per_letter = []
            powers = (2 ** torch.arange(length, device=DEVICE)).unsqueeze(0)
            for letter in range(26):
                code = ((mat == letter).to(torch.int64) * powers).sum(dim=1)
                _, dense = torch.unique(code, return_inverse=True)
                per_letter.append(dense)
            self.classes[length] = per_letter

    def consistent(self, length, boards, present, absent) -> torch.Tensor:
        onehot, counts = self.onehot[length], self.counts[length]
        revealed = boards != HOLE
        n_revealed = revealed.sum(dim=1).to(torch.float16)
        query = torch.zeros((boards.shape[0], length * 26), dtype=torch.float16, device=DEVICE)
        offsets = torch.arange(0, length * 26, 26, device=DEVICE).unsqueeze(0)
        query.scatter_(1, boards.clamp(0, 25) + offsets, revealed.to(torch.float16))
        ok = (query @ onehot.T) >= n_revealed.unsqueeze(1) - 0.5
        guessed = (present | absent).to(torch.float16)
        ok &= (guessed @ counts.T) <= n_revealed.unsqueeze(1) + 0.5
        return ok

    def guess(self, length, boards, present, absent, rule: str) -> torch.Tensor:
        ok = self.consistent(length, boards, present, absent)
        okf = ok.to(torch.float16)
        n_cand = okf.sum(dim=1).clamp_min(1.0).float()
        hits = (okf @ self.presence[length]).float()          # (S, 26)
        p_hit = hits / n_cand.unsqueeze(1)
        guessed = present | absent

        if rule == "marginal":
            return p_hit.masked_fill(guessed, -1.0).argmax(dim=1)

        # Expected surviving candidates per letter, exactly.
        expected = torch.zeros_like(p_hit)
        for letter in range(26):
            dense = self.classes[length][letter]
            k = int(dense.max().item()) + 1
            onehot_cls = torch.zeros((dense.numel(), k), dtype=torch.float16, device=DEVICE)
            onehot_cls[torch.arange(dense.numel(), device=DEVICE), dense] = 1.0
            per_class = (okf @ onehot_cls).float()             # (S, k)
            expected[:, letter] = (per_class ** 2).sum(dim=1) / n_cand

        if rule == "eliminate":
            score = -expected
        elif rule == "survive":
            # A miss costs a life, a hit does not: rank by P(hit) first and use
            # elimination only to separate letters of similar hit probability.
            score = p_hit - 0.02 * (expected / n_cand.unsqueeze(1))
        else:
            raise ValueError(rule)
        return score.masked_fill(guessed, -1e9).argmax(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--rules", nargs="*", default=["marginal", "eliminate", "survive"])
    args = parser.parse_args()
    ensure_dirs()

    test = load_test_words(DataConfig())
    train = load_words("Data/train.txt")
    rng = np.random.default_rng(0)
    sample = [test[i] for i in rng.choice(len(test), size=min(args.sample, len(test)), replace=False)]
    by_len: dict[int, list[str]] = collections.defaultdict(list)
    for w in sample:
        by_len[len(w)].append(w)

    results = {}
    for name, dictionary, note in (
        ("LEAKED (candidate set = test.txt, contains the answer)", test, "disqualifying"),
        ("LEGITIMATE (candidate set = train.txt, answer absent)", train, "allowed"),
    ):
        solver = LeakedSolver(dictionary)
        print(f"\n{name}  [{note}]")
        for rule in args.rules:
            n = won = 0
            for length, group in sorted(by_len.items()):
                if length not in solver.chars:
                    continue

                def fn(board, lengths, present, absent, lives, active, L=length, r=rule):
                    return solver.guess(L, board[:, :L].clamp_min(0), present, absent, r)

                res = play_batch(group, fn, device=DEVICE)
                n += len(group)
                won += int(res.won.sum().item())
            rate = 100 * won / max(n, 1)
            results[f"{note}/{rule}"] = rate
            print(f"    {rule:<10} {rate:6.2f}%   (n={n:,})")

    (METRIC_DIR / "leak_ceiling.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
