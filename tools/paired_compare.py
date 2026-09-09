#!/usr/bin/env python
"""Compare two guess policies on the same words, with a paired test.

Two win rates each carrying a +/-0.35 error bar look inseparable when they
differ by 0.6.  But they are measured on the *same* words, so the right
statistic is the paired one: of the games where the two disagree, how lopsided
is the split?  That is McNemar's test, and it is far more sensitive than
comparing two independent intervals.

    python tools/paired_compare.py --a artifacts/checkpoints/former_a_final.pt \
                                   --b artifacts/checkpoints/former_v2_final.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np
import torch

from hangman.config import METRIC_DIR, ensure_dirs
from hangman.pipeline import build_teacher, set_seed, split_validation
from hangman.policy import EnsembleSource, GreedyPolicy, ModelSource
from hangman.simulator import play_batch
from hangman.train import load_checkpoint


def play(words, source, device) -> np.ndarray:
    """Per-word win flags, in the original word order."""
    by_len: dict[int, list[int]] = collections.defaultdict(list)
    for i, w in enumerate(words):
        by_len[len(w)].append(i)
    won = np.zeros(len(words), dtype=bool)
    for length, ids in sorted(by_len.items()):
        group = [words[i] for i in ids]
        res = play_batch(group, GreedyPolicy(source), device=device)
        flags = res.won.cpu().numpy()
        for local, idx in enumerate(ids):
            won[idx] = flags[local]
    return won


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Returns (a-only wins, b-only wins, two-sided p) for paired binary outcomes."""
    only_a = int((a & ~b).sum())
    only_b = int((b & ~a).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    # Normal approximation to the binomial, continuity-corrected.
    z = (abs(only_a - only_b) - 1) / math.sqrt(n)
    p = math.erfc(z / math.sqrt(2))
    return only_a, only_b, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="baseline checkpoint")
    parser.add_argument("--b", default=None, help="second member; with --ensemble, a+b uniform")
    parser.add_argument("--nor-a", type=float, default=0.25)
    parser.add_argument("--nor-b", type=float, default=0.0)
    parser.add_argument("--fit-words", type=int, default=5000)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_a, cfg = load_checkpoint(args.a, use_ema=True)
    src_a = ModelSource(model_a, noisy_or_weight=args.nor_a)
    dicts, _, _ = build_teacher(cfg)
    _, words = split_validation(dicts.val, args.fit_words)
    print(f"comparing on {len(words):,} held-out words (disjoint from the fitting split)\n")

    configs = {"A: former_a alone": src_a}
    if args.b:
        model_b, _ = load_checkpoint(args.b, use_ema=True)
        src_b = ModelSource(model_b, noisy_or_weight=args.nor_b)
        configs["B: former_v2 alone"] = src_b
        configs["A+B uniform ensemble"] = EnsembleSource([src_a, src_b])

    outcomes = {}
    for name, source in configs.items():
        won = play(words, source, cfg.device)
        outcomes[name] = won
        print(f"  {name:<26} {100*won.mean():6.2f}%  ({won.sum():,}/{len(words):,})")

    base = "A: former_a alone"
    print()
    results = {"n_words": len(words),
               "win_rates": {k: float(100 * v.mean()) for k, v in outcomes.items()}}
    for name, won in outcomes.items():
        if name == base:
            continue
        only_a, only_b, p = mcnemar(outcomes[base], won)
        delta = 100 * (won.mean() - outcomes[base].mean())
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"  {name} vs {base}:")
        print(f"    delta {delta:+.2f} points   "
               f"{base.split(':')[0]} only: {only_a:,}   {name.split(':')[0]} only: {only_b:,}   "
               f"p = {p:.2e}  ({verdict})")
        results[f"vs_{name}"] = {"delta": delta, "only_base": only_a,
                                 "only_other": only_b, "p_value": p}

    (METRIC_DIR / "paired_compare.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
