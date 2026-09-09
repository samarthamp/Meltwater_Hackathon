#!/usr/bin/env python
"""Where does the shipped model lose games, and what would it take to win them?

Two questions:

1. **Where is the loss mass?**  Not the win *rate* by length - the raw count of
   losses, which is what a fix has to move.
2. **How close were the losses?**  A game lost with one letter still hidden is a
   different problem from one lost with five.

It also runs a deliberately illegitimate control - dictionary filtering where
the candidate set *contains the answer* - purely to establish what score that
approach yields, so a leaderboard number can be interpreted.  Nothing in the
submission path uses it; see the warning printed at the top of that section.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

import numpy as np
import pandas as pd
import torch

from hangman.config import METRIC_DIR, DataConfig, TeacherConfig, ensure_dirs
from hangman.data import build_dictionaries, build_length_buckets, load_test_words
from hangman.ngram import HOLE, NgramModel
from hangman.simulator import BatchGame, play_batch, reference_play
from hangman.teacher import BayesTeacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--oracle-sample", type=int, default=20000)
    args = parser.parse_args()
    ensure_dirs()

    test = load_test_words(DataConfig())
    frame = pd.read_csv(args.submission, dtype={"word_id": int, "guessed_letters_string": str})
    guesses = frame["guessed_letters_string"].fillna("").tolist()

    won, wrong, revealed_frac, missing = [], [], [], []
    for word, guess in zip(test, guesses):
        w, e, _ = reference_play(word, guess)
        won.append(w)
        wrong.append(e)
        letters = set(word)
        got = letters & set(guess)
        revealed_frac.append(len(got) / len(letters))
        missing.append(len(letters - got))
    won = np.array(won)
    wrong = np.array(wrong)
    revealed_frac = np.array(revealed_frac)
    missing = np.array(missing)
    lengths = np.array([len(w) for w in test])

    print("=" * 72)
    print("1. WHERE THE LOSSES ARE")
    print("=" * 72)
    total_losses = int((~won).sum())
    print(f"\n{total_losses:,} losses out of {len(test):,} games ({100*(~won).mean():.2f}%)\n")
    print(f"{'len':>4} {'games':>8} {'losses':>8} {'win%':>7} {'% of all losses':>16} {'cum%':>7}")
    cum = 0.0
    rows = []
    for L in range(1, 30):
        m = lengths == L
        if m.sum() == 0:
            continue
        losses = int((~won[m]).sum())
        share = 100 * losses / total_losses
        cum += share
        rows.append({"length": L, "games": int(m.sum()), "losses": losses,
                     "win_rate": float(100 * won[m].mean()), "loss_share": share})
        print(f"{L:>4} {m.sum():>8,} {losses:>8,} {100*won[m].mean():>6.1f}% "
              f"{share:>15.1f}% {cum:>6.1f}%")

    print("\n" + "=" * 72)
    print("2. HOW CLOSE WERE THE LOSSES")
    print("=" * 72)
    lost = ~won
    print(f"\nOf {total_losses:,} lost games, letters still hidden at the end:\n")
    hist = collections.Counter(missing[lost].tolist())
    for k in sorted(hist):
        print(f"  {k} letter{'s' if k != 1 else ' '} missing : {hist[k]:>7,}  "
              f"({100*hist[k]/total_losses:5.1f}%)")
    print(f"\n  mean fraction of the word's distinct letters found before losing: "
          f"{revealed_frac[lost].mean():.3f}")
    one_away = int((missing == 1).sum())
    print(f"  games lost with exactly ONE letter left: {one_away:,} "
          f"({100*one_away/len(test):.2f} points of win rate)")

    print("\n" + "=" * 72)
    print("3. WHAT DOES A LEAKED CANDIDATE SET SCORE?")
    print("=" * 72)
    print("""
    DIAGNOSTIC ONLY - NOT USED BY THE SUBMISSION.

    This runs plain dictionary filtering with the candidate set built from
    test.txt itself, so the answer is guaranteed to be in it.  The competition
    calls this a "lookup leak" and says it is disqualifying.  It is measured
    here for one reason: to establish what a leaderboard score of that shape
    implies, and to check whether such a score is reachable legitimately.
    """)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(test), size=min(args.oracle_sample, len(test)), replace=False)
    sample = [test[i] for i in sample_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TeacherConfig()
    leak_buckets = build_length_buckets(test)
    ngram = NgramModel(tcfg).fit(test, max_len=29, min_count=2)
    leak = BayesTeacher(leak_buckets, ngram, tcfg, device=device)

    by_len: dict[int, list[str]] = collections.defaultdict(list)
    for w in sample:
        by_len[len(w)].append(w)

    def factory(length: int):
        def fn(board, lengths, present, absent, lives, active):
            trimmed = board[:, :length].clamp_min(0)
            # No leave-one-out: the answer stays in the candidate set.
            return leak.posterior(length, trimmed, present, absent).presence.argmax(dim=1)
        return fn

    n = w_ = 0
    for length, group in sorted(by_len.items()):
        res = play_batch(group, factory(length), device=device)
        n += len(group)
        w_ += int(res.won.sum().item())
    leak_rate = 100 * w_ / n
    print(f"    leaked-dictionary filtering: {leak_rate:.2f}%  (n={n:,})")
    print(f"    our submitted model:         {100*won.mean():.2f}%")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "failure_analysis.json").write_text(json.dumps({
        "submission_win_rate": float(100 * won.mean()),
        "total_losses": total_losses,
        "by_length": rows,
        "missing_letters_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "lost_with_one_letter_left": one_away,
        "mean_revealed_fraction_when_lost": float(revealed_frac[lost].mean()),
        "leaked_dictionary_win_rate": float(leak_rate),
        "leaked_dictionary_n": int(n),
    }, indent=2))


if __name__ == "__main__":
    main()
