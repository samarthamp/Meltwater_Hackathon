#!/usr/bin/env python
"""Break down where the shipped model loses games.

Reports the raw loss count by word length (what a fix has to move) and how
close each lost game came: how many distinct letters were still hidden.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import pandas as pd

from hangman.config import METRIC_DIR, DataConfig, ensure_dirs
from hangman.data import load_test_words
from hangman.simulator import reference_play


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="submission.csv")
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

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "failure_analysis.json").write_text(json.dumps({
        "submission_win_rate": float(100 * won.mean()),
        "total_losses": total_losses,
        "by_length": rows,
        "missing_letters_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "lost_with_one_letter_left": one_away,
        "mean_revealed_fraction_when_lost": float(revealed_frac[lost].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
