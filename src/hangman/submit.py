"""Produce and validate ``submission.csv``.

The competition wants, for each test word, the chronological string of letters
our policy would guess if it played that word.  We therefore *simulate* every
game locally under the exact scoring rules and record what the policy actually
did, exactly as the published blueprint describes.

Because the public test words are known, the same simulation yields our own
score, which is the number reported in the model report.  The test words are
never used to fit anything - see reports/COMPLIANCE.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .config import MAX_WRONG_GUESSES
from .simulator import play_batch, reference_play

EXPECTED_ROWS = 250_000
ALPHABET = set("abcdefghijklmnopqrstuvwxyz")


@dataclass
class SubmissionStats:
    n_rows: int
    n_won: int
    total_wrong: int
    mean_guesses: float

    @property
    def win_rate(self) -> float:
        return 100.0 * self.n_won / max(self.n_rows, 1)

    @property
    def leaderboard_score(self) -> float:
        return self.win_rate - self.total_wrong / 1e8

    def summary(self) -> str:
        return (
            f"rows={self.n_rows:,}  win={self.win_rate:.3f}%  "
            f"E={self.total_wrong:,}  leaderboard={self.leaderboard_score:.6f}  "
            f"mean guesses={self.mean_guesses:.2f}"
        )


def generate_submission(
    words: list[str],
    policy_factory,
    out_path: str | Path,
    device: str | torch.device = "cuda",
    max_batch: int = 8192,
    verbose: bool = True,
) -> tuple[Path, SubmissionStats]:
    """Play every test word and write the guess strings to ``out_path``.

    Words are grouped by length for batching, then restored to the original
    order: ``word_id`` must match the line index of ``test.txt`` exactly.
    """
    by_length: dict[int, list[int]] = defaultdict(list)
    for idx, word in enumerate(words):
        by_length[len(word)].append(idx)

    guesses: list[str | None] = [None] * len(words)
    n_won = total_wrong = total_guesses = 0

    for length in sorted(by_length):
        ids = by_length[length]
        policy = policy_factory(length)
        for offset in range(0, len(ids), max_batch):
            chunk_ids = ids[offset : offset + max_batch]
            chunk_words = [words[i] for i in chunk_ids]
            result = play_batch(chunk_words, policy, device=device)
            for local, word_id in enumerate(chunk_ids):
                guesses[word_id] = result.guess_strings()[local]
            n_won += int(result.won.sum().item())
            total_wrong += result.total_wrong()
            total_guesses += int(result.n_guesses.sum().item())
        if verbose:
            print(f"  len {length:>2}: {len(ids):>7,} words done", flush=True)

    missing = [i for i, g in enumerate(guesses) if g is None]
    if missing:
        raise RuntimeError(f"{len(missing)} words produced no guess string")

    frame = pd.DataFrame(
        {"word_id": range(len(words)), "guessed_letters_string": guesses}
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    stats = SubmissionStats(
        n_rows=len(words),
        n_won=n_won,
        total_wrong=total_wrong,
        mean_guesses=total_guesses / max(len(words), 1),
    )
    return out_path, stats


def validate_submission(
    path: str | Path,
    words: list[str],
    expected_rows: int = EXPECTED_ROWS,
) -> SubmissionStats:
    """Re-read the CSV and re-score it from scratch against the rules.

    This deliberately does not trust the generator: it reloads the file, replays
    every string with the naive reference scorer, and re-derives the win rate.
    A mismatch between this and the generator's own count would mean the
    submission does not say what we think it says.
    """
    frame = pd.read_csv(path, dtype={"word_id": int, "guessed_letters_string": str})
    frame["guessed_letters_string"] = frame["guessed_letters_string"].fillna("")

    problems: list[str] = []
    if list(frame.columns) != ["word_id", "guessed_letters_string"]:
        problems.append(f"unexpected columns {list(frame.columns)}")
    if len(frame) != expected_rows:
        problems.append(f"expected {expected_rows:,} rows, found {len(frame):,}")
    if frame["word_id"].tolist() != list(range(len(frame))):
        problems.append("word_id is not 0..N-1 in order")

    n_won = total_wrong = total_guesses = 0
    for row_id, (word, guess_string) in enumerate(zip(words, frame["guessed_letters_string"])):
        if not set(guess_string) <= ALPHABET:
            problems.append(f"row {row_id}: non-alphabet character in {guess_string!r}")
            break
        if len(set(guess_string)) != len(guess_string):
            problems.append(f"row {row_id}: repeated letter in {guess_string!r}")
            break
        won, wrong, consumed = reference_play(word, guess_string)
        n_won += int(won)
        total_wrong += wrong
        total_guesses += consumed
        if not won and wrong < MAX_WRONG_GUESSES:
            problems.append(
                f"row {row_id}: string exhausted at {wrong} wrong guesses without solving "
                f"{word!r} - the policy gave up early"
            )
            break

    if problems:
        raise ValueError("submission validation failed:\n  " + "\n  ".join(problems))

    return SubmissionStats(
        n_rows=len(frame),
        n_won=n_won,
        total_wrong=total_wrong,
        mean_guesses=total_guesses / max(len(frame), 1),
    )


def save_stats(stats: SubmissionStats, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_rows": stats.n_rows,
                "n_won": stats.n_won,
                "win_rate": stats.win_rate,
                "total_wrong": stats.total_wrong,
                "leaderboard_score": stats.leaderboard_score,
                "mean_guesses": stats.mean_guesses,
            },
            indent=2,
        )
    )
    return path
