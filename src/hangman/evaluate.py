"""Evaluation: win rate, efficiency, and per-length breakdowns.

Plays complete games under the competition rules rather than scoring
next-letter accuracy.  Words are grouped by length so policies receive
same-length batches; the grouping does not affect the result.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .simulator import GameResult, play_batch


@dataclass
class EvalReport:
    """Aggregated results of a full evaluation pass."""

    n_words: int
    n_won: int
    total_wrong: int
    total_guesses: int
    per_length: dict[int, tuple[int, int]] = field(default_factory=dict)
    seconds: float = 0.0
    tag: str = ""

    @property
    def win_rate(self) -> float:
        return 100.0 * self.n_won / max(self.n_words, 1)

    @property
    def avg_wrong(self) -> float:
        return self.total_wrong / max(self.n_words, 1)

    @property
    def leaderboard_score(self) -> float:
        """``Score - E / 1e8``, the published ranking formula."""
        return self.win_rate - self.total_wrong / 1e8

    @property
    def stderr(self) -> float:
        """Standard error of the win rate, in percentage points."""
        p = self.win_rate / 100.0
        return 100.0 * (p * (1.0 - p) / max(self.n_words, 1)) ** 0.5

    def summary(self) -> str:
        return (
            f"{self.tag or 'eval'}: win {self.win_rate:.2f}% +/- {self.stderr:.2f}  "
            f"avg wrong {self.avg_wrong:.2f}  E={self.total_wrong:,}  "
            f"n={self.n_words:,}  ({self.seconds:.1f}s)"
        )

    def length_table(self, min_n: int = 5) -> str:
        lines = [f"{'len':>4} {'n':>7} {'win%':>7}"]
        for length in sorted(self.per_length):
            n, won = self.per_length[length]
            if n >= min_n:
                lines.append(f"{length:>4} {n:>7,} {100.0 * won / n:>7.2f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "n_words": self.n_words,
            "n_won": self.n_won,
            "win_rate": self.win_rate,
            "stderr": self.stderr,
            "avg_wrong": self.avg_wrong,
            "total_wrong": self.total_wrong,
            "leaderboard_score": self.leaderboard_score,
            "seconds": self.seconds,
            "per_length": {str(k): list(v) for k, v in sorted(self.per_length.items())},
        }


def evaluate_words(
    words: list[str],
    policy_factory,
    device: str | torch.device = "cuda",
    tag: str = "",
    max_batch: int = 8192,
    verbose: bool = False,
) -> EvalReport:
    """Play every word in ``words`` and aggregate the outcome.

    ``policy_factory(length)`` returns a :class:`~hangman.simulator.GuessFn` for
    that word length.
    """
    by_length: dict[int, list[str]] = defaultdict(list)
    for word in words:
        by_length[len(word)].append(word)

    report = EvalReport(n_words=0, n_won=0, total_wrong=0, total_guesses=0, tag=tag)
    start = time.time()
    for length in sorted(by_length):
        group = by_length[length]
        policy = policy_factory(length)
        won = wrong = guesses = 0
        for offset in range(0, len(group), max_batch):
            chunk = group[offset : offset + max_batch]
            result: GameResult = play_batch(chunk, policy, device=device)
            won += int(result.won.sum().item())
            wrong += result.total_wrong()
            guesses += int(result.n_guesses.sum().item())
        report.per_length[length] = (len(group), won)
        report.n_words += len(group)
        report.n_won += won
        report.total_wrong += wrong
        report.total_guesses += guesses
        if verbose:
            print(f"    len {length:>2}: n={len(group):>6,} win={100.0*won/len(group):5.1f}%", flush=True)
    report.seconds = time.time() - start
    return report


def save_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


class AblationTable:
    """Accumulates one evaluation row per configuration for an ablation table."""

    def __init__(self) -> None:
        self.rows: list[EvalReport] = []

    def add(self, report: EvalReport) -> EvalReport:
        self.rows.append(report)
        print(report.summary(), flush=True)
        return report

    def to_markdown(self) -> str:
        header = (
            "| Configuration | Win rate % | +/- | Avg wrong | E | n |\n"
            "|---|---:|---:|---:|---:|---:|"
        )
        lines = [header]
        for row in self.rows:
            lines.append(
                f"| {row.tag} | {row.win_rate:.2f} | {row.stderr:.2f} | "
                f"{row.avg_wrong:.2f} | {row.total_wrong:,} | {row.n_words:,} |"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([r.to_dict() for r in self.rows], indent=2)
        )
        return path
