"""Batched lock-step Hangman engine implementing the competition rules exactly.

Rules encoded here (from the problem statement)
-----------------------------------------------
* A correct letter reveals **all** of its occurrences at once.
* "Any character in your string that does not actively reveal a new letter in
  the hidden word - including an incorrect letter, a repeated/duplicate letter,
  a space, a number, or an invalid symbol - counts as exactly 1 wrong guess."
  So a *repeat* of an already-guessed letter is a strike even if that letter is
  in the word.  The engine implements this; our policies additionally never
  repeat, because a repeat can only ever cost a life.
* "The simulation loop for each word terminates instantly the moment your
  guessed_letters_string triggers a 6th wrong guess or completes the word."
  Six wrong guesses is therefore a **loss**; five is survivable.

Note that the blueprint snippet in the problem statement uses
``while wrong_guesses <= 6``, which would permit a seventh guess.  That
contradicts the normative rule text quoted above, so we implement the strict
rule - the conservative choice, since a policy tuned for six survivable misses
would lose games under the strict scorer but never the reverse.

Every game in a batch advances in lock-step so that a single model forward pass
serves thousands of games.  Finished games are simply masked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch

from .config import MAX_WORD_LEN, MAX_WRONG_GUESSES
from .ngram import HOLE

#: Marker used in the board tensor for positions past the end of a short word.
PAD_POS = -1


class GuessFn(Protocol):
    """Signature of a guess policy consumed by :func:`play_batch`."""

    def __call__(
        self,
        board: torch.Tensor,      # (B, Lmax) int64, HOLE where hidden, PAD_POS past the end
        lengths: torch.Tensor,    # (B,) int64
        present: torch.Tensor,    # (B, 26) bool
        absent: torch.Tensor,     # (B, 26) bool
        lives: torch.Tensor,      # (B,) int64, guesses remaining before loss
        active: torch.Tensor,     # (B,) bool
    ) -> torch.Tensor:            # (B,) int64 letter ids
        ...


@dataclass
class GameResult:
    """Outcome of a batch of games."""

    won: torch.Tensor            # (B,) bool
    wrong: torch.Tensor          # (B,) int64  strikes accumulated
    n_guesses: torch.Tensor      # (B,) int64  length of the guess string
    guess_seq: torch.Tensor      # (B, 32) int8, -1 padded, in order

    def win_rate(self) -> float:
        return float(self.won.to(torch.float64).mean().item())

    def total_wrong(self) -> int:
        """``E`` in the leaderboard tiebreak ``Score - E / 1e8``."""
        return int(self.wrong.sum().item())

    def guess_strings(self) -> list[str]:
        seq = self.guess_seq.detach().cpu().numpy()
        counts = self.n_guesses.detach().cpu().numpy()
        out = []
        for row, n in zip(seq, counts):
            out.append("".join(chr(97 + int(c)) for c in row[:n]))
        return out


class BatchGame:
    """Mutable state for a batch of simultaneous games."""

    def __init__(self, words: list[str], device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.words = words
        batch = len(words)
        lengths = np.array([len(w) for w in words], dtype=np.int64)
        lmax = int(lengths.max()) if batch else 1
        if lmax > MAX_WORD_LEN:
            raise ValueError(f"word longer than MAX_WORD_LEN={MAX_WORD_LEN}")

        truth = np.full((batch, lmax), PAD_POS, dtype=np.int64)
        presence = np.zeros((batch, 26), dtype=bool)
        for i, word in enumerate(words):
            ids = np.frombuffer(word.encode("ascii"), dtype=np.uint8).astype(np.int64) - 97
            truth[i, : ids.size] = ids
            presence[i, ids] = True

        self.truth = torch.from_numpy(truth).to(self.device)
        self.lengths = torch.from_numpy(lengths).to(self.device)
        self.truth_presence = torch.from_numpy(presence).to(self.device)

        self.present = torch.zeros((batch, 26), dtype=torch.bool, device=self.device)
        self.absent = torch.zeros((batch, 26), dtype=torch.bool, device=self.device)
        self.wrong = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.n_guesses = torch.zeros(batch, dtype=torch.long, device=self.device)
        # 26 distinct letters plus up to 5 survivable strikes; 32 is a safe cap.
        self.guess_seq = torch.full((batch, 32), -1, dtype=torch.int8, device=self.device)
        self._rows = torch.arange(batch, device=self.device)

    # -- observations --------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.truth.shape[0])

    def board(self) -> torch.Tensor:
        """``(B, Lmax)`` board: true letter where revealed, ``HOLE`` where hidden."""
        real = self.truth >= 0
        revealed = self.present.gather(1, self.truth.clamp_min(0)) & real
        board = torch.where(revealed, self.truth, torch.full_like(self.truth, HOLE))
        return torch.where(real, board, torch.full_like(self.truth, PAD_POS))

    def solved(self) -> torch.Tensor:
        """``(B,)`` bool: every letter of the word has been revealed."""
        return (self.truth_presence & ~self.present).sum(dim=1) == 0

    def lives(self) -> torch.Tensor:
        """``(B,)`` remaining strikes before the game is lost."""
        return MAX_WRONG_GUESSES - self.wrong

    def active(self) -> torch.Tensor:
        """``(B,)`` bool: game still in progress."""
        return (self.wrong < MAX_WRONG_GUESSES) & ~self.solved()

    # -- transition ----------------------------------------------------------

    def apply(self, guesses: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """Apply one guess per game.  Returns the ``(B,)`` bool hit mask.

        A guess counts as a hit only if the letter is in the word **and** has
        not been guessed before; anything else is a strike, per the rules.
        """
        if active is None:
            active = self.active()
        rows = self._rows[active]
        if rows.numel() == 0:
            return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        letters = guesses[active]
        already = self.present[rows, letters] | self.absent[rows, letters]
        in_word = self.truth_presence[rows, letters]
        hit = in_word & ~already

        # Clamp the write index: 26 distinct letters plus five survivable
        # strikes cannot exceed 31, but a stochastic policy that somehow
        # repeated itself would index out of bounds, and an out-of-bounds
        # scatter on CUDA is a device-side assert that surfaces asynchronously
        # somewhere else entirely.
        slot = self.n_guesses[rows].clamp(max=self.guess_seq.shape[1] - 1)
        self.guess_seq[rows, slot] = letters.to(torch.int8)
        self.n_guesses[rows] += 1

        self.present[rows[hit], letters[hit]] = True
        # A letter not in the word is recorded absent whether or not it is a
        # repeat; a repeat of a *correct* letter needs no state change, it just
        # costs a life.
        not_in_word = ~in_word
        self.absent[rows[not_in_word], letters[not_in_word]] = True
        self.wrong[rows[~hit]] += 1

        full_hit = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        full_hit[rows] = hit
        return full_hit

    def result(self) -> GameResult:
        return GameResult(
            won=self.solved(),
            wrong=self.wrong.clone(),
            n_guesses=self.n_guesses.clone(),
            guess_seq=self.guess_seq.clone(),
        )


@torch.no_grad()
def play_batch(
    words: list[str],
    guess_fn: GuessFn,
    device: str | torch.device = "cuda",
    max_turns: int = 26,
    on_state: Callable[[BatchGame, torch.Tensor], None] | None = None,
) -> GameResult:
    """Play a batch of games to completion under ``guess_fn``.

    ``on_state`` is invoked before each turn with the game and the active mask,
    which is how DAgger collects the on-policy state distribution.
    """
    game = BatchGame(words, device=device)
    for _ in range(max_turns):
        active = game.active()
        if not bool(active.any()):
            break
        if on_state is not None:
            on_state(game, active)
        guesses = guess_fn(
            board=game.board(),
            lengths=game.lengths,
            present=game.present,
            absent=game.absent,
            lives=game.lives(),
            active=active,
        )
        game.apply(guesses, active)
    return game.result()


def reference_play(word: str, guess_order: str) -> tuple[bool, int, int]:
    """Naive single-game scorer, used to validate the batched engine.

    Returns ``(won, wrong, n_consumed)`` for a fixed guess string, applying the
    6-strike lockout exactly as the competition scorer describes it.
    """
    revealed: set[str] = set()
    guessed: set[str] = set()
    wrong = 0
    for i, ch in enumerate(guess_order):
        if wrong >= MAX_WRONG_GUESSES or not (set(word) - revealed):
            return (not (set(word) - revealed)), wrong, i
        if ch in word and ch not in guessed:
            revealed.add(ch)
        else:
            wrong += 1
        guessed.add(ch)
    return (not (set(word) - revealed)), wrong, len(guess_order)
