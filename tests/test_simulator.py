"""Correctness tests for the batched engine and the state encoding.

The batched engine is the thing that decides what our submission string says,
so it is validated against a naive single-game reference implementation of the
published rules, including the edge cases the rules call out explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hangman.config import MAX_WRONG_GUESSES
from hangman.ngram import HOLE
from hangman.simulator import BatchGame, PAD_POS, play_batch, reference_play
from hangman.state import encode_globals, encode_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fixed_sequence_policy(sequence: str):
    """A policy that plays a predetermined letter sequence, for testing."""
    step = {"i": 0}

    def fn(board, lengths, present, absent, lives, active):
        letter = ord(sequence[min(step["i"], len(sequence) - 1)]) - 97
        step["i"] += 1
        return torch.full((board.shape[0],), letter, dtype=torch.long, device=board.device)

    return fn


def test_win_and_loss_basics():
    game = BatchGame(["cat"], device=DEVICE)
    assert not bool(game.solved().item())
    for letter in "cat":
        game.apply(torch.tensor([ord(letter) - 97], device=DEVICE))
    assert bool(game.solved().item())
    assert int(game.wrong.item()) == 0


def test_sixth_wrong_guess_is_a_loss():
    """Five misses are survivable; the sixth ends the game as a loss."""
    game = BatchGame(["cat"], device=DEVICE)
    for letter in "bdfghj":                     # six letters, none in "cat"
        game.apply(torch.tensor([ord(letter) - 97], device=DEVICE))
    assert int(game.wrong.item()) == MAX_WRONG_GUESSES
    assert not bool(game.solved().item())
    assert not bool(game.active().item())


def test_five_wrong_still_active():
    game = BatchGame(["cat"], device=DEVICE)
    for letter in "bdfgh":
        game.apply(torch.tensor([ord(letter) - 97], device=DEVICE))
    assert int(game.wrong.item()) == 5
    assert bool(game.active().item())


def test_repeated_letter_costs_a_life():
    """A duplicate guess reveals nothing new, so the rules make it a strike."""
    game = BatchGame(["cat"], device=DEVICE)
    game.apply(torch.tensor([2], device=DEVICE))          # 'c', a hit
    assert int(game.wrong.item()) == 0
    game.apply(torch.tensor([2], device=DEVICE))          # 'c' again
    assert int(game.wrong.item()) == 1


def test_all_occurrences_revealed_at_once():
    game = BatchGame(["aaa"], device=DEVICE)
    game.apply(torch.tensor([0], device=DEVICE))
    assert bool(game.solved().item())
    assert int(game.n_guesses.item()) == 1


@pytest.mark.parametrize("word", ["a", "zz", "hangman", "zyzzyva", "a" * 29])
def test_batched_matches_reference(word: str):
    """The batched engine must agree with the naive scorer on every outcome."""
    sequence = "etaonrishdlfcmugypwbvkxjqz"
    result = play_batch([word], fixed_sequence_policy(sequence), device=DEVICE)
    ref_won, ref_wrong, _ = reference_play(word, sequence)
    assert bool(result.won.item()) == ref_won
    assert int(result.wrong.item()) == ref_wrong


def test_batched_matches_reference_on_random_words():
    rng = np.random.default_rng(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = [
        "".join(rng.choice(list(alphabet), size=int(rng.integers(2, 20))))
        for _ in range(2000)
    ]
    sequence = "etaonrishdlfcmugypwbvkxjqz"
    result = play_batch(words, fixed_sequence_policy(sequence), device=DEVICE)
    won = result.won.cpu().numpy()
    wrong = result.wrong.cpu().numpy()
    strings = result.guess_strings()
    for i, word in enumerate(words):
        ref_won, ref_wrong, ref_n = reference_play(word, sequence)
        assert bool(won[i]) == ref_won, word
        assert int(wrong[i]) == ref_wrong, word
        assert strings[i] == sequence[:ref_n], word


def test_guess_string_never_repeats_under_greedy_sources():
    """Repeats are always a wasted life, so policies must never emit one."""
    game = BatchGame(["encyclopedia"], device=DEVICE)
    seen = set()
    for _ in range(12):
        active = game.active()
        if not bool(active.any()):
            break
        probs = torch.rand((1, 26), device=DEVICE)
        probs = probs.masked_fill(game.present | game.absent, 0.0)
        guess = int(probs.argmax(dim=1).item())
        assert guess not in seen
        seen.add(guess)
        game.apply(torch.tensor([guess], device=DEVICE), active)


def test_board_rendering():
    game = BatchGame(["banana", "ox"], device=DEVICE)
    game.apply(torch.tensor([0, 23], device=DEVICE))       # 'a' hits, 'x' hits
    board = game.board().cpu().numpy()
    assert board[0].tolist()[:6] == [HOLE, 0, HOLE, 0, HOLE, 0]
    assert board[1].tolist()[:2] == [HOLE, 23]
    assert board[1].tolist()[2] == PAD_POS


def test_state_encoding_shapes_and_lives():
    game = BatchGame(["hangman", "ox"], device=DEVICE)
    game.apply(torch.tensor([16, 16], device=DEVICE))      # 'q', a miss for both
    tokens, attn = encode_tokens(game.board())
    globals_vec = encode_globals(game.lengths, game.present, game.absent)
    # Width is sized to the batch (CLS + longest word), not to the global
    # maximum of 32 - padding every batch to 32 would triple the model's cost
    # on this bandwidth-bound hardware.
    assert tokens.shape == (2, 1 + len("hangman"))
    assert encode_tokens(game.board(), seq_len=32)[0].shape == (2, 32)
    assert attn[:, 0].all()
    lives_onehot = globals_vec[:, 52:58]
    assert torch.allclose(lives_onehot.sum(dim=1), torch.ones(2, device=DEVICE))
    # One miss taken, so five lives remain -> index 5 of the one-hot.
    assert int(lives_onehot[0].argmax().item()) == 5
    assert bool(globals_vec[0, 16].item() == 1.0)          # 'q' flagged absent
