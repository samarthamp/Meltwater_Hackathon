"""Tests for the compact record format.

Training records store only ``(word_id, guessed-letter bitmask)`` plus the soft
target; the board, the present/absent split, the remaining lives, the
per-position labels and the count buckets are all *derived* on the GPU at batch
time.  That derivation is load-bearing - if it disagrees with the game engine by
even one position, every gradient in the project is computed against the wrong
state.  So it is checked against ``BatchGame`` directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hangman.collect import FLAG_ONPOLICY, RECORD_DTYPE
from hangman.config import MAX_WRONG_GUESSES
from hangman.dataset import RecordDataset
from hangman.ngram import HOLE
from hangman.simulator import BatchGame, PAD_POS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORDS = [
    "cat", "banana", "hangman", "zyzzyva", "ox", "encyclopedia",
    "aa", "abcdefghijklmnopqrstuvwxyza", "mississippi", "queue",
]


def make_dataset(tmp_path, records: np.ndarray) -> RecordDataset:
    path = tmp_path / "records_0000.npy"
    np.save(path, records)
    return RecordDataset([path], WORDS, device=DEVICE)


def build_records(word_ids: list[int], guessed_bits: list[int]) -> np.ndarray:
    rec = np.zeros(len(word_ids), dtype=RECORD_DTYPE)
    rec["word_id"] = np.array(word_ids, dtype=np.uint32)
    rec["guessed"] = np.array(guessed_bits, dtype=np.uint32)
    rec["target"] = 0.0
    rec["value"] = -1.0
    rec["flags"] = FLAG_ONPOLICY
    return rec


def test_board_reconstruction_matches_the_game_engine(tmp_path):
    """Reconstructed boards must equal what BatchGame produces for the same guesses."""
    rng = np.random.default_rng(0)
    word_ids, bits, guess_sets = [], [], []
    for _ in range(200):
        wid = int(rng.integers(0, len(WORDS)))
        word = WORDS[wid]
        # A realistic guess set: some of the word's letters, some outside it.
        inside = sorted({ord(c) - 97 for c in word})
        outside = [i for i in range(26) if i not in inside]
        take_in = rng.choice(inside, size=int(rng.integers(0, len(inside) + 1)), replace=False)
        n_out = min(int(rng.integers(0, MAX_WRONG_GUESSES)), len(outside))
        take_out = (
            rng.choice(outside, size=n_out, replace=False)
            if n_out
            else np.zeros(0, dtype=np.int64)
        )
        chosen = sorted(set(take_in.tolist()) | set(take_out.tolist()))
        mask = 0
        for letter in chosen:
            mask |= 1 << letter
        word_ids.append(wid)
        bits.append(mask)
        guess_sets.append(chosen)

    dataset = make_dataset(tmp_path, build_records(word_ids, bits))
    batch = dataset.batch(np.arange(len(word_ids)))

    # ``batch`` sorts by record index, which for arange is the identity.
    for row, (wid, chosen) in enumerate(zip(word_ids, guess_sets)):
        word = WORDS[wid]
        game = BatchGame([word], device=DEVICE)
        for letter in chosen:
            game.apply(torch.tensor([letter], device=DEVICE), active=torch.ones(1, dtype=torch.bool, device=DEVICE))

        want_board = game.board()[0]
        got_board = batch["word"][row]
        real = got_board >= 0
        revealed = batch["present"][row].gather(0, got_board.clamp_min(0)) & real
        rebuilt = torch.where(revealed, got_board, torch.full_like(got_board, HOLE))
        rebuilt = torch.where(real, rebuilt, torch.full_like(got_board, PAD_POS))

        n = len(word)
        assert torch.equal(rebuilt[:n], want_board[:n]), (word, chosen)
        assert torch.equal(batch["present"][row], game.present[0]), (word, chosen)
        assert torch.equal(batch["absent"][row], game.absent[0]), (word, chosen)


def test_targets_are_consistent_with_the_word(tmp_path):
    dataset = make_dataset(tmp_path, build_records([1, 8], [0, (1 << 8) | (1 << 18)]))
    batch = dataset.batch(np.arange(2))

    # "banana" contains b, a, n
    hard = batch["target_hard"][0]
    for letter in "ban":
        assert hard[ord(letter) - 97] == 1.0
    assert hard.sum() == 3.0

    # "mississippi": m1 i4 s4 p2 -> buckets {0,1,2,>=3}
    counts = batch["count_target"][1]
    assert int(counts[ord("m") - 97]) == 1
    assert int(counts[ord("i") - 97]) == 3      # clamped to >=3
    assert int(counts[ord("s") - 97]) == 3
    assert int(counts[ord("p") - 97]) == 2
    assert int(counts[ord("z") - 97]) == 0


def test_char_target_supervises_only_hidden_positions(tmp_path):
    # "banana" with 'a' guessed: positions 1,3,5 revealed, 0,2,4 hidden.
    dataset = make_dataset(tmp_path, build_records([1], [1 << 0]))
    batch = dataset.batch(np.arange(1))
    labels = batch["char_target"][0]
    assert labels[0].item() == ord("b") - 97
    assert labels[1].item() == -1               # revealed 'a'
    assert labels[2].item() == ord("n") - 97
    assert labels[3].item() == -1
    assert labels[4].item() == ord("n") - 97
    assert labels[5].item() == -1
    assert (labels[6:] == -1).all()             # past the end of the word


def test_length_bucketed_batches_share_one_length(tmp_path):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, len(WORDS), size=5000)
    dataset = make_dataset(tmp_path, build_records(ids.tolist(), [0] * len(ids)))
    for _ in range(20):
        batch = dataset.batch(dataset.sample_indices(64))
        assert batch["lengths"].unique().numel() == 1
    mixed = dataset.batch(dataset.sample_indices_mixed(512))
    assert mixed["lengths"].unique().numel() > 1
