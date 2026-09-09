"""Tests for the posterior oracle and the n-gram backoff.

The matmul candidate filter is an optimisation of an obvious elementwise
predicate, so it is tested against that predicate directly rather than against
recorded outputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hangman.config import DataConfig, TeacherConfig
from hangman.data import build_dictionaries, build_length_buckets
from hangman.ngram import HOLE, NgramModel
from hangman.teacher import BayesTeacher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_teacher():
    words = [
        "cat", "car", "can", "bat", "bar", "ban", "cab", "tab", "tan", "rat",
        "cot", "cut", "bit", "big", "bog", "dog", "dot", "cog", "cop", "top",
    ]
    buckets = build_length_buckets(words)
    cfg = TeacherConfig(max_ngram=4)
    ngram = NgramModel(cfg).fit(words, max_len=8, min_count=1)
    return BayesTeacher(buckets, ngram, cfg, device=DEVICE), words


def elementwise_reference(teacher, length, boards, present, absent, tier):
    """The obvious O(L) predicate the matmul filter is meant to reproduce."""
    chars = teacher._chars[length]
    ok = torch.ones((boards.shape[0], chars.shape[0]), dtype=torch.bool, device=boards.device)
    for i in range(length):
        revealed = boards[:, i : i + 1] != HOLE
        ok &= (~revealed) | (chars[:, i].unsqueeze(0) == boards[:, i : i + 1])
        if tier <= 1:
            forbid = (absent | present) if tier == 0 else absent
            ok &= revealed | (~forbid.index_select(1, chars[:, i]))
    return ok


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_matmul_filter_matches_elementwise(tiny_teacher, tier):
    teacher, words = tiny_teacher
    boards = torch.tensor(
        [
            [2, HOLE, HOLE],      # c _ _
            [HOLE, 0, HOLE],      # _ a _
            [HOLE, HOLE, HOLE],   # _ _ _
        ],
        device=DEVICE,
    )
    present = torch.zeros((3, 26), dtype=torch.bool, device=DEVICE)
    absent = torch.zeros((3, 26), dtype=torch.bool, device=DEVICE)
    present[0, 2] = True                 # 'c' revealed
    present[1, 0] = True                 # 'a' revealed
    absent[0, 19] = True                 # 't' known absent
    absent[2, 1] = True                  # 'b' known absent

    got = teacher._consistency(3, boards, present, absent, tier)
    want = elementwise_reference(teacher, 3, boards, present, absent, tier)
    assert torch.equal(got, want)


def test_posterior_is_a_probability_and_masks_guesses(tiny_teacher):
    teacher, _ = tiny_teacher
    boards = torch.tensor([[2, HOLE, HOLE]], device=DEVICE)
    present = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    absent = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    present[0, 2] = True
    absent[0, 19] = True

    out = teacher.posterior(3, boards, present, absent)
    probs = out.presence[0]
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert float(probs[2]) == 0.0        # already guessed
    assert float(probs[19]) == 0.0       # already guessed


def test_leave_one_out_removes_the_played_word(tiny_teacher):
    """Without exclusion the teacher can read the answer off its own dictionary."""
    teacher, words = tiny_teacher
    row = words.index("cat") - words.index("cat")  # row inside the length-3 bucket
    bucket_words = [w for w in words if len(w) == 3]
    row = bucket_words.index("cat")

    boards = torch.tensor([[2, 0, HOLE]], device=DEVICE)   # c a _
    present = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    absent = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    present[0, 2] = True
    present[0, 0] = True

    exclude = torch.tensor([row], device=DEVICE)
    with_word = teacher.posterior(3, boards, present, absent).presence[0, 19]
    without = teacher.posterior(3, boards, present, absent, exclude_rows=exclude).presence[0, 19]
    assert float(with_word) > float(without)


def test_ngram_presence_is_calibrated_scale():
    """Noisy-OR pooling must return per-letter probabilities, not a simplex."""
    words = ["hangman", "handbag", "handles", "hangers", "bandage"]
    cfg = TeacherConfig(max_ngram=4)
    model = NgramModel(cfg).fit(words, max_len=8, min_count=1)
    boards = np.array([[7, 0, 13, HOLE, HOLE, HOLE, HOLE]], dtype=np.int64)
    lengths = np.array([7], dtype=np.int64)
    present = np.zeros((1, 26), dtype=bool)
    absent = np.zeros((1, 26), dtype=bool)
    present[0, [7, 0, 13]] = True
    probs = model.presence_batch(boards, lengths, present, absent)
    assert probs.shape == (1, 26)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert probs.sum() > 1.0            # marginals, not a distribution
    assert probs[0, 7] == 0.0           # guessed letters suppressed
