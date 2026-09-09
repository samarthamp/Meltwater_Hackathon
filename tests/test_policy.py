"""Tests for the inference policies.

The exact-search policy replaces a per-state Python loop with batched matrix
algebra, using the identity that two 0/1 indicator rows are equal iff

    rowsum(a) + rowsum(b) - 2 * <a, b> == 0

so the number of same-signature candidate pairs - and hence the expected
surviving candidate count - is a matmul away.  That identity is the thing worth
testing, so it is checked against a brute-force count.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hangman.config import PolicyConfig, TeacherConfig
from hangman.data import build_length_buckets
from hangman.ngram import HOLE, NgramModel
from hangman.policy import ExactSearchPolicy, GreedyPolicy, ModelSource
from hangman.teacher import BayesTeacher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WORDS = [
    "cat", "car", "can", "bat", "bar", "ban", "cab", "tab", "tan", "rat",
    "cot", "cut", "bit", "big", "bog", "dog", "dot", "cog", "cop", "top",
    "tot", "tit", "tat", "tab", "nab", "nag", "bag", "bad", "cad", "dab",
]


@pytest.fixture(scope="module")
def teacher():
    unique = list(dict.fromkeys(WORDS))
    cfg = TeacherConfig(max_ngram=4)
    ngram = NgramModel(cfg).fit(unique, max_len=8, min_count=1)
    return BayesTeacher(build_length_buckets(unique), ngram, cfg, device=DEVICE), unique


def brute_force_expected_left(candidates: list[str], letter: str) -> float:
    """``sum_s n_s^2 / n`` computed the obvious way, by grouping signatures."""
    groups: dict[tuple[bool, ...], int] = {}
    for word in candidates:
        signature = tuple(ch == letter for ch in word)
        groups[signature] = groups.get(signature, 0) + 1
    n = len(candidates)
    return sum(count * count for count in groups.values()) / n


def test_pair_counting_identity_matches_brute_force():
    """The matmul formulation must reproduce the grouped count exactly."""
    rng = np.random.default_rng(0)
    alphabet = "abcdefg"
    for trial in range(25):
        length = int(rng.integers(3, 8))
        n = int(rng.integers(2, 40))
        candidates = ["".join(rng.choice(list(alphabet), size=length)) for _ in range(n)]
        letter = str(rng.choice(list(alphabet)))

        ids = torch.tensor(
            [[ord(c) - 97 for c in word] for word in candidates], device=DEVICE
        ).unsqueeze(0)
        valid = torch.ones((1, n), dtype=torch.float32, device=DEVICE)
        indicator = (ids == (ord(letter) - 97)).to(torch.float32)

        occurrences = indicator.sum(dim=2)
        gram = indicator @ indicator.transpose(1, 2)
        rowsum = occurrences.unsqueeze(2) + occurrences.unsqueeze(1)
        same = (rowsum - 2.0 * gram).abs() < 0.5
        pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1)
        got = float((same.to(torch.float32) * pair_mask).sum(dim=(1, 2)).item() / n)

        want = brute_force_expected_left(candidates, letter)
        assert abs(got - want) < 1e-4, (trial, letter, candidates, got, want)


def test_exact_policy_only_touches_small_candidate_sets(teacher):
    """States with an empty or oversized candidate set must pass through."""
    oracle, words = teacher
    boards = torch.tensor([[HOLE, HOLE, HOLE]], device=DEVICE)
    present = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    absent = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    lengths = torch.tensor([3], device=DEVICE)
    lives = torch.tensor([6], device=DEVICE)
    active = torch.ones(1, dtype=torch.bool, device=DEVICE)

    fixed = torch.tensor([7], device=DEVICE)          # always guess 'h'

    class Fallback:
        source = None

        def __call__(self, *args, **kwargs):
            return fixed.clone()

    # max_candidates = 0 disables the refinement entirely.
    policy = ExactSearchPolicy(oracle, fallback=Fallback(), max_candidates=0, length=3)
    assert torch.equal(policy(boards, lengths, present, absent, lives, active), fixed)


def test_exact_policy_prefers_a_letter_the_candidates_contain(teacher):
    """With a small, informative candidate set the refinement should not pick a dud."""
    oracle, words = teacher
    # Board '_o_' - candidates are cot, dog, dot, cog, cop, top, tot, bog.
    boards = torch.tensor([[HOLE, 14, HOLE]], device=DEVICE)
    present = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    absent = torch.zeros((1, 26), dtype=torch.bool, device=DEVICE)
    present[0, 14] = True                              # 'o' revealed
    lengths = torch.tensor([3], device=DEVICE)
    lives = torch.tensor([4], device=DEVICE)
    active = torch.ones(1, dtype=torch.bool, device=DEVICE)

    class UniformSource:
        def presence(self, board, lengths, present, absent):
            probs = torch.full((board.shape[0], 26), 0.5, device=board.device)
            return probs.masked_fill(present | absent, 0.0)

    class Fallback:
        source = UniformSource()

        def __call__(self, board, lengths, present, absent, lives, active):
            return torch.tensor([16], device=board.device)     # 'q', a certain miss

    policy = ExactSearchPolicy(
        oracle, fallback=Fallback(), max_candidates=120, length=3,
        topk=8, source=UniformSource(),
    )
    chosen = int(policy(boards, lengths, present, absent, lives, active).item())
    consistent = [w for w in words if len(w) == 3 and w[1] == "o"]
    assert any(chr(97 + chosen) in w for w in consistent), (
        f"chose {chr(97 + chosen)!r}, which no candidate of {consistent} contains"
    )
