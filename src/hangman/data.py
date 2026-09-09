"""Dictionary loading and the disjoint fit/validation split.

The provided ``train.txt`` (225,300 words) and ``test.txt`` (250,000 words)
share **zero** words.  Any honest local estimate of test performance must
reproduce that condition, so we split ``train.txt`` by word into a *fit*
dictionary (used for both model training and as the teacher's candidate set)
and a *validation dictionary* whose words the model has never seen.

``test.txt`` is loaded only to simulate games for a public-leaderboard estimate.
It is never used to fit anything - see reports/COMPLIANCE.md.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig, MAX_WORD_LEN

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def load_words(path: str | Path) -> list[str]:
    """Read a newline-delimited word list, lowercased and stripped."""
    with open(path, "r", encoding="utf-8") as fh:
        words = [line.strip().lower() for line in fh]
    return [w for w in words if w]


def words_digest(words: list[str]) -> str:
    """Stable short hash of a word list, recorded in checkpoints for provenance."""
    h = hashlib.sha256()
    for w in words:
        h.update(w.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass
class LengthBuckets:
    """Words grouped by length as dense arrays, for vectorised candidate filtering.

    Attributes
    ----------
    chars : dict[int, np.ndarray]
        ``chars[L]`` has shape ``(n_words_of_length_L, L)``, dtype ``uint8``,
        holding 0-based letter indices.
    presence : dict[int, np.ndarray]
        ``presence[L]`` has shape ``(n_words_of_length_L, 26)``, dtype ``bool``:
        whether each letter occurs at least once in that word.
    index : dict[str, tuple[int, int]]
        Maps a word to ``(length, row)`` so it can be excluded leave-one-out.
    """

    chars: dict[int, np.ndarray]
    presence: dict[int, np.ndarray]
    index: dict[str, tuple[int, int]]

    @property
    def lengths(self) -> list[int]:
        return sorted(self.chars)

    def size(self, length: int) -> int:
        arr = self.chars.get(length)
        return 0 if arr is None else int(arr.shape[0])


def build_length_buckets(words: list[str]) -> LengthBuckets:
    """Group ``words`` by length into dense numpy arrays."""
    by_len: dict[int, list[str]] = {}
    for w in words:
        by_len.setdefault(len(w), []).append(w)

    chars: dict[int, np.ndarray] = {}
    presence: dict[int, np.ndarray] = {}
    index: dict[str, tuple[int, int]] = {}

    for length, group in by_len.items():
        flat = np.frombuffer("".join(group).encode("ascii"), dtype=np.uint8)
        mat = (flat - 97).reshape(len(group), length)
        chars[length] = np.ascontiguousarray(mat)

        pres = np.zeros((len(group), 26), dtype=bool)
        rows = np.repeat(np.arange(len(group)), length)
        pres[rows, mat.ravel()] = True
        presence[length] = pres

        for row, w in enumerate(group):
            index[w] = (length, row)

    return LengthBuckets(chars=chars, presence=presence, index=index)


@dataclass
class Dictionaries:
    """The words available to a run, split to mirror train/test disjointness."""

    fit: list[str]
    val: list[str]
    buckets: LengthBuckets
    digest: str

    def summary(self) -> str:
        return (
            f"fit={len(self.fit):,} words  val={len(self.val):,} words  "
            f"overlap={len(set(self.fit) & set(self.val))}  digest={self.digest}"
        )


def build_dictionaries(cfg: DataConfig) -> Dictionaries:
    """Load train.txt and produce the fit / validation dictionaries.

    With ``cfg.use_full_dictionary`` the fit dictionary is all of train.txt and
    the validation list is empty - the configuration used only for the final
    submission model, after the recipe has been frozen on the honest split.
    """
    words = load_words(cfg.train_path)
    words = [w for w in words if 1 <= len(w) <= MAX_WORD_LEN]

    if cfg.use_full_dictionary:
        fit, val = list(words), []
    else:
        rng = random.Random(cfg.split_seed)
        shuffled = list(words)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * cfg.val_fraction))
        val, fit = shuffled[:n_val], shuffled[n_val:]
        fit.sort()
        val.sort()

    return Dictionaries(
        fit=fit,
        val=val,
        buckets=build_length_buckets(fit),
        digest=words_digest(sorted(words)),
    )


def load_test_words(cfg: DataConfig) -> list[str]:
    """Load the public test words.

    Used exclusively to *simulate* games and self-score; never as training data.
    """
    return load_words(cfg.test_path)


def unique_letter_frequency(words: list[str]) -> np.ndarray:
    """Fraction of words containing each letter - the classic hangman prior."""
    counts = np.zeros(26, dtype=np.float64)
    for w in words:
        for ch in set(w):
            counts[LETTER_TO_IDX[ch]] += 1.0
    return counts / max(len(words), 1)


def length_histogram(words: list[str], max_len: int = MAX_WORD_LEN) -> np.ndarray:
    """Counts of words per length, index 0 unused."""
    hist = np.zeros(max_len + 1, dtype=np.int64)
    for w in words:
        if len(w) <= max_len:
            hist[len(w)] += 1
    return hist
