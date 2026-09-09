"""Record shards -> model batches.

Training records hold only ``(word_id, guessed_bitmask)`` plus the teacher's
soft target (66 bytes total, see :mod:`hangman.collect`).  Everything else -
the board, the present/absent split, remaining lives, per-position labels,
count buckets - is *derived*, and derived on the GPU, in one vectorised pass
per batch.

The payoff is that a 50M-state training set is ~3.3 GB on disk and can be
memory-mapped, while the expansion cost is negligible next to the forward and
backward passes it feeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import MAX_WRONG_GUESSES, SEQ_LEN, ModelConfig
from .collect import FLAG_ONPOLICY, RECORD_DTYPE
from .ngram import HOLE
from .simulator import PAD_POS
from .state import encode_globals, encode_tokens

#: Occurrence-count buckets for the count head: 0, 1, 2, 3-or-more.
COUNT_BUCKETS = 4


class RecordDataset:
    """A memory-mapped pool of labelled states with weighted sampling.

    Parameters
    ----------
    shards : list of paths to ``.npy`` record arrays.
    words : the fit dictionary, indexed by ``record['word_id']``.
    short_word_range, short_word_boost : oversample the length band where the
        measured baseline loses most of its games (lengths 4-9).
    onpolicy_mix : target fraction of each batch drawn from on-policy records.
        ``None`` samples the pool as-is.
    """

    def __init__(
        self,
        shards: list[str | Path],
        words: list[str],
        device: str | torch.device = "cuda",
        short_word_range: tuple[int, int] = (4, 9),
        short_word_boost: float = 1.0,
        onpolicy_mix: float | None = None,
        seed: int = 0,
    ) -> None:
        if not shards:
            raise ValueError("no record shards supplied")
        self.device = torch.device(device)
        self.records = np.concatenate(
            [np.load(p, mmap_mode="r") for p in shards]
        ) if len(shards) > 1 else np.load(shards[0], mmap_mode="r")
        if self.records.dtype != RECORD_DTYPE:
            raise ValueError(f"unexpected record dtype {self.records.dtype}")
        self.rng = np.random.default_rng(seed)
        self.n = int(self.records.size)

        lengths = np.array([len(w) for w in words], dtype=np.int64)
        self.word_lengths = lengths
        lmax = int(lengths.max())
        packed = np.full((len(words), lmax), PAD_POS, dtype=np.int16)
        for i, word in enumerate(words):
            packed[i, : len(word)] = np.frombuffer(word.encode("ascii"), np.uint8) - 97
        self.words_gpu = torch.from_numpy(packed.astype(np.int64)).to(self.device)
        self.lengths_gpu = torch.from_numpy(lengths).to(self.device)

        # Sampling weights.
        weight = np.ones(self.n, dtype=np.float32)
        rec_len = lengths[self.records["word_id"]]
        low, high = short_word_range
        if short_word_boost != 1.0:
            weight[(rec_len >= low) & (rec_len <= high)] *= short_word_boost
        if onpolicy_mix is not None:
            on = (self.records["flags"] & FLAG_ONPOLICY) != 0
            n_on, n_off = int(on.sum()), int((~on).sum())
            if n_on and n_off:
                weight[on] *= onpolicy_mix / (n_on / self.n)
                weight[~on] *= (1.0 - onpolicy_mix) / (n_off / self.n)
        self._uniform = bool(np.allclose(weight, weight[0]))
        # A cumulative table sampled by searchsorted costs O(batch * log n) per
        # draw; ``rng.choice(p=...)`` is O(n) per call, which at 50M records
        # would dominate the training step itself.
        self._cdf = None if self._uniform else np.cumsum(weight / weight.sum(), dtype=np.float64)

        # Length-bucketed index.  Drawing a whole batch from one word length
        # keeps the padded sequence at ``L + 1`` instead of the global 32, and
        # on this machine (273 GB/s against ~100 TFLOPS bf16) step time is set
        # by activation traffic, so padded width is the dominant cost lever.
        # RMSNorm normalises per sample, so a batch sharing one length
        # introduces none of the statistical coupling batch norm would suffer.
        order = np.argsort(rec_len, kind="stable").astype(np.int64)
        sorted_len = rec_len[order]
        self._bucket_lengths = np.unique(sorted_len)
        starts = np.searchsorted(sorted_len, self._bucket_lengths, side="left")
        ends = np.searchsorted(sorted_len, self._bucket_lengths, side="right")
        self._buckets: dict[int, np.ndarray] = {}
        self._bucket_cdf: dict[int, np.ndarray | None] = {}
        bucket_weight = np.zeros(self._bucket_lengths.size, dtype=np.float64)
        for slot, (length, start, end) in enumerate(zip(self._bucket_lengths, starts, ends)):
            members = order[start:end]
            self._buckets[int(length)] = members
            member_weight = weight[members]
            bucket_weight[slot] = float(member_weight.sum())
            self._bucket_cdf[int(length)] = (
                None
                if bool(np.allclose(member_weight, member_weight[0]))
                else np.cumsum(member_weight / member_weight.sum(), dtype=np.float64)
            )
        self._bucket_probs = bucket_weight / bucket_weight.sum()

    def __len__(self) -> int:
        return self.n

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Draw record indices that all share one word length.

        The length itself is drawn in proportion to that bucket's total
        sampling weight, so the marginal distribution over states is unchanged
        from :meth:`sample_indices_mixed` - only the grouping differs.
        """
        length = int(self.rng.choice(self._bucket_lengths, p=self._bucket_probs))
        pool = self._buckets[length]
        cdf = self._bucket_cdf[length]
        if cdf is None:
            return pool[self.rng.integers(0, pool.size, size=batch_size)]
        draws = self.rng.random(batch_size) * cdf[-1]
        return pool[np.searchsorted(cdf, draws).clip(0, pool.size - 1)]

    def sample_indices_mixed(self, batch_size: int) -> np.ndarray:
        """Draw record indices across all word lengths, for diagnostics."""
        if self._uniform:
            return self.rng.integers(0, self.n, size=batch_size)
        draws = self.rng.random(batch_size) * self._cdf[-1]
        return np.searchsorted(self._cdf, draws).clip(0, self.n - 1)

    def batch(self, indices: np.ndarray, model_cfg: ModelConfig | None = None) -> dict[str, torch.Tensor]:
        """Expand records into a full model batch on the device."""
        rec = self.records[np.sort(indices)]
        word_ids = torch.from_numpy(rec["word_id"].astype(np.int64)).to(self.device)
        bits = torch.from_numpy(rec["guessed"].astype(np.int64)).to(self.device)
        soft = torch.from_numpy(rec["target"].astype(np.float32)).to(self.device)
        value = torch.from_numpy(rec["value"].astype(np.float32)).to(self.device)
        n_cand = torch.from_numpy(rec["n_cand"].astype(np.int64)).to(self.device)

        words = self.words_gpu[word_ids]                    # (B, Lmax)
        lengths = self.lengths_gpu[word_ids]                # (B,)
        lmax = int(lengths.max().item())
        words = words[:, :lmax]

        shifts = torch.arange(26, device=self.device, dtype=torch.long)
        guessed = ((bits.unsqueeze(1) >> shifts) & 1).to(torch.bool)

        real = words >= 0
        safe = words.clamp_min(0)
        word_presence = torch.zeros_like(guessed)
        word_presence.scatter_(1, safe, True)
        # Padded slots scatter a spurious 'a'; correct it.
        word_presence[:, 0] &= (words == 0).any(dim=1)

        present = guessed & word_presence
        absent = guessed & ~word_presence
        revealed = present.gather(1, safe) & real
        board = torch.where(revealed, words, torch.full_like(words, HOLE))
        board = torch.where(real, board, torch.full_like(words, PAD_POS))

        tokens, attn = encode_tokens(board)
        globals_vec = encode_globals(lengths, present, absent)

        counts = torch.zeros_like(guessed, dtype=torch.long)
        counts.scatter_add_(1, safe, real.to(torch.long))
        count_bucket = counts.clamp(max=COUNT_BUCKETS - 1)

        # Masked-language-model target: the true letter at every *hidden*
        # position, -1 elsewhere so cross-entropy ignores it.  Widths are
        # aligned to the model's position head, which covers slots 1..SEQ_LEN-1.
        hidden = real & ~revealed
        width = SEQ_LEN - 1
        char_target = torch.full((words.shape[0], width), -1, dtype=torch.long, device=self.device)
        usable = min(lmax, width)
        char_target[:, :usable] = torch.where(
            hidden[:, :usable], words[:, :usable], torch.full_like(words[:, :usable], -1)
        )

        return {
            "tokens": tokens,
            "attn_mask": attn,
            "globals": globals_vec,
            "guessed": guessed,
            "target_soft": soft,
            "target_hard": word_presence.to(torch.float32),
            "char_target": char_target,
            "word": words,
            "hidden_mask": (board == HOLE),
            "count_target": count_bucket,
            "value_target": value,
            "n_candidates": n_cand,
            "lengths": lengths,
            "present": present,
            "absent": absent,
        }

    def iter_batches(self, batch_size: int, steps: int):
        """Yield ``steps`` sampled batches."""
        for _ in range(steps):
            yield self.batch(self.sample_indices(batch_size))

    def stats(self) -> dict[str, float]:
        on = (self.records["flags"] & FLAG_ONPOLICY) != 0
        tiers = np.bincount(self.records["tier"], minlength=4)
        return {
            "n_records": float(self.n),
            "onpolicy_frac": float(on.mean()),
            "empty_tier1_frac": float((self.records["n_cand"] == 0).mean()),
            **{f"tier{i}_frac": float(tiers[i] / max(self.n, 1)) for i in range(4)},
        }


def shard_paths(directory: str | Path, prefix: str) -> list[Path]:
    """All shards for ``prefix`` in ``directory``, in numeric order."""
    return sorted(Path(directory).glob(f"{prefix}_*.npy"))
