"""Interpolated single-hole character n-gram model.

This is the *backoff tier* of the teacher (Tier 3) and one input to the
calibrated inference mixture.  It is deliberately not the submission model -
the competition down-weights statistical baselines - but it is the thing a
neural model has to beat, and it covers the game states where exact dictionary
reasoning is blind (measured: 44.4% of on-policy states have no consistent
dictionary word at all).

Model
-----
For every order ``n`` in ``2..max_ngram`` and every window of ``n`` characters
in a padded training word, we record, for each slot ``j`` in the window, the
conditional distribution of the letter at ``j`` given the other ``n-1``
characters.  At inference the same windows are read off the board; a window
contributes only when exactly one of its slots is still hidden.

Evidence from different orders is interpolated with weight ``n ** power``,
following standard practice of trusting longer contexts more (Chen & Goodman,
1999, *An empirical study of smoothing techniques for language modeling*).

Two cheap priors cover the opening, when no window has exactly one hole:
a length-conditioned positional prior ``P(letter at i | length L)`` and a
co-occurrence prior ``P(letter present | letters known present)``.

Implementation
--------------
Contexts are encoded as base-28 integers and stored in a *sorted* int64 array
alongside a ``(n_contexts, 26)`` probability matrix.  Lookup is
``np.searchsorted``, which makes both construction and scoring fully
vectorised - construction over the whole dictionary at once, scoring over a
whole batch of game states at once.  This is what makes labelling tens of
millions of states tractable.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import TeacherConfig

BOUNDARY = 26          # sentinel letter id for positions outside the word
HOLE = 27              # sentinel for a slot that is still hidden
RADIX = 28             # 26 letters + BOUNDARY + HOLE


class NgramModel:
    """Interpolated single-hole n-gram scorer over a fixed dictionary."""

    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg
        self.max_n = cfg.max_ngram
        self.pad = self.max_n - 1
        #: sorted context codes and their conditional letter distributions
        self.keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self.probs: np.ndarray = np.zeros((0, 26), dtype=np.float32)
        self.order_weight = np.array(
            [float(n) ** cfg.ngram_power for n in range(self.max_n + 1)],
            dtype=np.float32,
        )
        self.pos_prior: np.ndarray = np.zeros((0, 0, 26), dtype=np.float32)
        self.unigram: np.ndarray = np.full(26, 1.0 / 26.0, dtype=np.float32)
        self.cooccur: np.ndarray = np.zeros((26, 26), dtype=np.float32)
        self._radix_pow = RADIX ** np.arange(self.max_n, dtype=np.int64)
        self._order_offset = RADIX ** self.max_n

    # -- construction -------------------------------------------------------

    def fit(self, words: list[str], max_len: int, min_count: int = 1) -> "NgramModel":
        """Estimate every table from ``words``.

        ``min_count`` prunes contexts seen fewer than this many times at orders
        >= 4, which removes the long tail of once-seen high-order contexts
        without measurably hurting quality.
        """
        by_len: dict[int, list[str]] = {}
        for w in words:
            by_len.setdefault(len(w), []).append(w)

        codes_chunks: list[np.ndarray] = []
        targets_chunks: list[np.ndarray] = []
        pos_counts = np.zeros((max_len + 1, max_len, 26), dtype=np.float64)
        uni = np.zeros(26, dtype=np.float64)
        co = np.zeros((26, 26), dtype=np.float64)
        pad = self.pad

        for length, group in by_len.items():
            flat = np.frombuffer("".join(group).encode("ascii"), dtype=np.uint8)
            mat = (flat.astype(np.int64) - 97).reshape(len(group), length)

            if length <= max_len:
                np.add.at(pos_counts[length], (np.arange(length)[None, :].repeat(len(group), 0), mat), 1.0)

            presence = np.zeros((len(group), 26), dtype=bool)
            rows = np.repeat(np.arange(len(group)), length)
            presence[rows, mat.ravel()] = True
            uni += presence.sum(axis=0)
            co += presence.T.astype(np.float64) @ presence.astype(np.float64)

            padded = np.concatenate(
                [
                    np.full((len(group), pad), BOUNDARY, dtype=np.int64),
                    mat,
                    np.full((len(group), pad), BOUNDARY, dtype=np.int64),
                ],
                axis=1,
            )
            width = padded.shape[1]
            for n in range(2, self.max_n + 1):
                base = self._order_offset * n
                for start in range(width - n + 1):
                    window = padded[:, start:start + n]              # (G, n)
                    for j in range(n):
                        target = window[:, j]
                        keep = target != BOUNDARY
                        if not keep.any():
                            continue
                        holed = window.copy()
                        holed[:, j] = HOLE
                        code = base + (holed * self._radix_pow[:n]).sum(axis=1)
                        codes_chunks.append(code[keep])
                        targets_chunks.append(target[keep])

        codes = np.concatenate(codes_chunks)
        targets = np.concatenate(targets_chunks)
        del codes_chunks, targets_chunks

        # Count (context, letter) pairs with one sort over a fused key.
        fused = codes * 26 + targets
        uniq, counts = np.unique(fused, return_counts=True)
        del fused
        uniq_codes = uniq // 26
        uniq_letters = (uniq % 26).astype(np.int64)

        key_ids, inverse = np.unique(uniq_codes, return_inverse=True)
        table = np.zeros((key_ids.size, 26), dtype=np.float64)
        np.add.at(table, (inverse, uniq_letters), counts.astype(np.float64))

        if min_count > 1:
            order = key_ids // self._order_offset
            totals = table.sum(axis=1)
            keep = (order < 4) | (totals >= min_count)
            key_ids, table = key_ids[keep], table[keep]

        self.keys = key_ids.astype(np.int64)
        self.probs = (table / np.maximum(table.sum(axis=1, keepdims=True), 1.0)).astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            self.pos_prior = np.nan_to_num(
                pos_counts / np.maximum(pos_counts.sum(axis=2, keepdims=True), 1.0)
            ).astype(np.float32)
        self.unigram = (uni / max(len(words), 1)).astype(np.float32)
        diag = np.maximum(np.diag(co), 1.0)
        self.cooccur = (co / diag[:, None]).astype(np.float32)
        return self

    # -- window enumeration -------------------------------------------------

    def _window_codes(
        self, padded: np.ndarray, pad: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Enumerate single-hole window codes for a batch of padded boards.

        Returns ``(rows, holes, codes, weights)`` where ``rows`` indexes the
        batch, ``holes`` is the *board* position the window predicts, ``codes``
        is the context code and ``weights`` the order interpolation weight.
        """
        width = padded.shape[1]
        rows_out: list[np.ndarray] = []
        holes_out: list[np.ndarray] = []
        codes_out: list[np.ndarray] = []
        weights_out: list[np.ndarray] = []

        for n in range(2, self.max_n + 1):
            base = self._order_offset * n
            weight = self.order_weight[n]
            for start in range(width - n + 1):
                window = padded[:, start:start + n]
                n_holes = (window == HOLE).sum(axis=1)
                for j in range(n):
                    sel = (n_holes == 1) & (window[:, j] == HOLE)
                    if not sel.any():
                        continue
                    idx = np.flatnonzero(sel)
                    code = base + (window[idx] * self._radix_pow[:n]).sum(axis=1)
                    rows_out.append(idx)
                    holes_out.append(np.full(idx.size, start + j - pad, dtype=np.int64))
                    codes_out.append(code)
                    weights_out.append(np.full(idx.size, weight, dtype=np.float32))

        if not rows_out:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, np.zeros(0, dtype=np.float32)
        return (
            np.concatenate(rows_out),
            np.concatenate(holes_out),
            np.concatenate(codes_out),
            np.concatenate(weights_out),
        )

    def _lookup(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve context codes to table rows.  Returns ``(positions, hit mask)``."""
        if self.keys.size == 0:
            return np.zeros_like(codes), np.zeros(codes.shape, dtype=bool)
        pos = np.searchsorted(self.keys, codes)
        np.clip(pos, 0, self.keys.size - 1, out=pos)
        return pos, self.keys[pos] == codes

    # -- scoring ------------------------------------------------------------

    def positional_distribution(
        self,
        boards: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        """``(B, Lmax, 26)`` letter distribution for every hidden position.

        Positions that are already revealed, or past the end of the word, get a
        zero row.  Hidden positions with no matching context fall back to the
        length-conditioned positional prior and then to the unigram prior, so
        the returned distribution is defined everywhere it matters.
        """
        batch, lmax = boards.shape
        pad = self.pad
        padded = np.full((batch, lmax + 2 * pad), BOUNDARY, dtype=np.int64)
        padded[:, pad:pad + lmax] = boards
        beyond = np.arange(lmax)[None, :] >= lengths[:, None]
        padded[:, pad:pad + lmax][beyond] = BOUNDARY

        out = np.zeros((batch, lmax, 26), dtype=np.float32)
        rows, holes, codes, weights = self._window_codes(padded, pad)
        if rows.size:
            pos, mask = self._lookup(codes)
            if mask.any():
                rows, holes, pos, weights = rows[mask], holes[mask], pos[mask], weights[mask]
                np.add.at(out, (rows, holes), self.probs[pos] * weights[:, None])

        hidden = (boards == HOLE) & ~beyond
        totals = out.sum(axis=2)
        needs_prior = hidden & (totals <= 0.0)
        if needs_prior.any():
            b_idx, p_idx = np.nonzero(needs_prior)
            valid = p_idx < self.pos_prior.shape[1]
            lens = lengths[b_idx]
            in_range = valid & (lens < self.pos_prior.shape[0])
            if in_range.any():
                out[b_idx[in_range], p_idx[in_range]] = self.pos_prior[
                    lens[in_range], p_idx[in_range]
                ]
            totals = out.sum(axis=2)
            still = hidden & (totals <= 0.0)
            if still.any():
                out[still] = self.unigram

        out[~hidden] = 0.0
        totals = out.sum(axis=2, keepdims=True)
        np.divide(out, np.maximum(totals, 1e-9), out=out)
        return out

    def presence_batch(
        self,
        boards: np.ndarray,
        lengths: np.ndarray,
        present: np.ndarray,
        absent: np.ndarray,
    ) -> np.ndarray:
        """``(B, 26)`` probability that each letter occupies at least one hole.

        Positional distributions are pooled with **noisy-OR**,
        ``P(present) = 1 - prod_i (1 - p_i(letter))``, which is the correct way
        to turn per-slot distributions into a presence marginal.  Earlier public
        hangman transformers max-pool here, which systematically underestimates
        letters that are likely in several places at once.

        The result is on the same scale as the exact dictionary posterior, so
        the two can be mixed directly.
        """
        per_pos = self.positional_distribution(boards, lengths)
        miss = np.clip(1.0 - per_pos, 0.0, 1.0)
        presence = 1.0 - np.prod(miss, axis=1)

        if present.any():
            boost = 0.15 * self.cooccur.T @ present.T.astype(np.float32)   # (26, B)
            presence = presence + boost.T * (presence <= 0.0)

        presence[present] = 0.0
        presence[absent] = 0.0
        return presence.astype(np.float32)

    def score_batch(
        self,
        boards: np.ndarray,
        lengths: np.ndarray,
        present: np.ndarray,
        absent: np.ndarray,
    ) -> np.ndarray:
        """Alias of :meth:`presence_batch`, kept for call-site readability."""
        return self.presence_batch(boards, lengths, present, absent)

    def score(
        self,
        board: np.ndarray,
        present: np.ndarray,
        absent: np.ndarray,
    ) -> np.ndarray:
        """Single-state convenience wrapper."""
        return self.presence_batch(
            board[None, :].astype(np.int64),
            np.array([board.size], dtype=np.int64),
            present[None, :],
            absent[None, :],
        )[0]

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "cfg": self.cfg,
                    "keys": self.keys,
                    "probs": self.probs,
                    "pos_prior": self.pos_prior,
                    "unigram": self.unigram,
                    "cooccur": self.cooccur,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(path: str | Path) -> "NgramModel":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        model = NgramModel(blob["cfg"])
        model.keys = blob["keys"]
        model.probs = blob["probs"]
        model.pos_prior = blob["pos_prior"]
        model.unigram = blob["unigram"]
        model.cooccur = blob["cooccur"]
        return model
