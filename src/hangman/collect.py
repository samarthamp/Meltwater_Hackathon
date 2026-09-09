"""State collection: the off-policy warm start and the DAgger on-policy rounds.

Why the state distribution matters more than the architecture
-------------------------------------------------------------
Every public hangman network I surveyed trains on *randomly masked* words: pick
a word, hide a random subset of letters, predict the rest.  That distribution
is not the one a game visits.  Two mismatches are severe:

* **Reachability.**  A real state can carry at most five absent letters, since
  the sixth ends the game.  Uniform random masking happily produces states with
  fifteen known-absent letters, which no game ever reaches, and starves the
  model of the tight five-strike endgames that decide win rate.
* **Correlation.**  A real absent set is biased toward *frequent* letters,
  because that is what a competent policy guesses first.  Random absent sets
  are dominated by ``j``, ``q``, ``x``, ``z`` - the letters a real game rarely
  wastes a guess on.

Behavioural cloning on the wrong state distribution is exactly the covariate
shift that DAgger was introduced to fix (Ross, Gordon & Bagnell, 2011, *A
Reduction of Imitation Learning and Structured Prediction to No-Regret Online
Learning*).  We therefore do both:

1. an off-policy warm start that samples only *reachable* states, with absent
   letters drawn in proportion to letter frequency; and
2. successive DAgger rounds that roll out the current model, record every state
   it actually visits, label them with the teacher, and **aggregate** them into
   the training set rather than replacing it.

Storage
-------
A state is fully determined by ``(word, guessed-letter set)``: the board, the
present/absent split and the remaining lives all follow.  Records therefore
store eight bytes of state plus the soft target, which keeps a 20M-state
dataset at ~1.4 GB and lets it be memory-mapped during training.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import CollectConfig, MAX_WRONG_GUESSES
from .data import Dictionaries
from .selfdistill import blend_targets
from .simulator import BatchGame, PAD_POS
from .teacher import BayesTeacher

#: Compact on-disk training record.  ``target`` is the teacher's soft presence
#: posterior; ``value`` is the Monte-Carlo win/loss return, or -1 when unknown
#: (off-policy states have no rollout attached).
RECORD_DTYPE = np.dtype(
    [
        ("word_id", np.uint32),     # row in the fit dictionary
        ("guessed", np.uint32),     # 26-bit guessed-letter mask
        ("n_cand", np.uint16),      # Tier-1 candidate count, clipped
        ("tier", np.uint8),         # which ladder rung supplied the posterior
        ("flags", np.uint8),        # bit 0: on-policy
        ("target", np.float16, 26),
        ("value", np.float16),
    ]
)

FLAG_ONPOLICY = 1 << 0
VALUE_UNKNOWN = np.float16(-1.0)


@dataclass
class CollectionStats:
    n_states: int = 0
    n_games: int = 0
    empty_tier1: int = 0
    seconds: float = 0.0

    def report(self, tag: str) -> str:
        rate = self.n_states / max(self.seconds, 1e-9)
        empty = 100.0 * self.empty_tier1 / max(self.n_states, 1)
        return (
            f"{tag}: {self.n_states:,} states from {self.n_games:,} games "
            f"in {self.seconds:.1f}s ({rate:,.0f} states/s), "
            f"Tier-1 empty {empty:.1f}%"
        )


class StateCollector:
    """Generates labelled game states for training."""

    def __init__(
        self,
        dicts: Dictionaries,
        teacher: BayesTeacher,
        cfg: CollectConfig,
        device: str | torch.device = "cuda",
        student=None,
        kappa_sd: float = 50.0,
    ) -> None:
        """
        Parameters
        ----------
        student : object with ``presence(board, lengths, present, absent)``,
            optional.  When supplied, the recorded target is a blend of the
            Bayes posterior and this model's predictions, gated on candidate
            count (see :mod:`hangman.selfdistill`).  Passing ``None`` reproduces
            the original pure-Bayes labelling exactly.
        kappa_sd : gate constant for that blend.
        """
        self.dicts = dicts
        self.teacher = teacher
        self.cfg = cfg
        self.device = torch.device(device)
        self.student = student
        self.kappa_sd = kappa_sd
        self.rng = np.random.default_rng(cfg.seed)

        # Global word index: word_id is the position in ``dicts.fit``.
        self.word_id = {w: i for i, w in enumerate(dicts.fit)}
        self.by_length: dict[int, np.ndarray] = {}
        grouped: dict[int, list[int]] = defaultdict(list)
        for word, idx in self.word_id.items():
            grouped[len(word)].append(idx)
        for length, ids in grouped.items():
            self.by_length[length] = np.array(sorted(ids), dtype=np.int64)

        # Sampling weights over lengths: proportional to the dictionary, with
        # the measured loss-heavy band (lengths 4-9) oversampled.
        lengths = sorted(self.by_length)
        weights = np.array([self.by_length[L].size for L in lengths], dtype=np.float64)
        low, high = cfg.short_word_range
        boost = np.array([cfg.short_word_boost if low <= L <= high else 1.0 for L in lengths])
        self.lengths = np.array(lengths, dtype=np.int64)
        self.length_probs = (weights * boost) / (weights * boost).sum()

        # Letter frequency prior, used to make sampled absent sets realistic.
        freq = np.zeros(26, dtype=np.float64)
        for word in dicts.fit:
            for ch in set(word):
                freq[ord(ch) - 97] += 1.0
        self.letter_freq = freq / freq.sum()

        # Per-length word matrices on device, for board reconstruction.
        self._word_rows: dict[int, torch.Tensor] = {}
        for length in lengths:
            rows = self.by_length[length]
            mat = np.stack(
                [np.frombuffer(dicts.fit[i].encode("ascii"), np.uint8).astype(np.int64) - 97
                 for i in rows]
            )
            self._word_rows[length] = torch.from_numpy(mat).to(self.device)

    # -- helpers -------------------------------------------------------------

    def _bucket_rows(self, length: int, word_ids: np.ndarray) -> np.ndarray:
        """Map global word ids to their row inside the length bucket."""
        return np.searchsorted(self.by_length[length], word_ids)

    def _label(
        self,
        length: int,
        word_ids: np.ndarray,
        guessed_bits: np.ndarray,
        values: np.ndarray,
        onpolicy: bool,
    ) -> tuple[np.ndarray, int]:
        """Run the teacher over a batch of same-length states and pack records."""
        bucket_rows = self._bucket_rows(length, word_ids)
        words = self._word_rows[length][torch.from_numpy(bucket_rows).to(self.device)]

        bits = torch.from_numpy(guessed_bits.astype(np.int64)).to(self.device)
        shifts = torch.arange(26, device=self.device, dtype=torch.long)
        guessed = ((bits.unsqueeze(1) >> shifts) & 1).to(torch.bool)

        word_presence = torch.zeros_like(guessed)
        word_presence.scatter_(1, words, True)
        present = guessed & word_presence
        absent = guessed & ~word_presence

        revealed = present.gather(1, words)
        board = torch.where(revealed, words, torch.full_like(words, 27))  # 27 == HOLE

        exclude = torch.from_numpy(bucket_rows.astype(np.int64)).to(self.device)
        out = self.teacher.posterior(length, board, present, absent, exclude_rows=exclude)

        target = out.presence
        if self.student is not None:
            lengths = torch.full(
                (word_ids.size,), length, dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                student_probs = self.student.presence(board, lengths, present, absent)
            target = blend_targets(
                target, student_probs, out.n_candidates, self.kappa_sd,
                guessed=present | absent,
            )

        n = word_ids.size
        rec = np.zeros(n, dtype=RECORD_DTYPE)
        rec["word_id"] = word_ids.astype(np.uint32)
        rec["guessed"] = guessed_bits.astype(np.uint32)
        rec["n_cand"] = np.clip(out.n_candidates.cpu().numpy(), 0, 65535).astype(np.uint16)
        rec["tier"] = out.tier.cpu().numpy().astype(np.uint8)
        rec["flags"] = FLAG_ONPOLICY if onpolicy else 0
        rec["target"] = target.cpu().numpy().astype(np.float16)
        rec["value"] = values.astype(np.float16)
        empty = int((out.n_candidates == 0).sum().item())
        return rec, empty

    def _sample_guessed(self, length: int, word_ids: np.ndarray) -> np.ndarray:
        """Vectorised reachable-state sampler for one length bucket.

        Returns a ``(S,)`` uint32 array of guessed-letter bitmasks.  Subsets are
        chosen by ranking random keys, which selects ``r`` items without
        replacement in one pass; the miss keys use the exponential race
        ``-log(U) / w``, giving frequency-weighted sampling without replacement.
        """
        rng = self.rng
        rows = self._bucket_rows(length, word_ids)
        chars = self._word_rows[length][torch.from_numpy(rows).to(self.device)].cpu().numpy()
        n = word_ids.size

        present = np.zeros((n, 26), dtype=bool)
        np.put_along_axis(present, chars, True, axis=1)
        k = present.sum(axis=1)

        # Reveal r of the word's own letters, 0 <= r <= k-1 (r == k is a win).
        n_reveal = np.floor(rng.random(n) * np.maximum(k, 1)).astype(np.int64)
        keys = rng.random((n, 26))
        keys[~present] = np.inf
        rank = keys.argsort(axis=1).argsort(axis=1)
        revealed = present & (rank < n_reveal[:, None])

        # Miss m letters not in the word, 0 <= m <= 5, weighted by frequency.
        n_miss = rng.integers(0, MAX_WRONG_GUESSES, size=n)
        weights = np.broadcast_to(self.letter_freq, (n, 26)).copy()
        race = -np.log(rng.random((n, 26)) + 1e-12) / np.maximum(weights, 1e-12)
        race[present] = np.inf
        rank_miss = race.argsort(axis=1).argsort(axis=1)
        missed = (~present) & (rank_miss < n_miss[:, None])

        bits = ((revealed | missed).astype(np.uint32) << np.arange(26, dtype=np.uint32)).sum(axis=1)
        return bits.astype(np.uint32)

    # -- off-policy warm start ------------------------------------------------

    def sample_offpolicy(
        self,
        n_states: int,
        batch_size: int = 8192,
        writer: "ShardWriter | None" = None,
    ) -> CollectionStats:
        """Sample reachable states uniformly over the game's state space.

        For a word with ``k`` distinct letters we draw the number of revealed
        letters ``r`` in ``0..k-1`` (``r == k`` is a finished game) and the
        number of misses ``m`` in ``0..MAX_WRONG_GUESSES-1``.  Revealed letters
        are a uniform subset of the word's letters; **misses are drawn from the
        letters not in the word in proportion to their corpus frequency**, so
        the absent sets look like the ones a competent policy produces.
        """
        stats = CollectionStats()
        start = time.time()
        produced = 0
        while produced < n_states:
            length = int(self.rng.choice(self.lengths, p=self.length_probs))
            rows = self.by_length[length]
            take = min(batch_size, n_states - produced)
            word_ids = self.rng.choice(rows, size=take, replace=True)

            guessed_bits = self._sample_guessed(length, word_ids)

            values = np.full(take, VALUE_UNKNOWN, dtype=np.float16)
            rec, empty = self._label(length, word_ids, guessed_bits, values, onpolicy=False)
            if writer is not None:
                writer.write(rec)
            produced += take
            stats.n_states += take
            stats.n_games += take
            stats.empty_tier1 += empty
        stats.seconds = time.time() - start
        return stats

    # -- DAgger on-policy rounds ----------------------------------------------

    def rollout_onpolicy(
        self,
        guess_fn_factory,
        n_states: int,
        batch_words: int = 4096,
        writer: "ShardWriter | None" = None,
    ) -> CollectionStats:
        """Roll out the current policy and record every state it visits.

        ``guess_fn_factory(length)`` must return a callable matching
        :class:`~hangman.simulator.GuessFn`.  Because the teacher needs
        same-length batches, rollouts are grouped by word length.

        Each recorded state receives the Monte-Carlo return of the game it came
        from (1.0 for a win, 0.0 for a loss), which is the training signal for
        the value head and the baseline for the RL stage.
        """
        stats = CollectionStats()
        start = time.time()
        produced = 0
        while produced < n_states:
            length = int(self.rng.choice(self.lengths, p=self.length_probs))
            rows = self.by_length[length]
            take = min(batch_words, rows.size)
            word_ids = self.rng.choice(rows, size=take, replace=False)
            words = [self.dicts.fit[i] for i in word_ids]

            game = BatchGame(words, device=self.device)
            guess_fn = guess_fn_factory(length)
            visited_guessed: list[np.ndarray] = []
            visited_rows: list[np.ndarray] = []

            for _ in range(32):
                active = game.active()
                if not bool(active.any()):
                    break
                idx = torch.nonzero(active, as_tuple=True)[0]
                bits = _mask_to_bits(game.present[idx] | game.absent[idx])
                visited_guessed.append(bits.cpu().numpy().astype(np.uint32))
                visited_rows.append(idx.cpu().numpy())
                guesses = guess_fn(
                    board=game.board(),
                    lengths=game.lengths,
                    present=game.present,
                    absent=game.absent,
                    lives=game.lives(),
                    active=active,
                )
                game.apply(guesses, active)

            won = game.solved().to(torch.float32).cpu().numpy()
            stats.n_games += take
            if not visited_guessed:
                continue

            all_rows = np.concatenate(visited_rows)
            all_bits = np.concatenate(visited_guessed)
            values = won[all_rows].astype(np.float16)
            rec, empty = self._label(
                length, word_ids[all_rows], all_bits, values, onpolicy=True
            )
            if writer is not None:
                writer.write(rec)
            produced += rec.size
            stats.n_states += rec.size
            stats.empty_tier1 += empty
        stats.seconds = time.time() - start
        return stats


def _mask_to_bits(mask: torch.Tensor) -> torch.Tensor:
    weights = 1 << torch.arange(26, device=mask.device, dtype=torch.long)
    return (mask.to(torch.long) * weights).sum(dim=1)


class ShardWriter:
    """Append-only writer that splits records into fixed-size ``.npy`` shards.

    Buffered records are written on ``flush``, on reaching ``shard_records``, on
    leaving the ``with`` block - and on SIGTERM/SIGINT.  The signal handler
    matters: a long collection pass killed by a timeout would otherwise discard
    every record it had produced, since a plain ``SIGTERM`` does not unwind the
    context manager.
    """

    def __init__(self, out_dir: str | Path, prefix: str, shard_records: int = 2_000_000) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_records = shard_records
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._shard = 0
        self.paths: list[Path] = []
        self.total = 0
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        import signal

        def _save_and_exit(signum, _frame):
            self.flush()
            print(f"ShardWriter: caught signal {signum}, flushed {self.total:,} records",
                  flush=True)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _save_and_exit)
            except (ValueError, OSError):
                # Not the main thread, or the platform disallows it: the context
                # manager still covers the ordinary exit paths.
                pass

    def write(self, records: np.ndarray) -> None:
        self._buffer.append(records)
        self._buffered += records.size
        self.total += records.size
        if self._buffered >= self.shard_records:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        data = np.concatenate(self._buffer)
        path = self.out_dir / f"{self.prefix}_{self._shard:04d}.npy"
        np.save(path, data)
        self.paths.append(path)
        self._shard += 1
        self._buffer = []
        self._buffered = 0

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

