"""Stage functions for the end-to-end pipeline.

Each stage is a plain function so the notebook and the CLI scripts drive the
same code path.  Artifacts land under ``artifacts/`` and every stage is
resumable: re-running a completed stage loads what is already on disk.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .collect import ShardWriter, StateCollector
from .config import (
    CHECKPOINT_DIR,
    DATASET_DIR,
    ExperimentConfig,
    METRIC_DIR,
    ensure_dirs,
)
from .data import Dictionaries, build_dictionaries
from .dataset import RecordDataset, shard_paths
from .evaluate import EvalReport, evaluate_words
from .ngram import NgramModel
from .policy import ModelSource, build_policy
from .teacher import BayesTeacher

NGRAM_FILENAME = "ngram.pkl"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_teacher(
    cfg: ExperimentConfig,
    dicts: Dictionaries | None = None,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[Dictionaries, NgramModel, BayesTeacher]:
    """Load or construct the dictionaries, the n-gram tables and the oracle."""
    ensure_dirs()
    dicts = dicts or build_dictionaries(cfg.data)
    cache_dir = Path(cache_dir or DATASET_DIR)
    ngram_path = cache_dir / f"{NGRAM_FILENAME.removesuffix('.pkl')}_{dicts.digest}_{len(dicts.fit)}.pkl"

    if ngram_path.exists() and not rebuild:
        ngram = NgramModel.load(ngram_path)
    else:
        started = time.time()
        ngram = NgramModel(cfg.teacher).fit(dicts.fit, max_len=29, min_count=2)
        ngram.save(ngram_path)
        print(
            f"n-gram fitted in {time.time() - started:.1f}s "
            f"({ngram.keys.size:,} contexts) -> {ngram_path.name}",
            flush=True,
        )
    teacher = BayesTeacher(dicts.buckets, ngram, cfg.teacher, device=cfg.device)
    return dicts, ngram, teacher


def stage_offpolicy_dataset(
    cfg: ExperimentConfig,
    dicts: Dictionaries,
    teacher: BayesTeacher,
    n_states: int | None = None,
    prefix: str = "offpolicy",
    overwrite: bool = False,
) -> list[Path]:
    """Generate the reachable-state warm-start pool."""
    existing = shard_paths(DATASET_DIR, prefix)
    if existing and not overwrite:
        print(f"reusing {len(existing)} existing {prefix} shards", flush=True)
        return existing

    n_states = n_states or cfg.collect.offpolicy_states
    collector = StateCollector(dicts, teacher, cfg.collect, device=cfg.device)
    with ShardWriter(DATASET_DIR, prefix) as writer:
        stats = collector.sample_offpolicy(n_states, batch_size=8192, writer=writer)
    print(stats.report(prefix), flush=True)
    (METRIC_DIR / f"collect_{prefix}.json").write_text(
        json.dumps(
            {
                "n_states": stats.n_states,
                "seconds": stats.seconds,
                "empty_tier1_frac": stats.empty_tier1 / max(stats.n_states, 1),
            },
            indent=2,
        )
    )
    return writer.paths


def stage_onpolicy_dataset(
    cfg: ExperimentConfig,
    dicts: Dictionaries,
    teacher: BayesTeacher,
    model: torch.nn.Module,
    round_idx: int,
    n_states: int | None = None,
    noisy_or_weight: float = 0.25,
    student=None,
    kappa_sd: float = float("inf"),
) -> list[Path]:
    """One DAgger round: roll the current policy out and label what it visits.

    Shards are namespaced by run, because DAgger states are on-policy *for a
    particular model* - sharing them across ensemble members would reintroduce
    exactly the distribution mismatch the round exists to remove.
    """
    n_states = n_states or cfg.collect.onpolicy_states_per_round
    prefix = f"onpolicy_{cfg.run_name}_r{round_idx}"
    existing = shard_paths(DATASET_DIR, prefix)
    if existing:
        print(f"reusing {len(existing)} existing {prefix} shards", flush=True)
        return existing

    collector = StateCollector(
        dicts, teacher, cfg.collect, device=cfg.device,
        student=student, kappa_sd=kappa_sd,
    )
    source = ModelSource(model.eval(), noisy_or_weight=noisy_or_weight)

    def factory(length: int):
        return build_policy(source, cfg.policy, teacher=None, length=length)

    with ShardWriter(DATASET_DIR, prefix) as writer:
        stats = collector.rollout_onpolicy(factory, n_states, batch_words=4096, writer=writer)
    print(stats.report(prefix), flush=True)
    return writer.paths


def evaluate_model(
    cfg: ExperimentConfig,
    model: torch.nn.Module,
    words: list[str],
    teacher: BayesTeacher | None = None,
    tag: str = "",
    noisy_or_weight: float = 0.25,
    verbose: bool = False,
) -> EvalReport:
    """Play a word list under the configured policy."""
    source = ModelSource(model.eval(), noisy_or_weight=noisy_or_weight)

    def factory(length: int):
        return build_policy(source, cfg.policy, teacher=teacher, length=length)

    return evaluate_words(words, factory, device=cfg.device, tag=tag, verbose=verbose)


def evaluate_teacher(
    cfg: ExperimentConfig,
    teacher: BayesTeacher,
    words: list[str],
    tag: str = "bayes-teacher",
) -> EvalReport:
    """Baseline: the label generator playing its own greedy policy."""

    def factory(length: int):
        def fn(board, lengths, present, absent, lives, active):
            trimmed = board[:, :length].clamp_min(0)
            return teacher.posterior(length, trimmed, present, absent).presence.argmax(dim=1)

        return fn

    return evaluate_words(words, factory, device=cfg.device, tag=tag)


def evaluate_ngram(
    cfg: ExperimentConfig,
    teacher: BayesTeacher,
    words: list[str],
    tag: str = "ngram-only",
) -> EvalReport:
    """Baseline: the interpolated n-gram backoff on its own."""

    def factory(length: int):
        def fn(board, lengths, present, absent, lives, active):
            trimmed = board[:, :length].clamp_min(0)
            return teacher._ngram_batch(length, trimmed, present, absent).argmax(dim=1)

        return fn

    return evaluate_words(words, factory, device=cfg.device, tag=tag)


def evaluate_frequency(cfg: ExperimentConfig, dicts: Dictionaries, words: list[str]) -> EvalReport:
    """Baseline: the static letter-frequency order, i.e. sample_submission.csv."""
    from .data import unique_letter_frequency

    order = torch.from_numpy(
        np.argsort(-unique_letter_frequency(dicts.fit))
    ).to(cfg.device)

    def factory(length: int):
        def fn(board, lengths, present, absent, lives, active):
            guessed = present | absent
            rank = torch.full((26,), 1e9, device=guessed.device)
            rank[order] = torch.arange(26, device=guessed.device, dtype=torch.float32)
            scores = rank.unsqueeze(0).expand_as(guessed).clone()
            scores[guessed] = 1e9
            return scores.argmin(dim=1)

        return fn

    return evaluate_words(words, factory, device=cfg.device, tag="static-frequency")


def dataset_for(
    cfg: ExperimentConfig,
    dicts: Dictionaries,
    shards: list[Path],
    onpolicy_mix: float | None,
    seed: int = 0,
) -> RecordDataset:
    return RecordDataset(
        shards,
        dicts.fit,
        device=cfg.device,
        short_word_range=cfg.collect.short_word_range,
        short_word_boost=cfg.collect.short_word_boost,
        onpolicy_mix=onpolicy_mix,
        seed=seed,
    )


def split_validation(
    words: list[str], n_fit: int, seed: int = 7
) -> tuple[list[str], list[str]]:
    """Split the validation dictionary into a fitting set and a reporting set.

    The split must be **random, not positional**.  ``build_dictionaries`` sorts
    the validation list for reproducibility, so ``val[:8000]`` is the words
    beginning a-f and ``val[-4000:]`` is s-z - and those differ substantially in
    difficulty.  Measured with the primary member: 65.29% on the alphabetical
    head against 71.88% on the alphabetical tail, a 6.6-point gap that has
    nothing to do with the model.  Fitting on one and reporting on the other
    would be comparing two different test sets.

    Returns ``(fit_words, report_words)``, disjoint, from a seeded permutation.
    """
    rng = random.Random(seed)
    shuffled = list(words)
    rng.shuffle(shuffled)
    n_fit = min(n_fit, max(len(shuffled) // 4, 1))
    return shuffled[:n_fit], shuffled[n_fit:]


def checkpoint_path(run_name: str, tag: str = "final") -> Path:
    return CHECKPOINT_DIR / f"{run_name}_{tag}.pt"

