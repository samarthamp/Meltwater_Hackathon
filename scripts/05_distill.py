#!/usr/bin/env python
"""Stage 5: distil the Qwen3-4B teacher into the compact students.

Selective by design.  Scoring states with a 4B model costs roughly four orders
of magnitude more than scoring them with the dictionary oracle, so we spend it
only where the oracle is blind: states whose consistent candidate set is empty
(measured at 45% of the pool) and short words, where the measured baseline
loses most of its games.  Everywhere else the exact posterior is already the
right answer and an LLM opinion could only add noise.

The output is a new set of record shards whose target is the blend

    target = alpha * bayes_posterior + (1 - alpha) * qwen_posterior

so the students need no new loss plumbing: fine-tuning on these shards *is* the
distillation objective (Hinton et al., 2015), with the blend playing the role
of the usual hard/soft mixture.

    python scripts/05_distill.py --adapters artifacts/checkpoints/qwen_teacher_adapters.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from hangman.collect import RECORD_DTYPE, ShardWriter
from hangman.config import DATASET_DIR, ExperimentConfig, METRIC_DIR, ensure_dirs
from hangman.dataset import RecordDataset, shard_paths
from hangman.models.llm_teacher import QwenHangmanTeacher
from hangman.ngram import HOLE
from hangman.pipeline import build_teacher, set_seed
from hangman.simulator import PAD_POS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--keep-layers", type=int, default=18)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight on the Bayes posterior; 1-alpha goes to Qwen")
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--short-max-len", type=int, default=9)
    parser.add_argument("--prefix", default="distill")
    parser.add_argument("--source-prefixes", nargs="*", default=None,
                        help="record shard prefixes to draw candidate states from")
    return parser.parse_args()


def select_states(dataset: RecordDataset, max_states: int, short_max_len: int) -> np.ndarray:
    """Pick the states where an outside English prior can actually help.

    Two criteria, both measured rather than assumed:
      * ``n_cand == 0`` - no training-dictionary word is consistent, so the
        exact posterior has nothing to say (45% of the pool);
      * short words - lengths 4..9 carry almost all of the measured losses.
    """
    records = dataset.records
    lengths = dataset.word_lengths[records["word_id"]]
    blind = records["n_cand"] == 0
    short = (lengths >= 4) & (lengths <= short_max_len)
    chosen = np.flatnonzero(blind | short)
    if chosen.size > max_states:
        rng = np.random.default_rng(0)
        chosen = rng.choice(chosen, size=max_states, replace=False)
    return np.sort(chosen)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = ExperimentConfig(run_name=args.prefix)
    cfg.llm.keep_layers = args.keep_layers
    dicts, _, _ = build_teacher(cfg)

    prefixes = args.source_prefixes or ["offpolicy"] + [
        p.name.rsplit("_", 1)[0] for p in DATASET_DIR.glob("onpolicy_*_r*_0000.npy")
    ]
    shards: list = []
    for prefix in dict.fromkeys(prefixes):
        shards.extend(shard_paths(DATASET_DIR, prefix))
    if not shards:
        raise SystemExit("no source shards found")
    print(f"drawing from {len(shards)} shards", flush=True)

    dataset = RecordDataset(shards, dicts.fit, device=cfg.device)
    chosen = select_states(dataset, args.max_states, args.short_max_len)
    print(f"selected {chosen.size:,} of {len(dataset):,} states for LLM scoring", flush=True)

    model = QwenHangmanTeacher(cfg.llm).to(cfg.device)
    model.apply_lora()
    model.load_adapters(args.adapters)
    model.eval()

    lengths_all = dataset.word_lengths[dataset.records["word_id"]]
    by_length: dict[int, list[int]] = defaultdict(list)
    for idx in chosen:
        by_length[int(lengths_all[idx])].append(int(idx))

    started = time.time()
    done = 0
    agreement = []
    # Small shards, so an interrupted pass still leaves usable work on disk.
    with ShardWriter(DATASET_DIR, args.prefix, shard_records=100_000) as writer:
        for length in sorted(by_length):
            ids = np.array(by_length[length], dtype=np.int64)
            for offset in range(0, ids.size, args.batch_size):
                chunk = ids[offset : offset + args.batch_size]
                batch = dataset.batch(chunk)
                board = _board_from(batch)
                with torch.no_grad():
                    qwen = model.presence(
                        board, batch["lengths"], batch["present"], batch["absent"]
                    )
                bayes = batch["target_soft"]
                blended = args.alpha * bayes + (1.0 - args.alpha) * qwen
                blended = blended.masked_fill(batch["guessed"], 0.0)

                valid = ~batch["guessed"]
                truth = batch["target_hard"]
                pick_q = qwen.masked_fill(~valid, -1.0).argmax(dim=1)
                agreement.append(
                    float(truth.gather(1, pick_q.unsqueeze(1)).mean().item())
                )

                source = dataset.records[np.sort(chunk)]
                out = np.zeros(source.size, dtype=RECORD_DTYPE)
                for field in ("word_id", "guessed", "n_cand", "tier", "flags", "value"):
                    out[field] = source[field]
                out["target"] = blended.cpu().numpy().astype(np.float16)
                writer.write(out)
                done += source.size
            elapsed = time.time() - started
            print(
                f"  len {length:>2}: {done:,}/{chosen.size:,} scored "
                f"({done / max(elapsed, 1e-9):,.0f} states/s)",
                flush=True,
            )

    hit = float(np.mean(agreement)) if agreement else 0.0
    (METRIC_DIR / f"{args.prefix}.json").write_text(
        json.dumps(
            {
                "n_states": int(chosen.size),
                "alpha": args.alpha,
                "keep_layers": args.keep_layers,
                "qwen_top1_hit_rate": hit,
                "seconds": time.time() - started,
            },
            indent=2,
        )
    )
    print(f"Qwen top-1 hit rate on scored states: {hit:.4f}", flush=True)
    print(f"wrote {done:,} distillation records in {(time.time() - started) / 60:.1f} min", flush=True)


def _board_from(batch: dict) -> torch.Tensor:
    words = batch["word"]
    real = words >= 0
    revealed = batch["present"].gather(1, words.clamp_min(0)) & real
    board = torch.where(revealed, words, torch.full_like(words, HOLE))
    return torch.where(real, board, torch.full_like(words, PAD_POS))


if __name__ == "__main__":
    main()
