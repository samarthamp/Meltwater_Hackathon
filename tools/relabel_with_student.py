#!/usr/bin/env python
"""Relabel the training pool with a trained model's own predictions.

Generation one was labelled by an exact dictionary posterior.  Measured on the
same states, that posterior scores 0.6813 top-1 and the model it trained scores
0.7568 - so for generation two the model is the better teacher, and
``tools/blend_sweep.py`` found no gated blend of the two that beats the model
alone.  This writes a copy of the pool whose targets come from the model.

Only the ``target`` field changes.  ``word_id``, ``guessed``, ``n_cand``,
``tier``, ``flags`` and ``value`` are copied through unchanged, and that is
asserted rather than assumed.

    python tools/relabel_with_student.py --ckpt artifacts/checkpoints/former_a_final.pt
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from hangman.collect import RECORD_DTYPE, ShardWriter
from hangman.config import DATASET_DIR, METRIC_DIR, ensure_dirs
from hangman.dataset import RecordDataset, shard_paths
from hangman.ngram import HOLE
from hangman.pipeline import build_teacher, set_seed
from hangman.policy import ModelSource
from hangman.selfdistill import blend_targets
from hangman.simulator import PAD_POS
from hangman.train import load_checkpoint

PASSTHROUGH = ("word_id", "guessed", "n_cand", "tier", "flags", "value")


def board_from(batch: dict) -> torch.Tensor:
    words = batch["word"]
    real = words >= 0
    revealed = batch["present"].gather(1, words.clamp_min(0)) & real
    board = torch.where(revealed, words, torch.full_like(words, HOLE))
    return torch.where(real, board, torch.full_like(words, PAD_POS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="artifacts/checkpoints/former_a_final.pt")
    parser.add_argument("--noisy-or-weight", type=float, default=0.4)
    parser.add_argument("--kappa", type=float, default=float("inf"),
                        help="weight gate on the Bayes posterior; inf = pure student")
    parser.add_argument("--prefixes", nargs="*", default=None)
    parser.add_argument("--out-prefix", default="sd")
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_checkpoint(args.ckpt, use_ema=True)
    cfg.optim.compile = False
    dicts, _, _ = build_teacher(cfg)
    student = ModelSource(model, noisy_or_weight=args.noisy_or_weight)

    prefixes = args.prefixes
    if prefixes is None:
        prefixes = ["offpolicy"] + sorted(
            {p.name.rsplit("_", 1)[0] for p in DATASET_DIR.glob("onpolicy_former_a_r*.npy")}
        )
    print(f"relabelling prefixes: {prefixes}", flush=True)

    started = time.time()
    total = 0
    with ShardWriter(DATASET_DIR, args.out_prefix, shard_records=2_000_000) as writer:
        for prefix in prefixes:
            paths = shard_paths(DATASET_DIR, prefix)
            if not paths:
                print(f"  {prefix}: no shards, skipping", flush=True)
                continue
            for path in paths:
                source = np.load(path)
                dataset = RecordDataset([path], dicts.fit, device=cfg.device)
                out = np.zeros(source.size, dtype=RECORD_DTYPE)
                for field in PASSTHROUGH:
                    out[field] = source[field]

                for start in range(0, source.size, args.batch_size):
                    idx = np.arange(start, min(start + args.batch_size, source.size))
                    batch = dataset.batch(idx)
                    with torch.no_grad():
                        probs = student.presence(
                            board_from(batch), batch["lengths"],
                            batch["present"], batch["absent"],
                        )
                    target = blend_targets(
                        batch["target_soft"], probs, batch["n_candidates"],
                        args.kappa, guessed=batch["guessed"],
                    )
                    out["target"][idx] = target.cpu().numpy().astype(np.float16)

                # The only field that may differ is the target.
                for field in PASSTHROUGH:
                    assert np.array_equal(out[field], source[field]), f"{field} changed in {path.name}"
                writer.write(out)
                total += out.size
                rate = total / max(time.time() - started, 1e-9)
                print(f"  {path.name}: {out.size:,} records  "
                      f"({total:,} done, {rate:,.0f}/s)", flush=True)

    elapsed = time.time() - started
    print(f"\nrelabelled {total:,} records in {elapsed/60:.1f} min "
          f"-> {len(writer.paths)} shards with prefix {args.out_prefix!r}", flush=True)
    (METRIC_DIR / "relabel.json").write_text(json.dumps({
        "checkpoint": args.ckpt, "kappa": args.kappa, "n_records": total,
        "minutes": elapsed / 60, "prefixes": prefixes, "out_prefix": args.out_prefix,
    }, indent=2))


if __name__ == "__main__":
    main()
