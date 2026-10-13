#!/usr/bin/env python
"""Choose the self-distillation gate by measuring target quality.

Each candidate gate blends the Bayes teacher with the student and is scored by
the top-1 hit rate of the resulting target: how often its highest-ranked
unguessed letter is in the word, which mirrors a greedy policy trained on it.

    python tools/blend_sweep.py --ckpt artifacts/checkpoints/former_a_final.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hangman.config import DATASET_DIR, METRIC_DIR, ensure_dirs
from hangman.dataset import RecordDataset, shard_paths
from hangman.ngram import HOLE
from hangman.pipeline import build_teacher, set_seed
from hangman.policy import ModelSource
from hangman.selfdistill import blend_targets, describe, top1_hit_rate
from hangman.simulator import PAD_POS
from hangman.train import load_checkpoint

GATES = [0.0, 5.0, 20.0, 50.0, 200.0, 1000.0, float("inf")]


def board_from(batch: dict) -> torch.Tensor:
    words = batch["word"]
    real = words >= 0
    revealed = batch["present"].gather(1, words.clamp_min(0)) & real
    board = torch.where(revealed, words, torch.full_like(words, HOLE))
    return torch.where(real, board, torch.full_like(words, PAD_POS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="artifacts/checkpoints/former_a_final.pt")
    parser.add_argument("--noisy-or-weight", type=float, default=0.4,
                        help="the value fitted for this checkpoint on validation")
    parser.add_argument("--sample", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_checkpoint(args.ckpt, use_ema=True)
    cfg.optim.compile = False
    dicts, _, _ = build_teacher(cfg)
    student_src = ModelSource(model, noisy_or_weight=args.noisy_or_weight)

    prefixes = ["offpolicy"] + sorted(
        {p.name.rsplit("_", 1)[0] for p in DATASET_DIR.glob("onpolicy_former_a_r*.npy")}
    )
    paths: list = []
    for prefix in prefixes:
        paths.extend(shard_paths(DATASET_DIR, prefix))
    dataset = RecordDataset(paths, dicts.fit, device=cfg.device, seed=1)
    print(f"pool: {len(dataset):,} records from {len(paths)} shards", flush=True)

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(dataset), size=min(args.sample, len(dataset)), replace=False))

    sums = {g: 0.0 for g in GATES}
    sums["bayes"] = 0.0
    sums["student"] = 0.0
    n_batches = 0
    for start in range(0, idx.size, args.batch_size):
        chunk = idx[start : start + args.batch_size]
        batch = dataset.batch(chunk)
        with torch.no_grad():
            student = student_src.presence(
                board_from(batch), batch["lengths"], batch["present"], batch["absent"]
            )
        bayes = batch["target_soft"]
        guessed = batch["guessed"]
        truth = batch["target_hard"]
        n_cand = batch["n_candidates"]

        sums["bayes"] += top1_hit_rate(bayes, guessed, truth)
        sums["student"] += top1_hit_rate(student, guessed, truth)
        for gate in GATES:
            blended = blend_targets(bayes, student, n_cand, gate, guessed)
            sums[gate] += top1_hit_rate(blended, guessed, truth)
        n_batches += 1

    results = {k: v / n_batches for k, v in sums.items()}
    print(f"\nTarget quality on {idx.size:,} sampled states (top-1 hit rate)\n")
    print(f"  {'source':<24} {'top-1':>7}")
    print(f"  {'-'*24} {'-'*7}")
    print(f"  {'Bayes posterior alone':<24} {results['bayes']:>7.4f}")
    print(f"  {'student alone':<24} {results['student']:>7.4f}")
    for gate in GATES:
        print(f"  {describe(gate):<24} {results[gate]:>7.4f}")

    best_gate = max(GATES, key=lambda g: results[g])
    best = results[best_gate]
    floor = max(results["bayes"], results["student"])
    verdict = "blend wins" if best > floor + 1e-4 else "no blend beats the better single source"
    print(f"\n  best: {describe(best_gate)} at {best:.4f}   ({verdict})")
    if best <= floor + 1e-4:
        best_gate = float("inf") if results["student"] >= results["bayes"] else 0.0
        print(f"  -> falling back to {describe(best_gate)}")

    (METRIC_DIR / "blend_sweep.json").write_text(json.dumps({
        "checkpoint": args.ckpt,
        "n_states": int(idx.size),
        "bayes_alone": results["bayes"],
        "student_alone": results["student"],
        "gates": {describe(g): results[g] for g in GATES},
        "chosen_kappa": best_gate,
        "chosen_top1": results[best_gate],
    }, indent=2))
    print(f"\nCHOSEN_KAPPA={best_gate:g}")


if __name__ == "__main__":
    main()
