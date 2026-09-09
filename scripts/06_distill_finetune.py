#!/usr/bin/env python
"""Stage 6: fine-tune a trained student on the distillation shards.

The shards written by ``scripts/05_distill.py`` carry the blended target
``alpha * bayes + (1 - alpha) * qwen`` on the states where the dictionary
oracle is blind, so fine-tuning on a pool that includes them *is* the
distillation objective - no new loss term is needed.

The result is kept only if it beats the input checkpoint on held-out words.
Reporting a regression is more useful than shipping one.

    python scripts/06_distill_finetune.py --ckpt artifacts/checkpoints/former_a_final.pt --steps 3000
"""

from __future__ import annotations

import argparse
import json

import torch

from hangman.config import CHECKPOINT_DIR, DATASET_DIR, LOG_DIR, METRIC_DIR, ensure_dirs
from hangman.dataset import shard_paths
from hangman.pipeline import build_teacher, dataset_for, evaluate_model, set_seed
from hangman.train import TrainState, load_checkpoint, save_checkpoint, train_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="lower than the main run: this is a refinement, not a restart")
    parser.add_argument("--distill-prefix", default="distill")
    parser.add_argument("--distill-mix", type=float, default=0.35,
                        help="target fraction of each batch drawn from the distillation shards")
    parser.add_argument("--eval-words", type=int, default=8000)
    parser.add_argument("--w-pos", type=float, default=None,
                        help="override the per-position loss weight. The noisy-OR measurement "
                             "(MODEL_REPORT section 6.5) says that head is the better presence "
                             "estimator, so the training weight of 0.40 is probably too low.")
    parser.add_argument("--no-distill", action="store_true",
                        help="refine on the existing pool only; use with --w-pos to test a "
                             "loss-weight change in isolation")
    parser.add_argument("--tag", default="distilled",
                        help="checkpoint suffix, so competing refinements do not overwrite")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_checkpoint(args.ckpt, use_ema=True)
    cfg.optim.lr = args.lr
    cfg.optim.warmup_steps = min(200, args.steps // 10)
    if args.w_pos is not None:
        print(f"w_pos {cfg.loss.w_pos} -> {args.w_pos}", flush=True)
        cfg.loss.w_pos = args.w_pos
    dicts, _, teacher = build_teacher(cfg)
    words = dicts.val[: args.eval_words]

    before = evaluate_model(cfg, model, words, tag=f"{cfg.run_name}/before-{args.tag}")
    print(before.summary(), flush=True)

    distill = [] if args.no_distill else shard_paths(DATASET_DIR, args.distill_prefix)
    if not distill and not args.no_distill:
        raise SystemExit(f"no shards with prefix {args.distill_prefix!r}; run scripts/05_distill.py")
    pool = shard_paths(DATASET_DIR, "offpolicy")
    for rnd in range(1, cfg.collect.dagger_rounds + 1):
        pool.extend(shard_paths(DATASET_DIR, f"onpolicy_{cfg.run_name}_r{rnd}"))
    pool.extend(distill)

    dataset = dataset_for(cfg, dicts, pool, onpolicy_mix=None)
    print(f"pool: {int(dataset.stats()['n_records']):,} records "
          f"({len(distill)} distillation shards)", flush=True)

    state, ema, _ = train_steps(
        model, dataset, cfg, args.steps, state=TrainState(),
        log_path=LOG_DIR / f"{cfg.run_name}_{args.tag}.json", log_every=200,
    )

    from hangman.train import clone_for_eval

    candidate = clone_for_eval(model, ema)
    after = evaluate_model(cfg, candidate, words, tag=f"{cfg.run_name}/after-{args.tag}")
    print(after.summary(), flush=True)

    kept = after.win_rate >= before.win_rate
    if kept:
        out = CHECKPOINT_DIR / f"{cfg.run_name}_{args.tag}.pt"
        save_checkpoint(out, model, cfg, state, ema)
        print(f"kept the refined model (+{after.win_rate - before.win_rate:.2f} points) -> {out}",
              flush=True)
    else:
        print(f"{args.tag} regressed ({before.win_rate:.2f}% -> {after.win_rate:.2f}%); "
              "keeping the original checkpoint", flush=True)

    (METRIC_DIR / f"{cfg.run_name}_{args.tag}_finetune.json").write_text(
        json.dumps({"before": before.to_dict(), "after": after.to_dict(), "kept": kept}, indent=2)
    )


if __name__ == "__main__":
    main()
