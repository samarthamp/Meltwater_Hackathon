#!/usr/bin/env python
"""Stage 2: supervised warm start plus DAgger self-play rounds for one model.

Run once per ensemble member; ``scripts/03_train_all.sh`` trains the full set.

    python scripts/02_train.py --run hangmanformer_a --arch transformer --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hangman.config import (
    CHECKPOINT_DIR,
    ExperimentConfig,
    LOG_DIR,
    METRIC_DIR,
    ensure_dirs,
)
from hangman.dataset import shard_paths
from hangman.pipeline import (
    build_teacher,
    checkpoint_path,
    dataset_for,
    evaluate_model,
    set_seed,
    stage_onpolicy_dataset,
)
from hangman.models import build_model
from hangman.train import clone_for_eval, save_checkpoint, train_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name, used for artifact filenames")
    parser.add_argument("--arch", default="transformer",
                        choices=["transformer", "bigru", "dilated_cnn"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1408)
    parser.add_argument("--rnn-layers", type=int, default=4)
    parser.add_argument("--cnn-layers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--pretrain-steps", type=int, default=12000)
    parser.add_argument("--dagger-steps", type=int, default=6000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-states", type=int, default=6_000_000)
    parser.add_argument("--onpolicy-mix", type=float, default=0.70)
    parser.add_argument("--eval-words", type=int, default=6000,
                        help="validation words played after each stage (0 disables)")
    parser.add_argument("--no-film", action="store_true")
    parser.add_argument("--no-absent", action="store_true",
                        help="ablation: hide the absent-letter evidence from the model")
    parser.add_argument("--offpolicy-only", action="store_true",
                        help="ablation: skip DAgger, train on random reachable states only")
    parser.add_argument("--w-pos", type=float, default=None,
                        help="per-position loss weight (default 0.40)")
    parser.add_argument("--pool-prefix", default="offpolicy",
                        help="shard prefix for the warm-start pool; use 'sd' for the pool "
                             "relabelled by tools/relabel_with_student.py")
    parser.add_argument("--self-distill-ckpt", default="",
                        help="previous-generation checkpoint used to label the DAgger rounds")
    parser.add_argument("--kappa-sd", type=float, default=float("inf"),
                        help="gate on the Bayes posterior when self-distilling; inf = pure student")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(run_name=args.run, seed=args.seed)
    cfg.model.name = args.run
    cfg.model.arch = args.arch
    cfg.model.d_model = args.d_model
    cfg.model.n_layers = args.layers
    cfg.model.n_heads = args.heads
    cfg.model.d_ff = args.d_ff
    cfg.model.rnn_layers = args.rnn_layers
    cfg.model.cnn_layers = args.cnn_layers
    cfg.model.use_film = not args.no_film
    cfg.optim.batch_size = args.batch_size
    cfg.optim.lr = args.lr
    cfg.optim.warmup_steps = args.warmup_steps
    cfg.collect.dagger_rounds = args.dagger_rounds
    cfg.collect.onpolicy_states_per_round = args.dagger_states
    cfg.collect.offpolicy_mix = 1.0 - args.onpolicy_mix
    if args.w_pos is not None:
        cfg.loss.w_pos = args.w_pos
    cfg.selfdistill.enabled = bool(args.self_distill_ckpt)
    cfg.selfdistill.kappa_sd = args.kappa_sd
    cfg.selfdistill.teacher_ckpt = args.self_distill_ckpt
    return cfg


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = build_config(args)
    dicts, _, teacher = build_teacher(cfg)
    print(f"[{args.run}] {dicts.summary()}", flush=True)

    model = build_model(cfg.model).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.run}] {args.arch}: {n_params / 1e6:.2f}M parameters", flush=True)
    for name, count, kind in model.parameter_report():
        print(f"    {name:<16} {count:>12,}  {kind}", flush=True)

    # Self-distillation teacher, reused for every DAgger round so new states share
    # the relabelled warm-start pool's target source.
    student = None
    if cfg.selfdistill.enabled:
        from hangman.policy import ModelSource
        from hangman.train import load_checkpoint as _load

        teacher_model, _ = _load(cfg.selfdistill.teacher_ckpt, use_ema=True)
        student = ModelSource(
            teacher_model.eval(),
            noisy_or_weight=cfg.selfdistill.teacher_noisy_or_weight,
        )
        print(f"[{args.run}] self-distilling from {cfg.selfdistill.teacher_ckpt} "
              f"(kappa={cfg.selfdistill.kappa_sd})", flush=True)

    eval_words = dicts.val[: args.eval_words] if args.eval_words else []
    metrics: list[dict] = []
    state = ema = optimizer = None

    # --- warm start on the reachable off-policy pool ------------------------
    shards = shard_paths(cfg_dataset_dir(), args.pool_prefix)
    if not shards:
        raise SystemExit(
            f"no shards with prefix {args.pool_prefix!r}; run scripts/01_build_data.py "
            "(or tools/relabel_with_student.py for the 'sd' pool)"
        )
    dataset = dataset_for(cfg, dicts, shards, onpolicy_mix=None, seed=args.seed)
    print(f"[{args.run}] warm-start pool: {dataset.stats()}", flush=True)

    started = time.time()
    state, ema, optimizer = train_steps(
        model, dataset, cfg, args.pretrain_steps,
        state=state, ema=ema, optimizer=optimizer,
        log_path=LOG_DIR / f"{args.run}_pretrain.json",
    )
    save_checkpoint(checkpoint_path(args.run, "pretrain"), model, cfg, state, ema)
    if eval_words:
        report = evaluate_model(
            cfg, clone_for_eval(model, ema), eval_words, teacher=None,
            tag=f"{args.run}/offpolicy-only",
        )
        print(f"[{args.run}] {report.summary()}", flush=True)
        metrics.append(report.to_dict())

    # --- DAgger rounds -------------------------------------------------------
    if not args.offpolicy_only:
        pool = list(shards)
        for rnd in range(1, args.dagger_rounds + 1):
            eval_model = clone_for_eval(model, ema)
            new_shards = stage_onpolicy_dataset(
                cfg, dicts, teacher, eval_model, round_idx=rnd,
                n_states=args.dagger_states,
                student=student, kappa_sd=cfg.selfdistill.kappa_sd,
            )
            pool.extend(new_shards)
            dataset = dataset_for(
                cfg, dicts, pool, onpolicy_mix=args.onpolicy_mix, seed=args.seed + rnd
            )
            print(f"[{args.run}] round {rnd} pool: {dataset.stats()}", flush=True)
            state, ema, optimizer = train_steps(
                model, dataset, cfg, args.dagger_steps,
                state=state, ema=ema, optimizer=optimizer,
                log_path=LOG_DIR / f"{args.run}_dagger{rnd}.json",
            )
            save_checkpoint(checkpoint_path(args.run, f"dagger{rnd}"), model, cfg, state, ema)
            if eval_words:
                report = evaluate_model(
                    cfg, clone_for_eval(model, ema), eval_words, teacher=None,
                    tag=f"{args.run}/dagger-r{rnd}",
                )
                print(f"[{args.run}] {report.summary()}", flush=True)
                metrics.append(report.to_dict())

    save_checkpoint(checkpoint_path(args.run, "final"), model, cfg, state, ema,
                    extra={"n_params": n_params})
    (METRIC_DIR / f"{args.run}_train.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{args.run}] done in {(time.time() - started) / 60:.1f} min", flush=True)


def cfg_dataset_dir() -> Path:
    from hangman.config import DATASET_DIR

    return DATASET_DIR


if __name__ == "__main__":
    main()
