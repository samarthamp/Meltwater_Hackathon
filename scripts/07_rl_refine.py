#!/usr/bin/env python
"""Stage 7: refine a trained model's policy by self-play policy gradient.

    python scripts/07_rl_refine.py --ckpt artifacts/checkpoints/former_a_final.pt --steps 800
"""

from __future__ import annotations

import argparse

import torch

from hangman.config import CHECKPOINT_DIR, LOG_DIR, METRIC_DIR, ensure_dirs
from hangman.evaluate import save_report
from hangman.pipeline import build_teacher, evaluate_model, set_seed
from hangman.rl import refine
from hangman.train import load_checkpoint, save_checkpoint, TrainState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--games-per-step", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-words", type=int, default=8000)
    parser.add_argument("--tag", default="rl")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_checkpoint(args.ckpt, use_ema=True)
    cfg.rl.lr = args.lr
    cfg.rl.kl_coef = args.kl_coef
    dicts, _, teacher = build_teacher(cfg)

    words = dicts.val[: args.eval_words]
    before = evaluate_model(cfg, model, words, teacher=None, tag=f"{cfg.run_name}/before-rl")
    print(before.summary(), flush=True)

    reference, stats = refine(
        model,
        dicts,
        cfg.rl,
        steps=args.steps,
        games_per_step=args.games_per_step,
        temperature=args.temperature,
        device=cfg.device,
    )
    stats.save(LOG_DIR / f"{cfg.run_name}_{args.tag}.json")

    after = evaluate_model(cfg, model, words, teacher=None, tag=f"{cfg.run_name}/after-rl")
    print(after.summary(), flush=True)

    if after.win_rate >= before.win_rate:
        out = CHECKPOINT_DIR / f"{cfg.run_name}_rl.pt"
        save_checkpoint(out, model, cfg, TrainState(step=args.steps), ema=None)
        print(f"kept refined policy -> {out}", flush=True)
    else:
        # Policy-gradient refinement can overshoot; if it does, say so and keep
        # the supervised model rather than shipping a regression.
        print(
            f"RL regressed ({before.win_rate:.2f}% -> {after.win_rate:.2f}%); "
            "keeping the supervised checkpoint",
            flush=True,
        )

    for report in (before, after):
        save_report(report, METRIC_DIR / f"{cfg.run_name}_{report.tag.split('/')[-1]}.json")


if __name__ == "__main__":
    main()
