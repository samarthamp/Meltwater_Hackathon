#!/usr/bin/env python
"""Stage 4: adapt Qwen3-4B into a hangman posterior teacher.

Two phases, following the LLM2Vec recipe adapted to this task:

1. **MNTP** - masked-character adaptation.  The backbone has only ever been run
   causally; enabling bidirectional attention changes what every position sees,
   so it needs a short adaptation phase before its representations mean
   anything.  We train on the per-position head only, predicting the true letter
   at hidden board positions.
2. **SFT** - fit the presence head (and continue the per-position head) to the
   Bayes teacher's soft posteriors on the collected states.

Only LoRA adapters and the two heads are trained; the base weights stay frozen
in bf16.

    python scripts/04_finetune_llm.py --keep-layers 18 --mntp-steps 1500 --sft-steps 6000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hangman.config import (
    CHECKPOINT_DIR,
    ExperimentConfig,
    LOG_DIR,
    METRIC_DIR,
    ensure_dirs,
)
from hangman.dataset import shard_paths
from hangman.models.llm_teacher import QwenHangmanTeacher
from hangman.ngram import HOLE
from hangman.pipeline import build_teacher, dataset_for, set_seed
from hangman.simulator import PAD_POS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-layers", type=int, default=18)
    parser.add_argument("--mntp-steps", type=int, default=1500)
    parser.add_argument("--sft-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", default="qwen_teacher")
    parser.add_argument("--shard-prefix", default="offpolicy")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="wall-clock budget; training stops cleanly at the limit and keeps "
                             "the adapters trained so far")
    parser.add_argument("--save-every", type=int, default=250,
                        help="checkpoint the adapters every N steps, so an interrupted run is "
                             "not a wasted one")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = ExperimentConfig(run_name=args.out)
    cfg.llm.keep_layers = args.keep_layers
    cfg.llm.lora_r = args.lora_r
    cfg.llm.batch_size = args.batch_size
    cfg.llm.lr = args.lr

    dicts, _, _ = build_teacher(cfg)
    dataset = dataset_for(cfg, dicts, shard_paths_or_die(args.shard_prefix), onpolicy_mix=None)
    print(f"pool: {dataset.stats()}", flush=True)

    model = QwenHangmanTeacher(cfg.llm).to(cfg.device)
    model.apply_lora()
    trainable, total = model.trainable_parameters()
    layers = model.backbone
    for attr in ("base_model", "model"):
        layers = getattr(layers, attr, layers)
    n_layers = len(getattr(layers, "layers", []))
    print(
        f"Qwen3 teacher: {n_layers} of 36 blocks kept, {total / 1e9:.2f}B total, "
        f"{trainable / 1e6:.1f}M trainable ({100 * trainable / total:.2f}%)",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    history: list[dict] = []
    out_path = CHECKPOINT_DIR / f"{args.out}_adapters.pt"
    deadline = None if args.max_minutes is None else time.time() + args.max_minutes * 60.0

    def run_phase(name: str, steps: int, mntp_only: bool) -> None:
        model.train()
        running = {"loss": 0.0, "set": 0.0, "pos": 0.0}
        started = time.time()
        for step in range(steps):
            if deadline is not None and time.time() > deadline:
                print(f"  [{name}] wall-clock budget reached at step {step}; stopping cleanly",
                      flush=True)
                model.save_adapters(out_path)
                break
            warm = min(1.0, (step + 1) / max(args.warmup, 1))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warm

            batch = dataset.batch(dataset.sample_indices(args.batch_size))
            out = model(
                board_from(batch), batch["lengths"], batch["present"], batch["absent"]
            )

            labels = batch["char_target"]
            width = min(out["pos_logits"].shape[1], labels.shape[1])
            pos_loss = F.cross_entropy(
                out["pos_logits"][:, :width].reshape(-1, 26).float(),
                labels[:, :width].reshape(-1),
                ignore_index=-1,
            )
            if torch.isnan(pos_loss):
                pos_loss = pos_loss.new_zeros(())

            valid = ~batch["guessed"]
            set_loss = pos_loss.new_zeros(())
            if not mntp_only:
                bce = F.binary_cross_entropy_with_logits(
                    out["set_logits"], batch["target_soft"], reduction="none"
                )
                set_loss = (bce * valid).sum(dim=1).div(valid.sum(dim=1).clamp_min(1)).mean()

            loss = pos_loss if mntp_only else (set_loss + 0.3 * pos_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            running["loss"] += float(loss.detach())
            running["set"] += float(set_loss.detach())
            running["pos"] += float(pos_loss.detach())
            if (step + 1) % args.log_every == 0:
                row = {k: v / args.log_every for k, v in running.items()}
                row.update(phase=name, step=step + 1,
                           states_per_sec=(step + 1) * args.batch_size / (time.time() - started))
                history.append(row)
                print(
                    f"  [{name}] {step + 1:>6d}/{steps}  loss {row['loss']:.4f}  "
                    f"set {row['set']:.4f}  pos {row['pos']:.4f}  "
                    f"{row['states_per_sec']:.0f} states/s",
                    flush=True,
                )
                running = {k: 0.0 for k in running}
                (LOG_DIR / f"{args.out}_train.json").write_text(json.dumps(history, indent=1))
            if args.save_every and (step + 1) % args.save_every == 0:
                model.save_adapters(out_path)

    if args.mntp_steps:
        run_phase("mntp", args.mntp_steps, mntp_only=True)
    if args.sft_steps:
        run_phase("sft", args.sft_steps, mntp_only=False)

    model.save_adapters(out_path)
    (METRIC_DIR / f"{args.out}.json").write_text(
        json.dumps(
            {
                "keep_layers": args.keep_layers,
                "lora_r": args.lora_r,
                "trainable_params": trainable,
                "total_params": total,
                "mntp_steps": args.mntp_steps,
                "sft_steps": args.sft_steps,
                "history_tail": history[-5:],
            },
            indent=2,
        )
    )
    print(f"saved adapters -> {out_path}", flush=True)


def board_from(batch: dict) -> torch.Tensor:
    """Rebuild the board tensor from a record batch.

    ``RecordDataset`` hands back the word and the present/absent split; the
    board is one gather away and is what the LLM codec consumes.
    """
    words = batch["word"]
    real = words >= 0
    revealed = batch["present"].gather(1, words.clamp_min(0)) & real
    board = torch.where(revealed, words, torch.full_like(words, HOLE))
    return torch.where(real, board, torch.full_like(words, PAD_POS))


def shard_paths_or_die(prefix: str) -> list[Path]:
    from hangman.config import DATASET_DIR

    paths = shard_paths(DATASET_DIR, prefix)
    if not paths:
        raise SystemExit(f"no shards with prefix {prefix!r}; run scripts/01_build_data.py")
    return paths


if __name__ == "__main__":
    main()
