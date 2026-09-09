#!/usr/bin/env python
"""Compare the LLM teacher against the dictionary oracle on the states it scored.

The distillation stage reports the Qwen teacher's top-1 hit rate, but a bare
number is not interpretable: 0.53 is excellent if the alternative is 0.40 and
useless if the alternative is 0.60.  This measures all three sources on exactly
the same states -

* the **Bayes teacher**, whose target the records already carry;
* the **Qwen teacher**, recovered from the blend
  ``blended = alpha * bayes + (1 - alpha) * qwen``;
* the **student** network, run over the same boards;

and reports how often each one's top unguessed letter is genuinely in the word.
The states are the ones distillation selected: those where the dictionary
oracle is blind, plus short words.

    python tools/teacher_comparison.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from hangman.config import DATASET_DIR, METRIC_DIR, ensure_dirs
from hangman.dataset import RecordDataset, shard_paths
from hangman.pipeline import build_teacher, set_seed
from hangman.policy import ModelSource
from hangman.train import load_checkpoint


def top1_hit(scores: torch.Tensor, guessed: torch.Tensor, truth: torch.Tensor) -> float:
    """Fraction of states whose top unguessed letter is really in the word."""
    pick = scores.masked_fill(guessed, -1e9).argmax(dim=1)
    return float(truth.gather(1, pick.unsqueeze(1)).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="artifacts/checkpoints/former_a_final.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sample", type=int, default=60000)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = load_checkpoint(args.ckpt, use_ema=True)
    cfg.optim.compile = False
    dicts, _, _ = build_teacher(cfg)

    distill_paths = shard_paths(DATASET_DIR, "distill")
    if not distill_paths:
        raise SystemExit("no distillation shards found")
    distill = np.concatenate([np.load(p) for p in distill_paths])

    source_paths = shard_paths(DATASET_DIR, "offpolicy")
    for prefix in sorted({p.name.rsplit("_", 1)[0] for p in DATASET_DIR.glob("onpolicy_*_r*.npy")}):
        source_paths.extend(shard_paths(DATASET_DIR, prefix))
    source = np.concatenate([np.load(p, mmap_mode="r") for p in source_paths])

    # Match distillation records back to their source rows on (word_id, guessed).
    key_src = source["word_id"].astype(np.int64) * (1 << 26) + source["guessed"].astype(np.int64)
    order = np.argsort(key_src, kind="stable")
    key_sorted = key_src[order]
    key_dst = distill["word_id"].astype(np.int64) * (1 << 26) + distill["guessed"].astype(np.int64)

    rng = np.random.default_rng(0)
    take = rng.choice(distill.size, size=min(args.sample, distill.size), replace=False)
    pos = np.searchsorted(key_sorted, key_dst[take])
    pos = np.clip(pos, 0, key_sorted.size - 1)
    matched = key_sorted[pos] == key_dst[take]
    take, rows = take[matched], order[pos[matched]]
    print(f"matched {take.size:,} of {distill.size:,} distillation records to their source rows",
          flush=True)

    blended = torch.from_numpy(distill["target"][take].astype(np.float32)).cuda()
    bayes = torch.from_numpy(source["target"][rows].astype(np.float32)).cuda()
    qwen = ((blended - args.alpha * bayes) / (1.0 - args.alpha)).clamp(0.0, 1.0)

    # Rebuild the boards so the student can be run on exactly these states.
    tmp = DATASET_DIR / "_cmp_shard.npy"
    np.save(tmp, distill[take])
    dataset = RecordDataset([tmp], dicts.fit, device=cfg.device)

    src = ModelSource(model, noisy_or_weight=0.5)
    student_scores, guessed_all, truth_all, idx = [], [], [], 0
    step = 8192
    for start in range(0, take.size, step):
        batch = dataset.batch(np.arange(start, min(start + step, take.size)))
        board = batch["word"]
        real = board >= 0
        revealed = batch["present"].gather(1, board.clamp_min(0)) & real
        rendered = torch.where(revealed, board, torch.full_like(board, 27))
        rendered = torch.where(real, rendered, torch.full_like(board, -1))
        student_scores.append(src.presence(rendered, batch["lengths"], batch["present"], batch["absent"]))
        guessed_all.append(batch["guessed"])
        truth_all.append(batch["target_hard"])
    tmp.unlink(missing_ok=True)

    student = torch.cat(student_scores)
    guessed = torch.cat(guessed_all)
    truth = torch.cat(truth_all)
    # dataset.batch sorts by record index, which for arange is the identity.
    n = truth.shape[0]

    results = {
        "n_states": int(n),
        "bayes_teacher": top1_hit(bayes[:n], guessed, truth),
        "qwen_teacher": top1_hit(qwen[:n], guessed, truth),
        "blended_target": top1_hit(blended[:n], guessed, truth),
        "student_former_a": top1_hit(student, guessed, truth),
    }
    print()
    print("Top-1 hit rate on the states distillation selected")
    print("(the oracle-blind and short-word states, i.e. the hard ones)\n")
    for name, value in results.items():
        if name != "n_states":
            print(f"  {name:<20} {value:.4f}")
    (METRIC_DIR / "teacher_comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
