#!/usr/bin/env python
"""Stage 9: simulate every test word and write ``submission.csv``.

The public test words are known, so this both produces the submission and
self-scores it exactly.  They are used only to *play* games - never to fit
anything; see reports/COMPLIANCE.md.

    python scripts/09_make_submission.py --ckpts artifacts/checkpoints/*_final.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hangman.config import (
    CHECKPOINT_DIR,
    ExperimentConfig,
    METRIC_DIR,
    REPO_ROOT,
    ensure_dirs,
)
from hangman.data import load_test_words
from hangman.pipeline import build_teacher, set_seed
from hangman.policy import EnsembleSource, GreedyPolicy, ModelSource, build_policy
from hangman.submit import generate_submission, save_stats, validate_submission
from hangman.train import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--calibration", default=str(CHECKPOINT_DIR / "ensemble_calibration.pt"))
    parser.add_argument("--policy", default="fitted",
                        choices=["fitted", "greedy", "value", "exact", "auto"],
                        help="'fitted' uses the mode selected on the validation dictionary "
                             "by scripts/07_ensemble_eval.py")
    parser.add_argument("--noisy-or-weight", type=float, default=0.5,
                        help="fallback when the calibration file has no fitted value")
    parser.add_argument("--uniform", action="store_true",
                        help="average ensemble members with equal weight, ignoring the fitted "
                             "calibration. Measured on two members of similar strength, the "
                             "NLL-fitted weights scored 68.15%% against 69.61%% for a plain "
                             "average - fitting log-likelihood is not the same as fitting "
                             "win rate.")
    parser.add_argument("--out", default=str(REPO_ROOT / "submission.csv"))
    parser.add_argument("--full-dictionary", action="store_true",
                        help="the teacher's candidate set is all of train.txt (submission setting)")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    models, cfgs = [], []
    for path in args.ckpts:
        model, cfg = load_checkpoint(path, use_ema=True)
        models.append(model)
        cfgs.append(cfg)
        print(f"loaded {Path(path).name}: {cfg.model.arch}", flush=True)

    cfg = cfgs[0]
    cfg.data.use_full_dictionary = args.full_dictionary
    _, _, teacher = build_teacher(cfg)

    weights = temperatures = None
    resolved_policy = args.policy
    per_member_nor: dict[str, float] = {}
    calib_path = Path(args.calibration)
    blob = None
    if calib_path.exists():
        blob = torch.load(calib_path, map_location=cfg.device, weights_only=False)
        per_member_nor = blob.get("noisy_or_weights", {})

    sources = [
        ModelSource(m, noisy_or_weight=per_member_nor.get(Path(p).stem, args.noisy_or_weight))
        for p, m in zip(args.ckpts, models)
    ]
    if per_member_nor:
        print(f"using fitted noisy-OR weights: {per_member_nor}", flush=True)
    if blob is not None:
        if args.uniform:
            print("ignoring the fitted calibration; averaging members uniformly", flush=True)
        elif len(sources) > 1 and len(blob["weights"]) == len(sources):
            weights = blob["weights"].to(cfg.device)
            temperatures = blob["temperatures"].to(cfg.device)
            print(f"using fitted calibration: weights={weights.tolist()}", flush=True)
        if resolved_policy == "fitted":
            resolved_policy = blob.get("policy_mode", "greedy")
            print(f"using fitted policy: {resolved_policy}", flush=True)
    if resolved_policy == "fitted":
        # No calibration file: fall back to the policy that cannot regress.
        resolved_policy = "greedy"
        print("no calibration file found; defaulting to the greedy policy", flush=True)
    cfg.policy.mode = resolved_policy

    combined = (
        EnsembleSource(sources, weights=weights, temperatures=temperatures)
        if len(sources) > 1
        else sources[0]
    )

    def factory(length: int):
        return build_policy(
            combined, cfg.policy, teacher=teacher, length=length, primary=sources[0]
        )

    words = load_test_words(cfg.data)
    print(f"simulating {len(words):,} test words with policy={resolved_policy}", flush=True)
    started = time.time()
    out_path, stats = generate_submission(words, factory, args.out, device=cfg.device)
    print(f"wrote {out_path} in {time.time() - started:.0f}s", flush=True)
    print("generator:", stats.summary(), flush=True)

    verified = validate_submission(out_path, words)
    print("validator:", verified.summary(), flush=True)
    if abs(verified.win_rate - stats.win_rate) > 1e-9:
        raise SystemExit("generator and validator disagree - submission is not what we think")
    save_stats(verified, METRIC_DIR / "submission.json")


if __name__ == "__main__":
    main()
