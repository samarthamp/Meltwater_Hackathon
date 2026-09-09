#!/usr/bin/env python
"""Package the minimal inference artefacts for upload as a Kaggle Dataset.

A training checkpoint carries the raw weights, the EMA shadow copy and the full
training history - 229 MB for a 28.6M-parameter model.  Inference needs one set
of weights and the config.  This folds the EMA weights in (they are what
evaluation uses), drops the history, and writes a checkpoint the submission
notebook loads unchanged.

    python tools/export_for_kaggle.py --ckpts artifacts/checkpoints/former_a_final.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from hangman.config import CHECKPOINT_DIR, REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--calibration", default=str(CHECKPOINT_DIR / "ensemble_calibration.pt"))
    parser.add_argument("--out", default=str(REPO_ROOT / "kaggle_upload"))
    parser.add_argument("--half", action="store_true",
                        help="store weights in fp16, halving the upload at no measurable "
                             "cost to a model that already runs under bf16 autocast")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for path in args.ckpts:
        path = Path(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["model"]
        # Evaluation uses the EMA weights, so those are the ones to ship.
        if "ema" in blob:
            for name, value in blob["ema"].items():
                if name in state:
                    state[name] = value.to(state[name].dtype)
        if args.half:
            state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}

        slim = {"model": state, "config": blob["config"]}
        dest = out / path.name
        torch.save(slim, dest)
        before = path.stat().st_size / 1e6
        after = dest.stat().st_size / 1e6
        manifest.append({"name": path.name, "mb_before": before, "mb_after": after})
        print(f"  {path.name:<32} {before:6.1f} MB -> {after:6.1f} MB")

    calib = Path(args.calibration)
    if calib.exists():
        shutil.copy2(calib, out / calib.name)
        print(f"  {calib.name:<32} copied")

    (out / "MANIFEST.json").write_text(json.dumps({
        "checkpoints": manifest,
        "note": "Attach this directory as a Kaggle Dataset and point WEIGHTS_DIR at it "
                "in notebooks/kaggle_submission.ipynb.",
    }, indent=2))
    total = sum(m["mb_after"] for m in manifest)
    print(f"\nwrote {out}  ({total:.0f} MB total)")


if __name__ == "__main__":
    main()
