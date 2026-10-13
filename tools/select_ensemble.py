#!/usr/bin/env python
"""Choose the ensemble composition by evaluation on the held-out dictionary.

Candidates are the primary alone, the primary plus each other member, the
primary plus earlier DAgger snapshots of itself, and all members together.  A
composition must beat the primary alone by more than one standard error to be
chosen.  The result is written to ``artifacts/metrics/ensemble_choice.json``.

    python tools/select_ensemble.py --primary artifacts/checkpoints/former_a_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hangman.calibrate import collect_calibration_states, fit_calibration
from hangman.config import CHECKPOINT_DIR, METRIC_DIR, ensure_dirs
from hangman.evaluate import evaluate_words
from hangman.pipeline import build_teacher, set_seed, split_validation
from hangman.policy import EnsembleSource, GreedyPolicy, ModelSource
from hangman.train import load_checkpoint

NOISY_OR_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", default=str(CHECKPOINT_DIR / "former_a_final.pt"))
    parser.add_argument("--others", nargs="*", default=None)
    parser.add_argument("--snapshots", nargs="*", default=None)
    parser.add_argument("--eval-words", type=int, default=8000)
    parser.add_argument("--tune-words", type=int, default=4000)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    others = args.others or []
    if args.others is None:
        others = [str(p) for p in sorted(CHECKPOINT_DIR.glob("*_final.pt"))
                  if Path(p).resolve() != Path(args.primary).resolve()]
    snapshots = args.snapshots
    if snapshots is None:
        # Strip the refinement suffix to find the member's DAgger snapshots.
        stem = Path(args.primary).stem
        for suffix in ("_final", "_rl", "_distilled", "_wpos", "_pretrain"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        # The last DAgger checkpoint is identical to ``_final``, so only earlier
        # snapshots that differ from the primary are used.
        all_snaps = sorted(CHECKPOINT_DIR.glob(f"{stem}_dagger*.pt"))
        candidates = [str(p) for p in all_snaps[:-1]] if len(all_snaps) > 1 else []
        snapshots = [
            p for p in candidates
            if Path(p).stat().st_size != Path(args.primary).stat().st_size
            or Path(p).read_bytes()[:4096] != Path(args.primary).read_bytes()[:4096]
        ][-2:]

    paths = list(dict.fromkeys([args.primary] + others + snapshots))
    models, cfgs = {}, {}
    for path in paths:
        model, cfg = load_checkpoint(path, use_ema=True)
        models[path] = model
        cfgs[path] = cfg
        print(f"loaded {Path(path).stem}: {cfg.model.arch}", flush=True)

    cfg = cfgs[args.primary]
    dicts, _, teacher = build_teacher(cfg)
    tune_words, reportable = split_validation(dicts.val, args.tune_words)
    eval_words = reportable if not args.eval_words else reportable[: args.eval_words]

    # The optimal noisy-OR weight differs by architecture, so fit it per member.
    sources: dict[str, ModelSource] = {}
    fitted_nor: dict[str, float] = {}
    for path, model in models.items():
        best_w, best = 0.5, -1.0
        for w in NOISY_OR_GRID:
            probe = ModelSource(model, noisy_or_weight=w)
            score = evaluate_words(tune_words, lambda L, s=probe: GreedyPolicy(s),
                                   device=cfg.device, tag="tune").win_rate
            if score > best:
                best_w, best = w, score
        sources[path] = ModelSource(model, noisy_or_weight=best_w)
        fitted_nor[Path(path).stem] = best_w
        print(f"  {Path(path).stem}: noisy_or_weight={best_w} ({best:.2f}%)", flush=True)

    candidates: dict[str, list[str]] = {"primary alone": [args.primary]}
    for other in others:
        candidates[f"primary + {Path(other).stem}"] = [args.primary, other]
    for snap in snapshots:
        candidates[f"primary + snapshot {Path(snap).stem}"] = [args.primary, snap]
    if len(paths) > 2:
        candidates["all members"] = paths

    results = {}
    for name, members in candidates.items():
        subset = [sources[p] for p in members]
        if len(subset) == 1:
            combined, calib = subset[0], None
        else:
            uniform = EnsembleSource(subset)
            logits, labels = collect_calibration_states(
                tune_words, subset, lambda L: GreedyPolicy(uniform), device=cfg.device
            )
            calib = fit_calibration(logits, labels)
            combined = EnsembleSource(subset,
                                      weights=calib.weights.to(cfg.device),
                                      temperatures=calib.temperatures.to(cfg.device))
        report = evaluate_words(eval_words, lambda L, c=combined: GreedyPolicy(c),
                                device=cfg.device, tag=name)
        results[name] = {
            "members": members,
            "win_rate": report.win_rate,
            "stderr": report.stderr,
            "avg_wrong": report.avg_wrong,
            "weights": calib.weights.tolist() if calib else None,
            "temperatures": calib.temperatures.tolist() if calib else None,
        }
        print(f"{name:<44} {report.win_rate:6.2f}% ±{report.stderr:.2f}  "
              f"avgwrong {report.avg_wrong:.2f}", flush=True)

    baseline = results["primary alone"]["win_rate"]
    margin = results["primary alone"]["stderr"]
    best_name = max(results, key=lambda k: results[k]["win_rate"])
    if results[best_name]["win_rate"] <= baseline + margin:
        best_name = "primary alone"
    chosen = results[best_name]
    print(f"\nchosen: {best_name}  ({chosen['win_rate']:.2f}% vs {baseline:.2f}% for the primary "
          f"alone; a member must clear +{margin:.2f} to be worth its inference cost)", flush=True)

    (METRIC_DIR / "ensemble_choice.json").write_text(
        json.dumps({"chosen": best_name, "results": results,
                    "noisy_or_weights": fitted_nor}, indent=2)
    )
    if chosen["weights"] is not None:
        torch.save(
            {"weights": torch.tensor(chosen["weights"]),
             "temperatures": torch.tensor(chosen["temperatures"]),
             "checkpoints": chosen["members"],
             "noisy_or_weights": fitted_nor},
            CHECKPOINT_DIR / "ensemble_calibration.pt",
        )
    else:
        torch.save({"weights": torch.ones(1), "temperatures": torch.ones(1),
                    "checkpoints": chosen["members"], "noisy_or_weights": fitted_nor},
                   CHECKPOINT_DIR / "ensemble_calibration.pt")
    print(" ".join(chosen["members"]))


if __name__ == "__main__":
    main()
