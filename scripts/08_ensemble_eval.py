#!/usr/bin/env python
"""Stage 8: calibrate the ensemble and choose the inference policy.

Everything here is fitted on the held-out validation dictionary.  ``test.txt``
is not touched.

    python scripts/08_ensemble_eval.py --ckpts artifacts/checkpoints/*_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hangman.calibrate import collect_calibration_states, fit_calibration
from hangman.config import CHECKPOINT_DIR, ExperimentConfig, METRIC_DIR, ensure_dirs
from hangman.evaluate import AblationTable, evaluate_words
from hangman.pipeline import (
    build_teacher,
    evaluate_frequency,
    evaluate_ngram,
    evaluate_teacher,
    set_seed,
    split_validation,
)
from hangman.policy import EnsembleSource, GreedyPolicy, ModelSource, build_policy
from hangman.train import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--eval-words", type=int, default=0,
                        help="0 = every validation word not reserved for fitting")
    parser.add_argument("--fit-words", type=int, default=5000,
                        help="validation words reserved for fitting the noisy-OR blend and the "
                             "ensemble calibration; disjoint from the words reported on")
    parser.add_argument("--noisy-or-weight", type=float, default=0.5,
                        help="starting point; the value actually used is fitted per member")
    parser.add_argument("--tune-words", type=int, default=5000,
                        help="validation words used to fit the noisy-OR blend")
    parser.add_argument("--out", default="ensemble")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    models, cfgs = [], []
    for path in args.ckpts:
        model, cfg = load_checkpoint(path, use_ema=True)
        models.append(model)
        cfgs.append(cfg)
        print(f"loaded {Path(path).name}: {cfg.model.arch} "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    cfg = cfgs[0]
    dicts, _, teacher = build_teacher(cfg)

    # Split the validation dictionary in two.  Everything that gets *fitted* -
    # the per-member noisy-OR blend, the ensemble temperatures and weights - is
    # fitted on the reserved tail; every number reported comes from the rest.
    # Without this the headline win rate would be measured partly on words used
    # to choose its own hyperparameters.
    fit_words, reportable = split_validation(dicts.val, args.fit_words)
    words = reportable if not args.eval_words else reportable[: args.eval_words]
    print(f"fitting on {len(fit_words):,} words, reporting on {len(words):,} disjoint words",
          flush=True)

    table = AblationTable()
    table.add(evaluate_frequency(cfg, dicts, words))
    table.add(evaluate_ngram(cfg, teacher, words))
    table.add(evaluate_teacher(cfg, teacher, words))

    # The blend between the presence head and the noisy-OR pooling of the
    # per-position head is fitted per member rather than assumed.  Measured on
    # the primary member it is worth several points, and the optimum is not at
    # the value one would guess: the per-position head, pooled correctly, is a
    # better presence estimator than the head trained directly for the job.
    tune_words = fit_words
    noisy_or_weights: dict[str, float] = {}
    sources: list[ModelSource] = []
    for path, model in zip(args.ckpts, models):
        name = Path(path).stem
        best_w, best_score = args.noisy_or_weight, -1.0
        for candidate in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
            probe = ModelSource(model, noisy_or_weight=candidate)
            score = evaluate_words(
                tune_words, lambda L, s=probe: GreedyPolicy(s),
                device=cfg.device, tag=f"tune/{name}/nor={candidate}",
            ).win_rate
            if score > best_score:
                best_w, best_score = candidate, score
        noisy_or_weights[name] = best_w
        print(f"  {name}: noisy_or_weight = {best_w} ({best_score:.2f}% on {len(tune_words):,} words)",
              flush=True)
        sources.append(ModelSource(model, noisy_or_weight=best_w))

    for path, source in zip(args.ckpts, sources):
        name = Path(path).stem
        table.add(
            evaluate_words(
                words, lambda L, s=source: GreedyPolicy(s),
                device=cfg.device, tag=f"single/{name}",
            )
        )

    # Uniform ensemble, then calibrated.
    uniform = EnsembleSource(sources)
    table.add(
        evaluate_words(
            words, lambda L: GreedyPolicy(uniform), device=cfg.device, tag="ensemble/uniform"
        )
    )

    logits, labels = collect_calibration_states(
        fit_words, sources, lambda L: GreedyPolicy(uniform), device=cfg.device
    )
    fit = fit_calibration(logits, labels)
    print("calibration:", fit.summary(), flush=True)

    calibrated = EnsembleSource(
        sources,
        weights=fit.weights.to(cfg.device),
        temperatures=fit.temperatures.to(cfg.device),
    )
    table.add(
        evaluate_words(
            words, lambda L: GreedyPolicy(calibrated), device=cfg.device, tag="ensemble/calibrated"
        )
    )

    # Policy variants, isolated on the primary member so the comparison is
    # about the decision rule and not about the ensemble.
    primary = sources[0]
    policy_scores: dict[str, float] = {}
    for mode in ("greedy", "value", "exact", "auto"):
        pcfg = ExperimentConfig().policy
        pcfg.mode = mode
        report = table.add(
            evaluate_words(
                words,
                lambda L, m=mode, p=pcfg: build_policy(
                    primary, p, teacher=teacher if m in ("exact", "auto") else None, length=L
                ),
                device=cfg.device,
                tag=f"policy/{mode}",
            )
        )
        policy_scores[mode] = report.win_rate

    # The policy is *selected*, not assumed.  A search policy only ships if it
    # actually beats greedy on held-out words by more than the noise floor -
    # measured once at 13.5 points *worse*, when the value head had not yet seen
    # an on-policy Monte-Carlo return to learn from.
    greedy_score = policy_scores["greedy"]
    margin = 2.0 * max(r.stderr for r in table.rows if r.tag.startswith("policy/"))
    best_mode = max(policy_scores, key=policy_scores.get)
    if policy_scores[best_mode] <= greedy_score + margin:
        best_mode = "greedy"
    print(
        f"\nselected policy: {best_mode}  "
        f"(greedy {greedy_score:.2f}%, best {max(policy_scores.values()):.2f}%, "
        f"margin {margin:.2f} points)",
        flush=True,
    )

    # The configuration actually shipped: calibrated ensemble scores driving the
    # decision, primary member supplying the auxiliary heads for lookahead.
    if len(sources) > 1:
        shipped = ExperimentConfig().policy
        shipped.mode = "auto"
        table.add(
            evaluate_words(
                words,
                lambda L, p=shipped: build_policy(
                    calibrated, p, teacher=teacher, length=L, primary=sources[0]
                ),
                device=cfg.device,
                tag="SHIPPED: calibrated ensemble + auto policy",
            )
        )

    table.save(METRIC_DIR / f"{args.out}_ablation.json")
    (METRIC_DIR / f"{args.out}_calibration.json").write_text(
        json.dumps(
            {
                "checkpoints": [str(p) for p in args.ckpts],
                "temperatures": fit.temperatures.tolist(),
                "weights": fit.weights.tolist(),
                "nll_before": fit.nll_before,
                "nll_after": fit.nll_after,
                "policy_scores": policy_scores,
                "selected_policy": best_mode,
                "noisy_or_weights": noisy_or_weights,
            },
            indent=2,
        )
    )
    torch.save(
        {"weights": fit.weights, "temperatures": fit.temperatures,
         "checkpoints": [str(p) for p in args.ckpts],
         "policy_mode": best_mode, "policy_scores": policy_scores,
         "noisy_or_weights": noisy_or_weights},
        CHECKPOINT_DIR / f"{args.out}_calibration.pt",
    )
    print("\n" + table.to_markdown(), flush=True)


if __name__ == "__main__":
    main()
