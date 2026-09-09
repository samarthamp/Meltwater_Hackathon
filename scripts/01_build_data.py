#!/usr/bin/env python
"""Stage 1: build dictionaries, fit the teacher, generate the warm-start pool.

    python scripts/01_build_data.py --states 20000000
"""

from __future__ import annotations

import argparse
import json
import time

from hangman.config import ExperimentConfig, METRIC_DIR, ensure_dirs
from hangman.pipeline import (
    build_teacher,
    evaluate_frequency,
    evaluate_ngram,
    evaluate_teacher,
    set_seed,
    stage_offpolicy_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=20_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--full-dictionary", action="store_true",
                        help="train the teacher on all of train.txt (final submission model)")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    cfg = ExperimentConfig()
    cfg.data.val_fraction = args.val_fraction
    cfg.data.use_full_dictionary = args.full_dictionary
    cfg.collect.offpolicy_states = args.states

    started = time.time()
    dicts, ngram, teacher = build_teacher(cfg)
    print(dicts.summary(), flush=True)

    if not args.skip_baselines and dicts.val:
        rows = {}
        for report in (
            evaluate_frequency(cfg, dicts, dicts.val),
            evaluate_ngram(cfg, teacher, dicts.val),
            evaluate_teacher(cfg, teacher, dicts.val),
        ):
            print(report.summary(), flush=True)
            rows[report.tag] = report.to_dict()
        (METRIC_DIR / "baselines.json").write_text(json.dumps(rows, indent=2))

    prefix = "offpolicy_full" if args.full_dictionary else "offpolicy"
    paths = stage_offpolicy_dataset(
        cfg, dicts, teacher, n_states=args.states, prefix=prefix, overwrite=args.overwrite
    )
    print(f"wrote {len(paths)} shards in {time.time() - started:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
