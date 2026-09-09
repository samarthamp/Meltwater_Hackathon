#!/usr/bin/env python
"""Splice measured results into reports/MODEL_REPORT.md at the <!-- RESULTS --> marker.

Reads whatever is present under artifacts/metrics/ and artifacts/logs/, so it can
be re-run at any point and will report what has actually finished.

    python tools/make_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "artifacts" / "metrics"
LOGS = REPO / "artifacts" / "logs"
REPORT = REPO / "reports" / "MODEL_REPORT.md"

MARKER = "<!-- RESULTS -->"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def row(tag: str, data: dict) -> str:
    return (
        f"| {tag} | {data['win_rate']:.2f} | ±{data['stderr']:.2f} | "
        f"{data['avg_wrong']:.2f} | {data['total_wrong']:,} | {data['n_words']:,} |"
    )


def main() -> None:
    lines: list[str] = []

    # --- headline table ----------------------------------------------------
    lines.append("### 10.1 Held-out dictionary (22,530 words, disjoint from training)\n")
    lines.append("| Configuration | Win % | s.e. | Avg wrong | E | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    baselines = load(METRICS / "baselines.json") or {}
    for tag in ("static-frequency", "ngram-only", "bayes-teacher"):
        if tag in baselines:
            lines.append(row(f"*baseline* — {tag}", baselines[tag]))

    ablation = load(METRICS / "ensemble_ablation.json")
    if ablation:
        seen = set()
        for entry in ablation:
            tag = entry["tag"]
            if tag in baselines or tag in seen:
                continue
            seen.add(tag)
            lines.append(row(tag, entry))
    else:
        for path in sorted(METRICS.glob("*_train.json")):
            entries = load(path) or []
            for entry in entries:
                lines.append(row(entry["tag"], entry))

    lines.append("")

    # --- per-stage progression --------------------------------------------
    stage_files = sorted(METRICS.glob("*_train.json"))
    if stage_files:
        lines.append("### 10.2 Stage-by-stage progression, per member\n")
        lines.append("Each row is a real evaluation of that member after the named stage, "
                     "played greedily so the numbers compare like with like.\n")
        lines.append("| Member | Stage | Win % | s.e. | Avg wrong |")
        lines.append("|---|---|---:|---:|---:|")
        for path in stage_files:
            member = path.stem.replace("_train", "")
            for entry in load(path) or []:
                stage = entry["tag"].split("/", 1)[-1]
                lines.append(
                    f"| `{member}` | {stage} | {entry['win_rate']:.2f} | "
                    f"±{entry['stderr']:.2f} | {entry['avg_wrong']:.2f} |"
                )
        lines.append("")

    # --- calibration -------------------------------------------------------
    calib = load(METRICS / "ensemble_calibration.json")
    if calib:
        lines.append("### 10.3 Ensemble calibration\n")
        lines.append("Temperatures and mixture weights fitted on the validation dictionary, "
                     "on states drawn from on-policy play.\n")
        lines.append("| Member | Temperature | Weight |")
        lines.append("|---|---:|---:|")
        for path, temp, weight in zip(
            calib["checkpoints"], calib["temperatures"], calib["weights"]
        ):
            lines.append(f"| `{Path(path).stem}` | {temp:.3f} | {weight:.3f} |")
        lines.append("")
        lines.append(
            f"Pooled negative log-likelihood improved from {calib['nll_before']:.4f} to "
            f"{calib['nll_after']:.4f}.\n"
        )

    # --- LLM ---------------------------------------------------------------
    llm = load(METRICS / "qwen_teacher.json")
    if llm:
        lines.append("### 10.4 Qwen3-4B teacher\n")
        lines.append("| Quantity | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Blocks kept | {llm['keep_layers']} of 36 |")
        lines.append(f"| LoRA rank | {llm['lora_r']} |")
        lines.append(f"| Trainable parameters | {llm['trainable_params'] / 1e6:.1f} M "
                     f"({100 * llm['trainable_params'] / llm['total_params']:.2f}%) |")
        lines.append(f"| MNTP steps | {llm['mntp_steps']:,} |")
        lines.append(f"| SFT steps | {llm['sft_steps']:,} |")
        lines.append("")

    distill = load(METRICS / "distill.json")
    if distill:
        lines.append(
            f"Distillation scored **{distill['n_states']:,}** selected states at "
            f"α = {distill['alpha']} in {distill['seconds'] / 60:.0f} min; the teacher's own "
            f"top-1 hit rate on those states was **{distill['qwen_top1_hit_rate']:.3f}**.\n"
        )

    # --- submission --------------------------------------------------------
    sub = load(METRICS / "submission.json")
    if sub:
        lines.append("### 10.5 Public test set (self-scored)\n")
        lines.append("Produced by simulating all 250,000 public test words and re-scoring the "
                     "written CSV independently with a naive per-game scorer.\n")
        lines.append("| Quantity | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Rows | {sub['n_rows']:,} |")
        lines.append(f"| Words solved | {sub['n_won']:,} |")
        lines.append(f"| **Win rate** | **{sub['win_rate']:.3f}%** |")
        lines.append(f"| Total wrong guesses (E) | {sub['total_wrong']:,} |")
        lines.append(f"| Leaderboard score (Score − E/10⁸) | {sub['leaderboard_score']:.6f} |")
        lines.append(f"| Mean guesses per word | {sub['mean_guesses']:.2f} |")
        lines.append("")

    body = "\n".join(lines) if lines else "_No metrics recorded yet._"
    text = REPORT.read_text()
    if MARKER in text:
        head, _, tail = text.partition(MARKER)
        # Drop anything previously spliced in, up to the next section heading.
        rest = tail.split("\n---\n", 1)
        remainder = "\n---\n" + rest[1] if len(rest) > 1 else ""
        REPORT.write_text(head + MARKER + "\n\n" + body + "\n" + remainder)
        print(f"spliced {len(lines)} lines into {REPORT}")
    else:
        print("marker not found; report left unchanged")


if __name__ == "__main__":
    main()
