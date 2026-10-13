#!/usr/bin/env python
"""Generate the result figures for the reports from recorded metrics.

    python tools/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "artifacts" / "metrics"
LOGS = REPO / "artifacts" / "logs"
FIGS = REPO / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

INK, MUTED, ACCENT, WARM, GREEN = "#1b2430", "#5b6b7c", "#2f6f9f", "#b4622a", "#2f7d5d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def training_curves() -> None:
    """Loss, teacher agreement and hit rate over training, per member.

    A dashed vertical line marks where DAgger begins for each member.
    """
    runs: dict[str, list] = {}
    boundaries: dict[str, int] = {}
    for path in sorted(LOGS.glob("*_pretrain.json")):
        member = path.stem.replace("_pretrain", "")
        rows = load(path) or []
        if rows:
            runs.setdefault(member, []).extend(rows)
            boundaries[member] = max(r["step"] for r in rows)
    for path in sorted(LOGS.glob("*_dagger*.json")):
        member = path.stem.split("_dagger")[0]
        if member in runs:
            runs[member].extend(load(path) or [])
    runs = {k: v for k, v in runs.items() if len(v) > 5}
    if not runs:
        print("no training logs yet")
        return

    def smooth(steps, values, window=9):
        """Centred rolling mean, trimmed so the edges are not pulled toward zero."""
        if len(values) < window:
            return steps, values
        kernel = np.ones(window) / window
        smoothed = np.convolve(values, kernel, mode="valid")
        half = window // 2
        return steps[half:len(steps) - half], smoothed

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    palette = [ACCENT, GREEN, WARM]
    for colour, (member, rows) in zip(palette, sorted(runs.items())):
        rows = sorted(rows, key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r["total"] for r in rows], label=member, lw=1.3, color=colour)
        axes[1].plot(steps, [r["teacher_agreement"] for r in rows], label=member, lw=1.3, color=colour)
        hit_x, hit_y = smooth(steps, [r["hit_rate"] for r in rows])
        axes[2].plot(hit_x, hit_y, label=member, lw=1.5, color=colour)
        if member in boundaries:
            for ax in axes:
                ax.axvline(boundaries[member], color=MUTED, lw=1, ls="--", alpha=0.7)

    axes[0].set_title("total loss")
    axes[1].set_title("agreement with the teacher's top-1")
    axes[2].set_title("hit rate (rolling mean of 9)")
    axes[2].set_ylim(0.3, 1.0)
    axes[0].annotate("DAgger begins", xy=(0.52, 0.9), xycoords="axes fraction",
                     fontsize=8.5, color=MUTED)
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2, lw=0.6)
    fig.suptitle(
        "The jump at the dashed line is DAgger: on-policy states are harder than random masks",
        fontsize=10, color=MUTED, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGS / "training_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("wrote training_curves.png")


def ablation_bars() -> None:
    """Win rate by configuration, with the baselines marked."""
    rows = load(METRICS / "ensemble_ablation.json")
    if not rows:
        print("no ablation table yet")
        return
    rows = [r for r in rows if r["n_words"] > 1000]
    labels = [r["tag"] for r in rows]
    values = [r["win_rate"] for r in rows]
    errors = [r["stderr"] for r in rows]
    colours = [
        WARM if "baseline" in t or t in ("static-frequency", "ngram-only", "bayes-teacher")
        else (GREEN if t.startswith("SHIPPED") else ACCENT)
        for t in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4))
    y = np.arange(len(rows))
    ax.barh(y, values, xerr=errors, color=colours, height=0.66, error_kw={"lw": 1, "ecolor": MUTED})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("win rate on the held-out dictionary (%)")
    ax.grid(axis="x", alpha=0.2, lw=0.6)
    for yi, value in zip(y, values):
        ax.text(value + 0.4, yi, f"{value:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(values) * 1.14)
    plt.tight_layout()
    plt.savefig(FIGS / "ablation.png", dpi=140)
    plt.close()
    print("wrote ablation.png")


def per_length() -> None:
    """Win rate by word length: baseline against the final model."""
    baselines = load(METRICS / "baselines.json") or {}
    rows = load(METRICS / "ensemble_ablation.json") or []
    final = next((r for r in reversed(rows) if r["tag"].startswith("SHIPPED")), None)
    final = final or (rows[-1] if rows else None)
    if not final or "bayes-teacher" not in baselines:
        print("not enough data for the per-length figure")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for source, label, colour in (
        (baselines["bayes-teacher"], "Bayes teacher (label source)", WARM),
        (final, final["tag"], GREEN),
    ):
        table = source["per_length"]
        lengths = sorted(int(k) for k in table)
        xs, ys = [], []
        for length in lengths:
            n, won = table[str(length)]
            if n >= 50:
                xs.append(length)
                ys.append(100 * won / n)
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=label, color=colour)
    ax.set_xlabel("word length")
    ax.set_ylabel("win rate (%)")
    ax.set_title("Where the games are won and lost")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, lw=0.6)
    plt.tight_layout()
    plt.savefig(FIGS / "win_rate_by_length.png", dpi=140)
    plt.close()
    print("wrote win_rate_by_length.png")


if __name__ == "__main__":
    training_curves()
    ablation_bars()
    per_length()
