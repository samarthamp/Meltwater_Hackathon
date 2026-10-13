#!/usr/bin/env python
"""Render the architecture and data-flow diagrams as standalone SVG.

All colours are explicit, so the output renders the same in light and dark
viewers.

    python tools/make_diagram.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures"

INK = "#1b2430"
MUTED = "#5b6b7c"
LINE = "#8fa3b8"
BG = "#ffffff"
BAND = "#eef3f8"
ACCENT = "#2f6f9f"
ACCENT_FILL = "#dceaf5"
WARM = "#b4622a"
WARM_FILL = "#f7e7da"
GREEN = "#2f7d5d"
GREEN_FILL = "#dcefe6"

FONT = "font-family='ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif'"


def box(x, y, w, h, label, sub="", fill=ACCENT_FILL, stroke=ACCENT, rx=7, size=13):
    out = (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' "
        f"fill='{fill}' stroke='{stroke}' stroke-width='1.4'/>"
    )
    if sub:
        out += (
            f"<text x='{x + w / 2}' y='{y + h / 2 - 5}' text-anchor='middle' {FONT} "
            f"font-size='{size}' font-weight='600' fill='{INK}'>{label}</text>"
            f"<text x='{x + w / 2}' y='{y + h / 2 + 12}' text-anchor='middle' {FONT} "
            f"font-size='11' fill='{MUTED}'>{sub}</text>"
        )
    else:
        out += (
            f"<text x='{x + w / 2}' y='{y + h / 2 + 5}' text-anchor='middle' {FONT} "
            f"font-size='{size}' font-weight='600' fill='{INK}'>{label}</text>"
        )
    return out


def text(x, y, s, size=12, fill=MUTED, anchor="start", weight="400"):
    return (
        f"<text x='{x}' y='{y}' text-anchor='{anchor}' {FONT} font-size='{size}' "
        f"font-weight='{weight}' fill='{fill}'>{s}</text>"
    )


def arrow(x1, y1, x2, y2, dashed=False, colour=LINE):
    dash = " stroke-dasharray='5 4'" if dashed else ""
    return (
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{colour}' "
        f"stroke-width='1.6' marker-end='url(#arrow)'{dash}/>"
    )


DEFS = f"""
<defs>
  <marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7'
          markerHeight='7' orient='auto-start-reverse'>
    <path d='M 0 0 L 10 5 L 0 10 z' fill='{LINE}'/>
  </marker>
</defs>
"""


def architecture() -> str:
    W, H = 1080, 760
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
         DEFS, f"<rect width='{W}' height='{H}' fill='{BG}'/>"]

    p.append(text(24, 34, "HangmanFormer — bidirectional character encoder", 18, INK, weight="700"))
    p.append(text(24, 56, "28.6 M parameters · d_model 512 · 8 blocks · 8 heads · SwiGLU 1408", 12.5, MUTED))

    # ---- inputs -----------------------------------------------------------
    p.append(f"<rect x='24' y='78' width='300' height='224' rx='9' fill='{BAND}' stroke='{LINE}' stroke-width='1'/>")
    p.append(text(40, 100, "GAME STATE", 11, MUTED, weight="700"))
    p.append(box(40, 112, 268, 46, "board   _ o _ _ e r", "29 char slots + [CLS], padded to batch width"))
    p.append(box(40, 168, 130, 44, "present {o,e}", "26-dim indicator", WARM_FILL, WARM, size=12))
    p.append(box(178, 168, 130, 44, "absent {q,x,z}", "26-dim indicator", WARM_FILL, WARM, size=12))
    p.append(box(40, 222, 268, 40, "lives = 3", "6-dim one-hot + 4 scalars", WARM_FILL, WARM, size=12))
    p.append(text(40, 284, "Guessed sets are permutation-invariant (Deep Sets),", 10.5, MUTED))
    p.append(text(40, 297, "not a guess-history sequence.", 10.5, MUTED))

    # ---- embedding / conditioning ----------------------------------------
    p.append(box(370, 112, 190, 46, "token + abs-pos embed", "vocab 29 → 512"))
    p.append(box(370, 190, 190, 62, "FiLM generator", "62 → 128 → 4·512 per block\nzero-init", WARM_FILL, WARM, size=12))
    p.append(arrow(310, 135, 366, 135))
    p.append(arrow(310, 190, 366, 200))
    p.append(arrow(310, 240, 366, 220))

    # ---- the block --------------------------------------------------------
    p.append(f"<rect x='610' y='88' width='300' height='250' rx='9' fill='{BAND}' stroke='{LINE}' stroke-width='1'/>")
    p.append(text(626, 110, "ENCODER BLOCK  × 8", 11, MUTED, weight="700"))
    p.append(box(626, 120, 268, 40, "RMSNorm → FiLM(γ, β)", "", "#ffffff", LINE, size=12))
    p.append(box(626, 168, 268, 40, "bidirectional MHSA + RoPE", "no causal mask", "#ffffff", LINE, size=12))
    p.append(box(626, 216, 268, 36, "RMSNorm → FiLM(γ, β)", "", "#ffffff", LINE, size=12))
    p.append(box(626, 260, 268, 40, "SwiGLU  512 → 1408 → 512", "", "#ffffff", LINE, size=12))
    p.append(text(626, 322, "pre-norm residuals · stochastic depth ramp 0 → 0.05", 10.5, MUTED))
    p.append(arrow(566, 135, 620, 135))
    p.append(arrow(566, 220, 620, 200, dashed=True, colour=WARM))

    # ---- heads ------------------------------------------------------------
    hy = 392
    p.append(text(24, hy - 14, "FOUR HEADS ON A SHARED TRUNK", 11, MUTED, weight="700"))
    p.append(box(24, hy, 246, 74, "set_head", "[CLS] → 26 sigmoid\nP(letter ∈ word) — the primary output",
                 GREEN_FILL, GREEN, size=13))
    p.append(box(286, hy, 246, 74, "pos_head", "per position → 26 softmax\ndense masked-LM supervision"))
    p.append(box(548, hy, 246, 74, "count_head", "[CLS] → 26 × {0,1,2,≥3}\nhow much a hit reveals"))
    p.append(box(810, hy, 246, 74, "value_head", "[CLS] → P(win from here)\nRL baseline + lookahead"))
    p.append(arrow(760, 330, 400, hy - 6))
    p.append(arrow(760, 330, 900, hy - 6))

    # ---- combination ------------------------------------------------------
    cy = 512
    p.append(box(286, cy, 246, 58, "noisy-OR pool", "1 − Π(1 − pᵢ)  —  not max-pool", GREEN_FILL, GREEN, size=12))
    p.append(arrow(409, hy + 74, 409, cy - 4))
    p.append(box(24, cy, 246, 58, "calibrated blend", "temperature + weights, fitted on val", GREEN_FILL, GREEN, size=12))
    p.append(arrow(147, hy + 74, 147, cy - 4))
    p.append(arrow(286, cy + 29, 274, cy + 29))

    # ---- policy -----------------------------------------------------------
    py_ = 606
    p.append(text(24, py_ - 12, "INFERENCE POLICY  (dispatched per state)", 11, MUTED, weight="700"))
    p.append(box(24, py_, 320, 62, "greedy", "argmax of the calibrated presence probability", "#ffffff", LINE, size=12))
    p.append(box(360, py_, 330, 62, "value-guided 1-ply", "lives ≤ 2 · EV over top-5 via value head", "#ffffff", LINE, size=12))
    p.append(box(706, py_, 350, 62, "exact information-set search",
                 "candidate set ≤ 120 · batched E[|C′|] re-rank", "#ffffff", LINE, size=12))
    p.append(arrow(147, cy + 58, 147, py_ - 4))

    p.append(text(24, 712, "Absent-letter evidence reaches every block through FiLM, alongside the "
                           "revealed board.", 11.5, WARM, weight="600"))
    p.append(text(24, 732, "Noisy-OR replaces the max-pooling used in prior work: measured +8.0 points on the "
                           "n-gram baseline alone.", 11.5, GREEN, weight="600"))
    p.append("</svg>")
    return "\n".join(p)


def pipeline() -> str:
    W, H = 1080, 560
    p = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}'>",
         DEFS, f"<rect width='{W}' height='{H}' fill='{BG}'/>"]
    p.append(text(24, 34, "Training pipeline — where the supervision comes from", 18, INK, weight="700"))

    p.append(box(24, 70, 210, 66, "train.txt", "225,300 words\nzero overlap with test.txt"))
    p.append(box(24, 158, 210, 78, "90 / 10 split by word", "fit 202,770\nval 22,530 (disjoint)"))
    p.append(arrow(129, 138, 129, 154))

    # teacher
    p.append(f"<rect x='272' y='60' width='356' height='250' rx='9' fill='{BAND}' stroke='{LINE}' stroke-width='1'/>")
    p.append(text(288, 82, "BAYES TEACHER  (label generator)", 11, MUTED, weight="700"))
    p.append(box(288, 92, 324, 42, "Tier 1 — exact leave-one-out posterior", "positional + no surplus guessed letter", "#ffffff", LINE, size=12))
    p.append(box(288, 142, 324, 34, "Tier 2 — drop the present-letter rule", "", "#ffffff", LINE, size=12))
    p.append(box(288, 184, 324, 34, "Tier 3 — positional match only", "", "#ffffff", LINE, size=12))
    p.append(box(288, 226, 324, 44, "Tier 4 — interpolated n-gram, noisy-OR", "covers the 45% with no candidates", WARM_FILL, WARM, size=12))
    p.append(text(288, 292, "blended by candidate count, λ = n/(n+300)", 10.5, MUTED))
    p.append(arrow(240, 190, 284, 190))

    # data stages
    p.append(box(668, 78, 388, 60, "Stage 0 — reachable-state warm start",
                 "20 M states · ≤5 absent letters · frequency-weighted misses"))
    p.append(box(668, 152, 388, 60, "Stage 1 — DAgger self-play × 3",
                 "roll out, label what the model visits, aggregate"))
    p.append(box(668, 226, 388, 60, "Stage 2/3 — Qwen3-4B teacher → distil",
                 "selective: only where the oracle is blind", WARM_FILL, WARM))
    p.append(box(668, 300, 388, 60, "Stage 4 — policy-gradient refinement",
                 "reward = win − 0.001·wrong, KL trust region", GREEN_FILL, GREEN))
    p.append(box(668, 374, 388, 60, "Stage 5 — calibrate + ensemble",
                 "transformer + Bi-GRU + dilated CNN, logit-space", GREEN_FILL, GREEN))
    for y in (138, 212, 286, 360):
        p.append(arrow(862, y, 862, y + 12))
    p.append(arrow(632, 130, 664, 108))

    p.append(box(24, 374, 596, 60, "Evaluation — play real games on the 22,530 held-out words",
                 "strict rules · 6th wrong guess is a loss · ±0.32 pt standard error", GREEN_FILL, GREEN))
    p.append(arrow(320, 340, 320, 370))

    p.append(text(24, 470, "test.txt is used only to simulate games for the submission — never to train "
                           "or tune anything.", 12, WARM, weight="600"))
    p.append(text(24, 494, "45.1% of on-policy states have no consistent training-dictionary word: "
                           "lookup approaches hit a wall, a generalising model does not.", 12, INK))
    p.append(text(24, 518, "Every state is stored as (word_id, guessed-letter bitmask) + soft target = 66 bytes; "
                           "20 M states fit in 1.3 GB.", 12, MUTED))
    p.append("</svg>")
    return "\n".join(p)


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "architecture.svg").write_text(architecture())
(OUT / "pipeline.svg").write_text(pipeline())
print(f"wrote {OUT / 'architecture.svg'}")
print(f"wrote {OUT / 'pipeline.svg'}")
