"""HangmanFormer: a bidirectional character transformer for hangman.

Design summary
--------------
* **Bidirectional, not causal.**  Reading ``_ o _ _ e r`` requires the suffix as
  much as the prefix, so the encoder attends both ways, BERT-style
  (Devlin et al., 2019) over a character alphabet.  Character granularity is
  the right level for a spelling task - the motivation behind CANINE
  (Clark et al., 2021), ByT5 (Xue et al., 2021) and Charformer (Tay et al.,
  2022).
* **Global constraints via FiLM.**  Guessed-letter sets and remaining lives
  modulate every block (Perez et al., 2018) instead of being appended to the
  sequence.
* **Four heads, one shared trunk.**  A presence head is what the game actually
  needs; a per-position head supplies dense supervision; a count head tells the
  policy how much a hit reveals; a value head estimates the win probability so
  the endgame can be played for win rate rather than for hit rate.
* **Noisy-OR pooling.**  The per-position distributions are pooled into a
  second presence estimate with ``1 - prod(1 - p)``, which is the correct
  probabilistic aggregation.  Public implementations max-pool here, which
  understates letters likely to occur in several places.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GLOBAL_FEATURE_DIM, ModelConfig, SEQ_LEN, VOCAB_SIZE
from .layers import (
    EncoderBlock,
    FiLMGenerator,
    RMSNorm,
    RotaryEmbedding,
    count_parameters,
    noisy_or_presence,
)


class HangmanOutput(dict):
    """Model outputs, exposed as attributes for readability."""

    set_logits: torch.Tensor
    pos_logits: torch.Tensor | None
    count_logits: torch.Tensor | None
    value: torch.Tensor | None

    def __getattr__(self, name: str):  # pragma: no cover - trivial
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class HangmanFormer(nn.Module):
    """Character transformer with set / position / count / value heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.abs_pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d)) if cfg.use_abs_pos else None
        if self.abs_pos is not None:
            nn.init.normal_(self.abs_pos, std=0.02)

        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.film = (
            FiLMGenerator(GLOBAL_FEATURE_DIM, d, cfg.n_layers) if cfg.use_film else None
        )
        rope = (
            RotaryEmbedding(d // cfg.n_heads, SEQ_LEN) if cfg.use_rope else None
        )
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    # Linearly ramped stochastic depth, deeper blocks drop more.
                    drop_path=cfg.stochastic_depth * i / max(cfg.n_layers - 1, 1),
                    rope=rope,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norm_out = RMSNorm(d)

        self.set_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 26))
        self.pos_head = nn.Linear(d, 26) if cfg.use_pos_head else None
        self.count_head = (
            nn.Linear(d, 26 * cfg.count_buckets) if cfg.use_count_head else None
        )
        self.value_head = (
            nn.Sequential(nn.Linear(d, d // 4), nn.SiLU(), nn.Linear(d // 4, 1))
            if cfg.use_value_head
            else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        globals: torch.Tensor,
        **_: torch.Tensor,
    ) -> HangmanOutput:
        x = self.tok_emb(tokens)
        if self.abs_pos is not None:
            x = x + self.abs_pos[:, : x.shape[1]]

        cond = self.global_mlp(globals)
        x = torch.cat([x[:, :1] + cond.unsqueeze(1), x[:, 1:]], dim=1)

        films = self.film(globals) if self.film is not None else [None] * len(self.blocks)
        for block, film in zip(self.blocks, films):
            x = block(x, attn_mask, film)
        x = self.norm_out(x)

        cls = x[:, 0]
        out = HangmanOutput(set_logits=self.set_head(cls))
        out["pos_logits"] = self.pos_head(x[:, 1:]) if self.pos_head is not None else None
        out["count_logits"] = (
            self.count_head(cls).view(-1, 26, self.cfg.count_buckets)
            if self.count_head is not None
            else None
        )
        out["value"] = (
            self.value_head(cls).squeeze(-1) if self.value_head is not None else None
        )
        return out

    # -- inference helpers ----------------------------------------------------

    noisy_or_presence = staticmethod(noisy_or_presence)

    @torch.no_grad()
    def presence_probs(
        self,
        batch: dict[str, torch.Tensor],
        hidden_mask: torch.Tensor | None = None,
        noisy_or_weight: float = 0.0,
    ) -> torch.Tensor:
        """``(B, 26)`` presence probability with guessed letters zeroed."""
        out = self(**batch)
        probs = torch.sigmoid(out.set_logits)
        if noisy_or_weight > 0.0 and out.pos_logits is not None and hidden_mask is not None:
            pooled = noisy_or_presence(out.pos_logits, hidden_mask)
            probs = (1.0 - noisy_or_weight) * probs + noisy_or_weight * pooled
        return probs.masked_fill(batch["guessed"], 0.0)

    # -- introspection ---------------------------------------------------------

    def parameter_report(self) -> list[tuple[str, int, str]]:
        """Layer-by-layer parameter table used in reports/MODEL_REPORT.md."""
        rows: list[tuple[str, int, str]] = []
        for name, child in self.named_children():
            n = count_parameters(child)
            if n:
                rows.append((name, n, child.__class__.__name__))
        rows.append(("TOTAL", count_parameters(self), ""))
        return rows
