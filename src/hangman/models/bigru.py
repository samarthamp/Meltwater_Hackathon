"""Bi-GRU ensemble member.

Architectural diversity contributes more to an ensemble than seed diversity
alone (Lakshminarayanan et al., 2017, *Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles*).  A recurrent model makes
different mistakes from an attention model on the same board - in particular it
is stronger on short words, where there is little context for attention to
exploit and where the measured baseline loses most of its games.

The conditioning, heads and output contract are identical to
:class:`~hangman.models.hangmanformer.HangmanFormer`, so the two are
interchangeable everywhere downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GLOBAL_FEATURE_DIM, ModelConfig, VOCAB_SIZE
from .hangmanformer import HangmanOutput
from .layers import RMSNorm, count_parameters, noisy_or_presence


class HangmanBiGRU(nn.Module):
    """Stacked bidirectional GRU with FiLM-style global conditioning."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        half = d // 2

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, d), nn.SiLU(), nn.Linear(d, 2 * d)
        )
        nn.init.zeros_(self.global_mlp[-1].weight)
        nn.init.zeros_(self.global_mlp[-1].bias)

        self.rnn = nn.GRU(
            input_size=d,
            hidden_size=half,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )
        self.norm_out = RMSNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

        self.set_head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 26))
        self.pos_head = nn.Linear(d, 26) if cfg.use_pos_head else None
        self.count_head = (
            nn.Linear(2 * d, 26 * cfg.count_buckets) if cfg.use_count_head else None
        )
        self.value_head = (
            nn.Sequential(nn.Linear(2 * d, d // 4), nn.SiLU(), nn.Linear(d // 4, 1))
            if cfg.use_value_head
            else None
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        globals: torch.Tensor,
        **_: torch.Tensor,
    ) -> HangmanOutput:
        x = self.tok_emb(tokens)
        gamma, beta = self.global_mlp(globals).chunk(2, dim=-1)
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = x * attn_mask.unsqueeze(-1).to(x.dtype)

        h, _ = self.rnn(x)
        h = self.norm_out(self.drop(h))

        valid = attn_mask.unsqueeze(-1).to(h.dtype)
        mean_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        summary = torch.cat([h[:, 0], mean_pool], dim=-1)

        out = HangmanOutput(set_logits=self.set_head(summary))
        out["pos_logits"] = self.pos_head(h[:, 1:]) if self.pos_head is not None else None
        out["count_logits"] = (
            self.count_head(summary).view(-1, 26, self.cfg.count_buckets)
            if self.count_head is not None
            else None
        )
        out["value"] = (
            self.value_head(summary).squeeze(-1) if self.value_head is not None else None
        )
        return out

    noisy_or_presence = staticmethod(noisy_or_presence)

    def parameter_report(self) -> list[tuple[str, int, str]]:
        rows = [
            (name, count_parameters(child), child.__class__.__name__)
            for name, child in self.named_children()
            if count_parameters(child)
        ]
        rows.append(("TOTAL", count_parameters(self), ""))
        return rows
