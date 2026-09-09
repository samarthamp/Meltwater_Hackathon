"""Dilated convolutional ensemble member.

A stack of dilated 1-D convolutions grows its receptive field exponentially
with depth (van den Oord et al., 2016, *WaveNet*; Kalchbrenner et al., 2016,
*Neural Machine Translation in Linear Time*).  On a 29-character board a depth
of eight already sees the whole word, while the inductive bias stays local -
which is exactly the bias that suits affixes and digraphs, and a genuinely
different bias from both attention and recurrence.

Conditioning, heads and output contract match the other two members.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GLOBAL_FEATURE_DIM, ModelConfig, VOCAB_SIZE
from .hangmanformer import HangmanOutput
from .layers import RMSNorm, count_parameters, noisy_or_presence


class _ResidualConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.norm = RMSNorm(d)
        self.conv = nn.Conv1d(d, 2 * d, kernel, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        value, gate = self.conv(h).chunk(2, dim=1)
        return x + self.drop(self.proj(torch.tanh(value) * torch.sigmoid(gate)))


class HangmanDilatedCNN(nn.Module):
    """Gated dilated residual CNN over the character axis."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualConvBlock(d, cfg.cnn_kernel, 2 ** (i % 5), cfg.dropout)
                for i in range(cfg.cnn_layers)
            ]
        )
        self.norm_out = RMSNorm(d)
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
        cond = self.global_mlp(globals)
        x = self.tok_emb(tokens).transpose(1, 2)          # (B, d, S)
        mask = attn_mask.unsqueeze(1).to(x.dtype)
        x = x * mask
        for block in self.blocks:
            x = block(x, cond) * mask
        h = self.norm_out(x.transpose(1, 2))              # (B, S, d)

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
