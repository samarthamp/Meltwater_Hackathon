"""Shared building blocks: RMSNorm, SwiGLU, rotary embeddings, FiLM modulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    Cheaper than LayerNorm and empirically as stable in pre-norm transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``sum(dtype=float32)`` accumulates in fp32 without materialising an
        # fp32 copy of ``x``.  On this machine (273 GB/s against ~100 TFLOPS)
        # the model is memory-bound, so avoiding that copy is worth more than
        # the arithmetic it saves.
        var = x.pow(2).sum(-1, keepdim=True, dtype=torch.float32) / x.shape[-1]
        return x * torch.rsqrt(var + self.eps).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward block (Shazeer, 2020, *GLU Variants Improve Transformer*)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021).

    Applied on the character axis so that the model sees *relative* offsets
    between letters - the signal that digraph and affix patterns live in -
    while the learned absolute embedding separately marks word-initial and
    word-final positions.
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("rotary head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = q.shape[-2]
        cos = self.cos[:, :, :seq, :].to(q.dtype)
        sin = self.sin[:, :, :seq, :].to(q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class FiLMGenerator(nn.Module):
    """Produces per-layer scale/shift pairs from the global conditioning vector.

    Feature-wise Linear Modulation (Perez et al., 2018).  The guessed-letter
    sets and remaining lives are *global* constraints on the whole board, so
    injecting them only at the ``[CLS]`` token would force the network to route
    them through attention.  FiLM instead lets them modulate every block
    directly.  Initialised to zero so the network starts as an unconditioned
    transformer and learns to use the conditioning.
    """

    def __init__(self, cond_dim: int, d_model: int, n_layers: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(bottleneck, 4 * d_model) for _ in range(n_layers)]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        h = self.trunk(cond)
        out = []
        for head in self.heads:
            gamma_a, beta_a, gamma_f, beta_f = head(h).chunk(4, dim=-1)
            out.append((gamma_a.unsqueeze(1), beta_a.unsqueeze(1),
                        gamma_f.unsqueeze(1), beta_f.unsqueeze(1)))
        return out


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer block with adaLN-style FiLM."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        drop_path: float,
        rope: RotaryEmbedding | None,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.drop_path = drop_path

        self.norm_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.resid_drop = nn.Dropout(dropout)

    def _stochastic_depth(self, branch: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path <= 0.0:
            return branch
        keep = 1.0 - self.drop_path
        mask = torch.rand(branch.shape[0], 1, 1, device=branch.device) < keep
        return branch * mask / keep

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        film: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        gamma_a, beta_a, gamma_f, beta_f = (
            film if film is not None else (0.0, 0.0, 0.0, 0.0)
        )

        h = self.norm_attn(x)
        h = h * (1.0 + gamma_a) + beta_a
        batch, seq, _ = h.shape
        qkv = self.qkv(h).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q, k)
        attn_bias = pad_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq, -1)
        x = x + self._stochastic_depth(self.resid_drop(self.proj(attended)))

        h = self.norm_ffn(x)
        h = h * (1.0 + gamma_f) + beta_f
        return x + self._stochastic_depth(self.ffn(h))


def noisy_or_presence(pos_logits: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
    """Pool per-position letter distributions into presence probabilities.

    ``P(letter present) = 1 - prod_over_hidden_positions (1 - p_i(letter))``,
    evaluated in log space for stability.  This is the correct probabilistic
    aggregation; the max-pooling used by public hangman transformers
    systematically understates letters that are likely to occur more than once.

    Parameters
    ----------
    pos_logits : (B, L, 26)
    hidden_mask : (B, L) bool, True at positions still hidden.
    """
    probs = pos_logits.float().softmax(dim=-1)
    log_miss = torch.log1p(-probs.clamp(max=1.0 - 1e-6))
    log_miss = log_miss * hidden_mask.unsqueeze(-1).to(log_miss.dtype)
    return -torch.expm1(log_miss.sum(dim=1))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def layer_table(model: nn.Module) -> list[tuple[str, str, int]]:
    """``(name, shape, count)`` for every top-level submodule, for the report."""
    rows: list[tuple[str, str, int]] = []
    for name, child in model.named_children():
        n = count_parameters(child)
        shapes = {tuple(p.shape) for p in child.parameters()}
        shape_str = ", ".join(str(s) for s in sorted(shapes)[:3])
        if len(shapes) > 3:
            shape_str += ", ..."
        rows.append((name, shape_str, n))
    return rows
