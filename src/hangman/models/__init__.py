"""Model zoo.

Every member exposes the same contract:

    forward(tokens, attn_mask, globals) -> HangmanOutput
        set_logits   (B, 26)      presence logits, the primary output
        pos_logits   (B, L, 26)   per-position letter logits (auxiliary)
        count_logits (B, 26, K)   occurrence-count buckets (auxiliary)
        value        (B,)         win-probability logit (auxiliary)

which lets the trainer, the policies and the ensemble treat them uniformly.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .bigru import HangmanBiGRU
from .dilated_cnn import HangmanDilatedCNN
from .hangmanformer import HangmanFormer, HangmanOutput

ARCHITECTURES = {
    "transformer": HangmanFormer,
    "bigru": HangmanBiGRU,
    "dilated_cnn": HangmanDilatedCNN,
}


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate the architecture named by ``cfg.arch``."""
    try:
        cls = ARCHITECTURES[cfg.arch]
    except KeyError as exc:
        raise ValueError(
            f"unknown architecture {cfg.arch!r}; choose from {sorted(ARCHITECTURES)}"
        ) from exc
    return cls(cfg)


__all__ = [
    "ARCHITECTURES",
    "HangmanBiGRU",
    "HangmanDilatedCNN",
    "HangmanFormer",
    "HangmanOutput",
    "build_model",
]
