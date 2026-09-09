"""Supervised trainer for the hangman models.

Nothing exotic: AdamW with a cosine schedule and linear warmup
(Loshchilov & Hutter, 2019), bf16 autocast, gradient clipping, and an
exponential moving average of the weights, which reliably buys a fraction of a
point on the win rate for free.

The loop is deliberately step-based rather than epoch-based, because the
training pool grows between DAgger rounds and "one epoch" would mean something
different each round.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .config import ExperimentConfig, OptimConfig
from .dataset import RecordDataset
from .losses import compute_loss, hit_rate, top1_agreement
from .models import build_model


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value.to(state[name].dtype))


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled on norms, biases and embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "emb" in name or "abs_pos" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def lr_at(step: int, total: int, cfg: OptimConfig) -> float:
    """Linear warmup into a cosine decay to 5% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@dataclass
class TrainState:
    step: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def log(self, row: dict[str, float]) -> None:
        self.history.append(row)


def train_steps(
    model: nn.Module,
    dataset: RecordDataset,
    cfg: ExperimentConfig,
    steps: int,
    state: TrainState | None = None,
    ema: EMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 100,
    log_path: Path | None = None,
) -> tuple[TrainState, EMA | None, torch.optim.Optimizer]:
    """Run ``steps`` optimisation steps, returning the updated handles."""
    device = torch.device(cfg.device)
    model.to(device).train()
    state = state or TrainState()
    optimizer = optimizer or build_optimizer(model, cfg.optim)
    if ema is None and cfg.optim.ema_decay > 0.0:
        ema = EMA(model, cfg.optim.ema_decay)

    if cfg.optim.compile and not getattr(model, "_hangman_compiled", False):
        # Compile in place rather than wrapping: ``Module.compile()`` keeps the
        # module identity, so ``state_dict``, the EMA and checkpointing all
        # continue to see the original parameter names.
        model.compile(dynamic=True)
        model._hangman_compiled = True

    amp_dtype = getattr(torch, cfg.optim.amp_dtype)
    started = time.time()
    running: dict[str, float] = {}

    for local in range(steps):
        lr = lr_at(state.step, steps, cfg.optim)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch = dataset.batch(dataset.sample_indices(cfg.optim.batch_size))
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            out = model(**batch)
        loss = compute_loss(out, batch, cfg.loss)

        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        state.step += 1

        for key, value in loss.as_floats().items():
            running[key] = running.get(key, 0.0) + value
        if (local + 1) % log_every == 0:
            row = {k: v / log_every for k, v in running.items()}
            row.update(
                step=state.step,
                lr=lr,
                grad_norm=float(grad_norm),
                teacher_agreement=top1_agreement(out, batch),
                hit_rate=hit_rate(out, batch),
                steps_per_sec=(local + 1) / (time.time() - started),
            )
            state.log(row)
            running = {}
            print(
                f"  step {state.step:>7d}  loss {row['total']:.4f}  "
                f"set {row['set_soft']:.4f}  pos {row['pos']:.4f}  "
                f"agree {row['teacher_agreement']:.3f}  hit {row['hit_rate']:.3f}  "
                f"{row['steps_per_sec']:.2f} it/s",
                flush=True,
            )
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(state.history, indent=1))

    return state, ema, optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: ExperimentConfig,
    state: TrainState,
    ema: EMA | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist weights, config and training history together.

    The EMA weights are stored alongside the raw ones so evaluation can compare
    them without a second training run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": state.step,
        "history": state.history,
        "extra": extra or {},
    }
    if ema is not None:
        payload["ema"] = {k: v.clone() for k, v in ema.shadow.items()}
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> tuple[nn.Module, ExperimentConfig]:
    """Rebuild a model from a checkpoint, preferring EMA weights when present."""
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig.load_from_dict(blob["config"])
    model = build_model(cfg.model).to(device)
    model.load_state_dict(blob["model"])
    if use_ema and "ema" in blob:
        state = model.state_dict()
        for name, value in blob["ema"].items():
            if name in state:
                state[name].copy_(value.to(state[name].dtype))
    model.eval()
    return model, cfg


def clone_for_eval(model: nn.Module, ema: EMA | None) -> nn.Module:
    """A frozen copy carrying the EMA weights, for mid-training evaluation."""
    shadow = copy.deepcopy(model)
    if ema is not None:
        ema.copy_to(shadow)
    return shadow.eval()
