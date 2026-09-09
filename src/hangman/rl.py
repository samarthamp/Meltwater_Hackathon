"""Self-play policy refinement.

Supervised training optimises agreement with a posterior.  The leaderboard
rewards *finishing words within six misses*, and those are not the same
objective: the supervised loss is indifferent to the order in which lives are
spent, while the game is not.  This stage closes that gap by optimising the
actual return with a policy gradient.

Setup
-----
*Action distribution.*  The presence head gives independent per-letter
probabilities, not a distribution over moves, so we induce one:
``pi(l | s) = softmax(set_logits / tau)`` restricted to unguessed letters.
Greedy play is the ``tau -> 0`` limit, so the supervised model is already a
sensible starting policy.

*Reward.*  ``win - efficiency_penalty * wrong_guesses``, matching the published
ranking ``Score - E / 1e8``: solving the word dominates, and among equal
outcomes fewer misses ranks higher.

*Baseline.*  The value head, trained in the supervised stage on Monte-Carlo
returns, is reused as the advantage baseline (Williams, 1992).

*Trust region.*  A KL penalty back to the frozen supervised policy.  Without it
the policy sharpens into a near-deterministic argmax within a few hundred
updates and the probabilities stop being usable by the value-guided policy;
with it the refinement stays local.  This is the PPO-style regularisation of
Schulman et al. (2017) in its penalty rather than clipped form, which is
sufficient here because we take one gradient step per batch of fresh rollouts
and therefore never go off-policy.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import MAX_WRONG_GUESSES, RLConfig
from .simulator import BatchGame
from .state import encode_globals, encode_tokens


@dataclass
class RLStats:
    history: list[dict] = field(default_factory=list)

    def log(self, row: dict) -> None:
        self.history.append(row)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.history, indent=1))
        return path


#: Mask value for already-guessed letters.  Deliberately a large finite number
#: rather than ``-inf``: a row that is entirely ``-inf`` softmaxes to NaN, and
#: ``torch.multinomial`` on a NaN row raises a device-side assert that surfaces
#: asynchronously, several calls later, pointing at innocent code.  A finite
#: floor keeps every row a valid distribution no matter what the model emits.
MASKED_LOGIT = -1e4


def _policy_logits(model, board, lengths, present, absent, amp_dtype=torch.bfloat16):
    tokens, attn = encode_tokens(board)
    globals_vec = encode_globals(lengths, present, absent)
    with torch.autocast(device_type=board.device.type, dtype=amp_dtype):
        out = model(tokens=tokens, attn_mask=attn, globals=globals_vec)
    logits = out.set_logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=MASKED_LOGIT)
    logits = logits.masked_fill(present | absent, MASKED_LOGIT)
    value = out.value.float() if out.value is not None else None
    return logits, value


def _sample_actions(logits: torch.Tensor, temperature: float, allowed: torch.Tensor) -> torch.Tensor:
    """Sample one letter per state, robustly.

    Guards the two ways a policy-gradient rollout can hand ``multinomial`` an
    invalid row - a non-finite probability, and a row that sums to zero - by
    falling back to a uniform draw over the letters still available.
    """
    probs = torch.softmax(logits / temperature, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    degenerate = probs.sum(dim=1) <= 1e-8
    if bool(degenerate.any()):
        fallback = allowed.to(probs.dtype)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = torch.where(degenerate.unsqueeze(1), fallback, probs)
    still_bad = probs.sum(dim=1) <= 1e-8
    if bool(still_bad.any()):
        # Every letter guessed: the game is over and the action is ignored.
        probs = torch.where(still_bad.unsqueeze(1), torch.full_like(probs, 1.0 / 26.0), probs)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


@torch.no_grad()
def rollout(
    model,
    reference,
    words: list[str],
    cfg: RLConfig,
    temperature: float,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Sample one game per word, recording the states visited and the outcome."""
    game = BatchGame(words, device=device)
    boards, lens, presents, absents, actions, rows = [], [], [], [], [], []

    for _ in range(32):
        active = game.active()
        if not bool(active.any()):
            break
        idx = torch.nonzero(active, as_tuple=True)[0]
        board = game.board()[idx]
        present, absent = game.present[idx], game.absent[idx]
        logits, _ = _policy_logits(model, board, game.lengths[idx], present, absent)
        action = _sample_actions(logits, temperature, ~(present | absent))

        boards.append(board)
        lens.append(game.lengths[idx])
        presents.append(game.present[idx].clone())
        absents.append(game.absent[idx].clone())
        actions.append(action)
        rows.append(idx)

        full = torch.zeros(game.batch_size, dtype=torch.long, device=device)
        full[idx] = action
        game.apply(full, active)

    won = game.solved().to(torch.float32)
    reward = won - cfg.efficiency_penalty * game.wrong.to(torch.float32)
    all_rows = torch.cat(rows)
    return {
        "board": torch.cat(boards),
        "lengths": torch.cat(lens),
        "present": torch.cat(presents),
        "absent": torch.cat(absents),
        "action": torch.cat(actions),
        "return": reward[all_rows],
        "won_mean": won.mean(),
        "wrong_mean": game.wrong.to(torch.float32).mean(),
    }


def refine(
    model,
    dicts,
    cfg: RLConfig,
    steps: int,
    games_per_step: int = 2048,
    temperature: float = 1.0,
    device: str | torch.device = "cuda",
    log_every: int = 25,
    seed: int = 0,
) -> tuple[object, RLStats]:
    """Refine ``model`` in place by policy gradient on self-play.

    Returns the frozen reference policy (for later comparison) and the log.
    """
    reference = copy.deepcopy(model).eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0
    )
    rng = np.random.default_rng(seed)
    by_length: dict[int, list[str]] = {}
    for word in dicts.fit:
        by_length.setdefault(len(word), []).append(word)
    lengths = np.array(sorted(by_length))
    counts = np.array([len(by_length[int(L)]) for L in lengths], dtype=np.float64)
    probs = counts / counts.sum()

    stats = RLStats()
    started = time.time()
    running: dict[str, float] = {}

    for step in range(steps):
        length = int(rng.choice(lengths, p=probs))
        pool = by_length[length]
        take = min(games_per_step, len(pool))
        words = [pool[i] for i in rng.integers(0, len(pool), size=take)]

        traj = rollout(model, reference, words, cfg, temperature, device=device)

        logits, value = _policy_logits(
            model, traj["board"], traj["lengths"], traj["present"], traj["absent"]
        )
        log_probs = torch.log_softmax(logits / temperature, dim=1)
        chosen = log_probs.gather(1, traj["action"].unsqueeze(1)).squeeze(1)
        chosen = torch.nan_to_num(chosen, nan=0.0, neginf=MASKED_LOGIT)

        returns = traj["return"]
        baseline = torch.sigmoid(value).detach() if value is not None else returns.mean()
        advantage = returns - baseline
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)

        policy_loss = -(chosen * advantage).mean()

        value_loss = logits.new_zeros(())
        if value is not None:
            value_loss = F.binary_cross_entropy_with_logits(
                value, returns.clamp(0.0, 1.0)
            )

        with torch.no_grad():
            ref_logits, _ = _policy_logits(
                reference, traj["board"], traj["lengths"], traj["present"], traj["absent"]
            )
        ref_log_probs = torch.log_softmax(ref_logits / temperature, dim=1)
        finite = (
            torch.isfinite(ref_log_probs)
            & torch.isfinite(log_probs)
            & (ref_log_probs > MASKED_LOGIT / 2)
        )
        kl = (
            (ref_log_probs.exp() * (ref_log_probs - log_probs))
            .masked_fill(~finite, 0.0)
            .sum(dim=1)
            .mean()
        )

        entropy = -(log_probs.exp() * log_probs).masked_fill(~finite, 0.0).sum(dim=1).mean()

        loss = (
            policy_loss
            + cfg.value_coef * value_loss
            + cfg.kl_coef * kl
            - cfg.entropy_coef * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for key, val in (
            ("loss", loss), ("policy", policy_loss), ("value", value_loss),
            ("kl", kl), ("entropy", entropy),
            ("sampled_win", traj["won_mean"]), ("wrong", traj["wrong_mean"]),
        ):
            running[key] = running.get(key, 0.0) + float(val.detach())

        if (step + 1) % log_every == 0:
            row = {k: v / log_every for k, v in running.items()}
            row.update(step=step + 1, steps_per_sec=(step + 1) / (time.time() - started))
            stats.log(row)
            print(
                f"  [rl] {step + 1:>5d}/{steps}  sampled-win {row['sampled_win']:.4f}  "
                f"wrong {row['wrong']:.2f}  kl {row['kl']:.4f}  "
                f"H {row['entropy']:.3f}  {row['steps_per_sec']:.2f} it/s",
                flush=True,
            )
            running = {}

    model.eval()
    return reference, stats
