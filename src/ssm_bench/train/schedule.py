"""Cosine LR schedule with linear warmup (resumable via scheduler.state_dict)."""
from __future__ import annotations

import math


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                       min_lr_ratio: float = 0.1):
    """LambdaLR: linear warmup to 1.0, then cosine decay to `min_lr_ratio`.

    The multiplier is applied to each param group's base lr, so resuming restores the
    exact schedule position from `scheduler.state_dict()`.
    """
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
