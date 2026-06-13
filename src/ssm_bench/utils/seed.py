"""Seed control for reproducible runs.

Sets every RNG we touch (python / numpy / torch-cpu / cuda). torch is imported
lazily so numpy-only code paths (e.g. synthetic generators, the loader) can use
this module without a torch install.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy, and (if available) torch + cuda.

    Call BEFORE model construction so weight init is reproducible.
    `deterministic=True` also forces deterministic cuDNN (slower).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # torch not installed (numpy-only context)
        pass


def capture_rng_state() -> dict:
    """Snapshot all RNG states for exact-resume checkpoints."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNG states captured by `capture_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch

        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass
