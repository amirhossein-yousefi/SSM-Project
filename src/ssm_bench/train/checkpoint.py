"""Checkpointing for bit-exact resume.

Saves everything needed to continue training as if never interrupted: model, optimizer,
LR scheduler, step/epoch/best, the data cursor, and all RNG states. Writes are atomic
(tmp + os.replace) so a crash mid-write can only corrupt the .tmp, never the live file.
On Colab the saver can stage to fast local disk first, then atomic-rename onto Drive.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
from typing import Optional, Tuple

import torch

from ..utils.seed import capture_rng_state, restore_rng_state


def atomic_save(obj, final_path: str, stage_dir: Optional[str] = None) -> None:
    """Atomically write `obj` to `final_path`.

    If `stage_dir` is given (e.g. '/content' on Colab), serialize there first then copy
    onto the (slow, FUSE-mounted) Drive path and atomic-rename — so a torn torch.save
    never touches the live checkpoint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    tmp = final_path + ".tmp"
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
        local = os.path.join(stage_dir, os.path.basename(final_path) + ".stage")
        torch.save(obj, local)
        shutil.copyfile(local, tmp)
        os.remove(local)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, final_path)


def build_state(model, optimizer, scheduler, global_step: int, epoch: int,
                best_val: float, loader, config: dict) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "best_val": best_val,
        "data": loader.state_dict(),
        "rng": capture_rng_state(),
        "config": config,
        "torch_version": torch.__version__,
    }


def save_checkpoint(state: dict, ckpt_dir: str, name: str,
                    stage_dir: Optional[str] = None) -> str:
    path = os.path.join(ckpt_dir, name)
    atomic_save(state, path, stage_dir=stage_dir)
    return path


def find_latest(ckpt_dir: str) -> Optional[str]:
    """Prefer last.pt; otherwise the highest-numbered stepN.pt."""
    last = os.path.join(ckpt_dir, "last.pt")
    if os.path.exists(last):
        return last
    snaps = glob.glob(os.path.join(ckpt_dir, "step*.pt"))
    if not snaps:
        return None
    return max(snaps, key=_step_of)


def _step_of(path: str) -> int:
    m = re.search(r"step(\d+)\.pt$", path)
    return int(m.group(1)) if m else -1


def maybe_resume(ckpt_dir: str, model, optimizer, scheduler, loader,
                 expected_arch: Optional[str] = None) -> Tuple[int, float]:
    """Restore exact state from the newest checkpoint. Returns (global_step, best_val)."""
    path = find_latest(ckpt_dir)
    if path is None:
        return 0, float("inf")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if expected_arch is not None:
        saved = ck.get("config", {}).get("arch")
        if saved is not None and saved != expected_arch:
            raise ValueError(f"resume arch mismatch: ckpt={saved} expected={expected_arch}")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    loader.load_state_dict(ck["data"])
    restore_rng_state(ck.get("rng", {}))
    print(f"[resume] restored from {path} at step {ck['global_step']}")
    return ck["global_step"], ck["best_val"]


def prune_snapshots(ckpt_dir: str, keep: int = 3) -> None:
    snaps = sorted(glob.glob(os.path.join(ckpt_dir, "step*.pt")), key=_step_of)
    for path in snaps[:-keep] if keep > 0 else snaps:
        try:
            os.remove(path)
        except OSError:
            pass


def write_done(run_dir: str, info: dict) -> None:
    import json

    with open(os.path.join(run_dir, "DONE"), "w") as f:
        json.dump(info, f, indent=2)


def is_done(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "DONE"))
