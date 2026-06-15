"""Resumable raw-PyTorch training loop for the LM runs.

One loop serves all three architectures (the registry returns a uniform forward). BF16
autocast (no GradScaler — fp32 AdamW master weights), gradient accumulation to a fixed
effective batch, cosine LR + warmup, a fixed TOTAL_STEPS budget so "done" is well-defined,
periodic held-out eval, JSONL logging, atomic checkpoints, and a DONE marker at the end.

Loss is computed manually from logits against the pre-shifted targets `y` (nanoGPT-style)
rather than via HF's `labels=` argument — passing already-shifted targets to HF would
double-shift (off-by-one). Both HF and our custom model expose `.logits`, so this is uniform.
"""
from __future__ import annotations

import atexit
import math
import os
import signal
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..data.packed_loader import PackedTokenLoader, RandomTokenLoader, load_manifest
from ..models.param_utils import active_params, count_params
from ..models.registry import build_model
from ..utils.logging import append_jsonl, dump_config
from ..utils.seed import set_all_seeds
from . import checkpoint as ckpt
from .schedule import cosine_with_warmup


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_loaders(cfg: Dict[str, Any], device: str, vocab_size: int):
    seed = cfg["seed"]
    bs = cfg["micro_batch_size"]
    block = cfg["block_size"]
    data_dir = cfg.get("data_dir")
    has_data = data_dir and load_manifest(data_dir) is not None
    if cfg.get("smoke") or not has_data:
        if not cfg.get("smoke"):
            print(f"[trainer] no tokenized data at {data_dir!r}; using RandomTokenLoader.")
        train = RandomTokenLoader(vocab_size, block, bs, device, seed=seed)
        val = RandomTokenLoader(vocab_size, block, bs, device, seed=seed + 999)
        return train, val
    train = PackedTokenLoader(data_dir, "train", block, bs, device, seed=seed)
    val = PackedTokenLoader(data_dir, "val", block, bs, device, seed=seed + 999)
    return train, val


@torch.no_grad()
def evaluate(model, val_loader, eval_steps: int, device: str, use_amp: bool) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y = val_loader.next_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    device = _device()
    use_amp = device == "cuda"
    run_dir = cfg["output_dir"]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if ckpt.is_done(run_dir):
        print(f"[trainer] {run_dir} already DONE — nothing to do.")
        return {"status": "already_done"}

    set_all_seeds(cfg["seed"])
    model_cfg = dict(cfg["model"])
    vocab_size = model_cfg.get("vocab_size", 50304)

    model = build_model(cfg["arch"], model_cfg).to(device)
    if cfg.get("grad_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False  # incompatible with checkpointing
            print("[trainer] gradient checkpointing enabled (trades compute for memory)")
        except Exception as e:
            print(f"[trainer] gradient checkpointing unavailable: {e}")
    n_total = count_params(model)
    n_active = active_params(
        model,
        num_experts=int(model_cfg.get("num_experts", 1)),
        num_experts_per_tok=int(model_cfg.get("num_experts_per_tok", 1)),
    )
    print(f"[trainer] {cfg['arch']}: {n_total/1e6:.1f}M total / {n_active/1e6:.1f}M active params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"], eps=1e-8,
    )
    scheduler = cosine_with_warmup(
        optimizer, cfg["warmup_steps"], cfg["total_steps"], cfg["min_lr_ratio"]
    )

    train_loader, val_loader = _build_loaders(cfg, device, vocab_size)

    # config.json captures the full run spec (param counts included for the results table)
    run_config = dict(cfg)
    run_config.update({"n_params_total": n_total, "n_params_active": n_active,
                       "device": device, "torch_version": torch.__version__})
    dump_config(os.path.join(run_dir, "config.json"), run_config)

    global_step, best_val = ckpt.maybe_resume(
        ckpt_dir, model, optimizer, scheduler, train_loader, expected_arch=cfg["arch"]
    )

    log_path = os.path.join(run_dir, "log.jsonl")
    stage_dir = cfg.get("stage_dir")  # e.g. '/content' on Colab for fast atomic staging

    def snapshot(name: str) -> None:
        state = ckpt.build_state(model, optimizer, scheduler, global_step,
                                 train_loader.epoch, best_val, train_loader, run_config)
        ckpt.save_checkpoint(state, ckpt_dir, name, stage_dir=stage_dir)

    # flush last.pt on SIGTERM / process exit (Colab idle/12h kill best-effort)
    _interrupted = {"flag": False}

    def _flush_and_exit(signum, frame):  # pragma: no cover (signal path)
        if not _interrupted["flag"]:
            _interrupted["flag"] = True
            print(f"\n[trainer] signal {signum} — flushing last.pt before exit.")
            try:
                snapshot("last.pt")
            finally:
                os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _flush_and_exit)
        except Exception:
            pass
    atexit.register(lambda: (not _interrupted["flag"]) and snapshot("last.pt"))

    tokens_per_step = train_loader.tokens_per_batch * cfg["grad_accum"]
    model.train()
    t_window = time.time()

    while global_step < cfg["total_steps"]:
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss_accum = 0.0
        for _ in range(cfg["grad_accum"]):
            x, y = train_loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids=x).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
                ) / cfg["grad_accum"]
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % cfg["log_every"] == 0:
            dt = time.time() - t_window
            tok_per_sec = tokens_per_step * cfg["log_every"] / max(1e-9, dt)
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
            append_jsonl(log_path, {
                "step": global_step, "loss": loss_accum,
                "lr": scheduler.get_last_lr()[0],
                "tokens_seen": global_step * tokens_per_step,
                "tok_per_sec": tok_per_sec, "peak_mem_gb": peak_gb,
                "epoch": train_loader.epoch, "wall_s": time.time(),
            })
            t_window = time.time()

        if global_step % cfg["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, cfg["eval_steps"], device, use_amp)
            ppl = math.exp(min(20.0, val_loss))
            append_jsonl(log_path, {
                "step": global_step, "val_loss": val_loss, "val_ppl": ppl,
                "val_bits_per_token": val_loss / math.log(2),
            })
            if val_loss < best_val:
                best_val = val_loss
                snapshot("best.pt")

        if global_step % cfg["save_every"] == 0:
            snapshot("last.pt")
        if cfg["snapshot_every"] and global_step % cfg["snapshot_every"] == 0:
            snapshot(f"step{global_step}.pt")
            ckpt.prune_snapshots(ckpt_dir, keep=cfg.get("keep_snapshots", 3))

    snapshot("last.pt")
    _interrupted["flag"] = True  # prevent atexit double-save
    ckpt.write_done(run_dir, {"global_step": global_step, "best_val": best_val,
                              "n_params_total": n_total, "n_params_active": n_active})
    print(f"[trainer] DONE {cfg['arch']} at step {global_step}, best_val={best_val:.4f}")
    return {"status": "done", "global_step": global_step, "best_val": best_val}
