#!/usr/bin/env python
"""Entry point for one LM training run (one arch, one seed). Resumable.

Usage:
    python scripts/train.py --arch transformer --seed 1337 \
        --data_dir data_cache/fineweb_edu_gpt2 \
        --output_dir runs/transformer_seed1337 --total_steps 6000

If --output_dir already contains a checkpoint, training resumes from it exactly. If it
contains a DONE marker, the run is skipped. With --smoke (or when no tokenized data is
present) a tiny model + random tokens are used to sanity-check the harness end to end.
"""
from __future__ import annotations

import argparse
import os
import sys

# make `ssm_bench` importable without `pip install -e .`
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import yaml  # noqa: E402

from ssm_bench.train.trainer import train  # noqa: E402


def tiny_config(arch: str) -> dict:
    """Minimal per-arch config for --smoke (fast, CPU-friendly)."""
    common = dict(vocab_size=512, hidden_size=64, max_position_embeddings=128,
                  tie_word_embeddings=True)
    if arch == "transformer":
        return {**common, "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 2, "intermediate_size": 128, "hidden_act": "silu"}
    if arch == "mamba":
        return {**common, "num_hidden_layers": 2, "num_heads": 2, "head_dim": 64,
                "state_size": 16, "expand": 2, "n_groups": 1, "conv_kernel": 4,
                "chunk_size": 32}
    if arch == "jamba":
        return {**common, "num_hidden_layers": 4, "num_attention_heads": 4,
                "num_key_value_heads": 2, "intermediate_size": 128, "mamba_d_state": 16,
                "mamba_d_conv": 4, "mamba_expand": 2, "mamba_dt_rank": "auto",
                "attn_layer_period": 2, "attn_layer_offset": 1, "expert_layer_period": 2,
                "expert_layer_offset": 1, "num_experts": 2, "num_experts_per_tok": 2}
    raise KeyError(arch)


def load_model_cfg(arch: str, config_dir: str) -> dict:
    path = os.path.join(config_dir, f"{arch}.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    assert doc["arch"] == arch, f"{path} arch mismatch"
    return dict(doc["model"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one LM arm (resumable).")
    ap.add_argument("--arch", required=True, choices=["transformer", "mamba", "jamba"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--config_dir", default="configs")
    ap.add_argument("--data_dir", default="data_cache/fineweb_edu_gpt2")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage_dir", default=None, help="fast local dir for atomic staging (Colab: /content)")
    # schedule / optimization
    ap.add_argument("--total_steps", type=int, default=6000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # cadence
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--snapshot_every", type=int, default=1000)
    ap.add_argument("--keep_snapshots", type=int, default=3)
    ap.add_argument("--ckpt_seconds", type=int, default=600,
                    help="also checkpoint at least this often (wall-clock); 0 disables")
    ap.add_argument("--grad_checkpointing", action="store_true",
                    help="trade compute for memory")
    ap.add_argument("--no_autocast", action="store_true",
                    help="disable bf16 autocast; run in the model's native dtype (fp32). "
                         "Needed to use the Jamba Mamba kernel (kernel + autocast mismatch dtypes).")
    ap.add_argument("--force_kernels", action="store_true",
                    help="force the fast Mamba CUDA kernel for Jamba (only safe with --no_autocast)")
    ap.add_argument("--smoke", action="store_true", help="tiny model + random tokens")
    args = ap.parse_args()

    if args.smoke:
        model_cfg = tiny_config(args.arch)
        args.block_size = min(args.block_size, 64)
        args.micro_batch_size = min(args.micro_batch_size, 4)
        args.grad_accum = 1
        args.eval_every = max(1, min(args.eval_every, 10))
        args.eval_steps = 2
        args.save_every = max(1, min(args.save_every, 10))
        args.snapshot_every = 0
        args.warmup_steps = min(args.warmup_steps, 2)
    else:
        model_cfg = load_model_cfg(args.arch, args.config_dir)

    if args.force_kernels:
        model_cfg["force_kernels"] = True

    cfg = vars(args)
    cfg["model"] = model_cfg
    train(cfg)


if __name__ == "__main__":
    main()
