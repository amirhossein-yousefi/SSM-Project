"""Language-modeling quality: strided sliding-window perplexity / bits-per-token.

Uses the HuggingFace `labels=` contract (the model shifts internally) with the context
portion of each window masked to -100, and aggregates summed-loss x counted-tokens (NOT a
mean of per-window means, which is biased by the shorter final window). Works for all three
HF models and the custom Mamba block alike.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, Optional

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def load_val_tokens(data_dir: str, max_tokens: Optional[int] = None) -> np.ndarray:
    """Concatenate the val_*.npy shards into one int64 token array (capped)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "val_*.npy")))
    if not paths:
        raise FileNotFoundError(f"no val_*.npy in {data_dir}")
    chunks = []
    total = 0
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        if max_tokens is not None and total + len(arr) > max_tokens:
            arr = arr[: max_tokens - total]
        chunks.append(np.asarray(arr, dtype=np.int64))
        total += len(arr)
        if max_tokens is not None and total >= max_tokens:
            break
    return np.concatenate(chunks)


def eval_ppl(model, ids, max_len: int = 1024, stride: int = 512,
             device: str = "cuda", use_amp: bool = True) -> Dict[str, float]:
    import torch

    model.eval()
    use_amp = use_amp and device == "cuda"
    nll_sum, n_tok = 0.0, 0
    prev_end = 0
    n = ids.size(0)
    with torch.no_grad():
        for begin in range(0, n, stride):
            end = min(begin + max_len, n)
            trg_len = end - prev_end
            input_ids = ids[begin:end].unsqueeze(0).to(device)
            target = input_ids.clone()
            target[:, :-trg_len] = -100  # only score the newly-revealed tokens
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(input_ids=input_ids, labels=target)
            counted = int((target != -100).sum().item()) - 1  # -1 for internal shift
            if counted > 0:
                nll_sum += out.loss.item() * counted
                n_tok += counted
            prev_end = end
            if end == n:
                break
    nll = nll_sum / max(1, n_tok)
    return {
        "ppl": math.exp(min(20.0, nll)),
        "nll_nats": nll,
        "bits_per_token": nll / math.log(2),
        "n_tok": n_tok,
    }


def evaluate_run(run_dir: str, data_dir: str, ckpt_name: str = "best.pt",
                 max_tokens: int = 2_000_000, max_len: int = 1024,
                 stride: int = 512) -> Dict[str, float]:
    import torch

    from ssm_bench.models.registry import build_model

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    arch = cfg["arch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Jamba's Mamba CUDA kernel mismatches dtypes under autocast, so eval it on the
    # torch path with autocast off (matching how it was trained). Strip the persisted
    # force_kernels control flag so the rebuild can't re-enable the kernel.
    model_cfg = dict(cfg["model"])
    model_cfg.pop("force_kernels", None)
    if arch == "jamba":
        model_cfg["force_torch"] = True
    use_amp = arch != "jamba"

    model = build_model(arch, model_cfg).to(device)
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", "last.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    ids = torch.from_numpy(load_val_tokens(data_dir, max_tokens))
    res = eval_ppl(model, ids, max_len=max_len, stride=stride, device=device, use_amp=use_amp)
    res.update({"arch": arch, "run_dir": run_dir,
                "n_params_total": cfg.get("n_params_total"),
                "n_params_active": cfg.get("n_params_active"),
                "ckpt": os.path.basename(ckpt_path)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Strided-window perplexity for a trained run.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data_cache/fineweb_edu_gpt2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--max_tokens", type=int, default=2_000_000)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", default="results/summaries/lm_quality.jsonl")
    args = ap.parse_args()

    res = evaluate_run(args.run_dir, args.data_dir, args.ckpt,
                       args.max_tokens, args.max_len, args.stride)
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
