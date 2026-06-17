"""Synthetic mechanistic-task sweep: MQAR / induction / selective-copy.

Small (2-layer) versions of the same backbones are trained per task cell and evaluated by
accuracy on the answer positions, swept over sequence length so the architectures diverge:
  * MQAR           — attention/Jamba stay high; pure Mamba degrades as recall load grows.
  * induction      — train short, EXTRAPOLATE long; Mamba generalizes far, attention doesn't.
  * selective copy — selectivity task; all solve it (sanity / lower bound).

One CLI invocation runs the full internal sweep for (arch, task), writing one idempotent
JSONL record per cell (skipped on re-run, so it's resumable on Colab).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ssm_bench.data.synthetics import (make_induction, make_mqar,  # noqa: E402
                                       make_selective_copy)
from ssm_bench.utils.logging import append_jsonl, read_jsonl, run_id  # noqa: E402
from ssm_bench.utils.seed import set_all_seeds  # noqa: E402

# ----------------------------------------------------------- small models ------

def small_config(arch: str, d_model: int, vocab: int, n_layer: int, max_pos: int) -> dict:
    common = dict(vocab_size=vocab, hidden_size=d_model, max_position_embeddings=max_pos,
                  tie_word_embeddings=True)
    if arch == "transformer":
        return {**common, "num_hidden_layers": n_layer, "num_attention_heads": 4,
                "num_key_value_heads": 4, "intermediate_size": 2 * d_model,
                "hidden_act": "silu", "rope_theta": 10000.0}
    if arch == "mamba":
        inner = 2 * d_model
        head_dim = 32
        return {**common, "num_hidden_layers": n_layer, "num_heads": inner // head_dim,
                "head_dim": head_dim, "state_size": 64, "expand": 2, "n_groups": 1,
                "conv_kernel": 4, "chunk_size": 32}
    if arch == "jamba":
        # force the torch path: the Jamba Mamba kernel mismatches dtypes under bf16 autocast
        # (used here), and these synthetic models are tiny so the torch path is plenty fast.
        return {**common, "num_hidden_layers": max(n_layer, 2), "num_attention_heads": 4,
                "num_key_value_heads": 2, "intermediate_size": 2 * d_model,
                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
                "mamba_dt_rank": "auto", "attn_layer_period": 2, "attn_layer_offset": 1,
                "expert_layer_period": 2, "expert_layer_offset": 1, "num_experts": 2,
                "num_experts_per_tok": 2, "force_torch": True}
    raise KeyError(arch)


# ------------------------------------------------------------- train/eval ------

def _to_torch(arr, device):
    import torch

    return torch.from_numpy(arr).to(device)


def train_cell(arch: str, gen: Callable, gkw: dict, d_model: int, vocab: int,
               max_pos: int, steps: int, lr: float, batch: int, seed: int,
               n_layer: int = 2) -> Dict:
    import torch
    import torch.nn.functional as F

    from ssm_bench.models.registry import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    set_all_seeds(seed)
    model = build_model(arch, small_config(arch, d_model, vocab, n_layer, max_pos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    for step in range(steps):
        inp, lab = gen(batch_size=batch, seed=seed + 1 + step, **gkw)
        x, y = _to_torch(inp, device), _to_torch(lab, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids=x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    n_params = sum(p.numel() for p in model.parameters())
    return {"model": model, "device": device, "n_params": n_params}


def eval_acc(model, gen: Callable, gkw: dict, vocab: int, batch: int,
             eval_seed: int, device: str, micro: int = 8) -> float:
    """Accuracy on answer positions, micro-batched and OOM-safe.

    Long induction eval lengths on Jamba's torch path materialize huge
    [B, d_inner, L, d_state] activations, so evaluate in micro-batches and halve the
    sub-batch on OOM (returning NaN if even a single example won't fit), rather than
    crashing the whole sweep.
    """
    import torch

    model.eval()
    inp, lab = gen(batch_size=batch, seed=eval_seed, **gkw)
    correct = total = 0
    i = 0
    while i < batch:
        bs = min(micro, batch - i)
        while bs >= 1:
            try:
                x = _to_torch(inp[i:i + bs], device)
                y = _to_torch(lab[i:i + bs], device)
                with torch.no_grad():
                    logits = model(input_ids=x).logits
                pred = logits[:, :-1].argmax(-1)
                gold = y[:, 1:]
                m = gold != -100
                if m.any():
                    correct += (pred[m] == gold[m]).sum().item()
                    total += int(m.sum().item())
                del logits
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                if "out of memory" not in str(ex).lower():
                    raise
                torch.cuda.empty_cache()
                bs //= 2
        if bs < 1:  # even one example won't fit
            torch.cuda.empty_cache()
            return float("nan") if total == 0 else correct / total
        i += bs
    return float("nan") if total == 0 else correct / total


# ----------------------------------------------------------------- sweeps ------

def _mqar_cells(d_models: List[int]) -> List[Tuple[int, int, int]]:
    pairs = [(64, 4), (128, 8), (256, 16), (512, 64)]
    return [(L, kv, d) for (L, kv) in pairs for d in d_models]


def run_sweep(arch: str, task: str, out: str, steps: int, lr: float, batch: int,
              seed: int, d_models: List[int], force: bool) -> None:
    done = set() if force else {r["run_id"] for r in read_jsonl(out) if "run_id" in r}

    def emit(cell: dict, acc: float, n_params: int, train_len: int) -> None:
        rid = run_id(task=task, arch=arch, **cell)
        if rid in done:
            return
        rec = {"run_id": rid, "task": task, "arch": arch, "test_acc": acc,
               "n_params": n_params, "train_len": train_len, **cell}
        append_jsonl(out, rec)
        done.add(rid)
        print(f"  [{task}/{arch}] {cell} -> acc={acc:.3f}")

    if task == "mqar":
        vocab = 8192
        for (L, kv, d) in _mqar_cells(d_models):
            cell = {"seq_len": L, "num_kv_pairs": kv, "d_model": d}
            rid = run_id(task=task, arch=arch, **cell)
            if rid in done:
                continue
            tr = train_cell(arch, make_mqar, {"seq_len": L, "num_kv_pairs": kv, "vocab_size": vocab},
                            d, vocab, max_pos=L, steps=steps, lr=lr, batch=batch, seed=seed)
            acc = eval_acc(tr["model"], make_mqar,
                           {"seq_len": L, "num_kv_pairs": kv, "vocab_size": vocab},
                           vocab, batch, eval_seed=10_000, device=tr["device"])
            emit(cell, acc, tr["n_params"], L)
            del tr

    elif task == "induction":
        vocab = 16
        train_len = 256
        eval_lens = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        d = d_models[-1] if d_models else 128
        # train ONCE at train_len, evaluate at many lengths (extrapolation)
        if not all(run_id(task=task, arch=arch, eval_len=el, d_model=d) in done for el in eval_lens):
            tr = train_cell(arch, make_induction, {"seq_len": train_len, "vocab_size": vocab},
                            d, vocab, max_pos=max(eval_lens), steps=steps, lr=lr,
                            batch=batch, seed=seed)
            for el in eval_lens:
                cell = {"eval_len": el, "d_model": d}
                acc = eval_acc(tr["model"], make_induction,
                               {"seq_len": el, "vocab_size": vocab},
                               vocab, batch, eval_seed=10_000, device=tr["device"])
                emit(cell, acc, tr["n_params"], train_len)
            del tr

    elif task == "selective_copy":
        vocab = 16
        d = d_models[-1] if d_models else 128
        for L in [256, 512, 1024]:
            for nt in [16, 32]:
                cell = {"seq_len": L, "num_tokens": nt, "d_model": d}
                rid = run_id(task=task, arch=arch, **cell)
                if rid in done:
                    continue
                gkw = {"seq_len": L, "num_tokens": nt, "vocab_size": vocab}
                tr = train_cell(arch, make_selective_copy, gkw, d, vocab,
                                max_pos=L + 1 + nt, steps=steps, lr=lr, batch=batch, seed=seed)
                acc = eval_acc(tr["model"], make_selective_copy, gkw, vocab, batch,
                               eval_seed=10_000, device=tr["device"])
                emit(cell, acc, tr["n_params"], L)
                del tr
    else:
        raise KeyError(task)


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic mechanistic-task sweep.")
    ap.add_argument("--arch", required=True, choices=["transformer", "mamba", "jamba"])
    ap.add_argument("--task", required=True, choices=["mqar", "induction", "selective_copy"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--d_models", default="64,128,256")
    ap.add_argument("--out", default="results/summaries/synthetic.jsonl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    d_models = [int(x) for x in args.d_models.split(",") if x]
    run_sweep(args.arch, args.task, args.out, args.steps, args.lr, args.batch,
              args.seed, d_models, args.force)


if __name__ == "__main__":
    main()
