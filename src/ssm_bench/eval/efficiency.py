"""Efficiency benchmark — the headline architecture-level result.

For each ~125M model and sequence length, measures: train-step throughput, peak GPU memory,
prefill latency, and autoregressive decode latency. The story: Transformer decode latency and
KV-cache memory grow with L; Mamba's recurrent state is ~constant; Jamba sits between.

Methodology: warmup iters, cuda.synchronize around timed regions, median over trials,
reset_peak_memory_stats for peak memory, and OOM caught + recorded (never crashes the sweep).
Each cell runs in a SUBPROCESS so a hard CUDA fault can't poison later cells; the driver runs
ascending L and skips larger L for an arch once it OOMs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DEFAULT_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
MODES = ["train", "prefill", "decode"]


def _load_full_cfg(arch: str, config_dir: str) -> dict:
    import yaml

    with open(os.path.join(config_dir, f"{arch}.yaml")) as f:
        doc = yaml.safe_load(f)
    return dict(doc["model"])


def _time_region(fn, n_warmup: int = 5, n_trials: int = 10) -> Dict[str, float]:
    import numpy as np
    import torch

    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))  # ms
    return {"ms_med": float(np.median(ts)),
            "ms_p10": float(np.percentile(ts, 10)),
            "ms_p90": float(np.percentile(ts, 90))}


def run_cell(arch: str, L: int, mode: str, batch: int, config_dir: str,
             gen: int = 128) -> Dict:
    import torch

    from ssm_bench.models.registry import build_model

    result = {"arch": arch, "L": L, "mode": mode, "batch": batch, "oom": False}
    if not torch.cuda.is_available():
        result.update({"oom": False, "error": "no_cuda"})
        return result
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = _load_full_cfg(arch, config_dir)
        cfg["max_position_embeddings"] = max(cfg.get("max_position_embeddings", 1024), L + gen + 1)
        vocab = cfg.get("vocab_size", 50304)
        model = build_model(arch, cfg).cuda()
        x = torch.randint(0, vocab, (batch, L), device="cuda")

        if mode == "train":
            model.train()

            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=x, labels=x)
                out.loss.backward()
                model.zero_grad(set_to_none=True)

            timing = _time_region(step)
            result["tok_per_sec"] = batch * L / (timing["ms_med"] / 1e3)

        elif mode == "prefill":
            model.eval()

            def fwd():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=x)

            timing = _time_region(fwd)
            result["tok_per_sec"] = batch * L / (timing["ms_med"] / 1e3)

        elif mode == "decode":
            model.eval()
            gen_kw = dict(max_new_tokens=gen, min_new_tokens=gen, do_sample=False,
                          num_beams=1, use_cache=True, pad_token_id=0)

            def prefill_only():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(input_ids=x)

            def generate():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.generate(x, **gen_kw)

            t_pre = _time_region(prefill_only, n_warmup=2, n_trials=5)["ms_med"]
            t_gen = _time_region(generate, n_warmup=2, n_trials=5)["ms_med"]
            per_tok_ms = max(1e-6, (t_gen - t_pre) / gen)
            timing = {"ms_med": per_tok_ms, "prefill_ms": t_pre, "generate_ms": t_gen}
            result["decode_tok_per_sec"] = batch / (per_tok_ms / 1e3)
        else:
            raise KeyError(mode)

        result.update(timing)
        result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        del model, x
        torch.cuda.empty_cache()
        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:  # type: ignore[attr-defined]
        msg = str(ex).lower()
        if isinstance(ex, RuntimeError) and "out of memory" not in msg:
            result.update({"oom": False, "error": str(ex)[:200]})
        else:
            result["oom"] = True
        torch.cuda.empty_cache()
        return result


def _run_subprocess(arch: str, L: int, mode: str, batch: int, config_dir: str) -> Dict:
    """Run one cell in a fresh process so a fatal CUDA fault is contained."""
    cmd = [sys.executable, os.path.abspath(__file__), "--single",
           "--arch", arch, "--L", str(L), "--mode", mode,
           "--batch", str(batch), "--config_dir", config_dir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"arch": arch, "L": L, "mode": mode, "batch": batch, "oom": False,
                "error": f"no_json (rc={proc.returncode}): {proc.stderr[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"arch": arch, "L": L, "mode": mode, "batch": batch, "error": "timeout"}


def driver(archs: List[str], lens: List[int], modes: List[str], batch: int,
           config_dir: str, out: str, isolate: bool) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from ssm_bench.utils.logging import append_jsonl, read_jsonl, run_id

    done = {r["run_id"] for r in read_jsonl(out) if "run_id" in r}
    for arch in archs:
        for mode in modes:
            oomed = False
            for L in sorted(lens):
                rid = run_id(arch=arch, L=L, mode=mode, batch=batch)
                if rid in done:
                    continue
                if oomed:
                    rec = {"run_id": rid, "arch": arch, "L": L, "mode": mode,
                           "batch": batch, "oom": True, "skipped": True}
                    append_jsonl(out, rec)
                    continue
                res = (_run_subprocess(arch, L, mode, batch, config_dir) if isolate
                       else run_cell(arch, L, mode, batch, config_dir))
                res["run_id"] = rid
                append_jsonl(out, res)
                print(f"[eff] {arch} {mode} L={L}: "
                      + ("OOM" if res.get("oom") else f"{res.get('ms_med', float('nan')):.2f}ms "
                         f"peak={res.get('peak_gb', 0):.2f}GB"))
                if res.get("oom"):
                    oomed = True  # larger L will also OOM


def main() -> None:
    ap = argparse.ArgumentParser(description="Efficiency benchmark (throughput/memory/latency).")
    ap.add_argument("--single", action="store_true", help="run ONE cell and print JSON (subprocess worker)")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--archs", default="transformer,mamba,jamba")
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--lens", default=",".join(str(x) for x in DEFAULT_LENS))
    ap.add_argument("--mode", default="train", choices=MODES)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--config_dir", default="configs")
    ap.add_argument("--out", default="results/summaries/efficiency.jsonl")
    ap.add_argument("--no_isolate", action="store_true", help="run cells in-process (no subprocess)")
    args = ap.parse_args()

    if args.single:
        res = run_cell(args.arch, args.L, args.mode, args.batch, args.config_dir)
        print(json.dumps(res))
        return

    archs = [a for a in args.archs.split(",") if a]
    lens = [int(x) for x in args.lens.split(",") if x]
    modes = [m for m in args.modes.split(",") if m]
    driver(archs, lens, modes, args.batch, args.config_dir, args.out,
           isolate=not args.no_isolate)


if __name__ == "__main__":
    main()
