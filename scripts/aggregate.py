#!/usr/bin/env python
"""Collate raw run logs + eval JSONL into plot-ready CSV summaries.

Outputs (results/summaries/):
  lm_quality.csv      one row per LM run: final val loss/ppl/bpt + strided-window ppl + params
  training_curves.csv per-eval val curve for each run (for the ppl-vs-tokens figure)
  synthetic.csv       one row per synthetic task cell
  efficiency.csv      one row per efficiency cell
"""
from __future__ import annotations

import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd  # noqa: E402

from ssm_bench.utils.logging import read_jsonl  # noqa: E402

RUNS = os.path.join(_ROOT, "runs")
SUMM = os.path.join(_ROOT, "results", "summaries")


def _run_meta(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return {"run": os.path.basename(run_dir), "arch": cfg.get("arch"),
            "seed": cfg.get("seed"), "n_params_total": cfg.get("n_params_total"),
            "n_params_active": cfg.get("n_params_active")}


def aggregate_lm() -> None:
    rows, curves = [], []
    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.isdir(run_dir):
            continue
        recs = read_jsonl(os.path.join(run_dir, "log.jsonl"))
        meta = _run_meta(run_dir)
        vals = [r for r in recs if "val_loss" in r]
        for r in vals:
            curves.append({**meta, "step": r["step"], "val_loss": r["val_loss"],
                           "val_ppl": r.get("val_ppl"),
                           "val_bits_per_token": r.get("val_bits_per_token")})
        if vals:
            last = vals[-1]
            best = min(vals, key=lambda r: r["val_loss"])
            rows.append({**meta, "final_val_loss": last["val_loss"],
                         "final_val_ppl": last.get("val_ppl"),
                         "best_val_loss": best["val_loss"],
                         "best_val_bits_per_token": best.get("val_bits_per_token")})

    # merge strided-window ppl produced by eval/lm_quality.py, keyed by arch
    strided = {r["arch"]: r for r in read_jsonl(os.path.join(SUMM, "lm_quality.jsonl"))}
    for row in rows:
        s = strided.get(row["arch"])
        if s:
            row["strided_ppl"] = s.get("ppl")
            row["strided_bits_per_token"] = s.get("bits_per_token")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(SUMM, "lm_quality.csv"), index=False)
        print(f"  lm_quality.csv: {len(rows)} runs")
    if curves:
        pd.DataFrame(curves).to_csv(os.path.join(SUMM, "training_curves.csv"), index=False)
        print(f"  training_curves.csv: {len(curves)} points")


def aggregate_jsonl(name: str) -> None:
    recs = read_jsonl(os.path.join(SUMM, f"{name}.jsonl"))
    if not recs:
        print(f"  {name}: no records")
        return
    pd.DataFrame(recs).to_csv(os.path.join(SUMM, f"{name}.csv"), index=False)
    print(f"  {name}.csv: {len(recs)} rows")


def main() -> None:
    os.makedirs(SUMM, exist_ok=True)
    print("Aggregating ->", SUMM)
    aggregate_lm()
    aggregate_jsonl("synthetic")
    aggregate_jsonl("efficiency")
    print("Done.")


if __name__ == "__main__":
    main()
