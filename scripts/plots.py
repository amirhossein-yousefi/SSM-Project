#!/usr/bin/env python
"""Render figures from results/summaries/*.csv into results/figures/*.png.

Figures (skipped gracefully if the underlying CSV is missing/empty):
  fig_ppl_curve.png    val bits-per-token vs tokens seen, per arch
  fig_mqar.png         MQAR accuracy vs sequence length (lines per arch, by d_model)
  fig_induction.png    induction accuracy vs eval length (length extrapolation)
  fig_selcopy.png      selective-copy accuracy vs sequence length
  fig_latency.png      decode per-token latency + prefill latency vs L (log-log)
  fig_memory.png       peak GPU memory vs L, with OOM markers
  fig_throughput.png   train throughput (tok/s) vs L
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM = os.path.join(_ROOT, "results", "summaries")
FIGS = os.path.join(_ROOT, "results", "figures")
COLORS = {"transformer": "#d62728", "mamba": "#1f77b4", "jamba": "#2ca02c"}


def _load(name: str):
    path = os.path.join(SUMM, f"{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _oom_mask(df):
    """Boolean Series: True where the cell OOM'd (handles a missing 'oom' column)."""
    if "oom" in df.columns:
        return df["oom"].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _save(fig, name: str) -> None:
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_ppl_curve() -> None:
    df = _load("training_curves")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in df.groupby("arch"):
        g = g.sort_values("step")
        y = g["val_bits_per_token"] if "val_bits_per_token" in g else g["val_loss"]
        ax.plot(g["step"], y, label=arch, color=COLORS.get(arch))
    ax.set_xlabel("training step")
    ax.set_ylabel("val bits / token")
    ax.set_title("LM quality vs training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_ppl_curve.png")


def plot_synthetic(task: str, xcol: str, fname: str, title: str) -> None:
    df = _load("synthetic")
    if df is None:
        return
    d = df[df["task"] == task]
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in d.groupby("arch"):
        if "d_model" in g and g["d_model"].nunique() > 1 and task == "mqar":
            for dm, gg in g.groupby("d_model"):
                gg = gg.sort_values(xcol)
                ax.plot(gg[xcol], gg["test_acc"], marker="o",
                        label=f"{arch} d={dm}", color=COLORS.get(arch),
                        alpha=0.4 + 0.6 * (dm / g["d_model"].max()))
        else:
            g = g.sort_values(xcol)
            ax.plot(g[xcol], g["test_acc"], marker="o", label=arch, color=COLORS.get(arch))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xcol)
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, fname)


def plot_latency() -> None:
    df = _load("efficiency")
    if df is None:
        return
    ok = ~_oom_mask(df)
    dec = df[(df["mode"] == "decode") & ok]
    pre = df[(df["mode"] == "prefill") & ok]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, d, title, ycol in [
        (axes[0], dec, "decode latency / token", "ms_med"),
        (axes[1], pre, "prefill latency", "ms_med"),
    ]:
        if len(d):
            for arch, g in d.groupby("arch"):
                g = g.sort_values("L")
                ax.plot(g["L"], g[ycol], marker="o", label=arch, color=COLORS.get(arch))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("ms")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_latency.png")


def plot_memory() -> None:
    df = _load("efficiency")
    if df is None:
        return
    d = df[df["mode"] == "train"]
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in d.groupby("arch"):
        g = g.sort_values("L")
        gmask = _oom_mask(g)
        ok = g[~gmask]
        ax.plot(ok["L"], ok["peak_gb"], marker="o", label=arch, color=COLORS.get(arch))
        oom = g[gmask]
        for _, r in oom.iterrows():
            ax.scatter(r["L"], ax.get_ylim()[1] * 0.95, marker="x", s=60,
                       color=COLORS.get(arch))
    ax.axhline(40, ls="--", color="gray", alpha=0.6, label="A100 40GB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title("Memory vs sequence length (x = OOM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_memory.png")


def plot_throughput() -> None:
    df = _load("efficiency")
    if df is None:
        return
    d = df[(df["mode"] == "train") & (~_oom_mask(df))]
    if not len(d) or "tok_per_sec" not in d:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, g in d.groupby("arch"):
        g = g.sort_values("L")
        ax.plot(g["L"], g["tok_per_sec"], marker="o", label=arch, color=COLORS.get(arch))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("train throughput (tok/s)")
    ax.set_title("Training throughput vs sequence length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_throughput.png")


def main() -> None:
    print("Rendering figures ->", FIGS)
    plot_ppl_curve()
    plot_synthetic("mqar", "seq_len", "fig_mqar.png", "MQAR: recall vs sequence length")
    plot_synthetic("induction", "eval_len", "fig_induction.png",
                   "Induction: length extrapolation")
    plot_synthetic("selective_copy", "seq_len", "fig_selcopy.png",
                   "Selective copy: accuracy vs length")
    plot_latency()
    plot_memory()
    plot_throughput()
    print("Done.")


if __name__ == "__main__":
    main()
