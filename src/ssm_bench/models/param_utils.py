"""Parameter counting + fairness check across the three architectures.

The closed-form math for Mamba2/Jamba param counts is unreliable, so this module is
the source of truth: build all three models and assert they land within a tolerance of
each other. Run `python -m ssm_bench.models.param_utils --check` on Colab (needs torch +
transformers) to finalize the per-arch layer counts before freezing the configs.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict


def count_params(model, only_trainable: bool = True) -> int:
    """Total parameter count (this is what optimizer state + VRAM scale with)."""
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not only_trainable)
    )


def moe_param_count(model) -> int:
    """Params living inside MoE experts (named with 'experts')."""
    return sum(p.numel() for n, p in model.named_parameters() if "experts" in n)


def active_params(model, num_experts: int = 1, num_experts_per_tok: int = 1) -> int:
    """FLOP-relevant params: total minus the experts not routed on an average token."""
    total = count_params(model)
    if num_experts <= 1:
        return total
    moe = moe_param_count(model)
    inactive_frac = (num_experts - num_experts_per_tok) / num_experts
    return int(total - inactive_frac * moe)


def assert_matched(counts: Dict[str, int], tol: float = 0.05, ref: str = "transformer") -> None:
    """Assert every model's TOTAL params is within `tol` of the reference arch."""
    ref_n = counts[ref]
    bad = []
    for name, n in counts.items():
        rel = abs(n - ref_n) / ref_n
        if rel > tol:
            bad.append(f"{name}={n/1e6:.1f}M ({rel:+.1%} vs {ref} {ref_n/1e6:.1f}M)")
    if bad:
        raise AssertionError("param mismatch > %.0f%%: %s" % (tol * 100, "; ".join(bad)))


# ------------------------------------------------------------------ CLI --------

def _load_cfgs(config_dir: str):
    import yaml

    cfgs = {}
    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        with open(path) as f:
            doc = yaml.safe_load(f)
        if not doc or "arch" not in doc:
            continue
        if doc["arch"] in ("transformer", "mamba", "jamba"):
            cfgs[doc["arch"]] = doc
    return cfgs


def _check(config_dir: str, tol: float) -> int:
    from .registry import build_model

    cfgs = _load_cfgs(config_dir)
    missing = {"transformer", "mamba", "jamba"} - set(cfgs)
    if missing:
        print(f"[check] missing configs for: {sorted(missing)} in {config_dir}")
        return 2

    counts, actives = {}, {}
    for arch in ("transformer", "mamba", "jamba"):
        doc = cfgs[arch]
        model = build_model(arch, dict(doc["model"]))
        counts[arch] = count_params(model)
        m = doc["model"]
        actives[arch] = active_params(
            model,
            num_experts=int(m.get("num_experts", 1)),
            num_experts_per_tok=int(m.get("num_experts_per_tok", 1)),
        )
        del model

    print(f"{'arch':<12}{'total (M)':>12}{'active (M)':>12}")
    for arch in ("transformer", "mamba", "jamba"):
        print(f"{arch:<12}{counts[arch]/1e6:>12.2f}{actives[arch]/1e6:>12.2f}")

    try:
        assert_matched(counts, tol=tol)
        print(f"\nOK: all archs within {tol:.0%} of transformer on TOTAL params.")
        return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        print("Adjust num_hidden_layers (mamba) / intermediate_size & experts (jamba) "
              "in configs/, then re-run --check.")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Param counting / fairness check.")
    ap.add_argument("--check", action="store_true", help="build all 3 models and verify matched params")
    ap.add_argument("--config_dir", default="configs")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()
    if args.check:
        raise SystemExit(_check(args.config_dir, args.tol))
    ap.print_help()


if __name__ == "__main__":
    main()
