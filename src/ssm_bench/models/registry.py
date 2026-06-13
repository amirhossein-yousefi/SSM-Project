"""Model factory: build_model(arch, cfg) -> nn.Module with a uniform forward.

All three architectures are returned as HuggingFace causal-LM models that share the
same call contract:

    out = model(input_ids=..., labels=...)
    out.loss      # mean cross-entropy (model shifts labels internally)
    out.logits    # [B, T, V]

so the trainer and eval code are identical across architectures — switching arms is
a single config line.

Tiering for the SSM path (Mamba / Jamba mixers):
  Tier 1  fast CUDA kernels (mamba-ssm + causal-conv1d) -> used if importable
  Tier 2  HuggingFace's built-in torch fallback         -> automatic when kernels absent
  Tier 3  our own pure-PyTorch Mamba block (mamba_torch) -> only if cfg.force_custom or
          the HF path fails a smoke forward (pure-SSM arm only)
"""
from __future__ import annotations

from typing import Any, Dict

ARCHS = ("transformer", "mamba", "jamba")


def kernels_available() -> bool:
    """True iff the fast Mamba CUDA kernels can be imported."""
    try:
        import causal_conv1d  # noqa: F401
        import mamba_ssm  # noqa: F401

        return True
    except Exception:
        return False


def build_model(arch: str, cfg: Dict[str, Any]):
    """Build one of {transformer, mamba, jamba} from a plain config dict.

    `cfg` is the `model:` block from configs/<arch>.yaml. Extra control keys
    (consumed here, not passed to HF): force_custom, force_torch.
    """
    if arch not in ARCHS:
        raise KeyError(f"unknown arch {arch!r}; expected one of {ARCHS}")
    cfg = dict(cfg)
    force_custom = cfg.pop("force_custom", False)
    force_torch = cfg.pop("force_torch", False)

    if arch == "transformer":
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaForCausalLM(LlamaConfig(**cfg))

    if arch == "mamba":
        if force_custom or not kernels_available():
            model = _build_mamba_hf(cfg)
            if model is not None and _smoke_ok(model):
                return model
            # Tier 3: our own block
            from .mamba_torch import MambaTorchForCausalLM

            return MambaTorchForCausalLM(**_to_torch_cfg(cfg))
        return _build_mamba_hf(cfg, require=True)

    if arch == "jamba":
        from transformers import JambaConfig, JambaForCausalLM

        c = dict(cfg)
        c["use_mamba_kernels"] = kernels_available() and not force_torch
        return JambaForCausalLM(JambaConfig(**c))


def _build_mamba_hf(cfg: Dict[str, Any], require: bool = False):
    """Instantiate HF Mamba2ForCausalLM; return None on failure unless `require`."""
    try:
        from transformers import Mamba2Config, Mamba2ForCausalLM

        return Mamba2ForCausalLM(Mamba2Config(**_mamba2_cfg(cfg)))
    except Exception as e:
        if require:
            raise
        print(f"[registry] HF Mamba2 unavailable ({e}); falling back to custom block.")
        return None


def _mamba2_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys Mamba2Config understands (drop transformer-only keys)."""
    allowed = {
        "vocab_size", "hidden_size", "state_size", "num_hidden_layers",
        "expand", "conv_kernel", "n_groups", "chunk_size", "num_heads",
        "head_dim", "tie_word_embeddings", "use_cache", "pad_token_id",
        "bos_token_id", "eos_token_id", "rms_norm", "residual_in_fp32",
        "time_step_rank", "use_conv_bias", "hidden_act", "initializer_range",
        "layer_norm_epsilon",
    }
    return {k: v for k, v in cfg.items() if k in allowed}


def _to_torch_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map shared config keys onto our pure-PyTorch Mamba block's signature."""
    return dict(
        vocab_size=cfg.get("vocab_size", 50304),
        d_model=cfg.get("hidden_size", 768),
        n_layer=cfg.get("num_hidden_layers", 24),
        d_state=cfg.get("state_size", 16),
        d_conv=cfg.get("conv_kernel", 4),
        expand=cfg.get("expand", 2),
        tie_embeddings=cfg.get("tie_word_embeddings", True),
    )


def _smoke_ok(model) -> bool:
    """Run a tiny CPU forward to catch HF-version-specific errors on the slow path."""
    try:
        import torch

        ids = torch.zeros((1, 8), dtype=torch.long)
        with torch.no_grad():
            model(input_ids=ids)
        return True
    except Exception as e:
        print(f"[registry] HF Mamba2 smoke forward failed ({e}); using custom block.")
        return False
