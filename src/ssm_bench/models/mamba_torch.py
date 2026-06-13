"""Pure-PyTorch Mamba-1 reference block (3rd-tier fallback / owned SSM code).

Correctness-first, not speed-first: the selective scan is a sequential loop over time,
so it is slow at long sequence lengths but runs on any device with no CUDA kernel build.
The registry only returns this when the fast kernels AND HuggingFace's own torch path are
both unavailable. It also serves as a readable reference for the SSM recurrence.

Interface mirrors HF causal-LM models:
    out = model(input_ids=..., labels=...)
    out.loss, out.logits
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaMixer(nn.Module):
    """Selective SSM mixer (Mamba-1), depthwise conv + input-dependent A/B/C/dt."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterized in log-space; D is the skip connection.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, d_model]
        b, l, _ = x.shape
        xz = self.in_proj(x)                       # [B, L, 2*d_inner]
        xin, z = xz.chunk(2, dim=-1)               # each [B, L, d_inner]

        # depthwise causal conv
        xin = xin.transpose(1, 2)                  # [B, d_inner, L]
        xin = self.conv1d(xin)[..., :l]
        xin = xin.transpose(1, 2)                  # [B, L, d_inner]
        xin = F.silu(xin)

        y = self._ssm(xin)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d_in = x.shape
        A = -torch.exp(self.A_log.float())                         # [d_in, d_state]
        x_dbl = self.x_proj(x)                                     # [B, L, dt_rank+2N]
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                          # [B, L, d_in]

        # discretize
        dA = torch.exp(dt.unsqueeze(-1) * A)                       # [B, L, d_in, N]
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [B, L, d_in, N]

        h = torch.zeros(b, d_in, self.d_state, device=x.device, dtype=dA.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB_x[:, t]                          # [B, d_in, N]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))      # [B, d_in]
        y = torch.stack(ys, dim=1)                                 # [B, L, d_in]
        return y + x * self.D


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, **mixer_kw):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, **mixer_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaTorchForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layer: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            MambaResidualBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layer)
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                **kwargs) -> CausalLMOutput:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)
