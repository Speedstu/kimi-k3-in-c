#!/usr/bin/env python3
"""Differentiable Q3 fake quantization for K3-Compact training.

The forward pass matches the deployed storage contract closely:
- signed Q3 codes clipped to [-3, 3];
- one positive scale per row/group of 128;
- scales rounded to BF16 before they are used.

Rounding uses straight-through estimators. This module is intentionally small enough to
unit-test on CPU; a distributed trainer can wrap real K3-Compact linear weights with the
same primitive without changing the checkpoint format.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from compact.q3 import GROUP, QMAX, QMIN


def _pad_groups(weight: Tensor, group_size: int) -> tuple[Tensor, int]:
    if weight.ndim != 2:
        raise ValueError("Q3 QAT expects a 2D [out,in] weight")
    cols = int(weight.shape[1])
    pad = (-cols) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))
    return weight.view(weight.shape[0], -1, group_size), pad


def initial_group_scales(weight: Tensor, group_size: int = GROUP) -> Tensor:
    """Deployment-oriented max-abs/3 initialization, [out, groups]."""
    grouped, _ = _pad_groups(weight.detach().float(), group_size)
    amax = grouped.abs().amax(dim=-1)
    floor = torch.finfo(torch.float32).tiny
    return (amax / float(QMAX)).clamp_min(floor)


def _round_ste(x: Tensor) -> Tensor:
    return x + (torch.round(x) - x).detach()


def _bf16_ste(x: Tensor) -> Tensor:
    bf = x.to(torch.bfloat16).to(torch.float32)
    return x + (bf - x).detach()


def fake_quant_q3(weight: Tensor, scales: Tensor, group_size: int = GROUP) -> Tensor:
    """Fake-quantize a 2D weight with learned positive scales.

    `scales` is [out, ceil(in/group_size)]. Returned tensor has the original shape.
    Gradients flow to weight and scales through STE rounding; values beyond the signed
    Q3 range saturate, as they will in the exported checkpoint.
    """
    grouped, pad = _pad_groups(weight.float(), group_size)
    expected = grouped.shape[:2]
    if tuple(scales.shape) != tuple(expected):
        raise ValueError(f"scales shape {tuple(scales.shape)} != expected {tuple(expected)}")
    if torch.any(scales <= 0):
        raise ValueError("Q3 scales must be positive")

    scale = _bf16_ste(scales.float()).unsqueeze(-1)
    normalized = grouped / scale
    codes = _round_ste(normalized).clamp(float(QMIN), float(QMAX))
    dequant = codes * scale
    flat = dequant.reshape(weight.shape[0], -1)
    if pad:
        flat = flat[:, :-pad]
    return flat.to(weight.dtype)


class Q3WeightQAT(nn.Module):
    """Learnable group scales for one matrix; the matrix itself remains caller-owned."""

    def __init__(self, reference_weight: Tensor, group_size: int = GROUP) -> None:
        super().__init__()
        self.group_size = int(group_size)
        init = initial_group_scales(reference_weight, self.group_size)
        # softplus(log_scale_raw) guarantees positivity throughout optimization.
        raw = torch.log(torch.expm1(init.clamp_min(1e-12)))
        raw = torch.where(torch.isfinite(raw), raw, torch.full_like(raw, -20.0))
        self.log_scale_raw = nn.Parameter(raw)

    def scales(self) -> Tensor:
        return F.softplus(self.log_scale_raw) + 1e-12

    def forward(self, weight: Tensor) -> Tensor:
        return fake_quant_q3(weight, self.scales(), self.group_size)

    @property
    def groups(self) -> int:
        return int(self.log_scale_raw.numel())

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}, groups={self.groups}, bits=3, scale=bf16"


def physical_bits_per_weight(group_size: int = GROUP) -> float:
    """Asymptotic full-group density: 3 code bits + 16 scale bits/group."""
    return 3.0 + 16.0 / float(group_size)


def estimated_matrix_bytes(rows: int, cols: int, group_size: int = GROUP) -> int:
    """Exact packed bytes for row-local groups, including BF16 scales and tails."""
    if rows < 0 or cols < 0:
        raise ValueError("negative matrix shape")
    groups = math.ceil(cols / group_size) if cols else 0
    # 3-bit codes are packed per group, so a partial group's final byte is local.
    full, rem = divmod(cols, group_size)
    code_bytes = full * ((3 * group_size + 7) // 8)
    if rem:
        code_bytes += (3 * rem + 7) // 8
    return rows * (code_bytes + groups * 2)
