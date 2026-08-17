#!/usr/bin/env python3
"""Storage-efficient K3 Max teacher traces for K3-Compact distillation.

Saving a full vocabulary distribution for every teacher token is prohibitively large.
Instead, preserve the teacher's top-K probabilities individually plus ONE exact tail-mass
bucket. The student is scored on the same K tokens plus its probability mass over every
other vocabulary token. This is an exact KL on a coarsened (K+1)-class distribution:
probability mass is conserved and the loss is zero when teacher/student distributions
match, while trace size is O(K) rather than O(vocab).

The trace is a training artifact only. It never changes K3 Max or the deployed compact
checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TopKTeacherTrace:
    indices: Tensor       # [..., K] int64
    log_probs: Tensor     # [..., K] log teacher probabilities
    log_tail_mass: Tensor # [...] log sum of all non-topK teacher probabilities

    @property
    def k(self) -> int:
        return int(self.indices.shape[-1])


def make_topk_trace(teacher_logits: Tensor, k: int) -> TopKTeacherTrace:
    """Create a probability-mass-preserving top-K + tail trace from teacher logits."""
    if teacher_logits.ndim < 1:
        raise ValueError("teacher logits need a vocabulary dimension")
    vocab = int(teacher_logits.shape[-1])
    if not 1 <= k < vocab:
        raise ValueError("k must satisfy 1 <= k < vocab")

    logp = F.log_softmax(teacher_logits.float(), dim=-1)
    top_logp, indices = torch.topk(logp, k=k, dim=-1, largest=True, sorted=True)
    top_mass = torch.exp(top_logp).sum(dim=-1)
    # Clamp only against tiny negative roundoff; a valid k<vocab always leaves tail mass.
    tail_mass = (1.0 - top_mass).clamp_min(torch.finfo(logp.dtype).tiny)
    return TopKTeacherTrace(
        indices=indices.to(torch.int64),
        log_probs=top_logp,
        log_tail_mass=torch.log(tail_mass),
    )


def _student_bucket_log_probs(student_logits: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    if student_logits.shape[:-1] != indices.shape[:-1]:
        raise ValueError("student logits and teacher indices leading shapes differ")
    vocab = int(student_logits.shape[-1])
    if indices.ndim != student_logits.ndim:
        raise ValueError("teacher indices must have one top-K dimension")
    if torch.any(indices < 0) or torch.any(indices >= vocab):
        raise ValueError("teacher trace contains vocabulary index out of range")

    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    selected = torch.gather(student_logp, -1, indices.to(student_logits.device))

    # Compute the tail probability robustly in probability space. Since top-K is usually
    # small (e.g. 128), gathering K entries is cheap and avoids materializing a V-wide mask.
    selected_mass = torch.exp(selected).sum(dim=-1)
    tail_mass = (1.0 - selected_mass).clamp_min(torch.finfo(student_logp.dtype).tiny)
    return selected, torch.log(tail_mass)


def bucketed_topk_kl_loss(
    student_logits: Tensor,
    trace: TopKTeacherTrace,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """KL(teacher_bucketed || student_bucketed) over K explicit tokens + exact tail mass."""
    if trace.log_probs.shape != trace.indices.shape:
        raise ValueError("teacher top-K indices/log_probs shape mismatch")
    if trace.log_tail_mass.shape != trace.indices.shape[:-1]:
        raise ValueError("teacher tail-mass shape mismatch")

    s_top, s_tail = _student_bucket_log_probs(student_logits, trace.indices)
    t_top = trace.log_probs.to(device=student_logits.device, dtype=torch.float32)
    t_tail = trace.log_tail_mass.to(device=student_logits.device, dtype=torch.float32)

    top_p = torch.exp(t_top)
    tail_p = torch.exp(t_tail)
    per_token = torch.sum(top_p * (t_top - s_top), dim=-1) + tail_p * (t_tail - s_tail)

    if mask is None:
        return per_token.mean()
    if mask.shape != per_token.shape:
        raise ValueError("mask shape must match token-loss shape")
    m = mask.to(device=per_token.device, dtype=per_token.dtype)
    return (per_token * m).sum() / m.sum().clamp_min(1.0)


def trace_physical_bytes_per_token(k: int, *, index_bytes: int = 4, prob_bytes: int = 2) -> int:
    """Intended serialized footprint: K ids + K logprobs + one tail logprob."""
    if k <= 0:
        raise ValueError("k must be positive")
    if index_bytes <= 0 or prob_bytes <= 0:
        raise ValueError("byte widths must be positive")
    return k * (index_bytes + prob_bytes) + prob_bytes
