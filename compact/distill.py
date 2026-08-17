#!/usr/bin/env python3
"""Training objectives for K3-Compact specialist distillation.

The goal is deliberately *not* pure imitation. Pure KL distillation can at best make the
student approximate the teacher. K3-Compact therefore combines retention losses with a
verified-improvement objective: on training examples where an executable verifier marks
one candidate as better than another, the student's preference margin is trained to beat
the K3 Max teacher's margin by a configurable amount.

This module contains no benchmark data and no cyber targets. Verifier labels are expected
to come from benchmark-clean training tasks: unit tests/compilers, tool task checks,
isolated CTF labs, defensive patch/detection tests, or other explicitly authorized labs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillWeights:
    logits: float = 1.0
    hidden: float = 0.25
    router: float = 0.20
    verified_improvement: float = 0.60


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    m = mask.to(dtype=values.dtype)
    while m.ndim < values.ndim:
        m = m.unsqueeze(-1)
    m = torch.broadcast_to(m, values.shape)
    denom = m.sum().clamp_min(1.0)
    return (values * m).sum() / denom


def logit_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Forward KL(teacher || student), scaled by T^2 as in standard distillation."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    teacher_p = F.softmax(teacher_logits.float() / t, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / t, dim=-1)
    student_logp = F.log_softmax(student_logits.float() / t, dim=-1)
    per_token = torch.sum(teacher_p * (teacher_logp - student_logp), dim=-1)
    return _masked_mean(per_token, mask) * (t * t)


def hidden_alignment_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Scale-insensitive hidden-state retention using 1-cosine similarity."""
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("student/teacher hidden states must have the same shape")
    s = F.normalize(student_hidden.float(), dim=-1, eps=1e-8)
    t = F.normalize(teacher_hidden.float(), dim=-1, eps=1e-8)
    per_token = 1.0 - torch.sum(s * t, dim=-1)
    return _masked_mean(per_token, mask)


def aggregate_teacher_router_probs(
    teacher_probs: Tensor,
    teacher_to_student: Tensor,
    n_student_experts: int,
) -> Tensor:
    """Aggregate 896-way teacher routing into the student's clustered expert space.

    `teacher_to_student[e]` is produced by `expert_cluster.py`. Probabilities belonging to
    all teacher experts in a cluster are summed; no renormalization trick or top-k deletion
    is used, so probability mass is conserved.
    """
    if teacher_probs.ndim < 1:
        raise ValueError("teacher_probs must have an expert dimension")
    n_teacher = teacher_probs.shape[-1]
    mapping = teacher_to_student.to(device=teacher_probs.device, dtype=torch.long)
    if mapping.shape != (n_teacher,):
        raise ValueError("teacher_to_student shape must equal teacher expert count")
    if n_student_experts <= 0:
        raise ValueError("n_student_experts must be positive")
    if torch.any(mapping < 0) or torch.any(mapping >= n_student_experts):
        raise ValueError("teacher_to_student contains an out-of-range cluster id")

    out_shape = (*teacher_probs.shape[:-1], n_student_experts)
    out = torch.zeros(out_shape, dtype=teacher_probs.dtype, device=teacher_probs.device)
    index = mapping.view(*([1] * (teacher_probs.ndim - 1)), n_teacher)
    index = index.expand_as(teacher_probs)
    out.scatter_add_(-1, index, teacher_probs)
    return out


def router_distill_loss(
    student_router_logits: Tensor,
    teacher_router_logits: Tensor,
    teacher_to_student: Tensor,
    *,
    mask: Tensor | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Distill the teacher's full router distribution after 896->48 cluster aggregation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n_student = student_router_logits.shape[-1]
    teacher_p = F.softmax(teacher_router_logits.float() / temperature, dim=-1)
    target = aggregate_teacher_router_probs(teacher_p, teacher_to_student, n_student)
    target = target.clamp_min(torch.finfo(target.dtype).tiny)
    student_logp = F.log_softmax(student_router_logits.float() / temperature, dim=-1)
    per_token = torch.sum(target * (torch.log(target) - student_logp), dim=-1)
    return _masked_mean(per_token, mask) * (temperature * temperature)


def verified_improvement_loss(
    student_chosen_logp: Tensor,
    student_rejected_logp: Tensor,
    teacher_chosen_logp: Tensor,
    teacher_rejected_logp: Tensor,
    *,
    margin_over_teacher: float = 0.25,
    beta: float = 1.0,
    weight: Tensor | None = None,
) -> Tensor:
    """Train the student to exceed the teacher's preference margin on verified pairs.

    For a verifier-confirmed pair (chosen passes, rejected fails), define each model's
    sequence-logprob preference margin. The target is:

        student_margin >= teacher_margin + margin_over_teacher

    A softplus hinge keeps gradients smooth. This is what gives the training objective a
    mechanism to become *better* than the teacher instead of merely copying it.
    """
    tensors = (
        student_chosen_logp,
        student_rejected_logp,
        teacher_chosen_logp,
        teacher_rejected_logp,
    )
    shape = tensors[0].shape
    if any(x.shape != shape for x in tensors[1:]):
        raise ValueError("all chosen/rejected logprob tensors must share a shape")
    if beta <= 0:
        raise ValueError("beta must be positive")
    student_margin = student_chosen_logp.float() - student_rejected_logp.float()
    teacher_margin = teacher_chosen_logp.float() - teacher_rejected_logp.float()
    gap = teacher_margin + float(margin_over_teacher) - student_margin
    loss = F.softplus(float(beta) * gap) / float(beta)
    if weight is not None:
        w = weight.to(device=loss.device, dtype=loss.dtype)
        if w.shape != loss.shape:
            raise ValueError("verified-pair weight shape mismatch")
        denom = w.sum().clamp_min(1e-8)
        return (loss * w).sum() / denom
    return loss.mean()


def specialist_distill_loss(
    *,
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    student_router_logits: Tensor,
    teacher_router_logits: Tensor,
    teacher_to_student: Tensor,
    token_mask: Tensor | None,
    student_chosen_logp: Tensor,
    student_rejected_logp: Tensor,
    teacher_chosen_logp: Tensor,
    teacher_rejected_logp: Tensor,
    pair_weight: Tensor | None = None,
    weights: DistillWeights = DistillWeights(),
    logit_temperature: float = 2.0,
    router_temperature: float = 1.0,
    margin_over_teacher: float = 0.25,
) -> dict[str, Tensor]:
    """Return named losses plus their weighted total for logging/training."""
    l_logits = logit_kl_loss(
        student_logits, teacher_logits, mask=token_mask, temperature=logit_temperature
    )
    l_hidden = hidden_alignment_loss(student_hidden, teacher_hidden, mask=token_mask)
    l_router = router_distill_loss(
        student_router_logits,
        teacher_router_logits,
        teacher_to_student,
        mask=token_mask,
        temperature=router_temperature,
    )
    l_verified = verified_improvement_loss(
        student_chosen_logp,
        student_rejected_logp,
        teacher_chosen_logp,
        teacher_rejected_logp,
        margin_over_teacher=margin_over_teacher,
        weight=pair_weight,
    )
    total = (
        weights.logits * l_logits
        + weights.hidden * l_hidden
        + weights.router * l_router
        + weights.verified_improvement * l_verified
    )
    return {
        "total": total,
        "logits": l_logits,
        "hidden": l_hidden,
        "router": l_router,
        "verified_improvement": l_verified,
    }
