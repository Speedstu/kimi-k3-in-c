#!/usr/bin/env python3
"""Tiny end-to-end optimizer smoke for the K3-Compact training objective.

This is NOT a model-quality benchmark. It catches integration mistakes that gradient-only
unit tests miss: wrong preference sign, objectives that cannot be optimized together,
router aggregation mistakes, or retention terms that explode during improvement training.

The student begins deliberately worse than a fixed synthetic teacher. Adam must recover
teacher logits/hidden/router behaviour while making its verified chosen-vs-rejected margin
strictly stronger than the teacher's.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from compact.distill import DistillWeights, specialist_distill_loss


class SmokeStudent(nn.Module):
    def __init__(self, teacher: dict[str, torch.Tensor], n_student_experts: int) -> None:
        super().__init__()
        # Perturb every retention channel so training has something real to recover.
        g = torch.Generator().manual_seed(1234)
        self.logits = nn.Parameter(
            teacher["logits"] + 0.45 * torch.randn(teacher["logits"].shape, generator=g)
        )
        self.hidden = nn.Parameter(
            teacher["hidden"] + 0.35 * torch.randn(teacher["hidden"].shape, generator=g)
        )
        shape = (*teacher["router"].shape[:-1], n_student_experts)
        self.router = nn.Parameter(torch.randn(shape, generator=g) * 0.8)

        # Teacher margin is +1.0. Student starts preferring the rejected answer (-0.5).
        batch = teacher["logits"].shape[0]
        self.chosen = nn.Parameter(torch.full((batch,), -3.0))
        self.rejected = nn.Parameter(torch.full((batch,), -2.5))


def build_fixture() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(77)
    batch, seq, vocab, hidden = 6, 5, 23, 12
    teacher_experts = 8
    teacher = {
        "logits": torch.randn((batch, seq, vocab), generator=g),
        "hidden": torch.randn((batch, seq, hidden), generator=g),
        "router": torch.randn((batch, seq, teacher_experts), generator=g),
        "chosen": torch.full((batch,), -2.0),
        "rejected": torch.full((batch,), -3.0),
    }
    mapping = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    mask = torch.ones((batch, seq), dtype=torch.bool)
    mask[-1, -1] = False
    return teacher, mapping, mask


def metrics(
    student: SmokeStudent,
    teacher: dict[str, torch.Tensor],
    mapping: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    losses = specialist_distill_loss(
        student_logits=student.logits,
        teacher_logits=teacher["logits"],
        student_hidden=student.hidden,
        teacher_hidden=teacher["hidden"],
        student_router_logits=student.router,
        teacher_router_logits=teacher["router"],
        teacher_to_student=mapping,
        token_mask=mask,
        student_chosen_logp=student.chosen,
        student_rejected_logp=student.rejected,
        teacher_chosen_logp=teacher["chosen"],
        teacher_rejected_logp=teacher["rejected"],
        weights=DistillWeights(),
        margin_over_teacher=0.25,
    )
    return {
        "total": float(losses["total"].detach()),
        "logits": float(losses["logits"].detach()),
        "hidden": float(losses["hidden"].detach()),
        "router": float(losses["router"].detach()),
        "verified": float(losses["verified_improvement"].detach()),
        "student_margin": float((student.chosen - student.rejected).mean().detach()),
        "teacher_margin": float((teacher["chosen"] - teacher["rejected"]).mean()),
    }


def run_smoke(*, steps: int = 300, lr: float = 0.08) -> dict[str, object]:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    teacher, mapping, mask = build_fixture()
    student = SmokeStudent(teacher, n_student_experts=3)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    before = metrics(student, teacher, mapping, mask)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = specialist_distill_loss(
            student_logits=student.logits,
            teacher_logits=teacher["logits"],
            student_hidden=student.hidden,
            teacher_hidden=teacher["hidden"],
            student_router_logits=student.router,
            teacher_router_logits=teacher["router"],
            teacher_to_student=mapping,
            token_mask=mask,
            student_chosen_logp=student.chosen,
            student_rejected_logp=student.rejected,
            teacher_chosen_logp=teacher["chosen"],
            teacher_rejected_logp=teacher["rejected"],
            weights=DistillWeights(),
            margin_over_teacher=0.25,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()

    after = metrics(student, teacher, mapping, mask)
    passed = (
        after["total"] < before["total"] * 0.20
        and after["logits"] < 1e-3
        and after["hidden"] < 1e-4
        and after["router"] < 1e-3
        and after["student_margin"] > after["teacher_margin"] + 0.25
        and after["verified"] < before["verified"] * 0.25
    )
    return {"pass": passed, "steps": steps, "lr": lr, "before": before, "after": after}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.08)
    args = p.parse_args()
    report = run_smoke(steps=args.steps, lr=args.lr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
