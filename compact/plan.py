#!/usr/bin/env python3
"""Deterministic size/active-parameter accounting for K3-Compact.

This does not claim quality. It answers the mechanical question: does a proposed
student architecture fit the disk/active-parameter envelope before expensive training?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def estimate(spec: dict[str, Any]) -> dict[str, float | int | bool]:
    t = spec["teacher"]
    s = spec["student"]
    routed_layers = int(s["routed_layers"])
    experts = int(s["experts_per_layer"])
    top_k = int(s["top_k"])
    trunk = int(t["trunk_params"])
    per_expert = int(t["params_per_expert"])

    total_expert_params = routed_layers * experts * per_expert
    active_expert_params = routed_layers * top_k * per_expert
    total_params = trunk + total_expert_params
    active_params = trunk + active_expert_params

    weight_bits = float(s["target_weight_bits"])
    group_size = int(s["scale_group_size"])
    scale_bits = int(s["scale_bits"])
    effective_bits = weight_bits + scale_bits / group_size
    packed_gb = total_params * effective_bits / 8.0 / 1e9
    estimated_checkpoint_gb = (
        packed_gb
        + float(s["unquantized_reserve_gb"])
        + float(s["packaging_reserve_gb"])
    )
    max_gb = float(s["max_checkpoint_gb"])

    teacher_total = trunk + (
        int(t["routed_layers"]) * int(t["experts_per_layer"]) * per_expert
    )
    teacher_active = trunk + int(t["routed_layers"]) * int(t["top_k"]) * per_expert

    return {
        "total_params": total_params,
        "active_params": active_params,
        "total_params_b": total_params / 1e9,
        "active_params_b": active_params / 1e9,
        "teacher_total_params_b": teacher_total / 1e9,
        "teacher_active_params_b": teacher_active / 1e9,
        "parameter_reduction_x": teacher_total / total_params,
        "active_reduction_x": teacher_active / active_params,
        "effective_bits_per_weight": effective_bits,
        "packed_weights_gb": packed_gb,
        "estimated_checkpoint_gb": estimated_checkpoint_gb,
        "fits_checkpoint_budget": estimated_checkpoint_gb <= max_gb,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("spec", nargs="?", default="compact/k3_compact.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    result = estimate(load_spec(args.spec))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "K3-Compact: "
            f"{result['total_params_b']:.1f}B total, "
            f"{result['active_params_b']:.1f}B active, "
            f"~{result['estimated_checkpoint_gb']:.1f} GB checkpoint"
        )
        print(
            f"reduction: {result['parameter_reduction_x']:.2f}x params, "
            f"{result['active_reduction_x']:.2f}x active"
        )
    return 0 if result["fits_checkpoint_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
