#!/usr/bin/env python3
"""Fail-closed K3-Compact vs K3 Max score gate.

Input JSON must contain a measured student section and may override teacher values.
Cyber has no hard-coded public K3 Max baseline, so both teacher and student Cybench
scores are mandatory before the cyber domain can pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CODE_METRICS = ("deepswe", "programbench", "terminal_bench_2_1", "kimi_code_bench_2_0")
AGENT_METRICS = ("browsecomp", "deepsearchqa_f1", "researchrubrics")
CYBER_METRICS = ("cybench",)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _domain_gate(
    teacher: dict[str, Any], student: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    missing: list[str] = []
    wins = True
    for metric in metrics:
        t = _num(teacher.get(metric))
        s = _num(student.get(metric))
        if t is None or s is None:
            missing.append(metric)
            wins = False
            rows[metric] = {"teacher": t, "student": s, "delta": None, "win": False}
            continue
        delta = s - t
        win = delta > 0.0
        wins = wins and win
        rows[metric] = {"teacher": t, "student": s, "delta": delta, "win": win}
    return {"pass": wins and not missing, "missing": missing, "metrics": rows}


def evaluate(
    baselines: dict[str, Any], results: dict[str, Any], max_checkpoint_gb: float = 100.0
) -> dict[str, Any]:
    teacher_override = results.get("teacher", {})
    student = results.get("student", {})

    teacher_code = dict(baselines["code"])
    teacher_code.update(teacher_override.get("code", {}))
    teacher_agent = dict(baselines["agentic"])
    teacher_agent.update(teacher_override.get("agentic", {}))
    teacher_cyber = dict(baselines["cyber"])
    teacher_cyber.update(teacher_override.get("cyber", {}))

    code = _domain_gate(teacher_code, student.get("code", {}), CODE_METRICS)
    agentic = _domain_gate(teacher_agent, student.get("agentic", {}), AGENT_METRICS)
    cyber = _domain_gate(teacher_cyber, student.get("cyber", {}), CYBER_METRICS)

    checkpoint_gb = _num(results.get("checkpoint_gb"))
    storage_pass = checkpoint_gb is not None and checkpoint_gb <= max_checkpoint_gb
    general_retention = _num(results.get("general_retention_relative"))
    retention_floor = _num(results.get("general_retention_floor")) or 0.97
    retention_pass = general_retention is not None and general_retention >= retention_floor

    passed = code["pass"] and agentic["pass"] and cyber["pass"] and storage_pass and retention_pass
    return {
        "claim": "K3-Compact > K3 Max on code/cyber/agentic" if passed else "NOT PROVEN",
        "pass": passed,
        "code": code,
        "agentic": agentic,
        "cyber": cyber,
        "storage": {
            "checkpoint_gb": checkpoint_gb,
            "max_checkpoint_gb": max_checkpoint_gb,
            "pass": storage_pass,
        },
        "general_retention": {
            "relative": general_retention,
            "floor": retention_floor,
            "pass": retention_pass,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--baselines", default="compact/k3_max_target_baselines.json")
    p.add_argument("--out")
    args = p.parse_args()

    baselines = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = evaluate(baselines, results)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
