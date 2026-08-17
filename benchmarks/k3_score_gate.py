#!/usr/bin/env python3
"""Fail closed if measured local K3 scores regress below the pinned K3 Max references.

Input is JSON with a `scores` object using the same category/benchmark keys as
`benchmarks/k3_max_reference.json`. Missing results fail by default: this prevents a
partial run from being presented as full benchmark parity. Use --allow-missing only while
bringing up a harness/capability; the summary remains explicitly PARTIAL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_value(name: str, got: Any, ref: Any, tolerance: float) -> list[str]:
    failures: list[str] = []
    if isinstance(ref, list):
        if not isinstance(got, list) or len(got) != len(ref):
            return [f"{name}: expected {len(ref)} score(s), got {got!r}"]
        for i, (g, r) in enumerate(zip(got, ref)):
            failures += compare_value(f"{name}[{i}]", g, r, tolerance)
        return failures
    if not isinstance(got, (int, float)):
        return [f"{name}: non-numeric result {got!r}"]
    if float(got) + tolerance < float(ref):
        failures.append(f"{name}: {got} < reference {ref} (tolerance {tolerance})")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("k3_max_reference.json"),
    )
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument(
        "--categories",
        default="reasoning_knowledge,coding,agentic,vision",
        help="comma-separated categories to gate",
    )
    args = ap.parse_args()

    ref = json.loads(args.reference.read_text())
    got_doc = json.loads(args.results.read_text())
    got_scores = got_doc.get("scores", got_doc)
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    failures: list[str] = []
    missing: list[str] = []
    passed = 0
    for category in categories:
        expected = ref["scores"].get(category)
        if expected is None:
            failures.append(f"unknown reference category: {category}")
            continue
        actual_cat = got_scores.get(category, {})
        for bench, target in expected.items():
            if bench not in actual_cat:
                missing.append(f"{category}/{bench}")
                continue
            errs = compare_value(
                f"{category}/{bench}", actual_cat[bench], target, args.tolerance
            )
            if errs:
                failures.extend(errs)
            else:
                passed += 1

    print(f"K3 MAX SCORE GATE: passed={passed} missing={len(missing)} failed={len(failures)}")
    for line in failures:
        print("FAIL", line)
    for line in missing:
        print("MISSING", line)

    if failures:
        return 1
    if missing and not args.allow_missing:
        return 2
    if missing:
        print("STATUS: PARTIAL (missing allowed explicitly)")
    else:
        print("STATUS: FULL PARITY / NO SCORE REGRESSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
