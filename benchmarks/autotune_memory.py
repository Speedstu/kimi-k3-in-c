#!/usr/bin/env python3
"""Tune K3 memory allocation under a fixed allocator budget.

The total tunable budget is held constant for every candidate:

    exact trunk GB + expert cache GB + optional draft trunk GB = allocator budget GB

Only performance knobs change. Every candidate must emit the exact same generated_ids and
full_ids or the tuner aborts without a recommendation. K3's own peak-RSS line is parsed
and recorded as an observed diagnostic; the allocator budget is not a hard OS RSS cap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


RSS_RE = re.compile(r"PEAK RSS for the whole run:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)")


@dataclass(frozen=True)
class Allocation:
    trunk_gb: float
    cache_gb: float
    draft_trunk_gb: float


@dataclass
class Run:
    phase: str
    allocation: Allocation
    repeat: int
    seconds_per_token: float
    peak_rss_gb: float | None
    generated_ids: list[int]
    full_ids: list[int]
    json_path: str
    log_path: str


def parse_float_candidates(text: str) -> list[float]:
    vals: list[float] = []
    for raw in text.replace(",", " ").split():
        try:
            value = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid memory candidate: {raw!r}") from exc
        if not (value > 0.0):
            raise argparse.ArgumentTypeError("memory candidates must be > 0")
        if not any(abs(value - old) < 1e-12 for old in vals):
            vals.append(value)
    vals.sort()
    if not vals:
        raise argparse.ArgumentTypeError("memory candidate list is empty")
    return vals


def candidate_arg_present(args: Iterable[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in args)


def nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda value: (abs(value - target), value))


def parse_peak_rss_gb(text: str) -> float | None:
    match = RSS_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    factors = {
        "B": 1e-9,
        "KB": 1e-6,
        "MB": 1e-3,
        "GB": 1.0,
        "TB": 1e3,
        "KiB": 1024.0 / 1e9,
        "MiB": 1024.0**2 / 1e9,
        "GiB": 1024.0**3 / 1e9,
        "TiB": 1024.0**4 / 1e9,
    }
    return value * factors[unit]


def allocation_for(budget: float, cache: float, draft: float, trunk_min: float) -> Allocation | None:
    trunk = budget - cache - draft
    if trunk + 1e-12 < trunk_min:
        return None
    # Round only to keep command lines/report keys stable; the sum remains within
    # floating-point noise of the requested budget.
    return Allocation(round(trunk, 6), round(cache, 6), round(draft, 6))


def median_for(runs: list[Run], phase: str, selector: str, value: float) -> float:
    samples: list[float] = []
    for run in runs:
        if run.phase != phase:
            continue
        actual = run.allocation.cache_gb if selector == "cache" else run.allocation.draft_trunk_gb
        if abs(actual - value) < 1e-9:
            samples.append(run.seconds_per_token)
    if not samples:
        raise RuntimeError(f"no samples for {phase} {selector}={value}")
    return statistics.median(samples)


def best_value(runs: list[Run], phase: str, selector: str) -> float:
    values = sorted(
        {
            run.allocation.cache_gb if selector == "cache" else run.allocation.draft_trunk_gb
            for run in runs
            if run.phase == phase
        }
    )
    if not values:
        raise RuntimeError(f"phase {phase!r} has no runs")
    return min(values, key=lambda value: median_for(runs, phase, selector, value))


def main() -> int:
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        tuner_argv = argv[:sep]
        k3_args = argv[sep + 1 :]
    else:
        tuner_argv = argv
        k3_args: list[str] = []

    parser = argparse.ArgumentParser(
        description="Tune exact-trunk / expert-cache / draft-trunk RAM under one fixed K3 allocator budget."
    )
    parser.add_argument("model_dir")
    parser.add_argument("--allocator-budget-gb", type=float, required=True)
    parser.add_argument("--k3-bin", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--strategy", choices=("coordinate", "grid"), default="coordinate")
    parser.add_argument("--cache-candidates", default="0.5,1,4,8,16,32")
    parser.add_argument("--draft-trunk-candidates", default="1,2,4,8,16")
    parser.add_argument("--cache-seed-gb", type=float, default=1.0)
    parser.add_argument("--draft-seed-gb", type=float, default=4.0)
    parser.add_argument("--trunk-min-gb", type=float, default=1.0)
    parser.add_argument("--out", default="k3-memory-autotune.json")
    parser.add_argument("--keep-run-files", action="store_true")
    parser.add_argument("--timeout", type=float, default=None)
    ns = parser.parse_args(tuner_argv)

    # Command-shape conflicts are more actionable than downstream numeric validation.
    # Report them first so a user who accidentally mixes a preset/manual budget with the
    # tuner is told exactly which option is competing with the controlled experiment.
    if not candidate_arg_present(k3_args, "--trunk"):
        parser.error("memory autotune requires --trunk after the bare --")
    controlled = ("--trunk-gb", "--cache-gb", "--draft-trunk-gb", "--preset", "--out")
    for option in controlled:
        if candidate_arg_present(k3_args, option):
            parser.error(f"do not pass {option} after --; memory autotune controls allocator budgets")

    if not (ns.allocator_budget_gb > 0.0):
        parser.error("--allocator-budget-gb must be > 0")
    if ns.repeats < 1:
        parser.error("--repeats must be >= 1")
    if not (ns.trunk_min_gb > 0.0):
        parser.error("--trunk-min-gb must be > 0")
    if ns.trunk_min_gb >= ns.allocator_budget_gb:
        parser.error("--trunk-min-gb must be smaller than --allocator-budget-gb")

    have_draft = candidate_arg_present(k3_args, "--draft-trunk")
    cache_values = parse_float_candidates(ns.cache_candidates)
    draft_values = parse_float_candidates(ns.draft_trunk_candidates) if have_draft else [0.0]

    if ns.k3_bin:
        k3_bin = Path(ns.k3_bin)
    elif os.name == "nt":
        k3_bin = Path("bin") / "k3.exe"
    else:
        k3_bin = Path("bin") / "k3"
    if not k3_bin.exists():
        parser.error(f"K3 executable not found: {k3_bin}")

    def valid_allocation(cache: float, draft: float) -> Allocation | None:
        return allocation_for(ns.allocator_budget_gb, cache, draft, ns.trunk_min_gb)

    valid_cache = [c for c in cache_values if any(valid_allocation(c, d) for d in draft_values)]
    if not valid_cache:
        parser.error("no cache candidate fits the allocator budget with the requested trunk minimum")
    valid_draft = [d for d in draft_values if any(valid_allocation(c, d) for c in valid_cache)]
    if not valid_draft:
        parser.error("no draft-trunk candidate fits the allocator budget")

    cache_seed = nearest(valid_cache, ns.cache_seed_gb)
    draft_seed = nearest(valid_draft, ns.draft_seed_gb) if have_draft else 0.0

    out_path = Path(ns.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_ctx = None
    if ns.keep_run_files:
        run_dir = out_path.with_suffix("").with_name(out_path.stem + "-runs")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="k3-memory-autotune-")
        run_dir = Path(temp_ctx.name)

    runs: list[Run] = []
    reference_generated: list[int] | None = None
    reference_full: list[int] | None = None

    def run_one(phase: str, alloc: Allocation, repeat: int) -> Run:
        nonlocal reference_generated, reference_full
        stem = (
            f"{phase}-t{alloc.trunk_gb:g}-c{alloc.cache_gb:g}"
            f"-d{alloc.draft_trunk_gb:g}-r{repeat}"
        )
        json_path = run_dir / f"{stem}.json"
        log_path = run_dir / f"{stem}.log"
        memory_args = [
            "--trunk-gb",
            f"{alloc.trunk_gb:g}",
            "--cache-gb",
            f"{alloc.cache_gb:g}",
        ]
        if have_draft:
            memory_args += ["--draft-trunk-gb", f"{alloc.draft_trunk_gb:g}"]
        cmd = [str(k3_bin), ns.model_dir, *k3_args, *memory_args, "--out", str(json_path)]
        print(
            f"  trunk={alloc.trunk_gb:>7.3f} cache={alloc.cache_gb:>7.3f}"
            + (f" draft={alloc.draft_trunk_gb:>7.3f}" if have_draft else "")
            + f" r={repeat}: ",
            end="",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=ns.timeout,
                check=False,
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"K3 failed with exit code {proc.returncode}\n" + "\n".join(log_text.splitlines()[-30:])
            )
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            seconds = float(data["seconds_per_token"])
            generated = [int(x) for x in data["generated_ids"]]
            full = [int(x) for x in data["full_ids"]]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot parse K3 result {json_path}: {exc}") from exc
        if not (seconds > 0.0):
            raise RuntimeError(f"invalid seconds_per_token={seconds!r}")
        if reference_generated is None:
            reference_generated = generated
            reference_full = full
        elif generated != reference_generated or full != reference_full:
            raise RuntimeError(
                "REFUSING RESULT: token stream changed while tuning memory allocation\n"
                f"reference generated_ids={reference_generated}\n"
                f"got       generated_ids={generated}\n"
                f"candidate={alloc}\nNo recommendation was produced."
            )
        peak = parse_peak_rss_gb(log_text)
        peak_text = f", peak RSS {peak:.3f} GB" if peak is not None else ""
        print(f"{seconds:.6f} s/token{peak_text}")
        run = Run(
            phase=phase,
            allocation=alloc,
            repeat=repeat,
            seconds_per_token=seconds,
            peak_rss_gb=peak,
            generated_ids=generated,
            full_ids=full,
            json_path=str(json_path) if ns.keep_run_files else "",
            log_path=str(log_path) if ns.keep_run_files else "",
        )
        runs.append(run)
        return run

    def sweep(phase: str, allocations: list[Allocation]) -> None:
        unique = list(dict.fromkeys(allocations))
        if not unique:
            raise RuntimeError(f"phase {phase!r} has no valid allocations")
        print(f"\n[{phase}] {len(unique)} allocation(s), {ns.repeats} repeat(s) each")
        for repeat in range(1, ns.repeats + 1):
            ordered = unique if repeat % 2 else list(reversed(unique))
            for alloc in ordered:
                run_one(phase, alloc, repeat)

    print("K3 fixed-budget memory autotune")
    print(f"  allocator budget : {ns.allocator_budget_gb:.3f} GB")
    print(f"  trunk minimum    : {ns.trunk_min_gb:.3f} GB")
    print(f"  cache candidates : {valid_cache}")
    print(f"  draft candidates : {valid_draft if have_draft else 'not active'}")
    print(f"  strategy         : {ns.strategy}")
    print("  exactness guard  : generated_ids + full_ids must match on every run")
    print("  note             : peak RSS is observed, not constrained by allocator-budget-gb")

    try:
        if ns.strategy == "grid":
            allocations = [
                alloc
                for cache in valid_cache
                for draft in valid_draft
                if (alloc := valid_allocation(cache, draft)) is not None
            ]
            sweep("grid", allocations)
            medians: dict[Allocation, float] = {}
            for alloc in set(allocations):
                samples = [r.seconds_per_token for r in runs if r.phase == "grid" and r.allocation == alloc]
                medians[alloc] = statistics.median(samples)
            best = min(medians, key=medians.get)
        elif not have_draft:
            allocations = [valid_allocation(cache, 0.0) for cache in valid_cache]
            sweep("cache", [a for a in allocations if a is not None])
            best_cache = best_value(runs, "cache", "cache")
            best = valid_allocation(best_cache, 0.0)
            assert best is not None
        else:
            cache_allocs = [valid_allocation(cache, draft_seed) for cache in valid_cache]
            sweep("cache-1", [a for a in cache_allocs if a is not None])
            best_cache_1 = best_value(runs, "cache-1", "cache")

            draft_allocs = [valid_allocation(best_cache_1, draft) for draft in valid_draft]
            sweep("draft", [a for a in draft_allocs if a is not None])
            best_draft = best_value(runs, "draft", "draft")

            if abs(best_draft - draft_seed) > 1e-12:
                cache_confirm = [valid_allocation(cache, best_draft) for cache in valid_cache]
                sweep("cache-2", [a for a in cache_confirm if a is not None])
                best_cache = best_value(runs, "cache-2", "cache")
            else:
                best_cache = best_cache_1
            best = valid_allocation(best_cache, best_draft)
            assert best is not None

        final_samples = [r.seconds_per_token for r in runs if r.allocation == best]
        if not final_samples:
            raise RuntimeError("internal error: final allocation has no samples")
        final_median = statistics.median(final_samples)
        peak_samples = [r.peak_rss_gb for r in runs if r.allocation == best and r.peak_rss_gb is not None]
        peak_median = statistics.median(peak_samples) if peak_samples else None

        summary = {
            "model_dir": ns.model_dir,
            "k3_bin": str(k3_bin),
            "k3_args": k3_args,
            "allocator_budget_gb": ns.allocator_budget_gb,
            "trunk_min_gb": ns.trunk_min_gb,
            "strategy": ns.strategy,
            "repeats": ns.repeats,
            "cache_candidates_gb": valid_cache,
            "draft_trunk_candidates_gb": valid_draft if have_draft else None,
            "recommended": {
                "trunk_gb": best.trunk_gb,
                "cache_gb": best.cache_gb,
                "draft_trunk_gb": best.draft_trunk_gb if have_draft else None,
                "allocator_sum_gb": best.trunk_gb + best.cache_gb + best.draft_trunk_gb,
                "median_seconds_per_token": final_median,
                "median_peak_rss_gb": peak_median,
                "samples_at_final_allocation": len(final_samples),
            },
            "reference_generated_ids": reference_generated,
            "reference_full_ids": reference_full,
            "runs": [
                {
                    **asdict(run),
                    "allocation": asdict(run.allocation),
                }
                for run in runs
            ],
        }
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        print("\nRECOMMENDED MEMORY ALLOCATION")
        print(f"  --trunk-gb {best.trunk_gb:g}")
        print(f"  --cache-gb {best.cache_gb:g}")
        if have_draft:
            print(f"  --draft-trunk-gb {best.draft_trunk_gb:g}")
        print(f"  fixed allocator sum: {best.trunk_gb + best.cache_gb + best.draft_trunk_gb:.3f} GB")
        if peak_median is not None:
            print(f"  observed median peak RSS: {peak_median:.3f} GB")
        print(f"  observed median: {final_median:.6f} s/token")
        print(f"  summary: {out_path}")
        print("\nThis is a same-budget performance comparison, not an OS hard-RSS guarantee.")
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no recommendation produced.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"memory autotune failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
