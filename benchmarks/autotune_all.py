#!/usr/bin/env python3
"""Orchestrate exact K3 hardware, draft and memory autotuning in one command.

This wrapper intentionally delegates measurement to the two existing exactness-guarded
Tuners instead of reimplementing their search logic:

  benchmarks/autotune.py         -> compute threads, async I/O, optional draft top-k
  benchmarks/autotune_memory.py  -> exact trunk / expert cache / draft trunk split

The stages alternate until the discrete memory allocation stabilises (or the configured
cycle limit is reached). Every child report must carry the same generated_ids and
full_ids reference. A mismatch anywhere aborts the orchestration without a final
recommendation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


EPS = 1e-9


def candidate_arg_present(args: Iterable[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in args)


def fmt(value: float) -> str:
    return f"{value:.9g}"


def same_memory(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("trunk_gb", "cache_gb", "draft_trunk_gb")
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            if av is not bv:
                return False
        elif abs(float(av) - float(bv)) > EPS:
            return False
    return True


def same_hardware(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        int(a["threads"]) == int(b["threads"])
        and int(a["async_io_threads"]) == int(b["async_io_threads"])
        and a.get("draft_topk") == b.get("draft_topk")
    )


def load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read child autotune report {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("recommended"), dict):
        raise RuntimeError(f"child autotune report {path} is missing recommended settings")
    return data


def run_child(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    print("$ " + shlex.join(command))
    sys.stdout.flush()
    proc = subprocess.run(command, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


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
        description=(
            "One-command exact K3 machine autotune: threads/I/O/draft-topk, then fixed-budget "
            "memory, iterated until the discrete memory split stabilises."
        )
    )
    parser.add_argument("model_dir")
    parser.add_argument("--allocator-budget-gb", type=float, required=True)
    parser.add_argument("--k3-bin", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--compute-candidates", default=None)
    parser.add_argument("--io-candidates", default="1,2,4,8,16")
    parser.add_argument("--io-seed", type=int, default=4)
    parser.add_argument("--draft-topk-candidates", default="auto")
    parser.add_argument("--draft-topk-seed", type=int, default=4)
    parser.add_argument("--cache-candidates", default="auto")
    parser.add_argument("--draft-trunk-candidates", default="auto")
    parser.add_argument("--auto-min-gb", type=float, default=0.25)
    parser.add_argument("--trunk-min-gb", type=float, default=1.0)
    parser.add_argument("--seed-cache-gb", type=float, default=None)
    parser.add_argument("--seed-draft-trunk-gb", type=float, default=None)
    parser.add_argument("--max-rss-gb", type=float, default=None)
    parser.add_argument("--hardware-strategy", choices=("coordinate", "grid"), default="coordinate")
    parser.add_argument("--memory-strategy", choices=("coordinate", "grid"), default="coordinate")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--keep-run-files", action="store_true")
    parser.add_argument("--out", default="k3-full-autotune.json")
    ns = parser.parse_args(tuner_argv)

    controlled = (
        "--threads",
        "--trunk-gb",
        "--cache-gb",
        "--draft-trunk-gb",
        "--draft-topk",
        "--preset",
        "--out",
    )
    if not candidate_arg_present(k3_args, "--trunk"):
        parser.error("full autotune requires --trunk after the bare --")
    for option in controlled:
        if candidate_arg_present(k3_args, option):
            parser.error(f"do not pass {option} after --; full autotune controls it")

    if not (ns.allocator_budget_gb > 0.0):
        parser.error("--allocator-budget-gb must be > 0")
    if not (ns.trunk_min_gb > 0.0) or ns.trunk_min_gb >= ns.allocator_budget_gb:
        parser.error("--trunk-min-gb must be > 0 and smaller than --allocator-budget-gb")
    if not (ns.auto_min_gb > 0.0):
        parser.error("--auto-min-gb must be > 0")
    if ns.repeats < 1:
        parser.error("--repeats must be >= 1")
    if ns.max_cycles < 1:
        parser.error("--max-cycles must be >= 1")
    if ns.max_rss_gb is not None and not (ns.max_rss_gb > 0.0):
        parser.error("--max-rss-gb must be > 0")

    have_draft = candidate_arg_present(k3_args, "--draft-trunk")
    available = ns.allocator_budget_gb - ns.trunk_min_gb
    if have_draft:
        seed_cache = ns.seed_cache_gb if ns.seed_cache_gb is not None else min(1.0, available / 4.0)
        seed_draft = (
            ns.seed_draft_trunk_gb
            if ns.seed_draft_trunk_gb is not None
            else min(4.0, available / 4.0)
        )
    else:
        seed_cache = ns.seed_cache_gb if ns.seed_cache_gb is not None else min(1.0, available / 4.0)
        seed_draft = 0.0
    if not (seed_cache > 0.0) or (have_draft and not (seed_draft > 0.0)):
        parser.error("automatic seed memory is non-positive; increase allocator budget or lower trunk minimum")
    if seed_cache + seed_draft > available + EPS:
        parser.error("seed cache + draft-trunk budgets leave less than --trunk-min-gb for exact trunk")
    seed_memory: dict[str, Any] = {
        "trunk_gb": round(ns.allocator_budget_gb - seed_cache - seed_draft, 6),
        "cache_gb": round(seed_cache, 6),
        "draft_trunk_gb": round(seed_draft, 6) if have_draft else None,
    }

    scripts = Path(__file__).resolve().parent
    hardware_script = scripts / "autotune.py"
    memory_script = scripts / "autotune_memory.py"
    for script in (hardware_script, memory_script):
        if not script.exists():
            parser.error(f"required child tuner not found: {script}")

    out_path = Path(ns.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = out_path.with_suffix("").with_name(out_path.stem + "-stages")
    stage_dir.mkdir(parents=True, exist_ok=True)

    reference_generated: list[int] | None = None
    reference_full: list[int] | None = None
    stages: list[dict[str, Any]] = []

    def validate_reference(report: dict[str, Any], label: str) -> None:
        nonlocal reference_generated, reference_full
        generated = report.get("reference_generated_ids")
        full = report.get("reference_full_ids")
        if not isinstance(generated, list) or not isinstance(full, list):
            raise RuntimeError(f"{label} report does not contain exact reference token streams")
        generated = [int(x) for x in generated]
        full = [int(x) for x in full]
        if reference_generated is None:
            reference_generated = generated
            reference_full = full
        elif generated != reference_generated or full != reference_full:
            raise RuntimeError(
                "REFUSING FINAL RESULT: child autotune phases disagree on exact K3 token stream\n"
                f"reference generated_ids={reference_generated}\n"
                f"{label} generated_ids={generated}"
            )

    def hardware_tune(memory: dict[str, Any], label: str) -> dict[str, Any]:
        report_path = stage_dir / f"{label}.json"
        command = [
            sys.executable,
            str(hardware_script),
            ns.model_dir,
            "--repeats",
            str(ns.repeats),
            "--strategy",
            ns.hardware_strategy,
            "--io-candidates",
            ns.io_candidates,
            "--io-seed",
            str(ns.io_seed),
            "--out",
            str(report_path),
        ]
        if ns.k3_bin:
            command += ["--k3-bin", ns.k3_bin]
        if ns.compute_candidates:
            command += ["--compute-candidates", ns.compute_candidates]
        if ns.timeout is not None:
            command += ["--timeout", str(ns.timeout)]
        if ns.keep_run_files:
            command += ["--keep-run-files"]
        if have_draft:
            command += [
                "--draft-topk-candidates",
                ns.draft_topk_candidates,
                "--draft-topk-seed",
                str(ns.draft_topk_seed),
            ]
        command += ["--", *k3_args]
        command += ["--trunk-gb", fmt(float(memory["trunk_gb"])), "--cache-gb", fmt(float(memory["cache_gb"]))]
        if have_draft:
            command += ["--draft-trunk-gb", fmt(float(memory["draft_trunk_gb"]))]
        run_child(f"{label}: hardware / I/O / draft-topk", command)
        report = load_report(report_path)
        validate_reference(report, label)
        recommended = report["recommended"]
        hardware: dict[str, Any] = {
            "threads": int(recommended["threads"]),
            "async_io_threads": int(recommended["async_io_threads"]),
            "draft_topk": int(recommended["draft_topk"]) if have_draft else None,
            "median_seconds_per_token": float(recommended["median_seconds_per_token"]),
        }
        stages.append(
            {
                "label": label,
                "kind": "hardware",
                "input_memory": memory,
                "recommended": hardware,
                "report": str(report_path),
            }
        )
        return hardware

    def memory_tune(hardware: dict[str, Any], seed: dict[str, Any], label: str) -> dict[str, Any]:
        report_path = stage_dir / f"{label}.json"
        command = [
            sys.executable,
            str(memory_script),
            ns.model_dir,
            "--allocator-budget-gb",
            fmt(ns.allocator_budget_gb),
            "--trunk-min-gb",
            fmt(ns.trunk_min_gb),
            "--cache-candidates",
            ns.cache_candidates,
            "--cache-seed-gb",
            fmt(float(seed["cache_gb"])),
            "--auto-min-gb",
            fmt(ns.auto_min_gb),
            "--repeats",
            str(ns.repeats),
            "--strategy",
            ns.memory_strategy,
            "--out",
            str(report_path),
        ]
        if ns.k3_bin:
            command += ["--k3-bin", ns.k3_bin]
        if ns.timeout is not None:
            command += ["--timeout", str(ns.timeout)]
        if ns.keep_run_files:
            command += ["--keep-run-files"]
        if ns.max_rss_gb is not None:
            command += ["--max-rss-gb", fmt(ns.max_rss_gb)]
        if have_draft:
            command += [
                "--draft-trunk-candidates",
                ns.draft_trunk_candidates,
                "--draft-seed-gb",
                fmt(float(seed["draft_trunk_gb"])),
            ]
        command += ["--", *k3_args, "--threads", str(hardware["threads"])]
        if have_draft:
            command += ["--draft-topk", str(hardware["draft_topk"])]
        env = os.environ.copy()
        env["K3_ASYNC_IO_THREADS"] = str(hardware["async_io_threads"])
        run_child(f"{label}: fixed-budget memory", command, env=env)
        report = load_report(report_path)
        validate_reference(report, label)
        recommended = report["recommended"]
        memory: dict[str, Any] = {
            "trunk_gb": float(recommended["trunk_gb"]),
            "cache_gb": float(recommended["cache_gb"]),
            "draft_trunk_gb": (
                float(recommended["draft_trunk_gb"])
                if have_draft and recommended.get("draft_trunk_gb") is not None
                else None
            ),
            "allocator_sum_gb": float(recommended["allocator_sum_gb"]),
            "median_seconds_per_token": float(recommended["median_seconds_per_token"]),
            "median_peak_rss_gb": recommended.get("median_peak_rss_gb"),
            "max_peak_rss_gb": recommended.get("max_peak_rss_gb"),
        }
        stages.append(
            {
                "label": label,
                "kind": "memory",
                "input_hardware": hardware,
                "recommended": memory,
                "report": str(report_path),
            }
        )
        return memory

    print("K3 full-machine exact autotune")
    print(f"  allocator budget : {ns.allocator_budget_gb:g} GB")
    print(f"  initial memory    : {seed_memory}")
    print(f"  draft active      : {'yes' if have_draft else 'no'}")
    print(f"  max cycles        : {ns.max_cycles}")
    if ns.max_rss_gb is not None:
        print(f"  measured RSS guard: {ns.max_rss_gb:g} GB")
    print("  exactness guard   : all child reference token streams must match")

    current_memory = seed_memory
    previous_hardware: dict[str, Any] | None = None
    final_hardware: dict[str, Any] | None = None
    converged = False
    cycles_completed = 0

    for cycle in range(1, ns.max_cycles + 1):
        cycles_completed = cycle
        hardware = hardware_tune(current_memory, f"cycle-{cycle}-hardware")
        memory = memory_tune(hardware, current_memory, f"cycle-{cycle}-memory")
        hardware_stable = previous_hardware is None or same_hardware(hardware, previous_hardware)
        memory_stable = same_memory(memory, current_memory)
        previous_hardware = hardware
        final_hardware = hardware
        current_memory = memory
        if memory_stable and hardware_stable:
            converged = True
            break

    # If memory moved in the last cycle, reconfirm the hardware/top-k winner on exactly
    # the final split. This makes the launch recommendation measured on its actual memory
    # configuration even when the alternating search hit the configured cycle limit.
    last_stage = stages[-1] if stages else None
    if not converged and last_stage and last_stage["kind"] == "memory":
        final_hardware = hardware_tune(current_memory, "final-hardware-confirm")

    if final_hardware is None or reference_generated is None or reference_full is None:
        raise RuntimeError("full autotune produced no final exact recommendation")

    final_k3_args = [*k3_args]
    final_k3_args += [
        "--trunk-gb",
        fmt(float(current_memory["trunk_gb"])),
        "--cache-gb",
        fmt(float(current_memory["cache_gb"])),
        "--threads",
        str(final_hardware["threads"]),
    ]
    if have_draft:
        final_k3_args += [
            "--draft-trunk-gb",
            fmt(float(current_memory["draft_trunk_gb"])),
            "--draft-topk",
            str(final_hardware["draft_topk"]),
        ]

    if ns.k3_bin:
        k3_exec = ns.k3_bin
    elif os.name == "nt":
        k3_exec = str(Path("bin") / "k3.exe")
    else:
        k3_exec = str(Path("bin") / "k3")
    launch = [k3_exec, ns.model_dir, *final_k3_args]

    result = {
        "model_dir": ns.model_dir,
        "allocator_budget_gb": ns.allocator_budget_gb,
        "max_rss_gb": ns.max_rss_gb,
        "cycles_completed": cycles_completed,
        "converged": converged,
        "reference_generated_ids": reference_generated,
        "reference_full_ids": reference_full,
        "stages": stages,
        "recommended": {
            "threads": final_hardware["threads"],
            "async_io_threads": final_hardware["async_io_threads"],
            "draft_topk": final_hardware.get("draft_topk"),
            "trunk_gb": current_memory["trunk_gb"],
            "cache_gb": current_memory["cache_gb"],
            "draft_trunk_gb": current_memory.get("draft_trunk_gb"),
            "allocator_sum_gb": (
                float(current_memory["trunk_gb"])
                + float(current_memory["cache_gb"])
                + (float(current_memory["draft_trunk_gb"]) if have_draft else 0.0)
            ),
            "median_peak_rss_gb": current_memory.get("median_peak_rss_gb"),
            "max_peak_rss_gb": current_memory.get("max_peak_rss_gb"),
            "environment": {"K3_ASYNC_IO_THREADS": str(final_hardware["async_io_threads"])},
            "k3_args": final_k3_args,
            "posix_command": shlex.join(launch),
            "windows_command": subprocess.list2cmdline(launch),
        },
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("FINAL EXACT K3 CONFIGURATION")
    print("=" * 76)
    print(f"  --threads {final_hardware['threads']}")
    print(f"  K3_ASYNC_IO_THREADS={final_hardware['async_io_threads']}")
    if have_draft:
        print(f"  --draft-topk {final_hardware['draft_topk']}")
    print(f"  --trunk-gb {current_memory['trunk_gb']:g}")
    print(f"  --cache-gb {current_memory['cache_gb']:g}")
    if have_draft:
        print(f"  --draft-trunk-gb {current_memory['draft_trunk_gb']:g}")
    print(f"  converged: {'yes' if converged else 'no (hardware reconfirmed on final memory)'}")
    print(f"  report: {out_path}")
    print("\nExactness: every child stage used the same generated_ids + full_ids reference.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no final recommendation produced.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"full autotune failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
