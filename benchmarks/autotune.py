#!/usr/bin/env python3
"""Tune real K3 compute, expert-I/O threads and optional draft top-k.

The tuner runs the SAME exact K3 request repeatedly, changing only performance knobs:
  * --threads N                  main OpenMP compute team
  * K3_ASYNC_IO_THREADS=N       background expert-read team
  * --draft-topk K              optional cheap-draft routed expert count

Draft top-k tuning is opt-in and requires a --draft-trunk in the K3 arguments. The exact
K3 model still verifies every emitted token; this script refuses to recommend anything
if generated_ids or full_ids change between any candidate runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Run:
    phase: str
    compute_threads: int
    io_threads: int
    draft_topk: int | None
    repeat: int
    seconds_per_token: float
    generated_ids: list[int]
    full_ids: list[int]
    json_path: str
    log_path: str


def parse_candidates(text: str, *, max_value: int | None = None) -> list[int]:
    vals: list[int] = []
    for raw in text.replace(",", " ").split():
        try:
            v = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer candidate: {raw!r}") from exc
        if v < 1:
            raise argparse.ArgumentTypeError("candidates must be >= 1")
        if max_value is not None and v > max_value:
            continue
        if v not in vals:
            vals.append(v)
    vals.sort()
    if not vals:
        raise argparse.ArgumentTypeError("candidate list is empty after validation")
    return vals


def default_compute_candidates(cpu_count: int) -> list[int]:
    vals = [1]
    x = 2
    while x < cpu_count:
        vals.append(x)
        x *= 2
    if cpu_count > 1:
        vals.append(cpu_count)
    return sorted(set(vals))


def candidate_arg_present(args: Iterable[str], name: str) -> bool:
    prefix = name + "="
    return any(a == name or a.startswith(prefix) for a in args)


def option_value(args: list[str], name: str) -> str | None:
    """Read a K3 option in either --name VALUE or --name=VALUE form."""
    prefix = name + "="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            if not value:
                raise ValueError(f"{name}= has no value")
            return value
        if arg == name:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise ValueError(f"{name} needs a value")
            return args[i + 1]
    return None


def exact_topk_from_config(model_dir: str, k3_args: list[str]) -> tuple[Path, int]:
    """Resolve the same config path K3 uses and read its exact routed top-k.

    K3 supports both the released nested config (text_config.num_experts_per_token) and
    the flat oracle/tiny fixture shape. We deliberately support exactly those two shapes
    here and never invent a default: an unreadable config must stop autotuning rather
    than silently generate inappropriate draft candidates.
    """
    raw = option_value(k3_args, "--config")
    path = Path(raw).expanduser() if raw else Path(model_dir).expanduser() / "config.json"
    path = path.resolve()
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read K3 config {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError(f"K3 config {path} is not a JSON object")
    base = root.get("text_config", root)
    if not isinstance(base, dict):
        raise ValueError(f"K3 config {path} has non-object text_config")
    value = base.get("num_experts_per_token")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"K3 config {path} has no integer num_experts_per_token")
    topk = int(value)
    if topk < 1:
        raise ValueError(f"K3 config {path} has invalid num_experts_per_token={topk}")
    return path, topk


def automatic_draft_topk_candidates(exact_topk: int) -> list[int]:
    """Powers of two up to exact top-k, always including the exact top-k itself."""
    vals: list[int] = []
    v = 1
    while v < exact_topk:
        vals.append(v)
        v *= 2
    if exact_topk not in vals:
        vals.append(exact_topk)
    return vals


def median_for(
    runs: list[Run],
    compute: int,
    io: int,
    draft_topk: int | None,
    phase: str | None = None,
) -> float:
    vals = [
        r.seconds_per_token
        for r in runs
        if r.compute_threads == compute
        and r.io_threads == io
        and r.draft_topk == draft_topk
        and (phase is None or r.phase == phase)
    ]
    if not vals:
        raise RuntimeError(
            f"no timings for compute={compute}, io={io}, draft_topk={draft_topk}, phase={phase}"
        )
    return statistics.median(vals)


def candidate_value(run: Run, vary: str) -> int:
    if vary == "compute":
        return run.compute_threads
    if vary == "io":
        return run.io_threads
    if vary == "draft_topk":
        if run.draft_topk is None:
            raise RuntimeError("draft-topk candidate requested for a run without draft-topk tuning")
        return run.draft_topk
    raise RuntimeError(f"unknown tuning dimension {vary!r}")


def best_candidate(runs: list[Run], *, phase: str, vary: str) -> int:
    subset = [r for r in runs if r.phase == phase]
    if not subset:
        raise RuntimeError(f"phase {phase!r} has no runs")
    values = sorted({candidate_value(r, vary) for r in subset})
    scored: list[tuple[float, int]] = []
    for v in values:
        samples = [r.seconds_per_token for r in subset if candidate_value(r, vary) == v]
        scored.append((statistics.median(samples), v))
    scored.sort()
    return scored[0][1]


def main() -> int:
    # argparse.REMAINDER would swallow tuner options placed after MODEL_DIR. Split the
    # command explicitly instead: everything before the first bare `--` belongs to this
    # tuner, everything after it is passed verbatim to K3. This works the same in Bash,
    # PowerShell and cmd.exe.
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        tuner_argv = argv[:sep]
        k3_args = argv[sep + 1 :]
    else:
        tuner_argv = argv
        k3_args: list[str] = []

    p = argparse.ArgumentParser(
        description=(
            "Autotune K3 compute + async expert-I/O threads, optionally including draft top-k, "
            "using a real exact request."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python benchmarks/autotune.py ~/k3model --repeats 2 -- \\\n"
            "    --trunk ~/k3trunk-lossless --preset laptop --incremental \\\n"
            "    --ids 1008,10484,318,15383,387 --gen 2 --temperature 0\n\n"
            "With an exact-verified draft trunk, use --draft-topk-candidates auto (recommended) "
            "or an explicit list such as 1,2,4 before the bare --, and pass \\\n"
            "  --draft-trunk PATH --spec-auto after it.\n\n"
            "For a quick first pass, use a short deterministic request. After tuning, run your "
            "normal longer prompt with the recommended settings."
        ),
    )
    p.add_argument("model_dir")
    p.add_argument(
        "--k3-bin",
        default=None,
        help="K3 executable (default ./bin/k3 or .\\bin\\k3.exe on Windows)",
    )
    p.add_argument("--repeats", type=int, default=2, help="runs per candidate (default 2)")
    p.add_argument(
        "--compute-candidates",
        default=None,
        help="comma/space list; default powers of two plus logical CPU count",
    )
    p.add_argument(
        "--io-candidates",
        default="1,2,4,8,16",
        help="async I/O thread candidates (default 1,2,4,8,16)",
    )
    p.add_argument(
        "--draft-topk-candidates",
        default=None,
        help=(
            "optional 'auto' or comma/space draft routed-expert counts; requires --draft-trunk "
            "after --. auto reads the exact model config and tries powers of two through its "
            "num_experts_per_token, including the exact top-k"
        ),
    )
    p.add_argument(
        "--draft-topk-seed",
        type=int,
        default=4,
        help="candidate nearest this value is used for the initial thread sweeps (default 4)",
    )
    p.add_argument(
        "--strategy",
        choices=("coordinate", "grid"),
        default="coordinate",
        help="coordinate is much cheaper; grid exhaustively tests every enabled dimension",
    )
    p.add_argument(
        "--io-seed", type=int, default=4, help="I/O threads used for the first compute sweep (default 4)"
    )
    p.add_argument("--out", default="k3-autotune.json", help="summary JSON path")
    p.add_argument("--keep-run-files", action="store_true", help="keep each K3 JSON/log beside the summary")
    p.add_argument("--timeout", type=float, default=None, help="optional timeout in seconds for each K3 run")
    ns = p.parse_args(tuner_argv)

    if ns.repeats < 1:
        p.error("--repeats must be >= 1")
    if ns.io_seed < 1 or ns.io_seed > 64:
        p.error("--io-seed must be in 1..64")
    if ns.draft_topk_seed < 1:
        p.error("--draft-topk-seed must be >= 1")

    if candidate_arg_present(k3_args, "--threads"):
        p.error("do not pass --threads after --; autotune controls it")
    if candidate_arg_present(k3_args, "--out"):
        p.error("do not pass --out after --; autotune needs a private JSON result per run")

    draft_topk: list[int] | None = None
    draft_topk_seed: int | None = None
    draft_topk_source_config: str | None = None
    draft_topk_exact_limit: int | None = None
    if ns.draft_topk_candidates:
        if not candidate_arg_present(k3_args, "--draft-trunk"):
            p.error("--draft-topk-candidates requires --draft-trunk after the bare --")
        if candidate_arg_present(k3_args, "--draft-topk"):
            p.error("do not pass --draft-topk after -- when --draft-topk-candidates is set")
        if ns.draft_topk_candidates.strip().lower() == "auto":
            try:
                config_path, draft_topk_exact_limit = exact_topk_from_config(ns.model_dir, k3_args)
            except ValueError as exc:
                p.error(str(exc))
            draft_topk_source_config = str(config_path)
            draft_topk = automatic_draft_topk_candidates(draft_topk_exact_limit)
        else:
            draft_topk = parse_candidates(ns.draft_topk_candidates)
        draft_topk_seed = min(draft_topk, key=lambda x: (abs(x - ns.draft_topk_seed), x))

    for disabled in ("K3_NOASYNC_PREFETCH", "K3_NOPREFETCH"):
        if os.environ.get(disabled):
            p.error(f"{disabled} is set; async I/O tuning would be meaningless. Unset it first.")

    cpu_count = max(1, os.cpu_count() or 1)
    compute = (
        parse_candidates(ns.compute_candidates, max_value=cpu_count)
        if ns.compute_candidates
        else default_compute_candidates(cpu_count)
    )
    io = parse_candidates(ns.io_candidates, max_value=64)
    io_seed = min(io, key=lambda x: (abs(x - ns.io_seed), x))

    if ns.k3_bin:
        k3_bin = Path(ns.k3_bin)
    elif os.name == "nt":
        k3_bin = Path("bin") / "k3.exe"
    else:
        k3_bin = Path("bin") / "k3"
    if not k3_bin.exists():
        p.error(f"K3 executable not found: {k3_bin}")

    out_path = Path(ns.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    permanent_dir = out_path.with_suffix("").with_name(out_path.stem + "-runs")
    temp_ctx = None
    if ns.keep_run_files:
        permanent_dir.mkdir(parents=True, exist_ok=True)
        run_dir = permanent_dir
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="k3-autotune-")
        run_dir = Path(temp_ctx.name)

    runs: list[Run] = []
    reference_generated: list[int] | None = None
    reference_full: list[int] | None = None

    def run_one(
        phase: str,
        cthreads: int,
        iothreads: int,
        dtopk: int | None,
        repeat: int,
    ) -> Run:
        nonlocal reference_generated, reference_full
        safe_phase = phase.replace("/", "_")
        ktag = f"-k{dtopk}" if dtopk is not None else ""
        stem = f"{safe_phase}-c{cthreads}-io{iothreads}{ktag}-r{repeat}"
        json_path = run_dir / f"{stem}.json"
        log_path = run_dir / f"{stem}.log"
        env = os.environ.copy()
        env["K3_ASYNC_IO_THREADS"] = str(iothreads)
        controlled = ["--threads", str(cthreads)]
        if dtopk is not None:
            controlled += ["--draft-topk", str(dtopk)]
        cmd = [str(k3_bin), ns.model_dir, *k3_args, *controlled, "--out", str(json_path)]
        ktxt = f" k={dtopk:>2}" if dtopk is not None else ""
        print(f"  c={cthreads:>3} io={iothreads:>2}{ktxt} r={repeat}: ", end="", flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=ns.timeout,
                check=False,
            )
        if proc.returncode != 0:
            tail = ""
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-30:])
            except OSError:
                pass
            raise RuntimeError(f"K3 failed with exit code {proc.returncode}\n{tail}")
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            sec = float(data["seconds_per_token"])
            generated = [int(x) for x in data["generated_ids"]]
            full = [int(x) for x in data["full_ids"]]
        except Exception as exc:  # noqa: BLE001 - report malformed result with path
            raise RuntimeError(f"cannot parse K3 result {json_path}: {exc}") from exc
        if not (sec > 0.0):
            raise RuntimeError(f"invalid seconds_per_token={sec!r} in {json_path}")

        if reference_generated is None:
            reference_generated = generated
            reference_full = full
        elif generated != reference_generated or full != reference_full:
            raise RuntimeError(
                "REFUSING RESULT: token stream changed while tuning performance knobs\n"
                f"reference generated_ids={reference_generated}\n"
                f"got       generated_ids={generated}\n"
                f"candidate compute={cthreads}, io={iothreads}, draft_topk={dtopk}, repeat={repeat}\n"
                "No recommendation was produced."
            )
        print(f"{sec:.6f} s/token")
        r = Run(
            phase=phase,
            compute_threads=cthreads,
            io_threads=iothreads,
            draft_topk=dtopk,
            repeat=repeat,
            seconds_per_token=sec,
            generated_ids=generated,
            full_ids=full,
            json_path=str(json_path) if ns.keep_run_files else "",
            log_path=str(log_path) if ns.keep_run_files else "",
        )
        runs.append(r)
        return r

    def sweep(phase: str, configs: list[tuple[int, int, int | None]]) -> None:
        print(f"\n[{phase}] {len(configs)} candidate configuration(s), {ns.repeats} repeat(s) each")
        # Alternate candidate order between repeats to reduce simple thermal/order bias.
        for rep in range(1, ns.repeats + 1):
            ordered = configs if rep % 2 else list(reversed(configs))
            for cthreads, iothreads, dtopk in ordered:
                run_one(phase, cthreads, iothreads, dtopk, rep)

    print("K3 real-hardware autotune")
    print(f"  logical CPUs       : {cpu_count}")
    print(f"  compute candidates : {compute}")
    print(f"  I/O candidates     : {io}")
    if draft_topk is not None:
        if draft_topk_exact_limit is not None:
            print(
                f"  draft top-k        : {draft_topk} (auto from exact top-{draft_topk_exact_limit}; "
                f"initial seed {draft_topk_seed})"
            )
            print(f"  source config      : {draft_topk_source_config}")
        else:
            print(f"  draft top-k        : {draft_topk} (initial seed {draft_topk_seed})")
    else:
        print("  draft top-k        : unchanged")
    print(f"  repeats            : {ns.repeats}")
    print(f"  strategy           : {ns.strategy}")
    print("  exactness guard    : generated_ids + full_ids must match on every run")

    try:
        best_topk: int | None = None
        if ns.strategy == "grid":
            topks: list[int | None] = draft_topk if draft_topk is not None else [None]
            configs = [(c, i, k) for c in compute for i in io for k in topks]
            sweep("grid", configs)
            config_medians = {
                (c, i, k): median_for(runs, c, i, k, "grid") for c, i, k in configs
            }
            best_c, best_io, best_topk = min(config_medians, key=config_medians.get)
        elif draft_topk is None:
            sweep("compute-1", [(c, io_seed, None) for c in compute])
            best_c1 = best_candidate(runs, phase="compute-1", vary="compute")
            sweep("io", [(best_c1, i, None) for i in io])
            best_io = best_candidate(runs, phase="io", vary="io")
            sweep("compute-2", [(c, best_io, None) for c in compute])
            best_c = best_candidate(runs, phase="compute-2", vary="compute")
            # If the second compute pass moved, confirm I/O at the final compute count.
            if best_c != best_c1:
                sweep("io-confirm", [(best_c, i, None) for i in io])
                best_io = best_candidate(runs, phase="io-confirm", vary="io")
        else:
            assert draft_topk_seed is not None
            # Cheap three-dimensional coordinate search. Tune compute/I/O around a sane
            # draft seed first, then let top-k change the workload and reconfirm both
            # thread dimensions once. If those move, reconfirm top-k at the final pair.
            sweep("compute-1", [(c, io_seed, draft_topk_seed) for c in compute])
            best_c1 = best_candidate(runs, phase="compute-1", vary="compute")
            sweep("io-1", [(best_c1, i, draft_topk_seed) for i in io])
            best_io1 = best_candidate(runs, phase="io-1", vary="io")
            sweep("draft-topk-1", [(best_c1, best_io1, k) for k in draft_topk])
            best_topk = best_candidate(runs, phase="draft-topk-1", vary="draft_topk")

            sweep("compute-2", [(c, best_io1, best_topk) for c in compute])
            best_c = best_candidate(runs, phase="compute-2", vary="compute")
            sweep("io-2", [(best_c, i, best_topk) for i in io])
            best_io = best_candidate(runs, phase="io-2", vary="io")

            if best_c != best_c1 or best_io != best_io1:
                sweep("draft-topk-2", [(best_c, best_io, k) for k in draft_topk])
                best_topk = best_candidate(runs, phase="draft-topk-2", vary="draft_topk")

        # Use every measurement at the final configuration when available; otherwise the
        # phase that selected it has at least `repeats` samples. Do not mix measurements
        # from a different draft top-k: that would bias the reported final median.
        final_samples = [
            r.seconds_per_token
            for r in runs
            if r.compute_threads == best_c
            and r.io_threads == best_io
            and r.draft_topk == best_topk
        ]
        if not final_samples:
            raise RuntimeError("internal error: final configuration has no timing samples")
        final_median = statistics.median(final_samples)
        recommended: dict[str, int | float] = {
            "threads": best_c,
            "async_io_threads": best_io,
            "median_seconds_per_token": final_median,
            # Backward-compatible two-dimensional field name retained for existing
            # report consumers. In 3D mode it means the same final exact configuration.
            "samples_at_final_pair": len(final_samples),
            "samples_at_final_configuration": len(final_samples),
        }
        if best_topk is not None:
            recommended["draft_topk"] = best_topk

        summary = {
            "model_dir": ns.model_dir,
            "k3_bin": str(k3_bin),
            "k3_args": k3_args,
            "logical_cpus": cpu_count,
            "strategy": ns.strategy,
            "repeats": ns.repeats,
            "compute_candidates": compute,
            "io_candidates": io,
            "draft_topk_candidates": draft_topk,
            "draft_topk_source_config": draft_topk_source_config,
            "draft_topk_exact_limit": draft_topk_exact_limit,
            "recommended": recommended,
            "reference_generated_ids": reference_generated,
            "reference_full_ids": reference_full,
            "runs": [asdict(r) for r in runs],
        }
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("\nRECOMMENDED")
        print(f"  --threads {best_c}")
        if best_topk is not None:
            print(f"  --draft-topk {best_topk}")
        if os.name == "nt":
            print(f"  PowerShell: $env:K3_ASYNC_IO_THREADS='{best_io}'")
            print(f"  cmd.exe   : set K3_ASYNC_IO_THREADS={best_io}")
        else:
            print(f"  K3_ASYNC_IO_THREADS={best_io}")
        print(f"  observed median at final configuration: {final_median:.6f} s/token")
        print(f"  summary: {out_path}")
        print("\nThe recommendation applies to this machine, storage path, cache/preset, draft and request shape.")
        print("Re-run after changing SSD, cache size, preset, draft trunk, or other major hardware/runtime conditions.")
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
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly with one concise message
        print(f"autotune failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
