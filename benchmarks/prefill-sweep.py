#!/usr/bin/env python3
"""Measure the real RAM-vs-trunk-I/O prefill tradeoff on this machine.

Each candidate starts one resident worker, runs the same tokenised prompt with greedy
exact K3, records the worker-selected chunk and request wall time, and refuses to rank a
candidate if its generated token differs from the baseline.  The checkpoint/trunk remain
local; no API or network inference is used.

Example:
  python benchmarks/prefill-sweep.py ~/k3model ~/k3trunk-lossless prompt.ids \
      --trunk-gb 3 --cache-gb 1 --budgets 64,128,256,512 --repeats 2
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
from pathlib import Path


CHUNK_RE = re.compile(r"resident prefill: chunk (\d+) tokens, ([0-9.]+) MiB transient")


def load_ids(path: Path) -> list[int]:
    text = path.read_text(encoding="ascii").replace(",", " ")
    ids = [int(x) for x in text.split()]
    if not ids:
        raise SystemExit(f"{path}: no token ids")
    if any(x < 0 for x in ids):
        raise SystemExit(f"{path}: negative token id")
    return ids


def one_run(args: argparse.Namespace, ids: list[int], budget: float) -> tuple[int, float, int]:
    context = args.context or max(1024, len(ids) + 8)
    cmd = [
        str(args.worker),
        str(args.model),
        "--trunk",
        str(args.trunk),
        "--context",
        str(context),
        "--trunk-gb",
        str(args.trunk_gb),
        "--cache-gb",
        str(args.cache_gb),
        "--prefill-mb",
        str(budget),
    ]
    if args.threads is not None:
        cmd += ["--threads", str(args.threads)]

    proc = subprocess.Popen(
        cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    selected = None
    startup_tail: list[str] = []
    for line in proc.stdout:
        startup_tail.append(line)
        if len(startup_tail) > 80:
            startup_tail.pop(0)
        match = CHUNK_RE.search(line)
        if match:
            selected = int(match.group(1))
        if line.startswith("@K3READY "):
            break
    else:
        raise RuntimeError("worker exited before READY:\n" + "".join(startup_tail))
    if selected is None:
        proc.kill()
        proc.wait()
        raise RuntimeError("worker did not report its selected prefill chunk")

    proc.stdin.write(f"REQ 1 {len(ids)} 1 0 1 1 -1\n")
    proc.stdin.write(" ".join(map(str, ids)) + "\n")
    proc.stdin.flush()
    token = None
    seconds = None
    for line in proc.stdout:
        if line.startswith("@K3TOKEN 1 "):
            token = int(line.split()[2])
        elif line.startswith("@K3ERROR 1 "):
            raise RuntimeError(line.strip())
        elif line.startswith("@K3DONE 1 "):
            fields = line.split()
            seconds = float(fields[5])
            break
    proc.stdin.write("QUIT\n")
    proc.stdin.flush()
    proc.wait(timeout=10)
    if proc.returncode != 0 or token is None or seconds is None:
        raise RuntimeError(f"worker failed (rc={proc.returncode}, token={token}, seconds={seconds})")
    return selected, seconds, token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("trunk", type=Path)
    ap.add_argument("ids_file", type=Path)
    ap.add_argument("--worker", type=Path, default=Path("bin/k3-worker"))
    ap.add_argument("--budgets", default="64,128,256,512,1024")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--context", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--trunk-gb", type=float, default=3.0)
    ap.add_argument("--cache-gb", type=float, default=1.0)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    if not budgets or any(x <= 0 for x in budgets):
        ap.error("--budgets must contain positive MiB values")
    ids = load_ids(args.ids_file)

    expected = None
    rows: list[tuple[float, int, float]] = []
    print("budget_MiB  chunk  median_seconds  runs")
    for budget in budgets:
        times: list[float] = []
        chunks: list[int] = []
        for _ in range(args.repeats):
            chunk, seconds, token = one_run(args, ids, budget)
            if expected is None:
                expected = token
            elif token != expected:
                raise SystemExit(
                    f"REFUSED: budget {budget:g} MiB emitted token {token}, baseline {expected}"
                )
            chunks.append(chunk)
            times.append(seconds)
        chunk = int(statistics.median(chunks))
        median = statistics.median(times)
        rows.append((budget, chunk, median))
        print(f"{budget:10g}  {chunk:5d}  {median:14.4f}  {','.join(f'{x:.3f}' for x in times)}")

    best = min(rows, key=lambda row: row[2])
    print(
        f"\nrecommendation: --prefill-mb {best[0]:g} "
        f"(chunk {best[1]}, median {best[2]:.4f} s for this prompt/machine)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
