#!/usr/bin/env python3
"""Measure dict7/dict15 exact storage ratios on small released K3 BF16 tensor ranges.

This is a network probe, not a permanent correctness oracle. It uses HTTP Range requests
and refuses full-file responses so a hosted runner can sample the real 1.56 TB checkpoint
without downloading shards.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lossless_trunk", ROOT / "tools/lossless_trunk.py")
lt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(lt)


def url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def rng(u: str, a: int, b: int, cap: int) -> bytes:
    if b < a or b - a + 1 > cap:
        raise ValueError((a, b, cap))
    r = requests.get(u, headers={"Range": f"bytes={a}-{b}", "Accept-Encoding": "identity"},
                     stream=True, allow_redirects=True, timeout=120)
    try:
        if r.status_code != 206:
            raise RuntimeError(f"Range ignored: HTTP {r.status_code} length={r.headers.get('content-length')}")
        want = b - a + 1
        out = bytearray()
        for chunk in r.iter_content(1 << 20):
            if chunk:
                out += chunk
                if len(out) > want:
                    raise RuntimeError("range overflow")
        if len(out) != want:
            raise RuntimeError((len(out), want))
        return bytes(out)
    finally:
        r.close()


def header(repo: str, rev: str, shard: str, cap: int) -> tuple[int, dict]:
    u = url(repo, rev, shard)
    hlen = int.from_bytes(rng(u, 0, 7, 8), "little")
    if hlen <= 0 or hlen > cap:
        raise RuntimeError(f"bad header len {hlen}")
    return 8 + hlen, json.loads(rng(u, 8, 8 + hlen - 1, cap))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="moonshotai/Kimi-K3")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--sample-mib", type=int, default=2)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    sample_n = args.sample_mib << 20

    idx = requests.get(url(args.repo, args.revision, "model.safetensors.index.json"), timeout=120)
    idx.raise_for_status()
    wm = idx.json()["weight_map"]
    # Dense/trunk candidates only. Expert MXFP4 is intentionally excluded; dict7 targets
    # the BF16 dense trunk that every token must sweep at low RAM budgets.
    names = [n for n in wm if ".layers." in n and "experts" not in n and n.endswith(".weight")]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by_shard[wm[n]].append(n)

    rows = []
    # Spread samples across shards/layers instead of cherry-picking one tensor family.
    for shard in sorted(by_shard):
        if len(rows) >= args.samples:
            break
        base, h = header(args.repo, args.revision, shard, 64 << 20)
        for name in sorted(by_shard[shard]):
            if len(rows) >= args.samples:
                break
            m = h.get(name)
            if not isinstance(m, dict) or m.get("dtype") != "BF16":
                continue
            a, b = map(int, m["data_offsets"])
            n = b - a
            if n < sample_n:
                continue
            rel = ((n - sample_n) // 2) & ~4095
            raw = rng(url(args.repo, args.revision, shard), base + a + rel,
                      base + a + rel + sample_n - 1, sample_n)
            p15, _, e15 = lt.encode_block_dict15(raw)
            p7, _, e7 = lt.encode_block_dict7(raw)
            r15 = len(p15) / len(raw)
            r7 = len(p7) / len(raw)
            chosen, payload, _, _ = lt.choose_block_codec(raw)
            rc = lt.align_up(len(payload)) / lt.align_up(len(raw))
            row = {"name": name, "shard": shard, "dict15": r15, "dict7": r7,
                   "adaptive_physical": rc, "chosen": chosen,
                   "dict15_escapes": e15, "dict7_escapes": e7}
            rows.append(row)
            print(f"{len(rows):02d} {chosen:6s} phys={rc:.4f} d7={r7:.4f} d15={r15:.4f} {name}")

    if len(rows) < min(args.samples, 4):
        raise SystemExit(f"only found {len(rows)} usable BF16 samples")
    result = {
        "schema": 1,
        "repo": args.repo,
        "revision": args.revision,
        "sample_bytes": sample_n,
        "samples": rows,
        "median_dict7": statistics.median(r["dict7"] for r in rows),
        "median_dict15": statistics.median(r["dict15"] for r in rows),
        "median_adaptive_physical": statistics.median(r["adaptive_physical"] for r in rows),
        "dict7_wins": sum(r["chosen"] == "dict7" for r in rows),
        "note": "range-sample evidence only; full-trunk ratio requires the full checkpoint",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
