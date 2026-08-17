#!/usr/bin/env python3
"""Probe exact fixed-bit BF16 dictionary codecs on released K3 ranges.

No model bytes are changed. For a BF16 block, the low byte plane stays verbatim; the high
byte is represented by a fixed-width codebook with the final code reserved as a literal
escape. We compare dict1/dict3/dict7/dict15 by *physical 4096-aligned bytes* and raw.

This is deliberately a range-only measurement: a hosted runner fetches only a few MiB from
each huge safetensors shard and refuses servers that ignore HTTP Range.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import requests

ALIGN = 4096
CODECS = {
    "dict1": (1, 1),
    "dict3": (2, 3),
    "dict7": (3, 7),
    "dict15": (4, 15),
}


def align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def hf_url(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def get_range(url: str, start: int, end: int, cap: int) -> bytes:
    want = end - start + 1
    if want <= 0 or want > cap:
        raise ValueError((start, end, cap))
    r = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        allow_redirects=True,
        timeout=120,
    )
    try:
        if r.status_code != 206:
            raise RuntimeError(
                f"HTTP Range not honored: status={r.status_code} "
                f"content-length={r.headers.get('content-length')}"
            )
        out = bytearray()
        for chunk in r.iter_content(1 << 20):
            if chunk:
                out += chunk
                if len(out) > want:
                    raise RuntimeError("range response exceeded requested byte count")
        if len(out) != want:
            raise RuntimeError(f"short range {len(out)} != {want}")
        return bytes(out)
    finally:
        r.close()


def shard_header(repo: str, rev: str, shard: str, cap: int = 64 << 20) -> tuple[int, dict]:
    u = hf_url(repo, rev, shard)
    hlen = int.from_bytes(get_range(u, 0, 7, 8), "little")
    if hlen <= 0 or hlen > cap:
        raise RuntimeError(f"implausible safetensors header {hlen}")
    return 8 + hlen, json.loads(get_range(u, 8, 8 + hlen - 1, cap))


def codec_stats(raw: bytes) -> dict[str, dict[str, float | int]]:
    if len(raw) & 1:
        raise ValueError("BF16 sample must have even byte count")
    n = len(raw) // 2
    high = raw[1::2]
    hist = Counter(high)
    ranked = [v for _, v in hist.most_common()]
    raw_phys = align(len(raw))
    out: dict[str, dict[str, float | int]] = {
        "raw": {"encoded": len(raw), "physical": raw_phys, "ratio": 1.0, "escapes": 0}
    }
    for name, (bits, dict_n) in CODECS.items():
        covered = sum(ranked[:dict_n])
        escapes = n - covered
        code_bytes = math.ceil(bits * n / 8)
        encoded = n + code_bytes + escapes
        physical = align(encoded)
        out[name] = {
            "encoded": encoded,
            "physical": physical,
            "ratio": physical / raw_phys,
            "escapes": escapes,
            "escape_fraction": escapes / n,
            "coverage": covered / n,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="moonshotai/Kimi-K3")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--sample-mib", type=int, default=4)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    sample_n = args.sample_mib << 20

    ri = requests.get(hf_url(args.repo, args.revision, "model.safetensors.index.json"), timeout=120)
    ri.raise_for_status()
    wm = ri.json()["weight_map"]
    names = [
        n for n in wm
        if ".layers." in n and "experts" not in n and n.endswith(".weight")
    ]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_shard[wm[name]].append(name)

    rows = []
    for shard in sorted(by_shard):
        if len(rows) >= args.samples:
            break
        data_base, header = shard_header(args.repo, args.revision, shard)
        # Spread across names in a shard. Sorting naturally mixes norm/gate/attn/projectors.
        for name in sorted(by_shard[shard]):
            if len(rows) >= args.samples:
                break
            meta = header.get(name)
            if not isinstance(meta, dict) or meta.get("dtype") != "BF16":
                continue
            a, b = map(int, meta["data_offsets"])
            size = b - a
            if size < sample_n:
                continue
            rel = ((size - sample_n) // 2) & ~4095
            raw = get_range(
                hf_url(args.repo, args.revision, shard),
                data_base + a + rel,
                data_base + a + rel + sample_n - 1,
                sample_n,
            )
            stats = codec_stats(raw)
            winner = min(stats, key=lambda k: (int(stats[k]["physical"]), k != "raw"))
            row = {"name": name, "shard": shard, "winner": winner, "stats": stats}
            rows.append(row)
            print(
                f"{len(rows):02d} {winner:6s} "
                + " ".join(f"{k}={float(stats[k]['ratio']):.4f}" for k in ("dict1","dict3","dict7","dict15"))
                + f" {name}"
            )

    if len(rows) < 8:
        raise SystemExit(f"only {len(rows)} usable BF16 samples")

    medians = {
        codec: statistics.median(float(r["stats"][codec]["ratio"]) for r in rows)
        for codec in ("dict1", "dict3", "dict7", "dict15")
    }
    adaptive = [min(float(v["ratio"]) for v in r["stats"].values()) for r in rows]
    wins = Counter(r["winner"] for r in rows)
    result = {
        "schema": 1,
        "repo": args.repo,
        "revision": args.revision,
        "sample_bytes": sample_n,
        "sample_count": len(rows),
        "median_ratios": medians,
        "adaptive_median_ratio": statistics.median(adaptive),
        "adaptive_mean_ratio": statistics.mean(adaptive),
        "winner_counts": dict(wins),
        "rows": rows,
        "note": "range-sample evidence only; full-trunk ratio requires full checkpoint",
    }
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
