#!/usr/bin/env python3
"""Probe whether released K3 MXFP4 experts have useful exact cross-expert redundancy.

Only small HTTP byte ranges are fetched from the public checkpoint. The script never
materializes a shard and never mutates model bytes. It compares raw-byte entropy and zlib
compressibility with XOR deltas between same-layer/same-projection experts. If XOR is not
materially lower-entropy/compressible, cross-expert lossless storage is not a credible path
to a dramatically smaller checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import requests

DEFAULT_REPO = "moonshotai/Kimi-K3"
EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?experts\.(?P<expert>\d+)\.(?P<proj>w[123])\.weight_packed$"
)


@dataclass(frozen=True)
class TensorLoc:
    name: str
    shard: str
    layer: int
    expert: int
    proj: str


def resolve_url(repo: str, path: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def get_range(url: str, start: int, end: int, *, max_bytes: int) -> bytes:
    if end < start or end - start + 1 > max_bytes:
        raise ValueError((start, end, max_bytes))
    r = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        allow_redirects=True,
        timeout=120,
    )
    try:
        if r.status_code != 206:
            # Never accidentally download a 17 GB shard if a backend ignores Range.
            raise RuntimeError(
                f"range request was not honored for {url}: HTTP {r.status_code}, "
                f"content-length={r.headers.get('content-length')}"
            )
        expected = end - start + 1
        out = bytearray()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                out += chunk
                if len(out) > expected:
                    raise RuntimeError("range backend returned more data than requested")
        if len(out) != expected:
            raise RuntimeError(f"short range: got {len(out)}, expected {expected}")
        return bytes(out)
    finally:
        r.close()


def safetensor_header(repo: str, shard: str, revision: str, max_header: int) -> tuple[int, dict]:
    url = resolve_url(repo, shard, revision)
    first = get_range(url, 0, 7, max_bytes=8)
    hlen = int.from_bytes(first, "little")
    if hlen <= 0 or hlen > max_header:
        raise RuntimeError(f"implausible safetensors header {hlen} in {shard}")
    raw = get_range(url, 8, 8 + hlen - 1, max_bytes=max_header)
    return 8 + hlen, json.loads(raw)


def entropy_bits(data: bytes) -> float:
    if not data:
        return 0.0
    c = Counter(data)
    n = len(data)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def zratio(data: bytes) -> float:
    return len(zlib.compress(data, 9)) / len(data)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError("xor lengths differ")
    return bytes(x ^ y for x, y in zip(a, b))


def tensor_sample(
    repo: str,
    shard: str,
    revision: str,
    data_base: int,
    meta: dict,
    sample_bytes: int,
) -> bytes:
    a, b = map(int, meta["data_offsets"])
    n = b - a
    take = min(n, sample_bytes)
    # Sample away from a possible tensor boundary while keeping deterministic alignment.
    rel = max(0, (n - take) // 2)
    rel -= rel % 4096
    if rel + take > n:
        rel = max(0, n - take)
    start = data_base + a + rel
    return get_range(resolve_url(repo, shard, revision), start, start + take - 1, max_bytes=sample_bytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--sample-mib", type=int, default=2)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--max-header-mib", type=int, default=64)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    sample_bytes = args.sample_mib << 20
    max_header = args.max_header_mib << 20
    idx = get_json(resolve_url(args.repo, "model.safetensors.index.json", args.revision))
    weight_map = idx.get("weight_map", {})
    candidates: list[TensorLoc] = []
    for name, shard in weight_map.items():
        m = EXPERT_RE.search(name)
        if m:
            candidates.append(
                TensorLoc(name, shard, int(m.group("layer")), int(m.group("expert")), m.group("proj"))
            )
    if not candidates:
        print("No expert weight_packed names matched. Example expert-like keys:", file=sys.stderr)
        for name in [k for k in weight_map if "expert" in k.lower()][:20]:
            print(name, file=sys.stderr)
        return 3

    groups: dict[tuple[int, str, str], list[TensorLoc]] = defaultdict(list)
    for x in candidates:
        groups[(x.layer, x.proj, x.shard)].append(x)
    viable = [v for v in groups.values() if len({x.expert for x in v}) >= args.experts]
    if not viable:
        raise SystemExit("No single shard contains enough same-layer/projection experts for the probe")
    viable.sort(key=lambda v: (-len(v), v[0].layer, v[0].proj, v[0].shard))
    chosen = sorted(viable[0], key=lambda x: x.expert)[: args.experts]
    shard = chosen[0].shard
    layer, proj = chosen[0].layer, chosen[0].proj
    print(f"probe group: layer={layer} proj={proj} shard={shard} experts={[x.expert for x in chosen]}")

    data_base, header = safetensor_header(args.repo, shard, args.revision, max_header)
    samples: list[tuple[TensorLoc, bytes]] = []
    for x in chosen:
        meta = header.get(x.name)
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            raise RuntimeError(f"tensor missing from shard header: {x.name}")
        data = tensor_sample(args.repo, shard, args.revision, data_base, meta, sample_bytes)
        samples.append((x, data))
        print(
            f"expert {x.expert:3d}: sample={len(data)/2**20:.2f} MiB "
            f"H={entropy_bits(data):.4f} bits/B zlib={zratio(data):.4f}"
        )

    # Compare every candidate to expert 0 in the chosen subset and also every pair; the
    # best pair is the optimistic exact-delta case a storage format could exploit.
    pair_rows = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            xa, a = samples[i]
            xb, b = samples[j]
            n = min(len(a), len(b))
            d = xor_bytes(a[:n], b[:n])
            zero = d.count(0) / n
            row = {
                "a": xa.expert,
                "b": xb.expert,
                "xor_zero": zero,
                "xor_entropy": entropy_bits(d),
                "xor_zlib": zratio(d),
                "raw_a_zlib": zratio(a[:n]),
                "raw_b_zlib": zratio(b[:n]),
            }
            pair_rows.append(row)
    pair_rows.sort(key=lambda r: (r["xor_zlib"], r["xor_entropy"]))
    best = pair_rows[0]
    med = statistics.median(r["xor_zlib"] for r in pair_rows)
    raw_med = statistics.median((r["raw_a_zlib"] + r["raw_b_zlib"]) / 2 for r in pair_rows)
    print(
        "best XOR pair:",
        f"{best['a']}->{best['b']}",
        f"zero={100*best['xor_zero']:.2f}%",
        f"H={best['xor_entropy']:.4f} bits/B",
        f"zlib={best['xor_zlib']:.4f}",
    )
    print(f"median XOR zlib={med:.4f}; median raw zlib={raw_med:.4f}; delta advantage={raw_med-med:+.4f}")

    # Conservative interpretation. A spectacular 15.6x checkpoint reduction would need
    # ~0.064 physical ratio overall. We do not extrapolate a tiny sample to the whole model;
    # this flag only says whether cross-expert XOR is promising enough to justify a larger
    # self-hosted probe.
    promising = best["xor_zlib"] <= 0.75 * raw_med or best["xor_zero"] >= 0.20
    result = {
        "schema": 1,
        "repo": args.repo,
        "revision": args.revision,
        "layer": layer,
        "projection": proj,
        "shard": shard,
        "sample_bytes_per_expert": sample_bytes,
        "experts": [x.expert for x, _ in samples],
        "pair_count": len(pair_rows),
        "best_pair": best,
        "median_xor_zlib": med,
        "median_raw_zlib": raw_med,
        "promising_for_larger_exact_probe": promising,
        "note": "sample-only; never extrapolate this directly to full-checkpoint size",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
