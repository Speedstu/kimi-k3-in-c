#!/usr/bin/env python3
"""Usage-weighted spherical clustering for K3 routed-expert compression.

The expensive teacher run should emit one compact behaviour sketch per expert for one
routed layer (for example random projections of expert outputs on a held-out training
stream) plus routing usage counts. This tool turns 896 teacher experts into K student
clusters and selects a REAL teacher medoid for initialization. It never averages packed
MXFP4 bytes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cluster_experts(
    sketches: np.ndarray,
    usage: np.ndarray,
    k: int,
    *,
    iterations: int = 40,
) -> dict[str, object]:
    sketches = np.asarray(sketches, dtype=np.float64)
    usage = np.asarray(usage, dtype=np.float64)
    if sketches.ndim != 2:
        raise ValueError("sketches must be [experts, sketch_dim]")
    n, _ = sketches.shape
    if usage.shape != (n,):
        raise ValueError("usage must be [experts]")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and number of experts")
    if np.any(usage < 0) or not np.all(np.isfinite(usage)):
        raise ValueError("usage must be finite and non-negative")
    if not np.all(np.isfinite(sketches)):
        raise ValueError("sketches must be finite")

    x = _normalize(sketches)
    w = usage + max(float(usage.max(initial=0.0)) * 1e-6, 1e-9)

    # Deterministic weighted farthest-first seeding. The busiest expert anchors the
    # first cluster, then every new seed covers behaviour not represented yet.
    seeds = [int(np.argmax(w))]
    best_sim = x @ x[seeds[0]]
    for _ in range(1, k):
        distance = np.clip(1.0 - best_sim, 0.0, 2.0)
        score = distance * np.sqrt(w)
        score[np.asarray(seeds, dtype=np.int64)] = -1.0
        nxt = int(np.argmax(score))
        seeds.append(nxt)
        best_sim = np.maximum(best_sim, x @ x[nxt])

    centers = x[np.asarray(seeds, dtype=np.int64)].copy()
    assign = np.full(n, -1, dtype=np.int64)

    for _ in range(iterations):
        sims = x @ centers.T
        new_assign = np.argmax(sims, axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        new_centers = np.zeros_like(centers)
        for cid in range(k):
            members = np.flatnonzero(assign == cid)
            if members.size == 0:
                # Re-seed with the worst represented expert, preserving determinism.
                represented = np.max(sims, axis=1)
                replacement = int(np.argmin(represented))
                assign[replacement] = cid
                members = np.array([replacement], dtype=np.int64)
            weighted = x[members] * w[members, None]
            center = weighted.sum(axis=0, keepdims=True)
            new_centers[cid] = _normalize(center)[0]
        centers = new_centers

    clusters: list[dict[str, object]] = []
    mapping = np.empty(n, dtype=np.int64)
    for cid in range(k):
        members = np.flatnonzero(assign == cid)
        if members.size == 0:
            raise RuntimeError("empty cluster after convergence")
        mapping[members] = cid
        sim = x[members] @ centers[cid]
        # Medoid remains an actual teacher expert; usage is only a gentle tie-breaker.
        weighted_score = sim + 1e-3 * np.log1p(w[members])
        medoid = int(members[int(np.argmax(weighted_score))])
        clusters.append(
            {
                "student_expert": cid,
                "teacher_medoid": medoid,
                "teacher_members": [int(v) for v in members],
                "usage": float(usage[members].sum()),
                "mean_cosine_to_center": float(sim.mean()),
            }
        )

    return {
        "teacher_experts": n,
        "student_experts": k,
        "mapping": [int(v) for v in mapping],
        "clusters": clusters,
        "usage_covered": float(usage.sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="NPZ containing sketches[E,D] and usage[E]")
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = np.load(args.npz)
    result = cluster_experts(data["sketches"], data["usage"], args.k, iterations=args.iterations)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"clustered {result['teacher_experts']} -> {result['student_experts']} experts; "
        f"usage={result['usage_covered']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
