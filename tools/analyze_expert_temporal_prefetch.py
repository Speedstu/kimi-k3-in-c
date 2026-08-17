#!/usr/bin/env python3
"""Measure how safely previous-token expert routes can predict the next K3 token.

Input is the committed real full-model expert trace: int32 (layer, expert) pairs in exact
request order. K3 routes top-16 experts through 92 routed layers, so 1472 requests form one
full decode-equivalent token in this trace.

No future information is used to construct the candidate policies. Metrics distinguish
coverage from precision because a wrong speculative disk read increases bytes/token even
though it cannot affect model output.
"""
from __future__ import annotations

import argparse
import os
import statistics
from array import array
from collections import Counter

TOPK = 16
ROUTED_LAYERS = 92
PER_TOKEN = TOPK * ROUTED_LAYERS
EXPERT_BYTES = 17_547_264


def load(path: str) -> list[list[set[int]]]:
    raw = array("i")
    with open(path, "rb") as f:
        raw.fromfile(f, os.path.getsize(path) // raw.itemsize)
    if len(raw) % 2:
        raise ValueError("odd int32 trace")
    pairs = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    if len(pairs) % PER_TOKEN:
        raise ValueError(f"{len(pairs)} requests is not divisible by {PER_TOKEN}")
    tokens: list[list[set[int]]] = []
    for base in range(0, len(pairs), PER_TOKEN):
        chunk = pairs[base : base + PER_TOKEN]
        order: list[int] = []
        by_layer: dict[int, set[int]] = {}
        for layer, expert in chunk:
            if layer not in by_layer:
                order.append(layer)
                by_layer[layer] = set()
            by_layer[layer].add(expert)
        if len(order) != ROUTED_LAYERS:
            raise ValueError(f"token has {len(order)} routed layers, expected {ROUTED_LAYERS}")
        rows = [by_layer[layer] for layer in order]
        if any(len(x) != TOPK for x in rows):
            bad = [len(x) for x in rows if len(x) != TOPK][:5]
            raise ValueError(f"layer top-k cardinality mismatch: {bad}")
        tokens.append(rows)
    return tokens


def policy_prev1(tokens, t, layer):
    return set(tokens[t - 1][layer])


def policy_intersection(tokens, t, layer, n):
    p = set(tokens[t - 1][layer])
    for lag in range(2, n + 1):
        p &= tokens[t - lag][layer]
    return p


def policy_count(tokens, t, layer, n, minimum):
    c = Counter()
    for lag in range(1, n + 1):
        c.update(tokens[t - lag][layer])
    return {e for e, k in c.items() if k >= minimum}


def evaluate(name, tokens, warmup, predict):
    correct = predicted = needed = false_positive = 0
    per_layer_precision = []
    per_layer_recall = []
    per_token_extra = []
    per_token_hidden = []
    for t in range(warmup, len(tokens)):
        extra_t = hidden_t = 0
        for layer in range(ROUTED_LAYERS):
            cur = tokens[t][layer]
            p = predict(tokens, t, layer)
            hit = len(cur & p)
            fp = len(p - cur)
            correct += hit
            predicted += len(p)
            needed += len(cur)
            false_positive += fp
            extra_t += fp
            hidden_t += hit
            if p:
                per_layer_precision.append(hit / len(p))
            per_layer_recall.append(hit / TOPK)
        per_token_extra.append(extra_t * EXPERT_BYTES / 1e9)
        per_token_hidden.append(hidden_t * EXPERT_BYTES / 1e9)
    precision = correct / predicted if predicted else 1.0
    recall = correct / needed if needed else 0.0
    base_gb = PER_TOKEN * EXPERT_BYTES / 1e9
    return {
        "name": name,
        "precision": precision,
        "recall": recall,
        "avg_predicted_per_layer": predicted / ((len(tokens) - warmup) * ROUTED_LAYERS),
        "extra_disk_gb_token": statistics.mean(per_token_extra),
        "potential_hidden_gb_token": statistics.mean(per_token_hidden),
        "baseline_expert_gb_token": base_gb,
        "disk_multiplier_if_prefetch_not_overlapped": 1.0 + statistics.mean(per_token_extra) / base_gb,
        "median_layer_precision": statistics.median(per_layer_precision) if per_layer_precision else 1.0,
        "median_layer_recall": statistics.median(per_layer_recall),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    args = ap.parse_args()
    tokens = load(args.trace)
    print(f"tokens={len(tokens)} layers/token={ROUTED_LAYERS} topk={TOPK}")
    policies = [
        ("prev1-all", 1, policy_prev1),
        ("intersection-2", 2, lambda x, t, l: policy_intersection(x, t, l, 2)),
        ("intersection-3", 3, lambda x, t, l: policy_intersection(x, t, l, 3)),
        ("seen>=2-of-3", 3, lambda x, t, l: policy_count(x, t, l, 3, 2)),
        ("seen>=3-of-4", 4, lambda x, t, l: policy_count(x, t, l, 4, 3)),
        ("seen>=2-of-4", 4, lambda x, t, l: policy_count(x, t, l, 4, 2)),
    ]
    print(
        f"{'policy':18s} {'prec':>7s} {'recall':>7s} {'pred/L':>7s} "
        f"{'hide GB/t':>10s} {'extra GB/t':>11s} {'disk x':>7s}"
    )
    results = []
    for name, warmup, fn in policies:
        r = evaluate(name, tokens, warmup, fn)
        results.append(r)
        print(
            f"{name:18s} {100*r['precision']:6.2f}% {100*r['recall']:6.2f}% "
            f"{r['avg_predicted_per_layer']:7.2f} {r['potential_hidden_gb_token']:10.2f} "
            f"{r['extra_disk_gb_token']:11.2f} {r['disk_multiplier_if_prefetch_not_overlapped']:7.3f}"
        )
    safe = [r for r in results if r["precision"] >= 0.90 and r["potential_hidden_gb_token"] >= 0.5]
    if safe:
        best = max(safe, key=lambda r: r["potential_hidden_gb_token"] - r["extra_disk_gb_token"])
        print("HIGH-PRECISION CANDIDATE", best)
    else:
        print("NO >=90% PRECISION PREFETCH POLICY WITH MEANINGFUL COVERAGE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
