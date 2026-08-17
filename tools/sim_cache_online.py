#!/usr/bin/env python3
"""Compare practical ONLINE expert-cache policies on a recorded K3 routing trace.

Unlike Belady, every policy in ONLINE_POLICIES uses only accesses that have already
happened. The model/router is never changed: this only decides which exact MXFP4 expert
bytes remain resident after they have been read once.

The purpose is to find a replacement/admission policy worth implementing in the C cache
before touching production inference. Equal-size expert objects make the comparison
especially clean.
"""
from __future__ import annotations

import argparse
import heapq
from array import array
from collections import Counter, OrderedDict, defaultdict, deque

EXPERT_BYTES = 17_547_264
PER_TOKEN = 16 * 92


def load_trace(path: str) -> list[int]:
    raw = array("i")
    with open(path, "rb") as f:
        raw.fromfile(f, __import__("os").path.getsize(path) // raw.itemsize)
    if len(raw) % 2:
        raise SystemExit("trace is not an even number of int32 values")
    return [(raw[i] << 20) | raw[i + 1] for i in range(0, len(raw), 2)]


def lru(trace: list[int], cap: int) -> int:
    q: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for k in trace:
        if k in q:
            hits += 1
            q.move_to_end(k)
        else:
            if len(q) >= cap:
                q.popitem(last=False)
            q[k] = None
    return hits


def slru(trace: list[int], cap: int, protected_frac: float = 0.80) -> int:
    """Segmented LRU: second touch promotes an object into a protected segment."""
    pcap = max(1, min(cap - 1, int(cap * protected_frac)))
    qcap = max(1, cap - pcap)
    probation: OrderedDict[int, None] = OrderedDict()
    protected: OrderedDict[int, None] = OrderedDict()
    hits = 0

    def trim_probation() -> None:
        while len(probation) > qcap:
            probation.popitem(last=False)

    for k in trace:
        if k in protected:
            hits += 1
            protected.move_to_end(k)
            continue
        if k in probation:
            hits += 1
            del probation[k]
            protected[k] = None
            if len(protected) > pcap:
                demote, _ = protected.popitem(last=False)
                probation[demote] = None
                trim_probation()
            continue
        probation[k] = None
        trim_probation()
    return hits


def tinylfu(trace: list[int], cap: int) -> int:
    """LRU residency plus an online frequency admission filter.

    The frequency sketch is represented exactly here (the real model has only ~82k
    expert keys, so exact 32-bit counters are sub-megabyte). A cold candidate cannot
    evict a resident item with a stronger observed history.
    """
    q: OrderedDict[int, None] = OrderedDict()
    freq: Counter[int] = Counter()
    hits = 0
    for k in trace:
        freq[k] += 1
        if k in q:
            hits += 1
            q.move_to_end(k)
            continue
        if len(q) < cap:
            q[k] = None
            continue
        victim = next(iter(q))
        if freq[k] > freq[victim]:
            del q[victim]
            q[k] = None
    return hits


def lfu(trace: list[int], cap: int) -> int:
    """Online LFU with recency tie-break and a lazy heap."""
    freq: Counter[int] = Counter()
    resident: set[int] = set()
    version: defaultdict[int, int] = defaultdict(int)
    heap: list[tuple[int, int, int, int]] = []
    hits = 0
    for t, k in enumerate(trace):
        freq[k] += 1
        version[k] += 1
        if k in resident:
            hits += 1
        elif len(resident) < cap:
            resident.add(k)
        else:
            while heap:
                _, _, v, victim = heapq.heappop(heap)
                if victim in resident and version[victim] == v:
                    resident.remove(victim)
                    break
            resident.add(k)
        heapq.heappush(heap, (freq[k], t, version[k], k))
    return hits


def lru2(trace: list[int], cap: int) -> int:
    """Online LRU-2: resist scans by evicting the oldest second-most-recent use.

    Keys seen only once are preferred victims. No future routing information is used.
    """
    hist: defaultdict[int, deque[int]] = defaultdict(lambda: deque(maxlen=2))
    resident: set[int] = set()
    version: defaultdict[int, int] = defaultdict(int)
    heap: list[tuple[int, int, int, int]] = []
    hits = 0

    def score(k: int) -> tuple[int, int]:
        h = hist[k]
        return (h[0] if len(h) >= 2 else -1, h[-1])

    for t, k in enumerate(trace):
        hist[k].append(t)
        version[k] += 1
        if k in resident:
            hits += 1
        elif len(resident) < cap:
            resident.add(k)
        else:
            while heap:
                _, _, v, victim = heapq.heappop(heap)
                if victim in resident and version[victim] == v:
                    resident.remove(victim)
                    break
            resident.add(k)
        s0, s1 = score(k)
        heapq.heappush(heap, (s0, s1, version[k], k))
    return hits


def gdf(trace: list[int], cap: int) -> int:
    """GreedyDual-Frequency with dynamic aging, specialized to equal-size experts."""
    resident: set[int] = set()
    version: defaultdict[int, int] = defaultdict(int)
    local_freq: defaultdict[int, int] = defaultdict(int)
    priority: defaultdict[int, float] = defaultdict(float)
    heap: list[tuple[float, int, int]] = []
    aging = 0.0
    hits = 0
    for k in trace:
        version[k] += 1
        if k in resident:
            hits += 1
            local_freq[k] += 1
            priority[k] = aging + local_freq[k]
        else:
            if len(resident) >= cap:
                while heap:
                    p, v, victim = heapq.heappop(heap)
                    if victim in resident and version[victim] == v:
                        resident.remove(victim)
                        aging = p
                        break
            resident.add(k)
            local_freq[k] = 1
            priority[k] = aging + 1.0
        heapq.heappush(heap, (priority[k], version[k], k))
    return hits


def layer_lru(trace: list[int], cap: int) -> int:
    """Equal per-layer LRU. Useful because a token revisits every MoE layer in order."""
    layers = sorted({k >> 20 for k in trace})
    base, extra = divmod(cap, len(layers))
    limits = {layer: base + (i < extra) for i, layer in enumerate(layers)}
    q = {layer: OrderedDict() for layer in layers}
    hits = 0
    for k in trace:
        layer = k >> 20
        d = q[layer]
        if k in d:
            hits += 1
            d.move_to_end(k)
        else:
            if len(d) >= limits[layer]:
                d.popitem(last=False)
            d[k] = None
    return hits


def layer_tinylfu(trace: list[int], cap: int) -> int:
    layers = sorted({k >> 20 for k in trace})
    base, extra = divmod(cap, len(layers))
    limits = {layer: base + (i < extra) for i, layer in enumerate(layers)}
    q = {layer: OrderedDict() for layer in layers}
    freq: Counter[int] = Counter()
    hits = 0
    for k in trace:
        layer = k >> 20
        d = q[layer]
        freq[k] += 1
        if k in d:
            hits += 1
            d.move_to_end(k)
            continue
        lim = limits[layer]
        if len(d) < lim:
            d[k] = None
            continue
        victim = next(iter(d))
        if freq[k] > freq[victim]:
            del d[victim]
            d[k] = None
    return hits


ONLINE_POLICIES = {
    "LRU": lru,
    "SLRU80": slru,
    "TinyLFU": tinylfu,
    "LFU": lfu,
    "LRU2": lru2,
    "GDF": gdf,
    "LayerLRU": layer_lru,
    "LayerTinyLFU": layer_tinylfu,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--disk-mbs", type=float, default=1234.0)
    ap.add_argument("--caps-gb", default="8,16,32,64,100,128")
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    args = ap.parse_args()
    trace = load_trace(args.trace)
    n = len(trace)
    ntok = n / PER_TOKEN
    caps = [float(x) for x in args.caps_gb.split(",") if x.strip()]
    print(f"requests={n} distinct={len(set(trace))} equivalent_tokens={ntok:.2f}")
    print("All tournament policies are ONLINE: past accesses only; routing/output unchanged.\n")

    header = f"{'CACHE':>7} {'POLICY':>14} {'HIT%':>8} {'GB/TOK':>9} {'IO S/TOK':>10} {'vs LRU':>9}"
    print(header)
    print("-" * len(header))
    winners = {}
    for gb in caps:
        cap = max(1, int(gb * 1e9 // args.expert_bytes))
        results = {}
        baseline = lru(trace, cap)
        for name, fn in ONLINE_POLICIES.items():
            h = fn(trace, cap)
            results[name] = h
            miss = n - h
            gb_tok = miss * args.expert_bytes / 1e9 / ntok
            sec = gb_tok * 1000.0 / args.disk_mbs
            delta = 100.0 * (h - baseline) / n
            print(f"{gb:6.0f}G {name:>14} {100*h/n:7.2f}% {gb_tok:9.2f} {sec:10.2f} {delta:+8.2f}p")
        winner = max(results, key=results.get)
        winners[gb] = (winner, results[winner], baseline)
        print(f"  -> winner {winner}: {100*results[winner]/n:.2f}% ({100*(results[winner]-baseline)/n:+.2f} points vs LRU)\n")

    print("SUMMARY")
    for gb, (name, h, base) in winners.items():
        saved = (h - base) * args.expert_bytes / 1e9 / ntok
        saved_s = saved * 1000.0 / args.disk_mbs
        print(f"  {gb:>5.0f} GB: {name:<14} {100*h/n:6.2f}%  saves {saved:5.2f} GB/token ~= {saved_s:5.2f} I/O s/token vs LRU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
