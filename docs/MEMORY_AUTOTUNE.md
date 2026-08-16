# Fixed-budget memory autotuning

K3 exposes three RAM dials when a draft trunk is active:

- `--trunk-gb`: lossless exact-model trunk residency;
- `--cache-gb`: routed-expert cache;
- `--draft-trunk-gb`: draft-trunk residency.

Giving more RAM to one of them necessarily leaves less RAM for the others on a fixed
machine. `benchmarks/autotune_memory.py` compares allocations at the **same allocator
budget** and refuses any result whose exact K3 token stream changes.

## What the budget means

`--allocator-budget-gb B` constrains the K3 allocator knobs, not the whole operating
system process:

```text
trunk_gb + cache_gb + draft_trunk_gb = B
```

For every candidate, the tuner chooses cache and optional draft-trunk budgets and assigns
**all remaining budget to the exact lossless trunk**. That keeps memory cost constant
between candidates and prioritizes the data that the exact model must read every token.

The whole process also contains embeddings, KV state, scratch, manifests and other fixed
overhead. Therefore allocator budget is not a hard RSS limit. The tuner parses K3's own
`PEAK RSS for the whole run` line and records the measured peak RSS of every candidate so
the report shows the real process footprint observed on that machine.

## Machine-fit mode

For the least manual setup, use automatic candidates plus an observed RSS ceiling:

```bash
python3 benchmarks/autotune_memory.py ~/k3model \
  --allocator-budget-gb 24 \
  --trunk-min-gb 4 \
  --cache-candidates auto \
  --draft-trunk-candidates auto \
  --max-rss-gb 30 \
  --repeats 2 -- \
  --trunk ~/k3trunk-lossless \
  --draft-trunk ~/k3draft --draft-topk K --spec-auto --spec 8 \
  --threads N \
  --ids 1008,10484,318,15383,387 --gen 12 --incremental --temperature 0
```

`auto` does not encode a claim such as “a certain cache size is always optimal”. It
simply creates a geometric set starting at `--auto-min-gb` (default 0.25 GB), doubles it
while space remains, and includes the upper boundary. Candidate pairs that would leave
less than `--trunk-min-gb` for the exact trunk are removed before K3 runs.

`--max-rss-gb` is deliberately a **post-run recommendation guard**. An allocation is
eligible only if every observed run of that exact allocation reports peak RSS at or below
the ceiling. If K3 does not print its RSS diagnostic while the guard is active, tuning
fails instead of pretending the limit was checked. This is not a preventive memory
sandbox: it cannot stop an allocation from OOMing before the process reaches the final
RSS report. Use `--allocator-budget-gb` conservatively enough to leave OS/application
headroom; use `--max-rss-gb` to reject measured configurations that still exceed your
target whole-process footprint.

The JSON report makes the distinction auditable:

```json
{
  "allocator_budget_gb": 24,
  "max_rss_gb": 30,
  "cache_candidates_auto": true,
  "draft_trunk_candidates_auto": true,
  "recommended": {
    "allocator_sum_gb": 24,
    "median_peak_rss_gb": 28.4,
    "max_peak_rss_gb": 28.9
  }
}
```

Every individual run also records `within_max_rss` when a ceiling is active.

## Explicit candidates with a draft trunk

For controlled experiments, explicit lists remain available:

```bash
python3 benchmarks/autotune_memory.py ~/k3model \
  --allocator-budget-gb 24 \
  --cache-candidates 0.5,1,4,8 \
  --draft-trunk-candidates 1,2,4,6,8 \
  --trunk-min-gb 4 \
  --repeats 2 -- \
  --trunk ~/k3trunk-lossless \
  --draft-trunk ~/k3draft --draft-topk K --spec-auto --spec 8 \
  --threads N \
  --ids 1008,10484,318,15383,387 --gen 12 --incremental --temperature 0
```

The tuner controls `--trunk-gb`, `--cache-gb`, and `--draft-trunk-gb`. Do not pass those
manually. Do not pass `--preset` either, because a preset would compete with the explicit
budgets being measured.

A recommendation looks like:

```text
RECOMMENDED MEMORY ALLOCATION
  --trunk-gb T
  --cache-gb C
  --draft-trunk-gb D
  fixed allocator sum: 24.000 GB
  observed median peak RSS: ... GB
  observed max peak RSS: ... GB
  observed median: ... s/token
```

The numerical winner is machine-, SSD-, draft- and request-specific. CI numbers are only
functional tests and are not full-K3 recommendations.

## Without a draft trunk

The same tool can tune exact trunk versus expert cache only. Automatic cache candidates
work here too:

```bash
python3 benchmarks/autotune_memory.py ~/k3model \
  --allocator-budget-gb 24 \
  --cache-candidates auto \
  --trunk-min-gb 4 --max-rss-gb 30 --repeats 2 -- \
  --trunk ~/k3trunk-lossless --threads N \
  --ids 1008,10484,318,15383,387 --gen 8 --incremental --temperature 0
```

Here `draft_trunk_gb` is absent and every candidate satisfies:

```text
trunk_gb + cache_gb = allocator_budget_gb
```

## Search strategy

The default `coordinate` strategy avoids a large Cartesian product on the 1.56 TB real
checkpoint.

With a draft trunk it:

1. sweeps cache candidates at a seed draft budget;
2. ignores allocations that violate the measured RSS ceiling, when one is active;
3. sweeps draft-trunk candidates at the best eligible cache budget;
4. if the draft winner changed, re-confirms cache with that draft budget.

An allocation measured above `--max-rss-gb` is never selected as the winner. If no
measured allocation in a required phase fits the ceiling, tuning fails and writes no
recommendation rather than silently relaxing the limit.

The exact-trunk budget is always the remainder. Candidate order reverses between repeats
to reduce simple thermal/order bias.

`--strategy grid` tests every valid cache × draft-trunk pair under the same total budget.
This is useful for small candidate sets or controlled benchmarking, but can be expensive
on the full model. In grid mode the RSS ceiling likewise removes over-limit allocations
from recommendation eligibility after they are measured.

Useful options:

```text
--allocator-budget-gb 24
--trunk-min-gb 4
--cache-candidates auto
--draft-trunk-candidates auto
--auto-min-gb 0.25
--max-rss-gb 30
--cache-seed-gb 1
--draft-seed-gb 4
--strategy coordinate
--repeats 2
--timeout 1800
--keep-run-files
--out k3-memory-autotune.json
```

Explicit lists such as `--cache-candidates 0.5,1,4,8,16` and
`--draft-trunk-candidates 1,2,4,6,8` remain supported. Candidates that leave less than
`--trunk-min-gb` for the exact trunk are removed before any K3 run starts.

## Exactness guard

Every allocation executes a complete K3 request. The first run establishes the reference
`generated_ids` and `full_ids`. Every later run must match both exactly. A single mismatch
causes an immediate failure and **no recommendation is written**.

This allows draft-trunk residency or expert-cache behavior to change performance without
letting either approximate mechanism become authoritative. The exact K3 model remains the
source of every committed token.

## Recommended tuning order

For a new machine, a practical order is:

1. choose an allocator memory budget that leaves enough RAM for the OS and non-K3 work;
2. run `benchmarks/autotune.py` to tune threads, async I/O and optional draft top-k;
3. keep those winners fixed and run `benchmarks/autotune_memory.py` with automatic memory
   candidates and, if useful, a conservative measured `--max-rss-gb` target;
4. optionally re-run `benchmarks/autotune.py` once using the winning RAM split, because a
   large memory-allocation change can alter I/O/CPU contention.

Re-run memory tuning after changing the SSD/storage path, total RAM budget, RSS target,
exact trunk, draft trunk or draft quantization.