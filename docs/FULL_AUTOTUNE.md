# One-command full-machine autotune

`benchmarks/autotune_all.py` combines the exactness-guarded hardware and memory tuners so
a new machine can be tuned with one command instead of manually transferring winners
between separate sweeps.

It does **not** implement a third independent performance model. It orchestrates:

- `benchmarks/autotune.py` for compute threads, async expert-I/O threads and optional
  draft top-k;
- `benchmarks/autotune_memory.py` for the fixed-budget exact-trunk / expert-cache /
  optional draft-trunk split.

Every child report must contain the same `generated_ids` and `full_ids`. If any phase
observes a different exact K3 stream, the full run aborts and writes no final
recommendation.

## Typical draft-trunk run

```bash
python3 benchmarks/autotune_all.py ~/k3model \
  --allocator-budget-gb 24 \
  --trunk-min-gb 4 \
  --max-rss-gb 30 \
  --draft-topk-candidates auto \
  --cache-candidates auto \
  --draft-trunk-candidates auto \
  --repeats 2 \
  --max-cycles 2 -- \
  --trunk ~/k3trunk-lossless \
  --draft-trunk ~/k3draft --spec-auto --spec 8 \
  --ids 1008,10484,318,15383,387 --gen 12 --incremental --temperature 0
```

Native Windows uses the same structure, for example:

```powershell
python benchmarks/autotune_all.py C:\k3model `
  --k3-bin .\bin\k3.exe `
  --allocator-budget-gb 24 `
  --trunk-min-gb 4 `
  --max-rss-gb 30 `
  --repeats 2 --max-cycles 2 -- `
  --trunk C:\k3trunk `
  --draft-trunk C:\k3draft --spec-auto --spec 8 `
  --ids 1008,10484,318,15383,387 --gen 12 --incremental --temperature 0
```

The wrapper owns all performance knobs it coordinates. Do not pass these after the bare
`--`:

```text
--threads
--trunk-gb
--cache-gb
--draft-trunk-gb
--draft-topk
--preset
--out
```

The model/trunk paths, prompt, generation length, tokenizer/config options and exact K3
behavior stay in the K3 argument section after `--`.

## Search sequence

The full tuner starts from a valid seed memory split under the requested allocator
budget. The seed is only a starting point, never a recommendation.

Each cycle does:

1. **hardware/draft phase** — with memory fixed, run `autotune.py` to choose compute
   threads, async I/O and, when a draft trunk is present, exact-config-derived draft
   top-k;
2. **memory phase** — keep those hardware/top-k winners fixed and run
   `autotune_memory.py` under the exact same allocator budget;
3. compare the new discrete memory allocation with the one used by the hardware phase.

If both the memory allocation and hardware winner are stable, the orchestration stops
early. Otherwise it starts another cycle up to `--max-cycles`.

If the memory split moved in the final allowed cycle, the wrapper always performs one
additional **final hardware confirmation** on that exact final split. This means the
launch settings for threads/I/O/top-k were actually measured using the RAM allocation
that will be launched, even when the alternating search did not reach a full fixed point.
The final JSON exposes `converged` so this distinction is visible rather than hidden.

## Exactness across stages

Each child tuner already checks every candidate against its own first exact K3 run. The
wrapper adds a second layer: the reference token streams from **all child reports** must
also be identical across hardware and memory phases.

That catches accidental changes in request shape or command construction between stages.
The full report contains one shared:

```json
{
  "reference_generated_ids": [],
  "reference_full_ids": []
}
```

and every stage report remains on disk beside it for audit.

## Output

If the final report is `k3-full-autotune.json`, stage reports are written to:

```text
k3-full-autotune-stages/
```

The final `recommended` block includes:

- `threads`;
- `async_io_threads`;
- optional `draft_topk`;
- `trunk_gb`;
- `cache_gb`;
- optional `draft_trunk_gb`;
- allocator sum;
- measured RSS fields from the winning memory stage when available;
- `environment` containing `K3_ASYNC_IO_THREADS`;
- final `k3_args`;
- POSIX and Windows command-line renderings.

The final command intentionally does not add an output path so it can be used for normal
interactive/inference workloads rather than only benchmarking.

## Useful controls

```text
--allocator-budget-gb B       required fixed allocator budget
--trunk-min-gb T              minimum exact-trunk allocation
--max-rss-gb R                optional measured post-run RSS guard
--repeats N                   repeats in both child tuners
--max-cycles N                alternating hardware/memory cycles
--compute-candidates ...      optional explicit CPU thread candidates
--io-candidates ...           async expert-I/O candidates
--draft-topk-candidates auto  recommended when a draft trunk is active
--cache-candidates auto       automatic memory candidates
--draft-trunk-candidates auto automatic draft memory candidates
--auto-min-gb X               smallest automatic memory candidate
--hardware-strategy coordinate|grid
--memory-strategy coordinate|grid
--timeout SECONDS             per-child K3-run timeout
--keep-run-files              retain child candidate JSON/log files
```

`coordinate` is the default for both search families because exhaustive joint search on
the released checkpoint is expensive. `grid` remains available for controlled small
candidate sets.

## Interpreting convergence

`converged: true` means a cycle re-selected the memory split used by its own hardware
phase while the hardware choice was also stable relative to the previous cycle (or the
first cycle's seed memory was already optimal).

`converged: false` is not an exactness failure. It means the configured cycle budget was
reached while the discrete optimum was still moving. In that case the wrapper still
reconfirms hardware/top-k on the final memory split before reporting it. Increase
`--max-cycles` if you want a stricter fixed-point search.

## Recommended order now

For normal use, `autotune_all.py` is the preferred entry point. The two lower-level tools
remain useful when diagnosing one dimension or doing a controlled benchmark:

- use `autotune.py` when memory is already fixed and only CPU/I/O/draft behavior matters;
- use `autotune_memory.py` when hardware/top-k is already fixed and only the RAM split
  should change;
- use `autotune_all.py` when setting up a machine from scratch.

As with the lower-level tuners, CI tiny-model winners are functional checks only. Real K3
settings must be measured on the target CPU, RAM capacity, SSD/storage path, exact trunk
and draft trunk.