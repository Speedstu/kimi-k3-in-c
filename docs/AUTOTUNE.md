# Real-hardware autotuning

K3's fastest settings are a property of the **whole machine**, not just the CPU. The
main OpenMP team competes for memory bandwidth while asynchronous expert readers compete
for NVMe queue depth and CPU time. When an exact-verified draft trunk is active, its
routed-expert count adds a third tradeoff: lower draft top-k is cheaper, but a worse
draft can waste exact verification work through lower acceptance.

`benchmarks/autotune.py` always tunes the two hardware knobs:

- `--threads N` for the main OpenMP compute team;
- `K3_ASYNC_IO_THREADS=N` for the background expert-read team.

With `--draft-topk-candidates`, it can additionally tune:

- `--draft-topk K` for the **draft model only**.

Draft top-k tuning requires `--draft-trunk` in the K3 command. It never changes the exact
K3 routing: the exact model still verifies every emitted token. The tuner does **not**
change weights, precision, cache size, prompt, sampling options, or the exact model.

## Quick run

Use a short deterministic request so each candidate is affordable. Two generated tokens
are usually enough for thread-only tuning; draft top-k tuning benefits from a somewhat
longer decode so `--spec-auto` can measure more than one speculative round. Use the same
trunk, draft trunk, cache/preset and storage paths that you intend to use afterward.

Linux / WSL, threads + I/O only:

```bash
python3 benchmarks/autotune.py ~/k3model --repeats 2 -- \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --ids 1008,10484,318,15383,387 --gen 2 --temperature 0
```

Linux / WSL, including draft top-k:

```bash
python3 benchmarks/autotune.py ~/k3model --repeats 2 \
  --draft-topk-candidates 1,2,4 -- \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --draft-trunk ~/k3draft --draft-trunk-gb 6 --spec-auto --spec 8 \
  --ids 1008,10484,318,15383,387 --gen 12 --temperature 0
```

Native Windows, when using the native `k3.exe` build:

```powershell
python benchmarks/autotune.py C:\k3model --k3-bin .\bin\k3.exe --repeats 2 `
  --draft-topk-candidates 1,2,4 -- `
  --trunk C:\k3trunk --preset laptop --incremental `
  --draft-trunk C:\k3draft --draft-trunk-gb 6 --spec-auto --spec 8 `
  --ids 1008,10484,318,15383,387 --gen 12 --temperature 0
```

The bare `--` is required: options before it belong to the tuner; options after it are
passed to K3. When draft top-k tuning is enabled, do not pass a manual `--draft-topk`
after the separator; the tuner owns that option and will refuse an ambiguous command.

A successful three-knob run ends with a recommendation similar to:

```text
RECOMMENDED
  --threads N
  --draft-topk K
  K3_ASYNC_IO_THREADS=M
```

On Windows it also prints the PowerShell and `cmd.exe` forms for the environment
variable. The exact `N`, `M` and `K` must come from your machine and draft: values from
CI or another computer are not transferable recommendations. The tiny CI result is only
a functional test of the tuner and must never be copied as a real-K3 hardware
recommendation.

## Correctness guard

Every candidate executes a normal K3 request and writes its ordinary JSON result. The
tuner compares both `generated_ids` and `full_ids` against the first run. If either
changes at any candidate or repeat, the tuner stops and **refuses to recommend a
setting**.

This is especially important for draft top-k: changing draft quality is allowed to
change proposals and acceptance, but it is not allowed to change the exact model's
committed token stream. Exact K3 remains the verifier for every candidate.

The permanent CI gate exercises both modes against a streamed tiny K3 checkpoint:
thread/I/O tuning preserves the known greedy token stream, and draft-topk tuning sweeps
multiple draft routing counts while requiring the same exact output.

## Search strategy

The default `coordinate` strategy is designed for a 1.56 TB model where every real run
can be expensive.

Without draft top-k tuning it keeps the original search:

1. sweep compute threads at a seed I/O count;
2. sweep I/O threads at the best compute count;
3. sweep compute again at the best I/O count;
4. if that changes the compute winner, confirm I/O one more time.

With `--draft-topk-candidates`, the tuner uses a three-dimensional coordinate search:

1. choose the candidate nearest `--draft-topk-seed` (default 4);
2. sweep compute threads and I/O threads at that seed top-k;
3. sweep all draft top-k candidates at the winning thread pair;
4. re-sweep compute and I/O using the winning top-k, because a cheaper/heavier draft can
   change CPU and storage contention;
5. if the winning thread pair moved, confirm draft top-k once at that final pair.

Candidate order reverses between repeats to reduce simple thermal/order bias.

For a smaller search space or a machine where measurement cost is acceptable, `grid`
exhaustively tests the Cartesian product of every enabled dimension. With draft top-k
enabled that means `compute × I/O × draft-topk`:

```bash
python3 benchmarks/autotune.py ~/k3model --strategy grid --repeats 3 \
  --compute-candidates 4,8 --io-candidates 2,4 \
  --draft-topk-candidates 1,2,4 -- \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --draft-trunk ~/k3draft --spec-auto --spec 8 \
  --ids 1008,10484,318,15383,387 --gen 12 --temperature 0
```

Useful tuner options:

```text
--compute-candidates 1,2,4,8,16
--io-candidates 1,2,4,8
--draft-topk-candidates 1,2,4
--draft-topk-seed 4
--repeats 3
--out k3-autotune.json
--keep-run-files
--timeout 1200
```

Without `--compute-candidates`, the tuner tries powers of two plus the machine's logical
CPU count. The default I/O candidates are `1,2,4,8,16`. Draft top-k is **not tuned by
default**: omitting `--draft-topk-candidates` preserves the previous thread/I/O-only
behavior exactly.

## Reading the result

The JSON report records every measured candidate, including `draft_topk` for each run
when that dimension is enabled. Apply all recommended knobs to your normal generation
command. On Linux/WSL, for example:

```bash
K3_ASYNC_IO_THREADS=M ./bin/k3 ~/k3model \
  --trunk ~/k3trunk-lossless --preset laptop --incremental --threads N \
  --draft-trunk ~/k3draft --draft-topk K --spec-auto --spec 8 \
  --ids 1008,10484,318,15383,387 --gen 32
```

Re-run the tuner after changing a major condition that affects the balance, especially:

- SSD/NVMe or storage location;
- cache or trunk memory budget;
- preset;
- CPU or memory configuration;
- draft trunk or draft quantization;
- a major kernel/I/O implementation change.

Avoid running the sweep while a large download, antivirus scan, backup, game update, or
other storage-heavy task is active; that would tune K3 to transient contention instead
of normal use.

## Limits

Autotuning does not remove the model's storage requirement. The released checkpoint is
still about **1.56 TB**, and low-RAM inference still depends heavily on local NVMe
bandwidth and latency. It also cannot substitute for a full-checkpoint benchmark: the CI
tiny model proves the tuner and exactness guard work, not what thread/I/O/top-k
combination will win on a particular real K3 machine.

`benchmarks/thread-sweep.sh` remains useful when only the compute thread count needs to be
measured. `benchmarks/autotune.py` is the preferred path when async expert streaming or
a draft trunk is active because it measures the interaction between the enabled
performance knobs.