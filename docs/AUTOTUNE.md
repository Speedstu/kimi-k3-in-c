# Real-hardware autotuning

K3's fastest thread settings are a property of the **whole machine**, not just the CPU.
The main OpenMP team competes for memory bandwidth while asynchronous expert readers
compete for NVMe queue depth and CPU time. After the exact MXFP4 and expert-I/O changes,
a universal `--threads N` or `K3_ASYNC_IO_THREADS=N` default is therefore less useful
than measuring the real request on the real storage path.

`benchmarks/autotune.py` tunes both knobs while keeping the model request unchanged.
It changes only:

- `--threads N` for the main OpenMP compute team;
- `K3_ASYNC_IO_THREADS=N` for the background expert-read team.

It does **not** change weights, precision, routing, cache size, prompt, sampling options,
or the exact-vs-draft model path.

## Quick run

Use a short deterministic request so each candidate is affordable. Two generated tokens
are usually enough to measure the decode path; use the same trunk, cache/preset and
storage path that you intend to use afterward.

Linux / WSL:

```bash
python3 benchmarks/autotune.py ~/k3model --repeats 2 -- \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --ids 1008,10484,318,15383,387 --gen 2 --temperature 0
```

Native Windows, when using the native `k3.exe` build:

```powershell
python benchmarks/autotune.py C:\k3model --k3-bin .\bin\k3.exe --repeats 2 -- `
  --trunk C:\k3trunk --preset laptop --incremental `
  --ids 1008,10484,318,15383,387 --gen 2 --temperature 0
```

The bare `--` is required: options before it belong to the tuner; options after it are
passed unchanged to K3.

A successful run ends with a recommendation similar to:

```text
RECOMMENDED
  --threads N
  K3_ASYNC_IO_THREADS=M
```

On Windows it also prints the PowerShell and `cmd.exe` forms for the environment
variable. The exact `N` and `M` must come from your machine; values from CI or another
computer are not transferable recommendations.

## Correctness guard

Every candidate executes a normal K3 request and writes its ordinary JSON result. The
tuner compares both `generated_ids` and `full_ids` against the first run. If either
changes at any candidate or repeat, the tuner stops and **refuses to recommend a
setting**.

This is intentionally stricter than comparing only elapsed time. Thread tuning is not
allowed to buy speed by silently changing the token stream.

The permanent CI gate exercises the tuner against a streamed tiny K3 checkpoint across
multiple compute/I/O candidates and verifies the expected greedy IDs.

## Search strategy

The default `coordinate` strategy is designed for a 1.56 TB model where every real run
can be expensive:

1. sweep compute threads at a seed I/O count;
2. sweep I/O threads at the best compute count;
3. sweep compute again at the best I/O count;
4. if that changes the compute winner, confirm I/O one more time.

Candidate order reverses between repeats to reduce simple thermal/order bias.

For a smaller search space or a machine where measurement cost is acceptable, use the
exhaustive Cartesian grid:

```bash
python3 benchmarks/autotune.py ~/k3model --strategy grid --repeats 3 -- \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --ids 1008,10484,318,15383,387 --gen 2 --temperature 0
```

Useful tuner options:

```text
--compute-candidates 1,2,4,8,16
--io-candidates 1,2,4,8
--repeats 3
--out k3-autotune.json
--keep-run-files
--timeout 1200
```

Without `--compute-candidates`, the tuner tries powers of two plus the machine's logical
CPU count. The default I/O candidates are `1,2,4,8,16`.

## Reading the result

The JSON report records every measured candidate and the final pair. Apply both knobs to
your normal generation command. On Linux/WSL, for example:

```bash
K3_ASYNC_IO_THREADS=M ./bin/k3 ~/k3model \
  --trunk ~/k3trunk-lossless --preset laptop --incremental --threads N \
  --ids 1008,10484,318,15383,387 --gen 8
```

Re-run the tuner after changing a major condition that affects the balance, especially:

- SSD/NVMe or storage location;
- cache or trunk memory budget;
- preset;
- CPU or memory configuration;
- a major kernel/I/O implementation change.

Avoid running the sweep while a large download, antivirus scan, backup, game update, or
other storage-heavy task is active; that would tune K3 to transient contention instead
of normal use.

## Limits

Autotuning does not remove the model's storage requirement. The released checkpoint is
still about **1.56 TB**, and low-RAM inference still depends heavily on local NVMe
bandwidth and latency. It also cannot substitute for a full-checkpoint benchmark: the CI
tiny model proves the tuner and exactness guard work, not what thread pair will win on a
particular real K3 machine.

`benchmarks/thread-sweep.sh` remains useful when only the compute thread count needs to be
measured. `benchmarks/autotune.py` is the preferred path when async expert streaming is
active because it measures the interaction between compute and I/O teams.
