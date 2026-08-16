# OpenMP thread tuning

K3 is not a workload where “all logical CPUs” is automatically optimal.

At low RAM budgets, decode is dominated by storage traffic. At larger budgets, dense matvecs become memory-bandwidth bound. In both regimes, adding threads can stop helping or even reduce throughput because of memory-controller saturation, scheduling overhead, NUMA traffic, or contention with the asynchronous expert I/O workers.

For that reason this project now exposes an exact thread control and a **measurement-based sweep** rather than an `auto = CPU count` guess.

## Set an exact compute thread count

```bash
./bin/k3 /path/to/model \
  --trunk /path/to/trunk \
  --preset laptop \
  --incremental \
  --threads 8 \
  --ids 1,2,3 \
  --gen 8
```

`--threads N` calls OpenMP with dynamic team resizing disabled, so repeated runs actually use the requested compute-team limit. The effective count is printed in the banner and stored in the output JSON as `"threads"`.

This option changes scheduling only. It is not a precision/quality setting.

The asynchronous expert reader has a separate control because it runs from a different host thread:

```bash
K3_ASYNC_IO_THREADS=4 ./bin/k3 ... --threads 8
```

## Measure the best count on the real machine

Use the same model, trunk, memory preset and prompt that matter for your workload:

```bash
K3_SWEEP_REPEATS=3 benchmarks/thread-sweep.sh /path/to/model \
  --trunk /path/to/trunk \
  --preset laptop \
  --incremental \
  --ids 1,2,3 \
  --gen 4
```

By default the script tests powers of two plus the machine CPU count. Override the candidates when useful:

```bash
K3_SWEEP_THREADS="2 4 6 8 12 16" \
K3_SWEEP_REPEATS=3 \
benchmarks/thread-sweep.sh /path/to/model ...
```

The winner is the lowest **median `seconds_per_token`** across repeats.

The script also compares `generated_ids` on every run and **refuses to publish a recommendation if a thread count changes the token stream**.

## Keep the comparison controlled

Thread sweeps are meaningful only when everything else stays fixed:

- same checkpoint and packed trunk;
- same raw/lossless trunk representation;
- same memory preset / trunk and expert-cache budgets;
- same prompt and generation length;
- same SSD and background system load as much as practical;
- same async-expert settings.

For serious measurements use several generated tokens and at least 3 repeats. Very short tiny-model runs are useful correctness gates but are too noisy to tune a real 2.78T checkpoint.

## Validation

During development the exact same generated tiny checkpoint was run with `--threads 1`, `2`, and `4`:

- dumped logits compared byte-for-byte;
- generated IDs were identical (`[92, 168, 13]`);
- the sweep script itself was executed in CI and verified the token IDs before producing a recommendation;
- normal `make test`, Shellcheck, and the strict `-Werror` build passed.

The real K3 best thread count is intentionally **not hardcoded** because it depends on CPU, RAM topology, storage and memory budget.
