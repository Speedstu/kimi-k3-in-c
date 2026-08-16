# Async expert I/O overlap

This optimization targets the streamed routed-expert path. It changes **when independent work runs**, not model arithmetic, expert bytes, routing, or the order in which routed contributions are accumulated.

## Before

For each MoE layer/token, decode effectively did:

```text
route top-k
↓
down-project hidden -> latent
↓
read/prefetch all selected experts and WAIT
↓
compute routed experts
↓
latent norm + up projection
↓
compute shared expert
↓
add shared expert
```

`getmany(top-k)` already gives the SSD queue depth, but it is synchronous from the caller's point of view: the model waits for the batch to finish before doing any more independent compute.

## Now

When the expert source implements the optional async callbacks:

```text
route top-k
↓
launch top-k reads in background
↓
down-project hidden -> latent        ┐
shared expert MLP                    ├─ overlaps storage I/O
↓                                    ┘
wait for background batch
↓
compute routed experts
↓
latent norm + up projection
↓
add the already-computed shared expert
```

The **shared expert add stays in its original place** after the routed up projection. Only the independent shared-expert calculation moves earlier in wall-clock time. Routed expert processing and weighted accumulation remain in the same top-k order.

## Safety / fallback

`K3ExpertSrc` now has optional `prefetch_begin` / `prefetch_wait` callbacks.

- if every selected expert is already resident, `begin` returns without launching a thread;
- if async launch is unavailable/fails, the old synchronous `getmany` path is used;
- the model always joins the background batch before the first routed expert is accessed;
- cache destruction waits for any active batch;
- a pthread join failure aborts instead of risking a partially loaded expert being consumed.

The cache itself remains single-writer during a batch: model compute performed while the reader thread is active does not touch the cache.

## Controls

Disable only the async overlap while keeping normal batched prefetch:

```bash
K3_NOASYNC_PREFETCH=1 ./bin/k3 ...
```

Disable expert batch prefetch entirely (existing behavior/diagnostic switch):

```bash
K3_NOPREFETCH=1 ./bin/k3 ...
```

Set the OpenMP worker count used by the background I/O batch (default `4`, allowed `1..K3_MAX_TOPK`):

```bash
K3_ASYNC_IO_THREADS=4 ./bin/k3 ...
```

The end-of-run cache report prints the number of async batches and how long the caller still had to wait **after** doing the independent down/shared computation. On a real checkpoint, this is the useful quantity to compare with `K3_NOASYNC_PREFETCH=1`.

## Correctness validation

The branch is gated by:

- cache-level async prefetch test: every asynchronously published expert is compared byte-for-byte with the safetensors source;
- normal `make test` model/kernel suite;
- strict warnings-as-errors build;
- end-to-end generated tiny checkpoint: async enabled vs `K3_NOASYNC_PREFETCH=1` produce binary-identical dumped logits and identical token IDs.

The tiny checkpoint test is a correctness test, **not a meaningful performance benchmark** because its expert tensors are intentionally tiny.

## Performance status

The full released checkpoint is still required to quantify the end-to-end speedup. Real K3 touches 16 routed experts in each of 92 routed layers, and the useful gain depends on SSD latency/bandwidth and how much of each batch is already cached.

This optimization is most useful when expert reads still take longer than the independent down/shared compute. If the SSD/cache finishes first, `prefetch_wait` should be near zero and the overlap is effectively free apart from thread-launch overhead.
