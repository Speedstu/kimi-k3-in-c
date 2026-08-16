# Async expert I/O overlap

This optimization targets the streamed routed-expert path. It changes **when independent work runs**, not model arithmetic, expert bytes, routing, or the order in which routed contributions are accumulated.

## Before

For each MoE layer/token, the first async version effectively did:

```text
route top-k
↓
launch all selected expert reads in background
↓
down-project hidden -> latent        ┐
shared expert MLP                    ├─ overlaps storage I/O
↓                                    ┘
WAIT FOR THE WHOLE expert batch
↓
compute routed expert 0
compute routed expert 1
...
compute routed expert K-1
↓
latent norm + up projection
↓
add shared expert
```

Batching already gives the SSD queue depth, but the whole-batch barrier leaves a bubble when some experts have arrived and a few stragglers are still loading. The CPU cannot start the routed work even though exact weights for the next top-k expert may already be available.

## Now: publish as ready, consume in exact order

The cache reserves all missing top-k slots before starting the background readers. Each expert is published independently as soon as **its own** read finishes.

```text
route top-k
↓
reserve missing top-k slots + launch reads
↓
down-project hidden -> latent        ┐
shared expert MLP                    ├─ overlaps storage I/O
↓                                    ┘
wait only for expert 0 -> compute expert 0   ┐
wait only for expert 1 -> compute expert 1   │ later reads keep running
...                                           ├ while routed MXFP4 compute runs
wait only for expert K-1 -> compute K-1      ┘
↓
join the already-nearly-finished batch
↓
latent norm + up projection
↓
add the already-computed shared expert
```

The important exactness property is that routed computation is still consumed and accumulated in the original `j = 0..K-1` top-k order. Read completion may be out of order; **model arithmetic is not**. The three MXFP4 matmuls for expert `j` can therefore hide storage latency for `j+1..K-1` without changing the mixture.

The shared expert add also stays in its original place after the routed up projection. Only the independent shared-expert calculation moves earlier in wall-clock time.

## API and synchronization

`K3ExpertSrc` has three optional async callbacks:

- `prefetch_begin(...)` reserves the misses and launches the background batch;
- `prefetch_get(..., expert, ...)` waits for one requested expert only and returns the exact cache bytes once published;
- `prefetch_wait(...)` performs the final batch join.

The cache's publication metadata is protected by a mutex/condition variable. A reader publishes `ref`, padding, slot ownership and LRU metadata under that lock, then wakes the model thread. The model uses the same lock while turning a ready slot into a normal cache `get`, so `slot_of`, `key_of`, `used_at` and the cache clock cannot race.

Every missing expert is assigned a distinct `K3_SLOT_INFLIGHT` slot **before** the reader thread starts. This prevents a later top-k miss from being assigned the buffer currently receiving another expert.

## Safety / fallback

- if every selected expert is already resident, `begin` returns without launching a thread;
- if async launch is unavailable/fails, the synchronous `getmany`/`get` path remains the fallback;
- if a per-expert asynchronous read fails, exact inference falls back to the normal synchronous `get` rather than dropping that expert;
- cache destruction joins any active batch before freeing its buffers or synchronization objects;
- a pthread join failure aborts instead of risking partially loaded weights;
- sources that do not implement `prefetch_get` retain the previous whole-batch barrier behavior.

## Controls

Disable only the new per-expert pipeline, while keeping the older asynchronous whole-batch overlap. This is the clean A/B switch because both modes use the **same binary**:

```bash
K3_NO_EXPERT_PIPELINE=1 ./bin/k3 ...
```

Disable async overlap entirely while keeping normal batched prefetch:

```bash
K3_NOASYNC_PREFETCH=1 ./bin/k3 ...
```

Disable expert batch prefetch entirely (existing diagnostic switch):

```bash
K3_NOPREFETCH=1 ./bin/k3 ...
```

Set the OpenMP worker count used by the background I/O batch (default `4`, allowed `1..K3_MAX_TOPK`):

```bash
K3_ASYNC_IO_THREADS=4 ./bin/k3 ...
```

The end-of-run cache report separates two waits:

- **next-expert waits**: time spent waiting for the next top-k expert before its MXFP4 work can start;
- **final join wait**: residual time left when all routed experts have already been consumed.

Those measurements are more informative than a raw device MB/s figure when tuning a real checkpoint.

## Correctness validation

The per-expert pipeline is gated by:

- a cache stress test under eviction pressure: 24 pipelined batches were launched and every expert handed to the model matched the safetensors source byte-for-byte;
- a strict warnings-as-errors build;
- end-to-end generated tiny checkpoint: pipeline enabled vs `K3_NO_EXPERT_PIPELINE=1` produced **binary-identical dumped logits** and the same eight greedy token IDs (`92,168,13,3,49,208,214,208`);
- the full weightless model/kernel suite, including the full-model oracle.

The tiny checkpoint is a correctness test, **not a meaningful performance benchmark** because its expert tensors are intentionally tiny.

## Performance status

The full released checkpoint is still required to quantify the end-to-end gain. Real K3 touches 16 routed experts in each routed MoE layer and one packed expert is about 17.55 MB, so whether this helps by a little or a lot depends on NVMe latency/bandwidth, the chosen I/O worker count, cache residency, and the ratio of routed MXFP4 compute time to remaining read time.

Do not infer a full-K3 tokens/second factor from the tiny gate. On real hardware, compare the same command with and without `K3_NO_EXPERT_PIPELINE=1`, repeat the runs, and use the reported wait counters to verify that storage latency is actually being hidden.