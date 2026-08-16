# Windows overlapped positioned-read benchmark — 2026-08

This is a narrow A/B for the Windows model-reader primitive, not an end-to-end K3
throughput claim.

## What changed

The first native-Windows port emulated `pread()` with `_lseeki64` + `_read` protected by a
per-descriptor SRW lock. That was exact, but two background expert reads targeting the
same safetensors shard could not overlap.

The new path opens model files with `FILE_FLAG_OVERLAPPED` and gives every `ReadFile`
request its own 64-bit `OVERLAPPED.Offset/OffsetHigh` and event. The Windows path remains
buffered; `FILE_FLAG_NO_BUFFERING` is deliberately not part of this measurement.

## Runner and workload

- GitHub-hosted `windows-latest`, Windows Server 2025 (10.0.26100)
- MSYS2 UCRT64 GCC
- one deterministic 512 MiB file
- 8 reader threads
- 8 MiB per positioned read
- 8 reads per thread, so every timed pass reads 512 MiB total
- both paths warmed before measurement
- five timed rounds with the order alternated
- both paths must produce the same checksum

Reproducer: `benchmarks/windows-pread-bench.c`.

## Result

```text
round 1 legacy 7648.8 MB/s  overlapped 9705.0 MB/s
round 2 legacy 8054.6 MB/s  overlapped 11884.7 MB/s
round 3 legacy 7752.0 MB/s  overlapped 11850.2 MB/s
round 4 legacy 7886.8 MB/s  overlapped 11415.0 MB/s
round 5 legacy 7717.3 MB/s  overlapped 11831.7 MB/s
median legacy     : 7752.0 MB/s
median overlapped : 11831.7 MB/s
overlapped/legacy : 1.526x
checksum parity   : PASS (536887232)
```

On this runner and this warm-cache same-file workload, removing the shared file-pointer
lock increased median aggregate read throughput by **1.526x**.

## What this does NOT prove

It does not mean K3 becomes 1.526x faster per token. The full model mixes trunk reads,
expert reads, cache hits, widening/decode and compute, and the real checkpoint is far
larger than the Windows page cache. The benchmark proves a narrower fact: same-shard
concurrent positioned reads were measurably serialized by the old compatibility path,
and the overlapped path removes that bottleneck without changing the bytes returned.

The permanent native-Windows model parity gate was also run against the overlapped backend
and remained token-identical for the tiny K3 one-shot engine, resident worker, seeded
sampling and Q4 speculative draft.