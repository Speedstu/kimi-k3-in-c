# Native Windows runtime

K3 has a native Windows x64 path through **MSYS2 UCRT64 + GCC**. It builds and runs
`bin/k3.exe` and `bin/k3-worker.exe` directly on Windows; WSL is not required for
correctness.

The Windows gate runs the same tiny K3 graph through the one-shot engine and the resident
worker, checks warm-prefix reuse and reset, checks seeded temperature/top-p sampling, and
checks a Q4 speculative draft whose output is still verified by the exact model. The
emitted token ids must match.

## What is tested

The permanent `Native Windows parity` workflow runs on GitHub's Windows runner with an
MSYS2 UCRT64 toolchain and requires all of the following:

- a warning-clean native build (`-Werror`) on the portable SSE2 path;
- every weightless kernel/model oracle test;
- a native tiny safetensors checkpoint generated and read on Windows;
- packed-trunk streaming;
- `bin/k3.exe` versus `bin/k3-worker.exe` token parity;
- a warm second turn with the expected number of prefix tokens reused;
- a divergent prompt/reset path;
- seeded `temperature=1` / top-p sampling parity;
- a resident Q4 speculative draft on two turns, still token-identical to the exact
  one-shot verifier.

A second `Windows lazy KV parity` gate starts both the exact worker and resident Q4 draft
with `--context 1048576`, verifies that the huge MLA address ranges are reserve-only at
startup, executes a sampled turn, decommits on `RESET`, recommits, and requires the same
exact output again.

The CI host is Windows Server 2025. The runtime uses ordinary Win32/UCRT APIs available on
modern x64 Windows, but Windows 10/11 client editions are not separate CI targets yet.

## Build

Install MSYS2, open an **UCRT64** shell, then install the toolchain:

```bash
pacman -S --needed git make mingw-w64-ucrt-x86_64-gcc
```

Clone the repository and build normally:

```bash
git clone <your-repository>
cd kimi-k3-in-c
make -j
make test
```

The binaries are native PE executables:

```text
bin/k3.exe
bin/k3-worker.exe
```

`make` uses the machine's native x86 ISA by default. To make a conservative binary for an
older x86-64 CPU, override `ARCH`, for example `make ARCH=-msse2`.

## Running the model

The command-line interface is the same as on Linux. Paths may be MSYS2-style paths or
paths that the UCRT64 shell can resolve.

Example one-shot run:

```bash
./bin/k3.exe /d/k3model \
  --trunk /d/k3trunk --preset laptop \
  --tok /d/k3model --prompt "Hello" --gen 16 --incremental
```

The local OpenAI-compatible bridge can use `bin/k3-worker.exe` in the same way it uses the
POSIX worker. The worker keeps the model index, trunk/cache and conversation state alive
between requests instead of starting a new inference process for every turn.

`--preset auto` uses the native Windows available-memory query (`GlobalMemoryStatusEx`),
not `/proc/meminfo`.

## Windows-specific implementation notes

### Positioned reads

The UCRT has no `pread()`. K3 therefore opens model files with
`FILE_FLAG_OVERLAPPED`, wraps the resulting HANDLE in the fd-shaped interface the rest of
the engine already uses, and supplies an explicit 64-bit offset to every `ReadFile` via
`OVERLAPPED.Offset` / `OffsetHigh`.

Each positioned read owns its event and never moves a shared file pointer. Multiple expert
prefetch workers can therefore have reads against the same safetensors shard in flight at
the same time. The trunk uses the same primitive. The files are still opened through the
Windows buffered I/O path; no `FILE_FLAG_NO_BUFFERING` claim is made here.

A reproducible warm-cache same-file microbenchmark on the Windows CI runner measured a
median **7,752.0 MB/s** for the old locked seek/read compatibility path and **11,831.7
MB/s** for overlapped reads, or **1.526x** aggregate throughput with identical checksums.
That is a positioned-I/O microbenchmark, not a 1.526x end-to-end token-speed claim. See
`docs/data/windows-overlapped-2026-08.md` and `benchmarks/windows-pread-bench.c`.

Every model descriptor is binary. This matters on Windows because model weights are
arbitrary bytes, not text.

### Resident KV virtual memory

The worker does not commit its full configured context on Windows. It reserves the MLA KV
and rope address ranges with `VirtualAlloc(..., MEM_RESERVE, PAGE_NOACCESS)` and commits
only the rows `[cached, cached + T)` immediately before a model `forward()` can touch
them. Exact and resident-draft models have independent reservations.

All worker inference paths go through one `worker_forward()` choke point: prefill, replay,
speculative proposal, exact verification, draft lockstep and the non-speculative fallback.
This keeps the memory policy separate from model math and prevents a path from touching a
reserved-but-uncommitted row.

On reset or conversation divergence, the worker decommits the rows it reached with
`MEM_DECOMMIT` while retaining the address reservation. A later turn recommits the rows as
needed.

The Windows 1M-capacity CI gate measured the following on the tiny K3 fixture while both
an exact model and Q4 resident draft were enabled:

```text
startup private context64  : 108.7 MiB
startup private context1M  : 148.4 MiB
startup private delta      : 39.7 MiB
virtual reservation/model  : 2.62 GiB
private after real request : 149.3 MiB
private after RESET        : 149.2 MiB
```

The gate then reran the same sampled request after reset/decommit/recommit and required
exact parity with the one-shot verifier. See `docs/data/windows-lazy-kv-2026-08.md`.

### Worker stdout

The worker protocol uses unbuffered stdout on Windows. `READY`, `TOKEN`, `DRAFT`, `DONE`
and reset markers must be immediately observable by the Python bridge when stdout is a
pipe.

### Memory reporting

Peak resident memory is read from the native process working-set counters. Page size comes
from `GetSystemInfo`.

## Current limitations

### Windows storage is overlapped, but still buffered

Linux can use `O_DIRECT` for the streamed trunk and experts. Windows now has true
explicit-offset overlapped reads and no longer serializes same-shard readers behind a CRT
file-position lock, but the storage path still uses the Windows system cache.

So there is still **no claim that native Windows beats Linux/WSL on the full 1.56 TB
checkpoint**. The next storage experiment is optional `FILE_FLAG_NO_BUFFERING`, and it
must first validate the volume's physical-sector alignment plus buffer, offset and length
alignment on every read. It should only become a default if a full-checkpoint benchmark
wins.

### One-million capacity does not mean one-million used tokens fit in RAM

`--context 1048576` is now cheap as an **address-space capacity** on Windows: untouched KV
rows carry no whole-reservation commit charge. It does not change the cost of rows that
are actually used.

On the released K3 architecture, MLA KV is still roughly **2.37 MiB per used position per
model**. A resident exact model plus a resident draft can therefore consume roughly twice
the KV memory if both advance through the same long conversation. Long contexts must still
fit the machine's real RAM/pagefile budget; the 1M setting only removes the artificial
startup cost of reserving capacity that may never be touched.

## What Windows support means here

Native Windows support means the engine and resident worker execute on Windows and pass
model-output parity gates. The Windows reader supports concurrent same-file positioned
I/O, and huge worker KV capacity is reserve-only until positions are reached. None of
these platform changes alter the model: the exact K3 verifier remains authoritative.