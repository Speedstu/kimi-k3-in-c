# Windows resident KV reserve/commit gate — 2026-08

This measurement validates **virtual-capacity cost**, not a claim that one million used
K3 tokens fit in a modest PC.

## Change under test

The resident worker used to obtain its Windows KV through the generic mmap compatibility
path, which maps anonymous memory with `MEM_RESERVE | MEM_COMMIT`. At very large configured
context capacities that makes Windows charge commit for rows that may never be touched.

The worker now uses an explicit VM policy instead:

1. reserve its complete MLA KV/rope address ranges with `MEM_RESERVE` only;
2. route every worker model invocation through one `worker_forward()` wrapper;
3. immediately before `forward()`, commit only rows `[cached, cached + T)` for every MLA
   layer that can be touched by that invocation;
4. on reset/divergence, decommit reached rows with `MEM_DECOMMIT` while retaining the
   address reservation;
5. recommit rows on the next request as needed.

The generic Windows mmap compatibility function remains reserve+commit, so this changes
only the resident worker policy and not unrelated anonymous mappings.

## CI configuration

GitHub-hosted `windows-latest`, Windows Server 2025 / UCRT64 GCC. The tiny K3 fixture used:

- `--context 64` as the small baseline;
- `--context 1048576` as the large-capacity case;
- resident exact trunk plus resident Q4 speculative draft;
- `--spec 2`, `--draft-topk 1`;
- seeded `temperature=1`, `top_p=.95` generation;
- reset followed by the same request again.

Private process memory was read with `psutil` after the worker emitted `@K3READY`.

## Measurement

```text
startup private context64  : 108.7 MiB
startup private context1M  : 148.4 MiB
startup private delta      : 39.7 MiB
virtual reservation/model  : 2.62 GiB
private after real request : 149.3 MiB
private after RESET        : 149.2 MiB
native Windows 1M reserve + on-demand commit + reset/recommit parity: PASS
```

The tiny fixture therefore grew private commit by only **39.7 MiB** when configured for
1,048,576 positions instead of 64, despite a **2.62 GiB virtual reservation per model**.
The exact and Q4 draft models were both enabled during this gate.

The tiny fixture's actual row size is deliberately small, so these MiB figures do not
predict full-checkpoint RAM usage. They establish the property being tested: **unused
configured context no longer causes commit proportional to the entire KV reservation**.

## Exactness gate

After the startup measurement, the 1M-capacity worker generated a real sampled response.
Its token ids were compared with the one-shot exact verifier using the same Q4 proposal
configuration and seed. The worker was then reset, its reached KV pages were decommitted,
and the same request was executed again after recommit. Both runs matched exactly.

No model arithmetic changed. The VM wrapper only makes pages writable before the existing
`forward()` accesses them and returns stale pages after state reset.

## Released-model interpretation

The released K3 architecture still uses roughly **2.37 MiB of MLA KV per used position per
model**. If both exact and resident draft advance through a very long conversation, their
used KV can therefore cost roughly twice that amount. Reserve-only 1M capacity removes an
artificial startup/commit wall; it does not compress the KV for positions that are really
used.