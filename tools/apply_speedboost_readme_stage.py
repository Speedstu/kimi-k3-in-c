#!/usr/bin/env python3
from pathlib import Path

p = Path('README.md')
s = p.read_text(encoding='utf-8')


def one(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {n}')
    s = s.replace(old, new, 1)

one(
    'https://github.com/FareedKhan-dev/kimi-k3-in-c/actions/workflows/ci.yml',
    'https://github.com/Speedstu/kimi-k3-in-c/actions/workflows/ci.yml',
    'CI badge URL',
)
one(
    'https://img.shields.io/github/actions/workflow/status/FareedKhan-dev/kimi-k3-in-c/ci.yml?branch=main&style=flat-square&label=CI',
    'https://img.shields.io/github/actions/workflow/status/Speedstu/kimi-k3-in-c/ci.yml?branch=main&style=flat-square&label=CI',
    'CI badge image',
)
# There are two clone commands in the README (Quick start and Full setup Step 0).
s = s.replace(
    'git clone https://github.com/FareedKhan-dev/kimi-k3-in-c.git',
    'git clone https://github.com/Speedstu/kimi-k3-in-c.git',
)

marker = '''---

# Part I: Getting started
'''
block = r'''---

## Speedboost fork: fastest path without changing exact K3

This fork keeps the original bf16/MXFP4 K3 as the **authoritative model**. The exact path
has been optimized around the real bottleneck (storage + memory bandwidth), while the
more aggressive Q4/top-k changes are confined to a speculative draft whose tokens are
verified by exact K3 before they can be emitted.

The main speed work in this fork is:

- **lossless exact trunk storage**: `tools/lossless_trunk.py` reconstructs the original
  packed bf16/f32 bytes before binding. A 256 MiB codec microbenchmark decoded at about
  17.2 GB/s mean on the development runner; a tiny end-to-end trunk stored 23.0% fewer
  bytes after O_DIRECT alignment and produced binary-identical logits;
- **streamed MoE router stays BF16** instead of widening the `896 x 7168` gate to FP32 on
  every streamed layer. Resident mode still keeps FP32 because that is faster when the
  gate is reused. The streamed router stage measured about 1.5x faster in a
  released-dimension microbenchmark, with bit-identical top-k/combining weights;
- **async expert prefetch** overlaps top-k expert reads with the independent down
  projection and shared-expert MLP, then joins before the first routed expert is touched;
- **measured thread tuning** via `--threads N` and `benchmarks/thread-sweep.sh` instead of
  assuming that all logical CPUs are fastest;
- **optional Q4 speculative draft** at 0.53125 bytes/weight for full groups plus
  `--draft-topk`, while exact bf16/MXFP4 K3 still verifies every emitted token.

### Recommended small-PC setup

The checkpoint itself is still **1.56 TB**. These changes reduce RAM and streamed trunk
traffic; they do not turn a 2.78T checkpoint into a small download. A fast local NVMe is
still the most important hardware component at low RAM budgets.

After the normal checkpoint download and trunk pack:

```bash
# 1. Create a byte-exact compressed storage representation of the exact trunk.
python3 tools/lossless_trunk.py ~/k3trunk ~/k3trunk-lossless

# 2. Find the best compute thread count on THIS machine/workload.
K3_SWEEP_REPEATS=3 benchmarks/thread-sweep.sh ~/k3model \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --ids 1008,10484,318,15383,387 --gen 4

# 3. Re-run with the recommended --threads N.
./bin/k3 ~/k3model --trunk ~/k3trunk-lossless --preset laptop \
  --incremental --threads N \
  --ids 1008,10484,318,15383,387 --gen 8
```

Async expert I/O overlap is enabled automatically when the real cache supports it. Useful
controls:

```bash
K3_NOASYNC_PREFETCH=1   # compare against the old synchronous caller behavior
K3_ASYNC_IO_THREADS=4   # background expert-read OpenMP team (default 4)
```

For an **optional faster speculative proposal path**, derive a Q4 trunk from the normal
packed bf16 trunk:

```bash
python3 tools/q4_trunk.py ~/k3trunk ~/k3draft-q4

./bin/k3 ~/k3model --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4 \
  --ids 1008,10484,318,15383,387 --gen 8
```

The Q4/top-k draft may change **proposal acceptance and speed**, but not the exact greedy
output: exact K3 verifies proposals before emission. Sweep `--draft-topk 2`, `4`, and `8`
on the real checkpoint rather than assuming one value is universally best.

More detail:
[`SPEEDBOOST.md`](docs/SPEEDBOOST.md) ·
[`LOSSLESS_TRUNK.md`](docs/LOSSLESS_TRUNK.md) ·
[`ASYNC_EXPERT_PREFETCH.md`](docs/ASYNC_EXPERT_PREFETCH.md) ·
[`THREAD_TUNING.md`](docs/THREAD_TUNING.md)

---

# Part I: Getting started
'''
one(marker, block, 'speedboost section insertion')

# Surface the exact thread knob in the existing generation table.
one(
'''| `--gen` | `N` | 8 | tokens to generate. Ceiling 4096; prompts may be up to 32768 tokens |
| `--incremental` | none | off | carry the KV cache and the recurrent state between tokens instead of re-running the whole prefix |''',
'''| `--gen` | `N` | 8 | tokens to generate. Ceiling 4096; prompts may be up to 32768 tokens |
| `--threads` | `N` | OpenMP default | exact compute-team size. Use `benchmarks/thread-sweep.sh` to measure the best value for this machine |
| `--incremental` | none | off | carry the KV cache and the recurrent state between tokens instead of re-running the whole prefix |''',
'usage threads row',
)

# Add current async controls to environment variable table.
one(
'''| `OMP_NUM_THREADS` | the engine | thread count, defaulting to all cores |
| `K3_TOK_FILES` | tokenizer tools and CI | directory holding `tiktoken.model`, when it is not in a default location |''',
'''| `OMP_NUM_THREADS` | the engine | OpenMP default when `--threads` is not supplied |
| `K3_NOASYNC_PREFETCH` | expert cache | set to disable only the background top-k overlap, useful for A/B timing |
| `K3_ASYNC_IO_THREADS` | expert cache | background expert-read OpenMP team, default 4 |
| `K3_TOK_FILES` | tokenizer tools and CI | directory holding `tiktoken.model`, when it is not in a default location |''',
'environment async controls',
)

p.write_text(s, encoding='utf-8', newline='\n')
print('README speedboost guide staged')
