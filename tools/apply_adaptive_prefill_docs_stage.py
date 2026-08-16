#!/usr/bin/env python3
from pathlib import Path


def once(s,old,new,label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1 got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'local/README.md'
s=p.read_text()
s=once(s,
'''The worker reserves KV address space lazily, and prompt prefill is processed in fixed
64-token chunks so hidden/residual/scratch buffers no longer scale with the whole configured
capacity. A conversation reset zeros only true KDA recurrent/ShortConv state: setting
''',
'''The worker reserves KV address space lazily. Prompt prefill is also RAM-bounded, but no
longer hard-coded to 64 tokens: by default `--prefill-mb 256` chooses the largest batch
(up to 8192 tokens) whose hidden/residual/scratch estimate fits that transient budget. If
an allocation still fails because of fragmentation/rlimits, the worker halves the chunk
until it fits. Larger chunks matter because every chunk is another whole-model/trunk sweep;
the right value is therefore a direct RAM-vs-I/O speed tradeoff. A conversation reset
zeros only true KDA recurrent/ShortConv state: setting
''','adaptive prefill explanation')
s=once(s,
'''worker is the low-latency path for the normal linear Kimi Code tool loop.

## Sampling correctness
''',
'''worker is the low-latency path for the normal linear Kimi Code tool loop.

### Prefill speed tuning

The localhost server exposes the same controls:

```bash
--prefill-mb 256        # automatic transient-RAM budget, normal path
--prefill-chunk 512     # manual override for measurement/debugging
```

Do not assume that the largest chunk is fastest: resident trunk pages, expert-cache size,
NVMe bandwidth and available RAM all matter. Measure the actual machine with an already
tokenised representative prompt:

```bash
python benchmarks/prefill-sweep.py ~/k3model ~/k3trunk-lossless prompt.ids \\
  --trunk-gb 3 --cache-gb 1 --budgets 64,128,256,512,1024 --repeats 2
```

The sweep starts the same local `k3-worker` for each budget, refuses to rank a candidate if
its greedy token differs from the baseline, and prints the median request time plus the
selected chunk. Use the recommended `--prefill-mb` for that PC/workload. Permanent CI also
checks manual 1/16/64/128 and automatic chunks against one-shot exact K3, including the
`temperature=1` sampled-draft path.

## Sampling correctness
''','prefill tuning section')
p.write_text(s)
print('adaptive prefill docs materialized')
