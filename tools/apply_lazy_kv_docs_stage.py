#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'local/README.md'
s=p.read_text()
s=once(s,
'''The **declared** model window and the **physically affordable** local context are separate.
The current exact C incremental cache stores expanded MLA K/V in fp32 (~2.37 MB per
position across the 24 MLA layers), so a small machine will correctly refuse a huge
context before allocation. The next performance target is a latent MLA cache; do not hide
this constraint by letting the OS OOM-kill the run.
''',
'''The **declared** model window and the **physically affordable** local context are separate.
The exact C incremental cache still stores expanded MLA K/V in fp32 (~2.37 MB per used
position across the 24 MLA layers). The resident worker now reserves its configured KV
capacity with lazy anonymous virtual memory on supported POSIX systems, so merely choosing
a larger capacity no longer faults every KV page into RAM. Physical use still grows as
positions are actually written. This removes an artificial startup/reset cost; it does
**not** make a million used tokens fit in laptop RAM.
''','benchmark window memory text')
s=once(s,
'''Raise it for long benchmark episodes only when RAM allows it. The expanded fp32 MLA cache
still costs roughly 2.37 MB per position across the 24 MLA layers, so context capacity is
a real memory decision, not a cosmetic API field.
''',
'''The worker reserves KV address space lazily, and prompt prefill is processed in fixed
64-token chunks so hidden/residual/scratch buffers no longer scale with the whole configured
capacity. A conversation reset zeros only true KDA recurrent/ShortConv state: setting
`cached=0` makes old MLA KV rows unreachable, and every row used by the next conversation
is fully overwritten before it can be read. This avoids sweeping gigabytes of stale KV on
a branch/reset while preserving exact output; permanent CI gates a 130-token prompt across
two chunk boundaries and a long-to-short reset against one-shot K3.

RAM is still proportional to **used** context. Expanded fp32 MLA KV costs roughly 2.37 MB
per used position across the 24 MLA layers (and a resident draft has its own state), so
`--worker-context` remains a real capacity decision rather than a promise that the full
advertised 1M-token model window is affordable locally.
''','resident context memory explanation')
p.write_text(s)
print('lazy KV docs materialized')
