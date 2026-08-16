#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1, got {n}')
    return s.replace(old,new,1)

root=Path(__file__).resolve().parents[1]

p=root/'local/k3_local.py'
s=p.read_text()
s=once(s,
'''        help="resident KV capacity in positions; raise for long benchmark sessions",
''',
'''        help=(
            "resident capacity in positions (2..1048576); virtual reservation is lazy, "
            "but RAM still grows with positions actually used"
        ),
''','python worker-context help')
p.write_text(s)

p=root/'local/README.md'
s=p.read_text()
s=once(s,
'''The resident KV capacity is explicit:

```bash
--worker-context 1024   # default
```

The worker reserves KV address space lazily, and prompt prefill is processed in fixed
''',
'''The resident KV capacity is explicit and can now be configured up to the declared K3
window:

```bash
--worker-context 1024      # conservative default
--worker-context 1048576   # maximum VIRTUAL capacity; not a laptop-RAM promise
```

On a 64-bit POSIX build, large exact/draft KV regions use anonymous lazy mappings and do
not fall back to a giant `calloc` if that virtual reservation fails. The worker prints the
per-used-position KV cost and total virtual reservation at startup. CI boots a tiny K3
worker with `--context 1048576`, performs an exact request, and rejects `1048577` before
model allocation. This validates the capacity mechanism, not the feasibility of filling a
million positions with the released model.

The worker reserves KV address space lazily, and prompt prefill is processed in fixed
''','README capacity block')
s=once(s,
'''capacity. A conversation reset zeros only true KDA recurrent/ShortConv state: setting
`cached=0` makes old MLA KV rows unreachable, and every row used by the next conversation
is fully overwritten before it can be read. This avoids sweeping gigabytes of stale KV on
a branch/reset while preserving exact output; permanent CI gates a 130-token prompt across
two chunk boundaries and a long-to-short reset against one-shot K3.
''',
'''capacity. A conversation reset zeros only true KDA recurrent/ShortConv state: setting
`cached=0` makes old MLA KV rows unreachable, and every row used by the next conversation
is fully overwritten before it can be read. On anonymous mappings the worker also gives
the actually-touched dead KV pages back to the OS with best-effort `MADV_DONTNEED`, rather
than writing zeros through gigabytes. This reclamation is a memory optimisation only;
correctness still comes from the cache-position invariant. Permanent CI gates a 130-token
prompt across two chunk boundaries and a long-to-short reset against one-shot K3.
''','README discard block')
p.write_text(s)
print('large-context docs/help materialized')
