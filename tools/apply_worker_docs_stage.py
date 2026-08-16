#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1 match, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'local/README.md'
s=p.read_text()
s=once(s,
'''./bin/k3               C inference engine
''',
'''./bin/k3               one-shot C inference engine
./bin/k3-worker        resident C inference worker (default localhost backend)
''','paths')
s=once(s,
'''  --preset laptop \\
  --threads N
```

It listens on `http://127.0.0.1:8000/v1` by default. The server refuses a non-loopback bind
''',
'''  --preset laptop \\
  --threads N \\
  --worker-context 1024
```

`k3_local.py` now starts **one resident `bin/k3-worker` process by default**. The
safetensors index, exact packed trunk mappings, model head, expert cache, recurrent KDA
state and MLA KV stay alive between HTTP/tool turns. If the next XTML prompt extends the
previous exact token sequence, only the pending last token plus the new suffix is fed; a
bifurcation resets conversation state but keeps weights and the warm expert cache open.
Use `--no-resident-worker` only for A/B testing or a feature not yet supported by the
worker.

It listens on `http://127.0.0.1:8000/v1` by default. The server refuses a non-loopback bind
''','server resident intro')
start=s.index('## Conversation state reuse\n')
end=s.index('\n## Sampling correctness\n',start)
new_section='''## Conversation state reuse

The default path is now **in-RAM resident reuse**, not a multi-GB state file per tool
turn. `k3-worker` retains the active exact sequence, KDA recurrence and MLA KV. Every new
request still sends the full canonical XTML token sequence; the worker reuses state only
when the entire previous sequence is an exact prefix. It reports the exact number of
prefix tokens reused. Any token mismatch causes a conversation-state reset, while the
loaded checkpoint/trunk and expert cache remain warm.

The resident KV capacity is explicit:

```bash
--worker-context 1024   # default
```

Raise it for long benchmark episodes only when RAM allows it. The expanded fp32 MLA cache
still costs roughly 2.37 MB per position across the 24 MLA layers, so context capacity is
a real memory decision, not a cosmetic API field.

The older disk-backed prefix cache remains available with `--no-resident-worker`:

```bash
--no-resident-worker
--state-cache-entries 1
--state-cache-dir PATH
--no-state-cache
```

That fallback is useful for A/B tests and can retain multiple prefixes, but the resident
worker is the low-latency path for the normal linear Kimi Code tool loop.
'''
s=s[:start]+new_section+s[end:]
s=once(s,
'''python local/k3_local.py serve \\
  --model-dir ~/k3model --trunk ~/k3trunk-lossless --preset laptop --threads N \\
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4
```

The draft can change acceptance rate and wall-clock speed, but exact K3 remains the
''',
'''python local/k3_local.py serve \\
  --model-dir ~/k3model --trunk ~/k3trunk-lossless --preset laptop --threads N \\
  --no-resident-worker \\
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4
```

The exact resident worker is the default, but the Q4 draft is still wired through the
one-shot backend in this revision; `--no-resident-worker` selects that path explicitly.
The next worker step is to keep **both** exact and draft trunks resident without changing
the probability-correct verifier.

The draft can change acceptance rate and wall-clock speed, but exact K3 remains the
''','draft resident note')
s=once(s,
'''- **Process startup per turn:** true token streaming is live now, and saved state avoids
  recomputing exact conversation prefixes, but the one-shot C process still reopens the
  checkpoint/index/trunk/cache for each HTTP request. A resident C worker is the next
  latency step and will also preserve the warm expert cache between tool turns.
''',
'''- **Draft + resident worker:** the exact path is resident now, but sampled Q4/I8 draft
  acceleration still uses the one-shot backend. Combining both resident states is the
  next throughput step; until then choose warm exact (`default`) or sampled draft
  (`--no-resident-worker --draft-trunk ...`) explicitly.
''','gap replacement')
p.write_text(s)
print('resident worker docs materialized')
