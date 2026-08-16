#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'local/README.md'
s=p.read_text()
s=once(s,
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
verification/target distribution. Sweep draft top-k/spec length on the real machine rather
than assuming one setting is universally fastest.
''',
'''python local/k3_local.py serve \\
  --model-dir ~/k3model --trunk ~/k3trunk-lossless --preset laptop --threads N \\
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4
```

With these flags the **exact trunk and the proposal trunk are both resident worker
resources**: their packed mappings and independent KDA/MLA states stay alive across Kimi
Code tool turns. The shared expert cache stays warm too. The worker only reuses a prior
conversation state when the entire previous XTML token sequence is an exact prefix; a
branch resets both exact and draft conversation states together without reopening either
trunk.

The worker deliberately mirrors the one-shot decoder's full-block scheduling as well as
its p/q mathematics. A fixed seed therefore gives the same output in one-shot and
resident modes, including near `max_tokens` where the decoder falls back to serial exact
decode instead of consuming extra speculative RNG draws. Permanent tiny-checkpoint CI
gates first-turn and warm second-turn parity against the one-shot engine.

The draft can change acceptance rate and wall-clock speed, but exact K3 remains the
verification/target distribution. The worker reports proposal rounds, proposed/accepted
tokens, draft time and exact verification time, so sweep draft top-k/spec length on the
real machine rather than assuming one setting is universally fastest.
''','resident draft example')
s=once(s,
'''- **Draft + resident worker:** the exact path is resident now, but sampled Q4/I8 draft
  acceleration still uses the one-shot backend. Combining both resident states is the
  next throughput step; until then choose warm exact (`default`) or sampled draft
  (`--no-resident-worker --draft-trunk ...`) explicitly.
''',
'''- **Draft quality is hardware/workload dependent:** Q4/I8 proposal inference is now
  resident and probability-correct, but a low acceptance rate can still make speculation
  slower than serial exact decode. Use the emitted draft metrics / sweep instead of
  treating one top-k/spec pair as a universal default.
''','current gap resident draft')
p.write_text(s)
print('resident sampled-draft docs materialized')
