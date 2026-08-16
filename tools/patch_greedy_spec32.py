#!/usr/bin/env python3
from pathlib import Path
import re

p = Path('src/cli/k3_run.c')
s = p.read_text()

if 'K3_SPEC_SAMPLE_MAX' in s:
    print('greedy spec32 already applied')
    raise SystemExit(0)

# 1) Increase only the general/greedy buffer ceiling. Sampling keeps the historical
# effective width 8 so old --spec 16 sampled commands still consume RNG exactly as they
# did when K3_SPEC_MAX itself was 8.
s, n = re.subn(r'#define\s+K3_SPEC_MAX\s+8\b',
               '#define K3_SPEC_SAMPLE_MAX 8\n#define K3_SPEC_MAX 32', s, count=1)
if n != 1:
    raise SystemExit(f'K3_SPEC_MAX anchor mismatch: {n}')

# Help text: only wording, no behavior.
s = s.replace('default ceiling 8', 'default greedy ceiling 32')
s = s.replace('Sampling keeps fixed --spec for seeded parity',
              'Sampling keeps fixed --spec capped at 8 for seeded parity')

# 2) Preserve sampled seeded behavior after the new generic clamp.
anchor = '''    if (spec_auto && spec_n <= 0) spec_n = K3_SPEC_MAX;
    if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;
    if (spec_n < 0) spec_n = 0;
'''
replacement = anchor + '''    if (temperature > 0.0 && spec_n > K3_SPEC_SAMPLE_MAX)
        spec_n = K3_SPEC_SAMPLE_MAX;
'''
if s.count(anchor) != 1:
    raise SystemExit(f'spec clamp anchor mismatch: {s.count(anchor)}')
s = s.replace(anchor, replacement, 1)

# 3) Track the actual largest proposal batch. This proves a run exercised >8 rather
# than merely accepting/parsing a larger command-line ceiling.
anchor = '''    long spec_auto_rounds = 0, spec_auto_proposed = 0, spec_auto_accepted = 0;
    int spec_auto_grows = 0, spec_auto_shrinks = 0;
'''
replacement = anchor + '    int spec_peak_width = 0;\n'
if s.count(anchor) != 1:
    raise SystemExit(f'spec counter anchor mismatch: {s.count(anchor)}')
s = s.replace(anchor, replacement, 1)

anchor = '''            if (nd > 0) {
                /* One sweep verifies the pending token plus nd drafts.'''
replacement = '''            if (nd > 0) {
                if (nd > spec_peak_width) spec_peak_width = nd;
                /* One sweep verifies the pending token plus nd drafts.'''
if s.count(anchor) != 1:
    raise SystemExit(f'verification anchor mismatch: {s.count(anchor)}')
s = s.replace(anchor, replacement, 1)

# Human diagnostic is also useful if a downstream JSON consumer predates the field.
anchor = '''    if (spec_auto && spec_auto_rounds > 0) {
'''
replacement = '''    if (spec_peak_width > 0)
        printf("\\nspeculative peak proposed width: %d\\n", spec_peak_width);
    if (spec_auto && spec_auto_rounds > 0) {
'''
if s.count(anchor) != 1:
    raise SystemExit(f'spec summary anchor mismatch: {s.count(anchor)}')
s = s.replace(anchor, replacement, 1)

# 4) Add a backwards-compatible JSON field. Locate the existing spec-auto tail rather
# than depending on exact surrounding line wrapping.
old = '"\\\"spec_auto_grows\\\":%d,\\\"spec_auto_shrinks\\\":%d,"\n                   "\\\"seconds_per_token\\\":%.4f}'
new = '"\\\"spec_auto_grows\\\":%d,\\\"spec_auto_shrinks\\\":%d,"\n                   "\\\"spec_peak_width\\\":%d,\\\"seconds_per_token\\\":%.4f}'
if old not in s:
    raise SystemExit('JSON spec-auto format anchor missing')
s = s.replace(old, new, 1)

old_args = '''                spec_auto_proposed, spec_auto_accepted, spec_cur, spec_limit,
                spec_auto_grows, spec_auto_shrinks, t_total / nout);'''
new_args = '''                spec_auto_proposed, spec_auto_accepted, spec_cur, spec_limit,
                spec_auto_grows, spec_auto_shrinks, spec_peak_width, t_total / nout);'''
if s.count(old_args) != 1:
    raise SystemExit(f'JSON spec-auto args anchor mismatch: {s.count(old_args)}')
s = s.replace(old_args, new_args, 1)

p.write_text(s)
print('applied greedy speculation ceiling 32 with sampled compatibility cap 8')
