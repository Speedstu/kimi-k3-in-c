#!/usr/bin/env python3
from pathlib import Path
import re

p = Path("src/cli/k3_run.c")
s = p.read_text()

if "K3_SPEC_SAMPLE_MAX" in s:
    print("greedy spec32 already applied")
    raise SystemExit(0)

# Raise only the generic/greedy speculative buffer ceiling. Seeded sampling keeps the
# historical effective width 8 so widening greedy verification cannot consume a different
# proposal/accept RNG sequence.
s, n = re.subn(
    r"#define\s+K3_SPEC_MAX\s+8\b",
    "#define K3_SPEC_SAMPLE_MAX 8\n#define K3_SPEC_MAX 32",
    s,
    count=1,
)
if n != 1:
    raise SystemExit(f"K3_SPEC_MAX anchor mismatch: {n}")

s = s.replace("default ceiling 8", "default greedy ceiling 32")
s = s.replace(
    "Sampling keeps fixed --spec for seeded parity",
    "Sampling keeps fixed --spec capped at 8 for seeded parity",
)

# Preserve sampled seeded behavior after the wider generic clamp.
anchor = """    if (spec_auto && spec_n <= 0) spec_n = K3_SPEC_MAX;
    if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;
    if (spec_n < 0) spec_n = 0;
"""
replacement = anchor + """    if (temperature > 0.0 && spec_n > K3_SPEC_SAMPLE_MAX)
        spec_n = K3_SPEC_SAMPLE_MAX;
"""
if s.count(anchor) != 1:
    raise SystemExit(f"spec clamp anchor mismatch: {s.count(anchor)}")
s = s.replace(anchor, replacement, 1)

# Track the actual largest proposal batch in the human log. This is deliberately not part
# of the JSON ABI: the previous staged patch failed because it depended on an older JSON
# layout after the cost-aware autotuner added fields.
anchor = """    long spec_auto_rounds = 0, spec_auto_proposed = 0, spec_auto_accepted = 0;
    int spec_auto_grows = 0, spec_auto_shrinks = 0;
"""
if s.count(anchor) != 1:
    raise SystemExit(f"spec counter anchor mismatch: {s.count(anchor)}")
s = s.replace(anchor, anchor + "    int spec_peak_width = 0;\n", 1)

anchor = """            if (nd > 0) {
                /* One sweep verifies the pending token plus nd drafts."""
replacement = """            if (nd > 0) {
                if (nd > spec_peak_width) spec_peak_width = nd;
                /* One sweep verifies the pending token plus nd drafts."""
if s.count(anchor) != 1:
    raise SystemExit(f"verification anchor mismatch: {s.count(anchor)}")
s = s.replace(anchor, replacement, 1)

anchor = """    if (spec_auto && spec_auto_rounds > 0) {
"""
replacement = """    if (spec_peak_width > 0)
        printf("\\nspeculative peak proposed width: %d\\n", spec_peak_width);
    if (spec_auto && spec_auto_rounds > 0) {
"""
if s.count(anchor) != 1:
    raise SystemExit(f"spec summary anchor mismatch: {s.count(anchor)}")
s = s.replace(anchor, replacement, 1)

p.write_text(s)
print("applied greedy speculation ceiling 32 with sampled compatibility cap 8")
