#!/usr/bin/env python3
"""Stage a greedy-only long-suffix n-gram drafter in k3_run.c.

The emitted model stays authoritative: this only changes which historical token spans are
PROPOSED for exact batched verification. Sampled decoding deliberately keeps the legacy
4/3-token matcher so seeded sampling behaviour does not move.
"""
from __future__ import annotations

from pathlib import Path

P = Path("src/cli/k3_run.c")
s = P.read_text()
start = s.index("static int spec_draft(")
end = s.index("\n#ifndef K3_VERSION", start)
old = s[start:end]
new = r'''#define K3_SPEC_LONG_MATCH_MIN 8
#define K3_SPEC_LONG_MATCH_MAX 64

static int spec_draft(const int *seq, int T, int cap, int *out,
                      int allow_long, int *match_len_out)
{
    /* Greedy-only long-suffix drafting. A long exact suffix match is strong evidence on
     * code/JSON/tool traces and costs essentially nothing compared with one K3 sweep.
     * It is still ONLY a proposal: the exact bf16/MXFP4 model verifies the whole batch
     * before a token can be emitted. If there are two historical matches their
     * continuations must agree; otherwise stop at the first divergence.
     *
     * Sampling intentionally skips this new path. Even distribution-correct speculative
     * sampling can consume RNG draws in a different grouping, so the historical 4/3
     * matcher remains untouched there for seeded behavioural parity. */
    if (match_len_out) *match_len_out = 0;
    if (cap > K3_SPEC_MAX) cap = K3_SPEC_MAX;

    if (allow_long) {
        int max_n = T - 1; /* an earlier occurrence needs at least one continuation id */
        if (max_n > K3_SPEC_LONG_MATCH_MAX) max_n = K3_SPEC_LONG_MATCH_MAX;
        for (int n = max_n; n >= K3_SPEC_LONG_MATCH_MIN; n--) {
            int m1 = -1, m2 = -1;
            const int suffix = T - n;
            for (int j = T - n - 1; j >= 0; j--) {
                /* Cheap boundary rejection matters at long contexts. */
                if (seq[j] != seq[suffix] || seq[j + n - 1] != seq[T - 1]) continue;
                int hit = 1;
                for (int i = 1; i < n - 1; i++)
                    if (seq[j + i] != seq[suffix + i]) { hit = 0; break; }
                if (!hit) continue;
                if (m1 < 0) m1 = j;
                else { m2 = j; break; }
            }
            if (m1 < 0) continue;

            int nd = 0;
            for (int i = 0; nd < cap && m1 + n + i < T; i++) {
                const int cand = seq[m1 + n + i];
                if (m2 >= 0) {
                    /* Do not compare through the newer historical occurrence itself;
                     * that would turn copied context into fake evidence. */
                    if (m2 + n + i >= m1 || seq[m2 + n + i] != cand) break;
                }
                out[nd++] = cand;
            }
            if (nd > 0) {
                if (match_len_out) *match_len_out = n;
                return nd;
            }
        }
    }

    /* Legacy evidence-gated fallback, byte-for-byte equivalent in policy to main:
     * match length 4 then 3; if a second occurrence exists, the continuations must
     * agree. Keeping this exact fallback protects non-repetitive and sampled behaviour. */
    for (int n = 4; n >= 3; n--) {
        if (T < n + 1) continue;
        int m1 = -1, m2 = -1;
        for (int j = T - n - 1; j >= 0; j--) {
            int hit = 1;
            for (int i = 0; i < n; i++)
                if (seq[j + i] != seq[T - n + i]) { hit = 0; break; }
            if (!hit) continue;
            if (m1 < 0) m1 = j;
            else { m2 = j; break; }
        }
        if (m1 < 0) continue;
        int nd = 0;
        for (int i = 0; nd < cap && m1 + n + i < T; i++) {
            const int cand = seq[m1 + n + i];
            if (m2 >= 0) {
                if (m2 + n + i >= m1 || seq[m2 + n + i] != cand) break;
            }
            out[nd++] = cand;
        }
        if (nd > 0) {
            if (match_len_out) *match_len_out = n;
            return nd;
        }
    }
    return 0;
}
'''
s = s[:start] + new + s[end:]

needle = """    int spec_peak_width = 0;\n    K3SpecAutoCost spec_cost;\n"""
repl = """    int spec_peak_width = 0;\n    long spec_long_rounds = 0, spec_long_proposed = 0;\n    int spec_long_match_max = 0;\n    K3SpecAutoCost spec_cost;\n"""
if needle not in s:
    raise SystemExit("spec counter anchor not found")
s = s.replace(needle, repl, 1)

needle = """                } else {\n                    nd = spec_draft(seq, T, spec_now, d);\n                    if (temperature > 0.0)\n"""
repl = """                } else {\n                    int ngram_match_len = 0;\n                    nd = spec_draft(seq, T, spec_now, d, temperature <= 0.0,\n                                    &ngram_match_len);\n                    if (ngram_match_len >= K3_SPEC_LONG_MATCH_MIN && nd > 0) {\n                        spec_long_rounds++;\n                        spec_long_proposed += nd;\n                        if (ngram_match_len > spec_long_match_max)\n                            spec_long_match_max = ngram_match_len;\n                    }\n                    if (temperature > 0.0)\n"""
if needle not in s:
    raise SystemExit("spec_draft call anchor not found")
s = s.replace(needle, repl, 1)

needle = """    if (spec_peak_width > 0)\n        printf("\\nspeculative peak proposed width: %d\\n", spec_peak_width);\n"""
repl = """    if (spec_peak_width > 0)\n        printf("\\nspeculative peak proposed width: %d\\n", spec_peak_width);\n    if (spec_long_rounds > 0)\n        printf("long-suffix ngram: %ld rounds, %ld proposed, max matched context %d tokens\\n",\n               spec_long_rounds, spec_long_proposed, spec_long_match_max);\n"""
if needle not in s:
    raise SystemExit("spec report anchor not found")
s = s.replace(needle, repl, 1)

P.write_text(s)
print("staged greedy long-suffix speculation")
