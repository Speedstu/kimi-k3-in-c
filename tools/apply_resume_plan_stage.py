#!/usr/bin/env python3
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, got {n}")
    return text.replace(old, new, 1)


p = Path(__file__).resolve().parents[1] / "src/cli/k3_run.c"
s = p.read_text()

s = once(
    s,
    '    if (np == 0) { fprintf(stderr, "no prompt ids parsed\\n"); return 2; }\n',
    '    if (np == 0 && !load_state) { fprintf(stderr, "no prompt ids parsed\\n"); return 2; }\n',
    "allow empty continuation only when resuming",
)

anchor = '''    for (int i = 0; i < np; i++)
        if (prompt[i] < 0 || prompt[i] >= c.vocab) {
            fprintf(stderr, "token id %d is outside the vocabulary of %d\\n", prompt[i], c.vocab);
            return 2;
        }

    /* Validate the request before allocating anything.
'''
replacement = '''    for (int i = 0; i < np; i++)
        if (prompt[i] < 0 || prompt[i] >= c.vocab) {
            fprintf(stderr, "token id %d is outside the vocabulary of %d\\n", prompt[i], c.vocab);
            return 2;
        }

    /* Peek a resume BEFORE every context / memory guard. The saved prefix is part of
     * this request's real working set even though it is not repeated on argv. The old
     * order only discovered `prior` after the 1.56 TB index/cache setup, so its guards
     * understated KV and scratch by the entire conversation history. */
    K3StateHdr shd;
    int prior = 0;
    if (load_state) {
        if (!incremental) {
            fprintf(stderr, "--load-state needs --incremental\\n");
            return 2;
        }
        if (k3_state_peek(load_state, &shd) != 0) return 1;
        prior = shd.nseq;
        if (prior < 0 || prior > K3_MAX_PROMPT + K3_MAX_GEN) {
            fprintf(stderr, "saved state reports an invalid sequence length %d\\n", prior);
            return 2;
        }
    }

    /* Validate the request before allocating anything.
'''
s = once(s, anchor, replacement, "early state peek")

s = once(
    s,
    "    if (np + gen + 1 > K3_MAX_PROMPT + K3_MAX_GEN) {\n"
    "        fprintf(stderr, \"prompt %d + gen %d + 1 exceeds the %d-position ceiling\\n\",\n"
    "                np, gen, K3_MAX_PROMPT + K3_MAX_GEN);\n",
    "    if (prior + np + gen + 1 > K3_MAX_PROMPT + K3_MAX_GEN) {\n"
    "        fprintf(stderr, \"saved %d + prompt %d + gen %d + 1 exceeds the %d-position ceiling\\n\",\n"
    "                prior, np, gen, K3_MAX_PROMPT + K3_MAX_GEN);\n",
    "sequence ceiling includes prior",
)

s = once(
    s,
    "        const double kv_need = (double)(np + gen + 1) * K3_KV_BYTES_PER_POS;\n",
    "        const int total_pos = prior + np + gen + 1;\n"
    "        const double kv_need = (double)total_pos * K3_KV_BYTES_PER_POS;\n",
    "KV guard total positions",
)
s = once(
    s,
    '        printf("  KV cache : %s for %d positions (%.2f MB/position)\\n",\n'
    "               kb, np + gen + 1, K3_KV_BYTES_PER_POS / 1e6);\n",
    '        printf("  KV cache : %s for %d positions (%.2f MB/position)\\n",\n'
    "               kb, total_pos, K3_KV_BYTES_PER_POS / 1e6);\n",
    "KV guard banner",
)
s = once(
    s,
    '                np + gen + 1, kb, ab);\n',
    '                total_pos, kb, ab);\n',
    "KV refusal total positions",
)

s = once(
    s,
    '    printf("  prompt   : %d tokens, generating %d\\n", np, gen);\n',
    '    if (prior) printf("  prompt   : %d saved + %d new tokens, generating %d\\n", prior, np, gen);\n'
    '    else       printf("  prompt   : %d tokens, generating %d\\n", np, gen);\n',
    "resume banner",
)

s = once(
    s,
    "        const int Tm = np + gen + 1;\n",
    "        const int Tm = prior + np + gen + 1;\n",
    "memory plan positions",
)

old_late = '''    K3StateHdr shd;
    int prior = 0;
    if (load_state) {
        if (!incremental) {
            fprintf(stderr, "--load-state needs --incremental\\n");
            return 2;
        }
        if (k3_state_peek(load_state, &shd) != 0) return 1;
        prior = shd.nseq;
        printf("resuming from %s: %d prior positions, %d new\\n\\n", load_state, prior, np);
    }
'''
new_late = '''    if (load_state)
        printf("resuming from %s: %d prior positions, %d new\\n\\n", load_state, prior, np);
'''
s = once(s, old_late, new_late, "remove late duplicate state peek")

p.write_text(s)
print("resume-aware memory planning patch applied")
