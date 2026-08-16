#!/usr/bin/env python3
from pathlib import Path

p = Path("src/cli/k3_run.c")
s = p.read_text()

if "--spec-auto" in s:
    print("greedy adaptive speculation already applied")
    raise SystemExit(0)

# Help text: fixed --spec remains unchanged; auto is explicitly greedy-only so seeded
# speculative sampling keeps its current deterministic RNG-consumption behaviour.
old = '''"  --spec N              speculative decode: draft up to N tokens by n-gram lookup and\\n"\n"                        verify them in ONE batched sweep. Output is identical to\\n"\n"                        serial decode by construction; needs --incremental. An extra\\n"\n"                        verified position costs ~22%% of a serial token when the trunk\\n"\n"                        streams, so repetitive text decodes up to several times faster\\n"\n'''
new = '''"  --spec N              speculative decode: draft up to N tokens by n-gram lookup and\\n"\n"                        verify them in ONE batched sweep. Output is identical to\\n"\n"                        serial decode by construction; needs --incremental. An extra\\n"\n"                        verified position costs ~22%% of a serial token when the trunk\\n"\n"                        streams, so repetitive text decodes up to several times faster\\n"\n"  --spec-auto           greedy-only adaptive horizon: start at 4 drafts, grow after a\\n"\n"                        fully accepted sweep and shrink after poor acceptance. Exact K3\\n"\n"                        still verifies every emitted token. --spec N becomes the ceiling\\n"\n"                        (default ceiling 8). Sampling keeps fixed --spec for seeded parity\\n"\n'''
if old not in s:
    raise SystemExit("usage anchor not found")
s = s.replace(old, new, 1)

old = '''    int spec_n = 0;\n    int tf_check = 0;\n'''
new = '''    int spec_n = 0;\n    int spec_auto = 0;\n    int tf_check = 0;\n'''
if old not in s:
    raise SystemExit("spec variable anchor not found")
s = s.replace(old, new, 1)

old = '''        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_n = atoi(argv[++i]);\n        else if (!strcmp(argv[i], "--tf-check")) tf_check = 1;\n'''
new = '''        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_n = atoi(argv[++i]);\n        else if (!strcmp(argv[i], "--spec-auto")) spec_auto = 1;\n        else if (!strcmp(argv[i], "--tf-check")) tf_check = 1;\n'''
if old not in s:
    raise SystemExit("spec parser anchor not found")
s = s.replace(old, new, 1)

# Initialise the adaptive ceiling/current width before the existing snapshot allocation.
old = '''    const size_t kperP  = (size_t)c.kda_heads * c.kda_head_dim;\n    const size_t kper_f = kperP * c.kda_head_dim + 3 * kperP * (c.conv_k - 1);\n    float *spec_snap = NULL;\n    if (spec_n > 0) {\n'''
new = '''    const size_t kperP  = (size_t)c.kda_heads * c.kda_head_dim;\n    const size_t kper_f = kperP * c.kda_head_dim + 3 * kperP * (c.conv_k - 1);\n\n    /* Adaptive speculative width is intentionally greedy-only in this first exact\n     * version. Standard speculative sampling is distribution-correct for any width,\n     * but changing width changes how many RNG draws proposal/acceptance streams consume;\n     * refusing auto there preserves seeded bit-for-bit behavioural parity. */\n    if (spec_auto && temperature > 0.0) {\n        fprintf(stderr, "--spec-auto currently supports greedy decode only; "\n                        "use fixed --spec N with --temperature > 0\\n");\n        return 2;\n    }\n    if (spec_auto && spec_n <= 0) spec_n = K3_SPEC_MAX;\n    if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;\n    if (spec_n < 0) spec_n = 0;\n    int spec_limit = spec_n;\n    int spec_cur = spec_auto ? (spec_limit < 4 ? spec_limit : 4) : spec_limit;\n    long spec_auto_rounds = 0, spec_auto_proposed = 0, spec_auto_accepted = 0;\n    int spec_auto_grows = 0, spec_auto_shrinks = 0;\n\n    float *spec_snap = NULL;\n    if (spec_n > 0) {\n'''
if old not in s:
    raise SystemExit("spec setup anchor not found")
s = s.replace(old, new, 1)

old = '''            if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;\n            spec_snap = (float *)malloc(kper_f * (size_t)w.n_bound * sizeof(float));\n            if (!spec_snap) { fprintf(stderr, "OOM for the --spec snapshot\\n"); return 1; }\n            printf("speculative decode: up to %d drafted tokens per sweep, n-gram lookup, "\n                   "verified batched\\n\\n", spec_n);\n'''
new = '''            spec_snap = (float *)malloc(kper_f * (size_t)w.n_bound * sizeof(float));\n            if (!spec_snap) { fprintf(stderr, "OOM for the --spec snapshot\\n"); return 1; }\n            if (spec_auto)\n                printf("speculative decode: adaptive greedy horizon starts at %d, ceiling %d; "\n                       "exact batched verification\\n\\n", spec_cur, spec_limit);\n            else\n                printf("speculative decode: up to %d drafted tokens per sweep, n-gram lookup, "\n                       "verified batched\\n\\n", spec_n);\n'''
if old not in s:
    raise SystemExit("spec snapshot print anchor not found")
s = s.replace(old, new, 1)

# If --spec was disabled because incremental is absent, disable the current auto width too.
old = '''            fprintf(stderr, "--spec needs --incremental; ignoring --spec\\n");\n            spec_n = 0;\n'''
new = '''            fprintf(stderr, "--spec needs --incremental; ignoring --spec\\n");\n            spec_n = 0;\n            spec_limit = 0;\n            spec_cur = 0;\n'''
if old not in s:
    raise SystemExit("incremental spec disable anchor not found")
s = s.replace(old, new, 1)

# Hybrid defaults: fixed mode still implies 4. Auto without an explicit ceiling was
# already initialised to 8 above and starts at 4.
old = '''            if (spec_n <= 0) spec_n = 4;\n            if (spec_n > K3_SPEC_MAX) spec_n = K3_SPEC_MAX;\n            if (!spec_snap) {\n'''
new = '''            if (spec_n <= 0) {\n                spec_n = 4;\n                spec_limit = spec_n;\n                spec_cur = spec_n;\n            }\n            if (!spec_snap) {\n'''
if old not in s:
    raise SystemExit("hybrid spec default anchor not found")
s = s.replace(old, new, 1)

old = '''            printf("hybrid decode: draft trunk %s (%.1f GB budget), top-%d%s, proposes "\n                   "up to %d tokens per sweep;\\n               the exact model verifies "\n                   "every one before it is emitted\\n\\n", draft_dir, draft_gb, draft_topk,\n                   draft_cache_only ? ", cache-only" : "", spec_n);\n'''
new = '''            printf("hybrid decode: draft trunk %s (%.1f GB budget), top-%d%s, proposes "\n                   "%s%d tokens per sweep;\\n               the exact model verifies "\n                   "every one before it is emitted\\n\\n", draft_dir, draft_gb, draft_topk,\n                   draft_cache_only ? ", cache-only" : "",\n                   spec_auto ? "adaptively up to " : "up to ",\n                   spec_auto ? spec_limit : spec_n);\n'''
if old not in s:
    raise SystemExit("hybrid print anchor not found")
s = s.replace(old, new, 1)

# Per-generation-step width. The ceiling remains spec_n/spec_limit for allocations; only
# the number of proposals made and verified changes.
old = '''            const int base = w.cached;\n            int d[K3_SPEC_MAX], nd = 0;\n            if (spec_snap && T + spec_n + 1 < Tmax && base + spec_n + 1 <= w.kv_cap) {\n'''
new = '''            const int base = w.cached;\n            int d[K3_SPEC_MAX], nd = 0;\n            const int spec_now = spec_auto ? spec_cur : spec_n;\n            if (spec_snap && spec_now > 0 &&\n                T + spec_now + 1 < Tmax && base + spec_now + 1 <= w.kv_cap) {\n'''
if old not in s:
    raise SystemExit("generation spec bound anchor not found")
s = s.replace(old, new, 1)

old = '''                    while (nd < spec_n) {\n'''
new = '''                    while (nd < spec_now) {\n'''
if old not in s:
    raise SystemExit("draft while anchor not found")
s = s.replace(old, new, 1)

old = '''                    nd = spec_draft(seq, T, spec_n, d);\n'''
new = '''                    nd = spec_draft(seq, T, spec_now, d);\n'''
if old not in s:
    raise SystemExit("ngram draft anchor not found")
s = s.replace(old, new, 1)

# Adapt only after exact verification has established m, and before the next generation
# round. A fully accepted sweep increases by one; a sweep accepting less than half cuts
# the width roughly in half. Medium acceptance leaves it stable. This reacts quickly to
# bad drafts without oscillating on one near-miss.
anchor = '''                    if (m == nd) {\n                        /* every fed position had true context; state is exact */\n                        w.cached = base + nd + 1;\n                    } else {\n'''
replacement = '''                    if (spec_auto) {\n                        spec_auto_rounds++;\n                        spec_auto_proposed += nd;\n                        spec_auto_accepted += m;\n                        if (m == nd && spec_cur < spec_limit) {\n                            spec_cur++;\n                            spec_auto_grows++;\n                        } else if (m * 2 < nd && spec_cur > 1) {\n                            int next = (spec_cur + 1) / 2;\n                            if (next < 1) next = 1;\n                            if (next < spec_cur) {\n                                spec_cur = next;\n                                spec_auto_shrinks++;\n                            }\n                        }\n                    }\n                    if (m == nd) {\n                        /* every fed position had true context; state is exact */\n                        w.cached = base + nd + 1;\n                    } else {\n'''
if anchor not in s:
    raise SystemExit("adaptive update anchor not found")
s = s.replace(anchor, replacement, 1)

# Human-readable end summary.
old = '''    if (dw.trunk && hyb_rounds > 0) {\n        printf("\\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "\n'''
new = '''    if (spec_auto && spec_auto_rounds > 0) {\n        printf("\\nadaptive speculation: %ld verified rounds, %ld/%ld drafts accepted (%.1f%%), "\n               "final horizon %d/%d, grows %d, shrinks %d\\n",\n               spec_auto_rounds, spec_auto_accepted, spec_auto_proposed,\n               spec_auto_proposed ? 100.0 * spec_auto_accepted / spec_auto_proposed : 0.0,\n               spec_cur, spec_limit, spec_auto_grows, spec_auto_shrinks);\n    }\n    if (dw.trunk && hyb_rounds > 0) {\n        printf("\\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "\n'''
if old not in s:
    raise SystemExit("adaptive summary anchor not found")
s = s.replace(old, new, 1)

# Machine-readable metrics let CI prove that the adaptive path actually changed width.
old = '''                   "\\\"draft_seconds\\\":%.6f,\\\"verify_seconds\\\":%.6f,"\n                   "\\\"seconds_per_token\\\":%.4f}\\n",\n                NL, compute_threads, temperature, top_p, sample_seed, stop_id,\n                hyb_rounds, hyb_drafted, hyb_accepted,\n                hyb_drafted ? (double)hyb_accepted / hyb_drafted : 0.0,\n                hyb_draft_s, hyb_verify_s, t_total / nout);\n'''
new = '''                   "\\\"draft_seconds\\\":%.6f,\\\"verify_seconds\\\":%.6f,"\n                   "\\\"spec_auto\\\":%d,\\\"spec_auto_rounds\\\":%ld,"\n                   "\\\"spec_auto_proposed\\\":%ld,\\\"spec_auto_accepted\\\":%ld,"\n                   "\\\"spec_auto_final\\\":%d,\\\"spec_auto_limit\\\":%d,"\n                   "\\\"spec_auto_grows\\\":%d,\\\"spec_auto_shrinks\\\":%d,"\n                   "\\\"seconds_per_token\\\":%.4f}\\n",\n                NL, compute_threads, temperature, top_p, sample_seed, stop_id,\n                hyb_rounds, hyb_drafted, hyb_accepted,\n                hyb_drafted ? (double)hyb_accepted / hyb_drafted : 0.0,\n                hyb_draft_s, hyb_verify_s, spec_auto, spec_auto_rounds,\n                spec_auto_proposed, spec_auto_accepted, spec_cur, spec_limit,\n                spec_auto_grows, spec_auto_shrinks, t_total / nout);\n'''
if old not in s:
    raise SystemExit("JSON metrics anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("applied greedy adaptive speculation")
