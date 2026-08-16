#!/usr/bin/env python3
"""Guarded one-shot patch for local K3 benchmark/API parity.

Refuses any non-unique source edit. Deleted before merge.
"""

from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {n}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]

# Make --------------------------------------------------------------------------
p = root / "Makefile"
s = p.read_text()
s = once(s, "ENGINE_SRC := src/core/k3_ops.c \\\n", "ENGINE_SRC := src/core/k3_ops.c src/core/k3_sample.c \\\n", "Make ENGINE_SRC")
s = once(
    s,
    "UNIT_TESTS := test_ops test_cache test_st test_cfg test_tok test_trunk_codec scale_test k3_model",
    "UNIT_TESTS := test_ops test_sample test_cache test_st test_cfg test_tok test_trunk_codec scale_test k3_model",
    "Make UNIT_TESTS",
)
s = once(
    s,
    "$(BIN)/test_ops: tests/unit/test_ops.c $(BUILD)/src/core/k3_ops.o | $(BIN)\n\t$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)\n",
    "$(BIN)/test_ops: tests/unit/test_ops.c $(BUILD)/src/core/k3_ops.o | $(BIN)\n"
    "\t$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)\n\n"
    "$(BIN)/test_sample: tests/unit/test_sample.c $(BUILD)/src/core/k3_sample.o | $(BIN)\n"
    "\t$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)\n",
    "Make test_sample target",
)
s = once(
    s,
    '\t@echo "== op kernels ==";        ./$(BIN)/test_ops $(FIXTURES)/ops\n',
    '\t@echo "== op kernels ==";        ./$(BIN)/test_ops $(FIXTURES)/ops\n'
    '\t@echo "== sampler ==";           ./$(BIN)/test_sample\n',
    "Make sampler gate",
)
p.write_text(s)

# CMake -------------------------------------------------------------------------
p = root / "CMakeLists.txt"
s = p.read_text()
s = once(s, "  src/core/k3_ops.c\n", "  src/core/k3_ops.c\n  src/core/k3_sample.c\n", "CMake source")
s = once(s, "k3_add_test(test_ops    tests/unit/test_ops.c)\n", "k3_add_test(test_ops    tests/unit/test_ops.c)\nk3_add_test(test_sample tests/unit/test_sample.c)\n", "CMake target")
s = once(s, "add_test(NAME ops        COMMAND test_ops   ${FIX}/ops)\n", "add_test(NAME ops        COMMAND test_ops   ${FIX}/ops)\nadd_test(NAME sampler    COMMAND test_sample)\n", "CMake ctest")
p.write_text(s)

# CLI ----------------------------------------------------------------------------
p = root / "src/cli/k3_run.c"
s = p.read_text()
s = once(
    s,
    '#include "k3_cfg.h"   /* read the checkpoint\'s own config rather than assuming it */\n',
    '#include "k3_cfg.h"   /* read the checkpoint\'s own config rather than assuming it */\n#include "k3_sample.h" /* benchmark-parity temperature/top-p sampling */\n',
    "CLI include",
)
s = once(
    s,
    '"  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\\n"\n',
    '"  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\\n"\n'
    '"  --ids-file PATH       raw token ids from a comma/space/newline separated file;\\n"\n'
    '"                        avoids argv limits for long official XTML prompts\\n"\n',
    "CLI usage ids-file",
)
s = once(
    s,
    '"  --gen N               tokens to generate (default 8)\\n"\n'
    '"  --threads N           OpenMP compute threads for this run. Exact output is unchanged;\\n"\n',
    '"  --gen N               tokens to generate (default 8)\\n"\n'
    '"  --temperature X       sampling temperature; 0 = legacy greedy (default 0)\\n"\n'
    '"  --top-p X             nucleus cutoff in (0,1], default 1.0\\n"\n'
    '"  --seed N              deterministic sampler seed (default 1)\\n"\n'
    '"  --stop-id N           stop after emitting this token id (default disabled)\\n"\n'
    '"  --threads N           OpenMP compute threads for this run. Exact output is unchanged;\\n"\n',
    "CLI usage sampling",
)
s = once(s, '    const char *ids_s = NULL, *outp = "k3_run.json", *trunk_dir = NULL;\n', '    const char *ids_s = NULL, *ids_file = NULL, *outp = "k3_run.json", *trunk_dir = NULL;\n', "CLI ids-file var")
s = once(
    s,
    "    int gen = 8, want_layers = -1;\n    int threads = 0;              /* 0 = OpenMP/runtime default */\n    double cache_gb = 64.0, trunk_gb = 16.0;\n",
    "    int gen = 8, want_layers = -1;\n    int threads = 0;              /* 0 = OpenMP/runtime default */\n    double temperature = 0.0, top_p = 1.0;\n    unsigned long long sample_seed = 1;\n    int stop_id = -1;\n    double cache_gb = 64.0, trunk_gb = 16.0;\n",
    "CLI sample vars",
)
s = once(s, '        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];\n', '        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_s = argv[++i];\n        else if (!strcmp(argv[i], "--ids-file") && i + 1 < argc) ids_file = argv[++i];\n', "CLI parse ids-file")
s = once(
    s,
    '        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);\n        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);\n',
    '        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) temperature = atof(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) sample_seed = strtoull(argv[++i], NULL, 10);\n'
    '        else if (!strcmp(argv[i], "--stop-id") && i + 1 < argc) stop_id = atoi(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);\n',
    "CLI parse sampler",
)
s = once(
    s,
    "        int nsrc = (ids_s != NULL) + (prompt_text != NULL) + (prompt_file != NULL);\n"
    "        if (nsrc == 0) {\n"
    '            fprintf(stderr, "one of --ids, --prompt or --prompt-file is required\\n");\n',
    "        int nsrc = (ids_s != NULL) + (ids_file != NULL) +\n"
    "                   (prompt_text != NULL) + (prompt_file != NULL);\n"
    "        if (nsrc == 0) {\n"
    '            fprintf(stderr, "one of --ids, --ids-file, --prompt or --prompt-file is required\\n");\n',
    "CLI prompt source count",
)
s = once(s, '            fprintf(stderr, "--ids, --prompt and --prompt-file are mutually exclusive\\n");\n', '            fprintf(stderr, "--ids, --ids-file, --prompt and --prompt-file are mutually exclusive\\n");\n', "CLI prompt mutual exclusion")
s = once(
    s,
    "    if (threads < 0 || threads > 4096) {\n",
    "    if (!isfinite(temperature) || temperature < 0.0) {\n"
    '        fprintf(stderr, "--temperature must be finite and >= 0\\n");\n        return 2;\n    }\n'
    "    if (!isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) {\n"
    '        fprintf(stderr, "--top-p must be finite and in (0,1]\\n");\n        return 2;\n    }\n'
    "    if (stop_id < -1) {\n"
    '        fprintf(stderr, "--stop-id must be >= -1\\n");\n        return 2;\n    }\n'
    "    if (temperature > 0.0 && (spec_n > 0 || draft_dir != NULL)) {\n"
    '        fprintf(stderr, "sampled speculative decoding is not implemented yet; remove --spec/--draft-trunk\\n"\n'
    '                        "rather than silently changing the requested sampling distribution.\\n");\n'
    "        return 2;\n    }\n\n"
    "    if (threads < 0 || threads > 4096) {\n",
    "CLI sampler validation",
)
s = once(
    s,
    "    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {\n",
    "    if (stop_id >= c.vocab) {\n"
    '        fprintf(stderr, "--stop-id %d is outside vocabulary [0,%d)\\n", stop_id, c.vocab);\n'
    "        return 2;\n    }\n"
    "    K3Sampler sampler;\n"
    "    k3_sampler_init(&sampler, temperature, top_p, (uint64_t)sample_seed);\n\n"
    "    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {\n",
    "CLI sampler init",
)
s = once(
    s,
    '    printf("  threads  : %d compute%s\\n", compute_threads,\n           threads > 0 ? " (explicit --threads)" : " (OpenMP/runtime default)");\n',
    '    printf("  threads  : %d compute%s\\n", compute_threads,\n           threads > 0 ? " (explicit --threads)" : " (OpenMP/runtime default)");\n'
    '    if (temperature <= 0.0) printf("  sampling : greedy\\n");\n'
    '    else printf("  sampling : temperature %.6g, top-p %.6g, seed %llu\\n",\n                temperature, top_p, sample_seed);\n',
    "CLI sampler banner",
)
old_parse = '''    } else {
        for (const char *p = ids_s; *p && np < K3_MAX_PROMPT; ) {
            prompt[np++] = (int)strtol(p, (char **)&p, 10);
            while (*p == ',' || *p == ' ') p++;
        }
    }
'''
new_parse = '''    } else {
        char *owned_ids = NULL;
        const char *src_ids = ids_s;
        if (ids_file) {
            FILE *idf = fopen(ids_file, "rb");
            if (!idf) { perror(ids_file); return 2; }
            if (fseek(idf, 0, SEEK_END) != 0) { fclose(idf); return 2; }
            long idlen = ftell(idf);
            if (idlen < 0 || fseek(idf, 0, SEEK_SET) != 0) { fclose(idf); return 2; }
            owned_ids = (char *)malloc((size_t)idlen + 1);
            if (!owned_ids) { fclose(idf); fprintf(stderr, "out of memory reading --ids-file\\n"); return 1; }
            if (fread(owned_ids, 1, (size_t)idlen, idf) != (size_t)idlen) {
                fclose(idf); free(owned_ids); fprintf(stderr, "%s is truncated while reading\\n", ids_file); return 2;
            }
            fclose(idf);
            owned_ids[idlen] = 0;
            src_ids = owned_ids;
        }
        for (const char *q = src_ids; *q && np < K3_MAX_PROMPT; ) {
            char *end = NULL;
            long id = strtol(q, &end, 10);
            if (end == q) {
                fprintf(stderr, "invalid token id near: %.32s\\n", q);
                free(owned_ids);
                return 2;
            }
            prompt[np++] = (int)id;
            q = end;
            while (*q == ',' || *q == ' ' || *q == '\\n' || *q == '\\r' || *q == '\\t') q++;
        }
        free(owned_ids);
    }
'''
s = once(s, old_parse, new_parse, "CLI ids-file reader")
s = once(s, "if (frc == 0) { w.cached = base + nT0; emit[emitn++] = argmax_(lg, c.vocab); }", "if (frc == 0) { w.cached = base + nT0; emit[emitn++] = k3_sample_token(&sampler, lg, c.vocab); }", "sample initial")
s = once(s, "if (frc == 0) { w.cached = base + 1; emit[emitn++] = argmax_(lg, c.vocab); }", "if (frc == 0) { w.cached = base + 1; emit[emitn++] = k3_sample_token(&sampler, lg, c.vocab); }", "sample incremental")
s = once(s, "if (frc == 0) emit[emitn++] = argmax_(lg, c.vocab);", "if (frc == 0) emit[emitn++] = k3_sample_token(&sampler, lg, c.vocab);", "sample recompute")
s = once(
    s,
    "        /* Abort the run rather than argmax a buffer the forward never wrote. */\n        if (frc != 0 || emitn == 0) {\n",
    "        for (int i = 0; i < emitn; i++) if (emit[i] < 0) frc = -1;\n"
    "        /* Abort the run rather than consume a buffer the forward/sampler never wrote. */\n        if (frc != 0 || emitn == 0) {\n",
    "sample failure",
)
s = once(
    s,
    "        for (int i = 0; i < emitn && nout < gen && T < Tmax; i++) {\n            seq[T++] = emit[i];\n            outtok[nout++] = emit[i];\n        }\n        if (T >= Tmax) break;\n",
    "        int stop_hit = 0;\n        for (int i = 0; i < emitn && nout < gen && T < Tmax; i++) {\n            seq[T++] = emit[i];\n            outtok[nout++] = emit[i];\n            if (stop_id >= 0 && emit[i] == stop_id) { stop_hit = 1; break; }\n        }\n        if (stop_hit || T >= Tmax) break;\n",
    "stop id",
)
s = once(
    s,
    '        fprintf(f, "],\\\"layers\\\":%d,\\\"threads\\\":%d,\\\"seconds_per_token\\\":%.4f}\\n",\n                NL, compute_threads, t_total / nout);\n',
    '        fprintf(f, "],\\\"layers\\\":%d,\\\"threads\\\":%d,\\\"temperature\\\":%.9g,"\n'
    '                   "\\\"top_p\\\":%.9g,\\\"seed\\\":%llu,\\\"stop_id\\\":%d,"\n'
    '                   "\\\"seconds_per_token\\\":%.4f}\\n",\n'
    '                NL, compute_threads, temperature, top_p, sample_seed, stop_id, t_total / nout);\n',
    "JSON sampling metadata",
)
p.write_text(s)

# Local Python parser cleanup ----------------------------------------------------
p = root / "local/k3_local.py"
s = p.read_text()
first = '''        call_re = re.compile(
            re.escape(O) + r"call(?P<attrs>[^" + re.escape(S[0]) + r"]*)" + re.escape(S)
            + r"(?P<body>.*?)" + re.escape(C + "call" + S),
            re.DOTALL,
        )
        # The generic regex above is deliberately permissive about attributes, but the
        # special-token delimiter itself is easier and safer to parse with finditer on a
        # second expression that cannot cross a call close marker.
'''
s = once(s, first, "        # Calls and arguments are delimited by K3's control-token separators.\n", "local parser cleanup")
p.write_text(s)

# CI: local Python is part of the product, so lint it. --------------------------
p = root / ".github/workflows/ci.yml"
s = p.read_text()
s = once(
    s,
    "      - run: ruff check tools/ --output-format=github\n      - name: Syntax check every tool\n        run: python -m compileall -q tools/\n",
    "      - run: ruff check tools/ local/ --output-format=github\n      - name: Syntax check every Python entrypoint\n        run: python -m compileall -q tools/ local/\n",
    "CI local lint",
)
p.write_text(s)

print("local max parity staging patch applied")
