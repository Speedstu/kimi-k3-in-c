#!/usr/bin/env python3
from pathlib import Path
import sys


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, found {n}")
    return text.replace(old, new, 1)


def patch_header(text):
    return replace_once(text,
'''    int          cache_only;
} K3MoeW;''',
'''    int          cache_only;
    /* Draft-only: reduce routed experts from config top-k to this value. Zero keeps the
     * checkpoint's exact top-k. The exact model never sets this: only a speculative
     * draft may use it, and every emitted token is still verified by exact K3. */
    int          topk_override;
} K3MoeW;''',
"K3MoeW.topk_override")


def patch_ops(text):
    start = text.find("void k3_kda_step(float *S, float *o, const float *q, const float *k,")
    if start < 0:
        raise RuntimeError("k3_kda_step start not found")
    end_marker = "    free(u);\n}\n\n/* ---------------------------------------------------------------- matmul ---- */"
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("k3_kda_step end not found")
    end += len("    free(u);\n}\n")
    new_kda = r'''/* Internal form with caller-owned u scratch. The hot model path has a D-float
 * slice that is dead after q has been pre-scaled, so using it here removes one
 * calloc/free pair per KDA head and token without changing any arithmetic. */
static void k3_kda_step_ws(float *S, float *o, const float *q, const float *k,
                           const float *v, const float *alpha, float beta,
                           int dk, int dv, float *u)
{
    /* 1. channel-wise decay */
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float a = alpha[i];
        for (int j = 0; j < dv; j++) row[j] *= a;
    }

    /* 2. read the state along k: u = S^T k */
    for (int j = 0; j < dv; j++) u[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) u[j] += ki * row[j];
    }

    /* 3. rank-one delta write */
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) row[j] += ki * beta * (v[j] - u[j]);
    }

    /* 4. output from the already-updated state */
    for (int j = 0; j < dv; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float qi = q[i];
        if (qi == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) o[j] += qi * row[j];
    }
}

void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv)
{
    /* Public compatibility wrapper. Standalone callers retain the old API; the model
     * hot path below supplies existing scratch and performs no heap allocation. */
    float *u = (float *)malloc((size_t)dv * sizeof(float));
    if (!u) k3_fatal_oom("KDA recurrence temporary", (size_t)dv * sizeof(float));
    k3_kda_step_ws(S, o, q, k, v, alpha, beta, dk, dv, u);
    free(u);
}
'''
    text = text[:start] + new_kda + text[end:]

    text = replace_once(text,
'''            for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
            k3_kda_step(S + (size_t)h * D * D, o + off, wh, k + off, v + off,
                        al + off, bt[(size_t)t * H + h], D, D);''',
'''            for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
            /* q[off:off+D] is dead after the copy above. Reuse it as the recurrence's
             * D-float u workspace, removing heap traffic from the real KDA hot path. */
            k3_kda_step_ws(S + (size_t)h * D * D, o + off, wh, k + off, v + off,
                           al + off, bt[(size_t)t * H + h], D, D, q + off);''',
"KDA workspace")

    text = replace_once(text,
'''{
    const int E = c->hidden, L = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared;

    float *z''',
'''{
    const int E = c->hidden, L = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared;
    const int K = (w->topk_override > 0 && w->topk_override < c->topk)
                ? w->topk_override : c->topk;

    float *z''',
"MoE effective K")

    text = replace_once(text,
'''k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, c->topk,
                  c->moe_renorm, c->routed_scale);

        int nk = c->topk;''',
'''k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);

        int nk = K;''',
"MoE router K")

    text = replace_once(text,
"            for (int j = 0; j < c->topk; j++) {\n",
"            for (int j = 0; j < K; j++) {\n",
"cache-only effective K")

    text = replace_once(text,
'''    const int E = c->hidden, Ll = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared, K = c->topk;''',
'''    const int E = c->hidden, Ll = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared;
    const int K = (w->topk_override > 0 && w->topk_override < c->topk)
                ? w->topk_override : c->topk;''',
"prefill effective K")
    return text


def patch_cli(text):
    text = replace_once(text,
'''"  --draft-trunk-gb X    trunk budget for the draft model (default 6)\\n"
"  --spec N''',
'''"  --draft-trunk-gb X    trunk budget for the draft model (default 6)\\n"
"  --draft-topk K        draft-only routed expert count (default 4); exact K3 stays\\n"
"                        at checkpoint top-k and verifies every emitted token\\n"
"  --draft-cache-only    additionally restrict draft routing to resident experts;\\n"
"                        zero expert reads, but useful only with a large cache\\n"
"  --spec N''',
"CLI help")

    text = replace_once(text,
'''    int          n_mla, kv_cap, cached;
    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
} Weights;''',
'''    int          n_mla, kv_cap, cached;
    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
    int          draft_topk;   /* 0 exact; >0 proposal-only reduced expert top-k */
} Weights;''',
"Weights.draft_topk")

    text = replace_once(text,
'''            w->lay[L].moe.cache_only = w->draft_mode;
        }''',
'''            w->lay[L].moe.cache_only = w->draft_mode;
            w->lay[L].moe.topk_override = w->draft_topk;
        }''',
"forward draft topk")

    text = replace_once(text,
'''    const char *draft_dir = NULL;
    double draft_gb = 6.0;''',
'''    const char *draft_dir = NULL;
    double draft_gb = 6.0;
    int draft_topk = 4;
    int draft_cache_only = 0;''',
"draft vars")

    text = replace_once(text,
'''        else if (!strcmp(argv[i], "--draft-trunk") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk-gb") && i + 1 < argc) draft_gb = atof(argv[++i]);''',
'''        else if (!strcmp(argv[i], "--draft-trunk") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk-gb") && i + 1 < argc) draft_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--draft-topk") && i + 1 < argc) draft_topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft-cache-only")) draft_cache_only = 1;''',
"draft parsing")

    text = replace_once(text,
'''    if (!real_cfg(&c, fa, 128, dir, cfg_path)) {
        fprintf(stderr, "ABORTED: the model config could not be read with confidence.\\n");
        return 2;
    }''',
'''    if (!real_cfg(&c, fa, 128, dir, cfg_path)) {
        fprintf(stderr, "ABORTED: the model config could not be read with confidence.\\n");
        return 2;
    }
    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {
        fprintf(stderr, "--draft-topk must be in [1,%d], got %d\\n", c.topk, draft_topk);
        return 2;
    }''',
"draft validation")

    text = replace_once(text,
'''            dw.cached = 0;
            dw.draft_mode = 1;   /* cache-only routing: draft tokens read no new experts */
            printf("hybrid decode: draft trunk %s (%.1f GB budget) proposes up to %d "
                   "tokens per sweep;\\n               the exact model verifies every one "
                   "before it is emitted\\n\\n", draft_dir, draft_gb, spec_n);''',
'''            dw.cached = 0;
            /* Reduced top-k is the default cheap draft. Cache-only remains opt-in:
             * tiny expert caches were measured to destroy draft acceptance. */
            dw.draft_mode = draft_cache_only;
            dw.draft_topk = draft_topk;
            printf("hybrid decode: draft trunk %s (%.1f GB budget), top-%d%s, proposes "
                   "up to %d tokens per sweep;\\n               the exact model verifies "
                   "every one before it is emitted\\n\\n", draft_dir, draft_gb, draft_topk,
                   draft_cache_only ? ", cache-only" : "", spec_n);''',
"draft init")
    return text


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    targets = [
        (root / "include/k3/k3.h", patch_header),
        (root / "src/core/k3_ops.c", patch_ops),
        (root / "src/cli/k3_run.c", patch_cli),
    ]
    changed = []
    for path, fn in targets:
        old = path.read_text(encoding="utf-8")
        new = fn(old)
        path.write_text(new, encoding="utf-8", newline="\n")
        changed.append(str(path.relative_to(root)))
    print("patched:", ", ".join(changed))


if __name__ == "__main__":
    main()
