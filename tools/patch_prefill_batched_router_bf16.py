#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

if "K3_NO_PREFILL_BATCH_ROUTER_BF16" in s:
    print("exact BF16 router prefill batching already applied")
    raise SystemExit(0)

# Insert the exact batched BF16 router immediately after the existing scalar router.
anchor = '''    free(score); free(choice);\n}\n\n/* --------------------------------------------------------------- AttnRes ---- */\n'''
helper = r'''    free(score); free(choice);
}

/* Exact BF16 router for a prompt chunk. This changes only the order in which TOKENS
 * are visited, never the arithmetic order inside any token's expert dot product.
 *
 * Scalar production path for token t / expert e:
 *     acc = 0; for i=0..hidden-1: acc += bf16(W[e,i]) * x[t,i]
 *
 * This path keeps one acc[t] per token and runs the SAME i=0..hidden-1 sequence for
 * each of them while sharing the BF16 widening/load of W[e,i]. Consequently every
 * score, sigmoid, bias correction, repeated-max top-k decision and combining weight is
 * bit-identical to k3_router. The streamed K3 binder keeps router gates in native BF16,
 * so this removes repeated ~12.8 MB gate reads per MoE layer during prompt prefill.
 */
static void k3_router_batch_bf16(int *idx, float *wt, const float *x,
                                  const uint16_t *W, const float *bias,
                                  int T, int hidden, int n_experts, int topk,
                                  int renorm, float routed_scale)
{
    if (T <= 0) return;
    if (T == 1 || T > K3_PREFILL_BATCH_MAX) {
        for (int t = 0; t < T; t++)
            k3_router(idx + (size_t)t * topk, wt + (size_t)t * topk,
                      x + (size_t)t * hidden, W, K3_WBF16, bias,
                      hidden, n_experts, topk, renorm, routed_scale);
        return;
    }

    const size_t N = (size_t)T * n_experts;
    float *score = (float *)malloc(N * sizeof(float));
    float *choice = (float *)malloc(N * sizeof(float));
    if (!score || !choice)
        k3_fatal_oom("batched BF16 router scores", N * sizeof(float) * 2);

#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int e = 0; e < n_experts; e++) {
        const uint16_t *row = W + (size_t)e * hidden;
        double acc[K3_PREFILL_BATCH_MAX] = {0};
        for (int i = 0; i < hidden; i++) {
            const double wi = (double)k3_bf16f(row[i]);
            for (int t = 0; t < T; t++)
                acc[t] += wi * (double)x[(size_t)t * hidden + i];
        }
        for (int t = 0; t < T; t++) {
            const size_t q = (size_t)t * n_experts + e;
            score[q] = 1.0f / (1.0f + expf(-(float)acc[t]));
            choice[q] = score[q] + (bias ? bias[e] : 0.0f);
        }
    }

    /* Preserve the scalar router's exact post-dot-product order independently for every
     * token, including deterministic first-index tie handling. */
    for (int t = 0; t < T; t++) {
        float *st = score + (size_t)t * n_experts;
        float *ct = choice + (size_t)t * n_experts;
        int *it = idx + (size_t)t * topk;
        float *ww = wt + (size_t)t * topk;
        for (int j = 0; j < topk; j++) {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < n_experts; e++)
                if (ct[e] > bv) { bv = ct[e]; best = e; }
            if (best < 0) { it[j] = 0; ww[j] = 0.0f; continue; }
            it[j] = best;
            ww[j] = st[best];
            ct[best] = -INFINITY;
        }
        if (renorm && topk > 1) {
            double z = 0.0;
            for (int j = 0; j < topk; j++) z += (double)ww[j];
            const float inv = (float)(1.0 / (z + 1e-20));
            for (int j = 0; j < topk; j++) ww[j] *= inv;
        }
        for (int j = 0; j < topk; j++) ww[j] *= routed_scale;
    }

    free(score); free(choice);
}

/* --------------------------------------------------------------- AttnRes ---- */
'''
if anchor not in s:
    raise SystemExit("router helper insertion anchor not found")
s = s.replace(anchor, helper, 1)

old = '''    static int no_prefill_batch_bf16 = -1;\n    if (no_prefill_batch_bf16 < 0)\n        no_prefill_batch_bf16 = getenv("K3_NO_PREFILL_BATCH_BF16") ? 1 : 0;\n    const int batch_bf16 = !no_prefill_batch_bf16 && T > 1 && w->wdt == K3_WBF16;\n\n    int nu = 0;\n    for (int t = 0; t < T; t++) {\n        const float *xt = x + (size_t)t * E;\n        int   *it = ridx + (size_t)t * K;\n        float *wtt = rwt + (size_t)t * K;\n        k3_router(it, wtt, xt, w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,\n                  c->moe_renorm, c->routed_scale);\n        if (!batch_bf16)\n            k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);\n        for (int j = 0; j < K; j++) {\n            const int e = it[j];\n            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }\n        }\n    }\n'''
new = '''    static int no_prefill_batch_bf16 = -1;\n    if (no_prefill_batch_bf16 < 0)\n        no_prefill_batch_bf16 = getenv("K3_NO_PREFILL_BATCH_BF16") ? 1 : 0;\n    const int batch_bf16 = !no_prefill_batch_bf16 && T > 1 && w->wdt == K3_WBF16;\n\n    static int no_prefill_batch_router_bf16 = -1;\n    if (no_prefill_batch_router_bf16 < 0)\n        no_prefill_batch_router_bf16 = getenv("K3_NO_PREFILL_BATCH_ROUTER_BF16") ? 1 : 0;\n    const int batch_router_bf16 = !no_prefill_batch_router_bf16 && T > 1\n                                && w->gate_wdt == K3_WBF16;\n    if (batch_router_bf16)\n        k3_router_batch_bf16(ridx, rwt, x, (const uint16_t *)w->gate, w->bias,\n                             T, E, c->n_experts, K, c->moe_renorm, c->routed_scale);\n\n    int nu = 0;\n    for (int t = 0; t < T; t++) {\n        const float *xt = x + (size_t)t * E;\n        int   *it = ridx + (size_t)t * K;\n        float *wtt = rwt + (size_t)t * K;\n        if (!batch_router_bf16)\n            k3_router(it, wtt, xt, w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,\n                      c->moe_renorm, c->routed_scale);\n        if (!batch_bf16)\n            k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);\n        for (int j = 0; j < K; j++) {\n            const int e = it[j];\n            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }\n        }\n    }\n'''
if old not in s:
    raise SystemExit("MoE router prefill anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("applied exact BF16 router prefill batching")
