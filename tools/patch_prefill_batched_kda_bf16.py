#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

if "K3_NO_PREFILL_BATCH_KDA_BF16" in s:
    print("exact KDA BF16 prefill batching already applied")
    raise SystemExit(0)

old = '''    return 3 * (size_t)T * P        /* q, k, v after conv            */
         + 2 * (size_t)T * P        /* z then alpha                  */
         + (size_t)T * c->kda_heads /* beta                          */
         + (size_t)T * P            /* recurrence output             */
         + 2 * P                    /* gate buffer and one work row  */
         + (size_t)c->kda_head_dim; /* f_a output                    */
'''
new = '''    return 3 * (size_t)T * P        /* q, k, v after conv            */
         + 2 * (size_t)T * P        /* z then alpha                  */
         + (size_t)T * c->kda_heads /* beta                          */
         + (size_t)T * P            /* recurrence output             */
         + 2 * P                    /* gate buffer and one work row  */
         + (size_t)T * c->kda_head_dim; /* batched f_a output          */
'''
if old not in s:
    raise SystemExit("KDA scratch anchor not found")
s = s.replace(old, new, 1)

old = '''    float *o  = bt + (size_t)T * H;      float *gb = o + (size_t)T * P;
    float *wr = gb + P;                  float *fa = wr + P;

    /* 1. projections */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        k3_mmw(q + (size_t)t * P, xt, w->q, w->wdt, E, P);
        k3_mmw(k + (size_t)t * P, xt, w->k, w->wdt, E, P);
        k3_mmw(v + (size_t)t * P, xt, w->v, w->wdt, E, P);
        k3_mmw(bt + (size_t)t * H, xt, w->b, w->wdt, E, H);
        /* ONE shared low-rank pair feeds every head: [E->D] then [D->H*D] */
        k3_mmw(fa, xt, w->f_a, w->wdt, E, D);
        k3_mmw(z + (size_t)t * P, fa, w->f_b, w->wdt, D, P);
    }
'''
new = '''    float *o  = bt + (size_t)T * H;      float *gb = o + (size_t)T * P;
    float *wr = gb + P;                  float *fa = wr + P; /* [T][D] */

    /* Exact prefill batching for the always-active BF16 KDA projections. The KDA
     * recurrence itself remains strictly sequential in token order. Only independent
     * matvecs share weight widening/cache traffic across the prompt chunk. */
    static int no_prefill_batch_kda_bf16 = -1;
    if (no_prefill_batch_kda_bf16 < 0)
        no_prefill_batch_kda_bf16 = getenv("K3_NO_PREFILL_BATCH_KDA_BF16") ? 1 : 0;
    const int batch_kda = !no_prefill_batch_kda_bf16 && T > 1
                       && T <= K3_PREFILL_BATCH_MAX && w->wdt == K3_WBF16;

    /* 1. projections */
    if (batch_kda) {
        const float *xp[K3_PREFILL_BATCH_MAX];
        const float *fp[K3_PREFILL_BATCH_MAX];
        for (int t = 0; t < T; t++) xp[t] = x + (size_t)t * E;

        k3_matmul_bf16_batch(q,  P, xp, T, (const uint16_t *)w->q, E, P);
        k3_matmul_bf16_batch(k,  P, xp, T, (const uint16_t *)w->k, E, P);
        k3_matmul_bf16_batch(v,  P, xp, T, (const uint16_t *)w->v, E, P);
        k3_matmul_bf16_batch(bt, H, xp, T, (const uint16_t *)w->b, E, H);
        /* ONE shared low-rank pair feeds every head: [E->D] then [D->H*D]. */
        k3_matmul_bf16_batch(fa, D, xp, T, (const uint16_t *)w->f_a, E, D);
        for (int t = 0; t < T; t++) fp[t] = fa + (size_t)t * D;
        k3_matmul_bf16_batch(z, P, fp, T, (const uint16_t *)w->f_b, D, P);
    } else {
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * E;
            k3_mmw(q + (size_t)t * P, xt, w->q, w->wdt, E, P);
            k3_mmw(k + (size_t)t * P, xt, w->k, w->wdt, E, P);
            k3_mmw(v + (size_t)t * P, xt, w->v, w->wdt, E, P);
            k3_mmw(bt + (size_t)t * H, xt, w->b, w->wdt, E, H);
            k3_mmw(fa, xt, w->f_a, w->wdt, E, D);
            k3_mmw(z + (size_t)t * P, fa, w->f_b, w->wdt, D, P);
        }
    }
'''
if old not in s:
    raise SystemExit("KDA projection anchor not found")
s = s.replace(old, new, 1)

old = '''    /* 7/8/9. head-wise RMSNorm, THEN the gate, THEN the output projection */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = o + (size_t)t * P;
        for (int h = 0; h < H; h++)
            k3_rmsnorm(ot + (size_t)h * D, ot + (size_t)h * D, w->o_norm, D, c->rms_eps);
        k3_mmw(gb, xt, w->g, w->wdt, E, P);
        for (int i = 0; i < P; i++) ot[i] *= sigmoidf_(gb[i]);
        k3_mmw(out + (size_t)t * E, ot, w->o, w->wdt, P, E);
    }
'''
new = '''    /* 7. head-wise RMSNorm. This remains per token/head and byte-identical. */
    for (int t = 0; t < T; t++) {
        float *ot = o + (size_t)t * P;
        for (int h = 0; h < H; h++)
            k3_rmsnorm(ot + (size_t)h * D, ot + (size_t)h * D,
                       w->o_norm, D, c->rms_eps);
    }

    /* 8/9. Gate and output projection. q is dead after the recurrence, so the batched
     * gate can reuse its [T][P] storage without increasing the real KDA scratch. */
    if (batch_kda) {
        const float *xp[K3_PREFILL_BATCH_MAX];
        const float *op[K3_PREFILL_BATCH_MAX];
        for (int t = 0; t < T; t++) xp[t] = x + (size_t)t * E;
        k3_matmul_bf16_batch(q, P, xp, T, (const uint16_t *)w->g, E, P);
        for (int t = 0; t < T; t++) {
            float *ot = o + (size_t)t * P;
            const float *gt = q + (size_t)t * P;
            for (int i = 0; i < P; i++) ot[i] *= sigmoidf_(gt[i]);
            op[t] = ot;
        }
        k3_matmul_bf16_batch(out, E, op, T, (const uint16_t *)w->o, P, E);
    } else {
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * E;
            float *ot = o + (size_t)t * P;
            k3_mmw(gb, xt, w->g, w->wdt, E, P);
            for (int i = 0; i < P; i++) ot[i] *= sigmoidf_(gb[i]);
            k3_mmw(out + (size_t)t * E, ot, w->o, w->wdt, P, E);
        }
    }
'''
if old not in s:
    raise SystemExit("KDA gate/output anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("applied exact KDA BF16 prefill batching")
