#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected one match, found {n}')
    return s.replace(old, new, 1)

# ---- public API/types ---------------------------------------------------------------
p = root / 'include/k3/k3.h'
s = p.read_text()
s = one(s,
'''void k3_router(int *idx, float *w, const float *x, const float *W,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale);''',
'''void k3_router(int *idx, float *w, const float *x, const void *W, int wdt,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale);''',
'router declaration')
s = one(s,
'''    /* gate stays fp32 on purpose. k3_router carries its own inline matmul rather than
     * calling k3_matmul, so tagging it would change that function's signature and its
     * fixture, to save 1.18 GB out of about 114 GB. Not worth the churn.
     * w1/w3/w2 also stay fp32: that resident bank is indexed by pointer arithmetic and
     * is only ever used by the fixtures, because the real model streams experts
     * through src and multiplies them straight out of MXFP4. */
    const float *gate, *bias;            /* router: [n_experts][hidden], [n_experts] */''',
'''    /* The router gate is a large matrix and follows wdt just like the other trunk
     * matrices. On the exact checkpoint it remains BF16 and is widened on read inside
     * k3_router, eliminating a full gate-sized fp32 expansion per streamed layer. Draft
     * I8R/Q4G gates are consumed in their native proposal-only formats too. */
    const void  *gate;                   /* router: [n_experts][hidden], tagged by wdt */
    const float *bias;                   /* [n_experts], elementwise: stays fp32       */''',
'K3MoeW gate type/comment')
p.write_text(s, newline='\n')

# ---- router kernel + call sites -----------------------------------------------------
p = root / 'src/core/k3_ops.c'
s = p.read_text()
old_start = s.find('void k3_router(int *idx, float *w, const float *x, const float *W,')
old_end = s.find('\n/* --------------------------------------------------------------- AttnRes ---- */', old_start)
if old_start < 0 or old_end < 0:
    raise RuntimeError('router function bounds not found')
old_router = s[old_start:old_end]
# Preserve top-k tail from current function verbatim after the score loop marker.
marker = '''    /* top-k by repeated max. n_experts is 896 and topk is 16, so this is 14k
'''
pos = old_router.find(marker)
if pos < 0:
    raise RuntimeError('router top-k marker not found')
tail = old_router[pos:]
new_head = r'''void k3_router(int *idx, float *w, const float *x, const void *W, int wdt,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale)
{
    /* Returning early here would leave idx[] and w[] untouched, and k3_moe forms
     * expert pointers from them immediately afterward. Abort on allocation failure. */
    float *score  = (float *)malloc((size_t)n_experts * sizeof(float));
    float *choice = (float *)malloc((size_t)n_experts * sizeof(float));
    if (!score || !choice) k3_fatal_oom("router scores", (size_t)n_experts * sizeof(float) * 2);

    /* Exact FP32/BF16 paths deliberately preserve the original sequential i=0..hidden-1
     * double accumulation. The checkpoint gate is BF16; the old streamed binder widened
     * those exact BF16 values into FP32 first, so widening each element here yields the
     * same operand and the same addition order, hence bit-identical scores/top-k.
     *
     * I8R/Q4G occur only on speculative draft trunks. They may use their fast matmul
     * kernels because their logits are proposals: exact BF16 K3 verifies emitted tokens. */
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int e = 0; e < n_experts; e++) {
        double acc;
        if (wdt == K3_WBF16) {
            const uint16_t *row = (const uint16_t *)W + (size_t)e * hidden;
            acc = 0.0;
            for (int i = 0; i < hidden; i++)
                acc += (double)k3_bf16f(row[i]) * (double)x[i];
        } else if (wdt == K3_WF32) {
            const float *row = (const float *)W + (size_t)e * hidden;
            acc = 0.0;
            for (int i = 0; i < hidden; i++)
                acc += (double)row[i] * (double)x[i];
        } else {
            const unsigned char *row = (const unsigned char *)W
                                     + (size_t)e * k3_row_bytes(wdt, hidden);
            float draft_logit = 0.0f;
            k3_mmw(&draft_logit, x, row, wdt, hidden, 1);
            acc = (double)draft_logit;
        }
        score[e]  = 1.0f / (1.0f + expf(-(float)acc));
        choice[e] = score[e] + (bias ? bias[e] : 0.0f);
    }

'''
s = s[:old_start] + new_head + tail + s[old_end:]
# There are two current router calls in MoE normal/prefill paths. Type tag is the same
# K3MoeW.wdt that describes every large matrix in the packed layer.
s = s.replace('k3_router(it, wtt, xt, w->gate, w->bias, E, c->n_experts, K,',
              'k3_router(it, wtt, xt, w->gate, w->wdt, w->bias, E, c->n_experts, K,')
s = s.replace('k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, K,',
              'k3_router(idx, wt, xt, w->gate, w->wdt, w->bias, E, c->n_experts, K,')
p.write_text(s, newline='\n')

# ---- binder: gate becomes normal narrow matrix -------------------------------------
p = root / 'src/model/k3_bind.c'
s = p.read_text()
s = one(s,
'''        /* gate stays fp32: k3_router has its own inline matmul. See k3.h. */
        reqw(p, &b->moe.gate, (int64_t)c->n_experts * H, -1,
             PRE "layers.%d.block_sparse_moe.gate.weight", L);''',
'''        /* Gate is a real matrix too. Exact BF16 is consumed directly by k3_router;
         * draft I8R/Q4G is proposal-only and follows the layer's tagged matrix format. */
        reqn(p, &b->moe.gate, (int64_t)c->n_experts * H,
             PRE "layers.%d.block_sparse_moe.gate.weight", L);''',
'binder gate reqn')
s = one(s,
'''    /* Only the BF16 vectors that kernels read elementwise are copied. Everything else
     * is pointed at in place. The router gate dominates: it is BF16 on disk but stays
     * fp32 in the engine because k3_router walks it with its own inline matmul. */
    const size_t H = (size_t)c->hidden;
    size_t n = 6 * H                       /* in/post norm, attn-res and mlp-res pair  */
             + (size_t)c->q_lora + c->kv_lora   /* MLA q_a/kv_a layernorms             */
             + (size_t)c->latent                /* routed_expert_norm                  */
             + (size_t)c->n_experts * H;        /* router gate                          */''',
'''    /* Only BF16 vectors that kernels read elementwise are copied. Large matrices,
     * including the router gate, remain in their native tagged representation. */
    const size_t H = (size_t)c->hidden;
    size_t n = 6 * H                       /* in/post norm, attn-res and mlp-res pair  */
             + (size_t)c->q_lora + c->kv_lora   /* MLA q_a/kv_a layernorms             */
             + (size_t)c->latent;               /* routed_expert_norm                  */''',
'widen bytes remove router')
p.write_text(s, newline='\n')

# ---- tests: call signature + BF16 exact parity folded into existing router test ------
p = root / 'tests/unit/test_ops.c'
s = p.read_text()
s = one(s,
'''            k3_router(gi, gw, x + (size_t)rw * hidden, W, bias, hidden, E, K, 1, 1.0f);''',
'''            k3_router(gi, gw, x + (size_t)rw * hidden, W, K3_WF32,
                      bias, hidden, E, K, 1, 1.0f);''',
'fixture router call')
# Add parity before existing PASS/FAIL decision, but keep one g_pass so CI suite count
# stays stable at 23.
needle = '''        if (set_ok && worst_w <= 1.0) {
            printf("  PASS  router         rows=%-4d k=%d  index sets match, "
                   "worst weight=%.2fx tol\\n", rows, K, worst_w);
            g_pass++;
        } else {
            printf("  FAIL  router         index_sets=%s worst weight=%.2fx tol\\n",
                   set_ok ? "ok" : "MISMATCH", worst_w);
            g_fail++;
        }
'''
replacement = '''        /* Exact BF16 parity: build a separate gate whose values are first rounded to
         * BF16, then compare the typed BF16 router against those SAME values widened to
         * fp32. idx and combining weights must match bit-for-bit, because the real old
         * binder did exactly that widening before calling the FP32 router. */
        int bf16_ok = 1;
        const int PE = 17, PH = 257, PK = 4, PR = 5;
        uint16_t *wb = (uint16_t *)malloc((size_t)PE * PH * sizeof(uint16_t));
        float *wf = (float *)malloc((size_t)PE * PH * sizeof(float));
        float *px = (float *)malloc((size_t)PR * PH * sizeof(float));
        float *pb = (float *)malloc((size_t)PE * sizeof(float));
        int ia[PK], ib[PK]; float wa[PK], wbw[PK];
        if (!wb || !wf || !px || !pb) bf16_ok = 0;
        if (bf16_ok) {
            unsigned st = 0x51A7E123u;
            for (size_t i = 0; i < (size_t)PE * PH; i++) {
                st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                const float z = ((int)(st & 0xffffu) - 32768) * (1.0f / 65536.0f);
                union { float f; uint32_t u; } v; v.f = z;
                wb[i] = (uint16_t)(v.u >> 16);
                wf[i] = k3_bf16f(wb[i]);
            }
            for (int i = 0; i < PR * PH; i++) {
                st ^= st << 13; st ^= st >> 17; st ^= st << 5;
                px[i] = ((int)(st & 0xffffu) - 32768) * (1.0f / 32768.0f);
            }
            for (int e = 0; e < PE; e++) pb[e] = (float)(e - 8) * 0.00031f;
            for (int r0 = 0; r0 < PR && bf16_ok; r0++) {
                k3_router(ia, wa, px + (size_t)r0 * PH, wf, K3_WF32,
                          pb, PH, PE, PK, 1, 1.0f);
                k3_router(ib, wbw, px + (size_t)r0 * PH, wb, K3_WBF16,
                          pb, PH, PE, PK, 1, 1.0f);
                if (memcmp(ia, ib, sizeof ia) != 0 || memcmp(wa, wbw, sizeof wa) != 0)
                    bf16_ok = 0;
            }
        }
        free(wb); free(wf); free(px); free(pb);

        if (set_ok && worst_w <= 1.0 && bf16_ok) {
            printf("  PASS  router         rows=%-4d k=%d  fixture matches; BF16 typed "
                   "gate is BIT-IDENTICAL\\n", rows, K);
            g_pass++;
        } else {
            printf("  FAIL  router         index_sets=%s worst weight=%.2fx tol bf16=%s\\n",
                   set_ok ? "ok" : "MISMATCH", worst_w, bf16_ok ? "exact" : "MISMATCH");
            g_fail++;
        }
'''
s = one(s, needle, replacement, 'router bf16 parity test')
p.write_text(s, newline='\n')

print('staged typed BF16/I8R/Q4 router optimization')
