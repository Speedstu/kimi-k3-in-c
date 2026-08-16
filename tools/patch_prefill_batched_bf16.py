#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

if "K3_NO_PREFILL_BATCH_BF16" in s:
    print("exact BF16 prefill batching already applied")
    raise SystemExit(0)

# 1) Prototype beside the existing exact prefill batch kernel.
anchor = '''#define K3_PREFILL_BATCH_MAX 64
static void k3_matmul_mxfp4_batch(float *y, int ystride,
                                   const float *const *xs, int batch,
                                   const unsigned char *packed,
                                   const unsigned char *scales,
                                   int in, int rows, int group);
'''
replacement = '''#define K3_PREFILL_BATCH_MAX 64
static void k3_matmul_bf16_batch(float *y, int ystride,
                                  const float *const *xs, int batch,
                                  const uint16_t *W, int in, int out);
static void k3_matmul_mxfp4_batch(float *y, int ystride,
                                   const float *const *xs, int batch,
                                   const unsigned char *packed,
                                   const unsigned char *scales,
                                   int in, int rows, int group);
'''
if anchor not in s:
    raise SystemExit("prototype anchor not found")
s = s.replace(anchor, replacement, 1)

# 2) Route first, but batch the exact BF16 down projection across the whole prefill chunk.
anchor = '''    int nu = 0;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        int   *it = ridx + (size_t)t * K;
        float *wtt = rwt + (size_t)t * K;
        k3_router(it, wtt, xt, w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);
        k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);
        for (int j = 0; j < K; j++) {
            const int e = it[j];
            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    }
'''
replacement = '''    static int no_prefill_batch_bf16 = -1;
    if (no_prefill_batch_bf16 < 0)
        no_prefill_batch_bf16 = getenv("K3_NO_PREFILL_BATCH_BF16") ? 1 : 0;
    const int batch_bf16 = !no_prefill_batch_bf16 && T > 1 && w->wdt == K3_WBF16;

    int nu = 0;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        int   *it = ridx + (size_t)t * K;
        float *wtt = rwt + (size_t)t * K;
        k3_router(it, wtt, xt, w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);
        if (!batch_bf16)
            k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);
        for (int j = 0; j < K; j++) {
            const int e = it[j];
            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    }
    if (batch_bf16) {
        const float *xp[K3_PREFILL_BATCH_MAX];
        for (int t = 0; t < T; t++) xp[t] = x + (size_t)t * E;
        k3_matmul_bf16_batch(zz, Ll, xp, T, (const uint16_t *)w->down, E, Ll);
    }
'''
if anchor not in s:
    raise SystemExit("MoE prefill route/down anchor not found")
s = s.replace(anchor, replacement, 1)

# 3) After routed-expert work, the temporary MXFP4 batch buffer is dead. Free it before
#    the larger shared-expert batch scratch so peak RSS stays small.
anchor = '''    /* 3. per token, sum contributions in the ORIGINAL top-k order, then the tail of the
     * MoE exactly as k3_moe does it, so every float matches the per-token path. */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;
        const float *wtt = rwt + (size_t)t * K;
        /* Reuse this token's now-dead down-projection slot as the aggregate. */
        float *acc = zz + (size_t)t * Ll;
        for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            const float wj = wtt[j];
            const float *cb = contrib + ((size_t)t * K + j) * Ll;
            for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
        }
        if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
        k3_mmw(ot, acc, w->up, w->wdt, Ll, E);

        float *sgu  = gu;                 /* [2*SI] */
        float *sact = sgu + 2 * SI;       /* [SI]   */
        float *sdn  = sact + SI;          /* [E]    */
        k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
        k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
        k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
        k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        for (int i = 0; i < E; i++) ot[i] += sdn[i];
    }

    free(bbuf);
    free(ridx); free(rwt); free(zz); free(contrib); free(uniq); free(seen);
'''
replacement = '''    /* Routed MXFP4 batch scratch is dead before the dense BF16 tail. Releasing it here
     * keeps the exact prefill optimization essentially flat in peak RSS. */
    free(bbuf);
    bbuf = NULL;

    /* 3. Sum routed contributions in each token's ORIGINAL top-k order. This phase is
     * unchanged arithmetically; it merely stops before the token-independent BF16
     * matrices so those matrices can reuse each weight row across the chunk. */
    if (batch_bf16) {
        for (int t = 0; t < T; t++) {
            const float *wtt = rwt + (size_t)t * K;
            float *acc = zz + (size_t)t * Ll;  /* down projection is dead now */
            for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
            for (int j = 0; j < K; j++) {
                const float wj = wtt[j];
                const float *cb = contrib + ((size_t)t * K + j) * Ll;
                for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
            }
            if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
        }

        const float *zp[K3_PREFILL_BATCH_MAX];
        const float *xp[K3_PREFILL_BATCH_MAX];
        const float *ap[K3_PREFILL_BATCH_MAX];
        for (int t = 0; t < T; t++) {
            zp[t] = zz + (size_t)t * Ll;
            xp[t] = x  + (size_t)t * E;
        }

        /* Routed latent -> hidden. Each output-row weight is widened once for every
         * token in the chunk; every token still follows k3_matmul_bf16's exact lanes. */
        k3_matmul_bf16_batch(out, E, zp, T, (const uint16_t *)w->up, Ll, E);

        /* Shared expert: [E->2*SI], SiTU per token, [SI->E]. At the released K3
         * dimensions this avoids rereading ~264 MB of shared-expert BF16 weights for
         * every prompt token. The temporary is ~6.6 MB at T=64. */
        const size_t sn = (size_t)T * ((size_t)3 * SI + E);
        float *sbuf = (float *)malloc(sn * sizeof(float));
        if (!sbuf) k3_fatal_oom("MoE prefill batched BF16 shared expert", sn * sizeof(float));
        float *sgu  = sbuf;                         /* [T][2*SI] */
        float *sact = sgu  + (size_t)T * 2 * SI;   /* [T][SI]   */
        float *sdn  = sact + (size_t)T * SI;       /* [T][E]    */

        k3_matmul_bf16_batch(sgu, 2 * SI, xp, T,
                             (const uint16_t *)w->sh1, E, SI);
        k3_matmul_bf16_batch(sgu + SI, 2 * SI, xp, T,
                             (const uint16_t *)w->sh3, E, SI);
        for (int t = 0; t < T; t++) {
            float *at = sact + (size_t)t * SI;
            k3_situ_glu(at, sgu + (size_t)t * 2 * SI,
                        SI, c->situ_b1, c->situ_b2);
            ap[t] = at;
        }
        k3_matmul_bf16_batch(sdn, E, ap, T,
                             (const uint16_t *)w->sh2, SI, E);
        for (int t = 0; t < T; t++) {
            float *ot = out + (size_t)t * E;
            const float *dt = sdn + (size_t)t * E;
            for (int i = 0; i < E; i++) ot[i] += dt[i];
        }
        free(sbuf);
    } else {
        /* Non-BF16 draft trunks and the A/B escape hatch retain the proven path. */
        for (int t = 0; t < T; t++) {
            const float *xt = x + (size_t)t * E;
            float *ot = out + (size_t)t * E;
            const float *wtt = rwt + (size_t)t * K;
            float *acc = zz + (size_t)t * Ll;
            for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
            for (int j = 0; j < K; j++) {
                const float wj = wtt[j];
                const float *cb = contrib + ((size_t)t * K + j) * Ll;
                for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
            }
            if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
            k3_mmw(ot, acc, w->up, w->wdt, Ll, E);

            float *sgu  = gu;
            float *sact = sgu + 2 * SI;
            float *sdn  = sact + SI;
            k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
            k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
            k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
            k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
            for (int i = 0; i < E; i++) ot[i] += sdn[i];
        }
    }

    free(ridx); free(rwt); free(zz); free(contrib); free(uniq); free(seen);
'''
if anchor not in s:
    raise SystemExit("MoE prefill tail anchor not found")
s = s.replace(anchor, replacement, 1)

# 4) Exact BF16 mini-batch kernel. It shares only weight widening/load. Each token keeps
#    exactly the same 16 FMA chains, per-lane iteration order, reduction tree and tail.
anchor = '''void k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
'''
helper = r'''static void k3_matmul_bf16_batch(float *y, int ystride,
                                  const float *const *xs, int batch,
                                  const uint16_t *W, int in, int out)
{
    if (batch <= 0) return;
    if (batch == 1) {
        k3_matmul_bf16(y, xs[0], W, in, out);
        return;
    }
    if (batch > K3_PREFILL_BATCH_MAX) {
        for (int b = 0; b < batch; b++)
            k3_matmul_bf16(y + (size_t)b * ystride, xs[b], W, in, out);
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int i = 0;
#if defined(__AVX2__)
        __m256d v0[K3_PREFILL_BATCH_MAX], v1[K3_PREFILL_BATCH_MAX];
        __m256d v2[K3_PREFILL_BATCH_MAX], v3[K3_PREFILL_BATCH_MAX];
        for (int b = 0; b < batch; b++) {
            v0[b] = _mm256_setzero_pd(); v1[b] = _mm256_setzero_pd();
            v2[b] = _mm256_setzero_pd(); v3[b] = _mm256_setzero_pd();
        }
        for (; i + 15 < in; i += 16) {
            /* Widen these 16 bf16 values ONCE, then replay the production kernel's
             * four FMA lanes independently for each activation in the batch. */
            const __m128i h0 = _mm_loadl_epi64((const __m128i *)(row + i));
            const __m128i h1 = _mm_loadl_epi64((const __m128i *)(row + i + 4));
            const __m128i h2 = _mm_loadl_epi64((const __m128i *)(row + i + 8));
            const __m128i h3 = _mm_loadl_epi64((const __m128i *)(row + i + 12));
            const __m256d w0 = _mm256_cvtps_pd(
                _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h0), 16)));
            const __m256d w1 = _mm256_cvtps_pd(
                _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h1), 16)));
            const __m256d w2 = _mm256_cvtps_pd(
                _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h2), 16)));
            const __m256d w3 = _mm256_cvtps_pd(
                _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h3), 16)));
            for (int b = 0; b < batch; b++) {
                const float *xb = xs[b];
                v0[b] = _mm256_fmadd_pd(w0, _mm256_cvtps_pd(_mm_loadu_ps(xb + i)),      v0[b]);
                v1[b] = _mm256_fmadd_pd(w1, _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 4)),  v1[b]);
                v2[b] = _mm256_fmadd_pd(w2, _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 8)),  v2[b]);
                v3[b] = _mm256_fmadd_pd(w3, _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 12)), v3[b]);
            }
        }
        for (int b = 0; b < batch; b++) {
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0[b], v1[b]),
                                             _mm256_add_pd(v2[b], v3[b]));
            double a[4];
            _mm256_storeu_pd(a, vt);
            double acc = (a[0] + a[1]) + (a[2] + a[3]);
            const float *xb = xs[b];
            for (int j = i; j < in; j++)
                acc = fma((double)k3_bf16f(row[j]), (double)xb[j], acc);
            y[(size_t)b * ystride + o] = (float)acc;
        }
#else
        double a[K3_PREFILL_BATCH_MAX][16] = {{0}};
        for (; i + 15 < in; i += 16) {
            double ww[16];
            for (int l = 0; l < 16; l++) ww[l] = (double)k3_bf16f(row[i + l]);
            for (int b = 0; b < batch; b++) {
                const float *xb = xs[b];
                for (int l = 0; l < 16; l++)
                    a[b][l] = fma(ww[l], (double)xb[i + l], a[b][l]);
            }
        }
        for (int b = 0; b < batch; b++) {
            double b0 = (a[b][0] + a[b][4]) + (a[b][8]  + a[b][12]);
            double b1 = (a[b][1] + a[b][5]) + (a[b][9]  + a[b][13]);
            double b2 = (a[b][2] + a[b][6]) + (a[b][10] + a[b][14]);
            double b3 = (a[b][3] + a[b][7]) + (a[b][11] + a[b][15]);
            double acc = (b0 + b1) + (b2 + b3);
            const float *xb = xs[b];
            for (int j = i; j < in; j++)
                acc = fma((double)k3_bf16f(row[j]), (double)xb[j], acc);
            y[(size_t)b * ystride + o] = (float)acc;
        }
#endif
    }
}

'''
if anchor not in s:
    raise SystemExit("BF16 kernel anchor not found")
s = s.replace(anchor, helper + anchor, 1)

p.write_text(s)
print("applied exact BF16 MoE prefill batching")
