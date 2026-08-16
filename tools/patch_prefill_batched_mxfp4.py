#!/usr/bin/env python3
from pathlib import Path

P = Path("src/core/k3_ops.c")
s = P.read_text()

if "K3_PREFILL_BATCH_MAX" in s:
    print("batched MXFP4 prefill already applied")
    raise SystemExit(0)

proto = '''static void moe_prefill_chunk(float *out, const float *x, const K3MoeW *w,
                              const K3Cfg *c, int T, float *scratch);
'''
proto_new = proto + '''
#define K3_PREFILL_BATCH_MAX 64
static void k3_matmul_mxfp4_batch(float *y, int ystride,
                                   const float *const *xs, int batch,
                                   const unsigned char *packed,
                                   const unsigned char *scales,
                                   int in, int rows, int group);
'''
if proto not in s:
    raise SystemExit("prototype anchor not found")
s = s.replace(proto, proto_new, 1)

scratch = '''    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */
    static int no_prefill_pipeline = -1;
'''
scratch_new = '''    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */

    /* Exact prefill compute batching: when several prompt tokens route to the same
     * expert, decode/read each MXFP4 weight row once and apply it to all of those
     * activations. Each token keeps the exact same per-row FMA lanes, group order and
     * scale accumulation as k3_matmul_mxfp4; only the weight decode/load is shared. */
    static int no_prefill_batch_mxfp4 = -1;
    if (no_prefill_batch_mxfp4 < 0)
        no_prefill_batch_mxfp4 = getenv("K3_NO_PREFILL_BATCH_MXFP4") ? 1 : 0;
    float *bbuf = NULL, *bgu = NULL, *bact = NULL, *bedn = NULL;
    if (!no_prefill_batch_mxfp4 && T > 1) {
        const size_t bn = (size_t)T * ((size_t)3 * I + Ll);
        bbuf = (float *)malloc(bn * sizeof(float));
        if (!bbuf) k3_fatal_oom("MoE prefill batched MXFP4", bn * sizeof(float));
        bgu = bbuf;                              /* [T][2*I] */
        bact = bgu + (size_t)T * 2 * I;         /* [T][I]   */
        bedn = bact + (size_t)T * I;             /* [T][Ll]  */
    }

    static int no_prefill_pipeline = -1;
'''
if scratch not in s:
    raise SystemExit("scratch anchor not found")
s = s.replace(scratch, scratch_new, 1)

old = '''        } else {
            for (int t = 0; t < T; t++) {
                const int   *it = ridx + (size_t)t * K;
                const float *zt = zz  + (size_t)t * Ll;
                for (int j = 0; j < K; j++) {
                    if (it[j] != e) continue;
                    k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                    k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                    k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                    k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                    memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
                }
            }
        }
'''
new = '''        } else {
            int bt[K3_PREFILL_BATCH_MAX], bj[K3_PREFILL_BATCH_MAX];
            int nb = 0, overflow = 0;
            if (bbuf) {
                for (int t = 0; t < T; t++) {
                    const int *it = ridx + (size_t)t * K;
                    for (int j = 0; j < K; j++) {
                        if (it[j] != e) continue;
                        if (nb < K3_PREFILL_BATCH_MAX) {
                            bt[nb] = t;
                            bj[nb] = j;
                            nb++;
                        } else {
                            overflow = 1;
                        }
                    }
                }
            }

            /* A single occurrence cannot share any weight traffic, and an impossible
             * duplicate-router overflow falls back to the proven scalar-token path. */
            if (!bbuf || nb <= 1 || overflow) {
                for (int t = 0; t < T; t++) {
                    const int   *it = ridx + (size_t)t * K;
                    const float *zt = zz  + (size_t)t * Ll;
                    for (int j = 0; j < K; j++) {
                        if (it[j] != e) continue;
                        k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                        k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                        k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                        k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                        memcpy(contrib + ((size_t)t * K + j) * Ll, edn,
                               (size_t)Ll * sizeof(float));
                    }
                }
            } else {
                const float *xp[K3_PREFILL_BATCH_MAX];
                const float *ap[K3_PREFILL_BATCH_MAX];
                for (int b = 0; b < nb; b++) xp[b] = zz + (size_t)bt[b] * Ll;

                k3_matmul_mxfp4_batch(bgu, 2 * I, xp, nb,
                                      q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4_batch(bgu + I, 2 * I, xp, nb,
                                      q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                for (int b = 0; b < nb; b++) {
                    float *ab = bact + (size_t)b * I;
                    k3_situ_glu(ab, bgu + (size_t)b * 2 * I,
                                I, c->situ_b1, c->situ_b2);
                    ap[b] = ab;
                }
                k3_matmul_mxfp4_batch(bedn, Ll, ap, nb,
                                      q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                for (int b = 0; b < nb; b++)
                    memcpy(contrib + ((size_t)bt[b] * K + bj[b]) * Ll,
                           bedn + (size_t)b * Ll, (size_t)Ll * sizeof(float));
            }
        }
'''
if old not in s:
    raise SystemExit("expert compute anchor not found")
s = s.replace(old, new, 1)

free_old = '''    free(ridx); free(rwt); free(zz); free(contrib); free(uniq); free(seen);
'''
free_new = '''    free(bbuf);
    free(ridx); free(rwt); free(zz); free(contrib); free(uniq); free(seen);
'''
if free_old not in s:
    raise SystemExit("free anchor not found")
s = s.replace(free_old, free_new, 1)

impl_anchor = '''void k3_mxfp4_dequant(float *out, const unsigned char *packed,
'''
impl = r'''/* Batched exact MXFP4 matvec for prompt prefill.
 *
 * Output is y[batch][rows] with caller-selected ystride. The packed weight row is
 * decoded once, then the exact single-vector arithmetic is replayed independently for
 * every activation. This preserves each token's floating-point operation order while
 * amortising expert-weight memory traffic across prompt tokens that chose the same
 * expert. Decode remains the ordinary single-token path. */
static void k3_matmul_mxfp4_batch(float *y, int ystride,
                                   const float *const *xs, int batch,
                                   const unsigned char *packed,
                                   const unsigned char *scales,
                                   int in, int rows, int group)
{
    if (batch <= 0) return;
    if (batch == 1) {
        k3_matmul_mxfp4(y, xs[0], packed, scales, in, rows, group);
        return;
    }
    if (batch > K3_PREFILL_BATCH_MAX) {
        for (int b = 0; b < batch; b++)
            k3_matmul_mxfp4(y + (size_t)b * ystride, xs[b],
                            packed, scales, in, rows, group);
        return;
    }

    const int pcols = in / 2;
    const int ngrp  = (in + group - 1) / group;
    const int gbyte = group / 2;
    if (!k3_pair_ready) k3_pair_init();
    if (!k3_e8m0_ready) k3_e8m0_init();

#if defined(__AVX2__)
    const __m128i mx_mask = _mm_set1_epi8(0x0f);
    const __m128i mx_half_units = _mm_setr_epi8(
         0,  1,  2,  3,  4,  6,  8, 12,
         0, -1, -2, -3, -4, -6, -8,-12);
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 64)
#endif
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        double acc[K3_PREFILL_BATCH_MAX] = {0};

        for (int g = 0; g < ngrp; g++) {
            const unsigned char sb = sr[g];
            if (sb == 255) continue;
            const unsigned char *pb = pr + (size_t)g * gbyte;
            int n = in - g * group;
            if (n > group) n = group;

#if defined(__AVX2__)
            if (group == 32 && n == 32) {
                /* Decode the row's 32 weights ONCE. The per-activation FMA lanes and
                 * reduction below are copied from k3_matmul_mxfp4 verbatim. */
                const __m128i wb = _mm_loadu_si128((const __m128i *)pb);
                const __m128i lo = _mm_shuffle_epi8(
                    mx_half_units, _mm_and_si128(wb, mx_mask));
                const __m128i hi = _mm_shuffle_epi8(
                    mx_half_units, _mm_and_si128(_mm_srli_epi16(wb, 4), mx_mask));
                const __m128i q0 = _mm_unpacklo_epi8(lo, hi);
                const __m128i q1 = _mm_unpackhi_epi8(lo, hi);
                const __m256i i0 = _mm256_cvtepi8_epi32(q0);
                const __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));
                const __m256i i2 = _mm256_cvtepi8_epi32(q1);
                const __m256i i3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));
                const double mult = (double)K3_E8M0[sb] * 0.5;

                for (int b = 0; b < batch; b++) {
                    const float *xg = xs[b] + (size_t)g * group;
                    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
#define K3_MXB_F4(V, I128, XOFF) do {                                         \
                    (V) = _mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),         \
                        _mm256_cvtps_pd(_mm_loadu_ps(xg + (XOFF))), (V));      \
                } while (0)
                    K3_MXB_F4(v0, _mm256_castsi256_si128(i0),       0);
                    K3_MXB_F4(v1, _mm256_extracti128_si256(i0, 1),  4);
                    K3_MXB_F4(v0, _mm256_castsi256_si128(i1),       8);
                    K3_MXB_F4(v1, _mm256_extracti128_si256(i1, 1), 12);
                    K3_MXB_F4(v0, _mm256_castsi256_si128(i2),      16);
                    K3_MXB_F4(v1, _mm256_extracti128_si256(i2, 1), 20);
                    K3_MXB_F4(v0, _mm256_castsi256_si128(i3),      24);
                    K3_MXB_F4(v1, _mm256_extracti128_si256(i3, 1), 28);
#undef K3_MXB_F4
                    double a[4];
                    _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
                    const double sub2 = (a[0] + a[1]) + (a[2] + a[3]);
                    acc[b] += sub2 * mult;
                }
                continue;
            }
#endif

            float wf[64];
            const int half = n >> 1;
            for (int j = 0; j < half; j++) {
                const float *pv = K3_E2M1_PAIR[pb[j]];
                wf[2 * j] = pv[0];
                wf[2 * j + 1] = pv[1];
            }
            if (n & 1) wf[n - 1] = K3_E2M1_PAIR[pb[half]][0];

            for (int b = 0; b < batch; b++) {
                const float *xg = xs[b] + (size_t)g * group;
                double sub;
                int i = 0;
#if defined(__AVX2__)
                {
                    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
                    for (; i + 7 < n; i += 8) {
                        v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i)),
                                             _mm256_cvtps_pd(_mm_loadu_ps(xg + i)), v0);
                        v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i + 4)),
                                             _mm256_cvtps_pd(_mm_loadu_ps(xg + i + 4)), v1);
                    }
                    double a[4];
                    _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
                    sub = (a[0] + a[1]) + (a[2] + a[3]);
                }
#else
                {
                    double ss[8] = {0};
                    for (; i + 7 < n; i += 8)
                        for (int l = 0; l < 8; l++)
                            ss[l] = fma((double)wf[i + l], (double)xg[i + l], ss[l]);
                    double b0 = ss[0] + ss[4], b1 = ss[1] + ss[5];
                    double b2 = ss[2] + ss[6], b3 = ss[3] + ss[7];
                    sub = (b0 + b1) + (b2 + b3);
                }
#endif
                for (; i < n; i++)
                    sub = fma((double)wf[i], (double)xg[i], sub);
                acc[b] += sub * (double)K3_E8M0[sb];
            }
        }

        for (int b = 0; b < batch; b++)
            y[(size_t)b * ystride + r] = (float)acc[b];
    }
}

'''
if impl_anchor not in s:
    raise SystemExit("implementation anchor not found")
s = s.replace(impl_anchor, impl + impl_anchor, 1)

P.write_text(s)
print("applied exact batched MXFP4 prefill transform")
