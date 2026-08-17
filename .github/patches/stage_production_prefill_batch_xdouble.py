from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text()

# 1) Forward declaration beside the existing exact float batch helper.
old = '''static void k3_matmul_mxfp4_batch(float *y, int ystride,
                                   const float *const *xs, int batch,
                                   const unsigned char *packed,
                                   const unsigned char *scales,
                                   int in, int rows, int group);
'''
new = old + '''
#if defined(__AVX2__)
static void k3_matmul_mxfp4_batch_xd(float *y, int ystride,
                                      const double *const *xs, int batch,
                                      const unsigned char *packed,
                                      const unsigned char *scales,
                                      int in, int rows, int group);
#endif
'''
assert s.count(old) == 1, ('forward', s.count(old))
s = s.replace(old, new, 1)

# 2) Allocate/widen one exact double copy of the chunk latents and one reusable
# activation batch only when the existing MXFP4 batch path is live.
old = '''    float *bbuf = NULL, *bgu = NULL, *bact = NULL, *bedn = NULL;
    if (!no_prefill_batch_mxfp4 && T > 1) {
        const size_t bn = (size_t)T * ((size_t)3 * I + Ll);
        bbuf = (float *)malloc(bn * sizeof(float));
        if (!bbuf) k3_fatal_oom("MoE prefill batched MXFP4", bn * sizeof(float));
        bgu = bbuf;                              /* [T][2*I] */
        bact = bgu + (size_t)T * 2 * I;         /* [T][I]   */
        bedn = bact + (size_t)T * I;             /* [T][Ll]  */
    }
'''
new = '''    float *bbuf = NULL, *bgu = NULL, *bact = NULL, *bedn = NULL;
    if (!no_prefill_batch_mxfp4 && T > 1) {
        const size_t bn = (size_t)T * ((size_t)3 * I + Ll);
        bbuf = (float *)malloc(bn * sizeof(float));
        if (!bbuf) k3_fatal_oom("MoE prefill batched MXFP4", bn * sizeof(float));
        bgu = bbuf;                              /* [T][2*I] */
        bact = bgu + (size_t)T * 2 * I;         /* [T][I]   */
        bedn = bact + (size_t)T * I;             /* [T][Ll]  */
    }

#if defined(__AVX2__)
    /* The exact batch kernel otherwise repeats float->double conversion for every
     * output row and every MXFP4 group. The chunk latents are reused by many experts,
     * so widen them once. `bactd` is reused expert-by-expert after SiTU. Released K3
     * needs at most 64*(3584+3072)*8 ~= 3.25 MiB here. */
    static int no_prefill_batch_xdouble = -1;
    if (no_prefill_batch_xdouble < 0)
        no_prefill_batch_xdouble = getenv("K3_NO_PREFILL_BATCH_XDOUBLE") ? 1 : 0;
    double *xdbuf = NULL, *zzd = NULL, *bactd = NULL;
    if (bbuf && !no_prefill_batch_xdouble) {
        const size_t xdn = (size_t)T * ((size_t)Ll + I);
        xdbuf = (double *)malloc(xdn * sizeof(double));
        if (!xdbuf)
            k3_fatal_oom("MoE prefill batched MXFP4 xdouble", xdn * sizeof(double));
        zzd = xdbuf;                              /* [T][Ll] */
        bactd = zzd + (size_t)T * Ll;            /* [T][I]  */
        for (size_t i = 0; i < (size_t)T * Ll; i++) zzd[i] = (double)zz[i];
    }
#endif
'''
assert s.count(old) == 1, ('buffer', s.count(old))
s = s.replace(old, new, 1)

# 3) Use the pre-widened exact activations for shared-expert occurrences. Keep the
# current float batch helper intact behind the A/B escape hatch/non-AVX2 build.
old = '''            } else {
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
'''
new = '''            } else {
                const float *xp[K3_PREFILL_BATCH_MAX];
                const float *ap[K3_PREFILL_BATCH_MAX];
#if defined(__AVX2__)
                const double *xpd[K3_PREFILL_BATCH_MAX];
                const double *apd[K3_PREFILL_BATCH_MAX];
#endif
                for (int b = 0; b < nb; b++) {
                    xp[b] = zz + (size_t)bt[b] * Ll;
#if defined(__AVX2__)
                    if (xdbuf) xpd[b] = zzd + (size_t)bt[b] * Ll;
#endif
                }

#if defined(__AVX2__)
                if (xdbuf) {
                    k3_matmul_mxfp4_batch_xd(bgu, 2 * I, xpd, nb,
                                             q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                    k3_matmul_mxfp4_batch_xd(bgu + I, 2 * I, xpd, nb,
                                             q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                } else
#endif
                {
                    k3_matmul_mxfp4_batch(bgu, 2 * I, xp, nb,
                                          q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                    k3_matmul_mxfp4_batch(bgu + I, 2 * I, xp, nb,
                                          q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                }
                for (int b = 0; b < nb; b++) {
                    float *ab = bact + (size_t)b * I;
                    k3_situ_glu(ab, bgu + (size_t)b * 2 * I,
                                I, c->situ_b1, c->situ_b2);
                    ap[b] = ab;
#if defined(__AVX2__)
                    if (xdbuf) {
                        double *abd = bactd + (size_t)b * I;
                        for (int i = 0; i < I; i++) abd[i] = (double)ab[i];
                        apd[b] = abd;
                    }
#endif
                }
#if defined(__AVX2__)
                if (xdbuf)
                    k3_matmul_mxfp4_batch_xd(bedn, Ll, apd, nb,
                                             q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                else
#endif
                    k3_matmul_mxfp4_batch(bedn, Ll, ap, nb,
                                          q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                for (int b = 0; b < nb; b++)
                    memcpy(contrib + ((size_t)bt[b] * K + bj[b]) * Ll,
                           bedn + (size_t)b * Ll, (size_t)Ll * sizeof(float));
            }
'''
assert s.count(old) == 1, ('expert_batch', s.count(old))
s = s.replace(old, new, 1)

# 4) Release the extra exact-copy buffer with the existing MXFP4 batch scratch.
old = '''    free(bbuf);
    bbuf = NULL;
'''
new = '''#if defined(__AVX2__)
    free(xdbuf);
    xdbuf = NULL;
#endif
    free(bbuf);
    bbuf = NULL;
'''
assert s.count(old) == 1, ('free', s.count(old))
s = s.replace(old, new, 1)

# 5) Promote the already-tested candidate helper to production AVX2 scope while
# keeping only its test wrappers behind K3_TEST_INTERNALS.
start = s.index('#if defined(__AVX2__) && defined(K3_TEST_INTERNALS)\n/* Candidate for prompt prefill only:')
end_marker = '\n/* Batched exact MXFP4 matvec for prompt prefill.\n'
end = s.index(end_marker, start)
old_section = s[start:end]
helper = r'''#if defined(__AVX2__)
/* Exact prompt-prefill companion to k3_matmul_mxfp4_batch. The caller has already
 * widened every activation from float to double once, eliminating repeated cvtps2pd
 * while preserving the float-batch kernel's per-token FMA lanes, group order, scale
 * accumulation and final float rounding exactly. */
static void k3_matmul_mxfp4_batch_xd(float *y, int ystride,
                                      const double *const *xs, int batch,
                                      const unsigned char *packed,
                                      const unsigned char *scales,
                                      int in, int rows, int group)
{
    if (batch <= 0) return;
    if (batch == 1) {
        k3_matmul_mxfp4_xd(y, xs[0], packed, scales, in, rows, group, 0);
        return;
    }
    if (batch > K3_PREFILL_BATCH_MAX || group != 32 || (in & 31)) {
        for (int b = 0; b < batch; b++)
            k3_matmul_mxfp4_xd(y + (size_t)b * ystride, xs[b],
                               packed, scales, in, rows, group, 0);
        return;
    }

    const int pcols = in / 2, ngrp = in / 32;
    if (!k3_e8m0_ready) k3_e8m0_init();
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i half_units = _mm_setr_epi8(
         0,  1,  2,  3,  4,  6,  8, 12,
         0, -1, -2, -3, -4, -6, -8,-12);

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
            const unsigned char *pb = pr + (size_t)g * 16;
            const __m128i wb = _mm_loadu_si128((const __m128i *)pb);
            const __m128i lo = _mm_shuffle_epi8(half_units, _mm_and_si128(wb, mask));
            const __m128i hi = _mm_shuffle_epi8(
                half_units, _mm_and_si128(_mm_srli_epi16(wb, 4), mask));
            const __m128i q0 = _mm_unpacklo_epi8(lo, hi);
            const __m128i q1 = _mm_unpackhi_epi8(lo, hi);
            const __m256i i0 = _mm256_cvtepi8_epi32(q0);
            const __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));
            const __m256i i2 = _mm256_cvtepi8_epi32(q1);
            const __m256i i3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));
            const double mult = (double)K3_E8M0[sb] * 0.5;

            for (int b = 0; b < batch; b++) {
                const double *xg = xs[b] + (size_t)g * 32;
                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
#define K3_MXBD_F4(V, I128, XOFF) do {                                        \
                    (V) = _mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),         \
                                           _mm256_loadu_pd(xg + (XOFF)), (V)); \
                } while (0)
                K3_MXBD_F4(v0, _mm256_castsi256_si128(i0),       0);
                K3_MXBD_F4(v1, _mm256_extracti128_si256(i0, 1),  4);
                K3_MXBD_F4(v0, _mm256_castsi256_si128(i1),       8);
                K3_MXBD_F4(v1, _mm256_extracti128_si256(i1, 1), 12);
                K3_MXBD_F4(v0, _mm256_castsi256_si128(i2),      16);
                K3_MXBD_F4(v1, _mm256_extracti128_si256(i2, 1), 20);
                K3_MXBD_F4(v0, _mm256_castsi256_si128(i3),      24);
                K3_MXBD_F4(v1, _mm256_extracti128_si256(i3, 1), 28);
#undef K3_MXBD_F4
                double a[4];
                _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
                const double sub2 = (a[0] + a[1]) + (a[2] + a[3]);
                acc[b] += sub2 * mult;
            }
        }

        for (int b = 0; b < batch; b++)
            y[(size_t)b * ystride + r] = (float)acc[b];
    }
}

#ifdef K3_TEST_INTERNALS
void k3_test_matmul_mxfp4_batch_xd(float *y, int ystride,
                                    const double *const *xs, int batch,
                                    const unsigned char *packed,
                                    const unsigned char *scales,
                                    int in, int rows, int group)
{
    k3_matmul_mxfp4_batch_xd(y, ystride, xs, batch,
                             packed, scales, in, rows, group);
}

void k3_test_matmul_mxfp4_batch_float(float *y, int ystride,
                                       const float *const *xs, int batch,
                                       const unsigned char *packed,
                                       const unsigned char *scales,
                                       int in, int rows, int group);
#endif
#endif
'''
s = s[:start] + helper + s[end:]

p.write_text(s)
