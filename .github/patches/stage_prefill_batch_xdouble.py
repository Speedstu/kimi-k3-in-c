from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text()

anchor = '/* Batched exact MXFP4 matvec for prompt prefill.\n'
assert s.count(anchor) == 1, s.count(anchor)
insert = r'''#if defined(__AVX2__) && defined(K3_TEST_INTERNALS)
/* Candidate for prompt prefill only: same exact batched MXFP4 arithmetic as
 * k3_matmul_mxfp4_batch, but the caller has already widened every activation from
 * float to double once. This removes cvtps2pd from every row/group replay while still
 * decoding each packed weight group only once across the activation batch. */
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

'''
s = s.replace(anchor, insert + anchor)

anchor2 = '\nvoid k3_mxfp4_dequant(float *out, const unsigned char *packed,\n'
assert s.count(anchor2) == 1, s.count(anchor2)
wrap = r'''
#if defined(__AVX2__) && defined(K3_TEST_INTERNALS)
void k3_test_matmul_mxfp4_batch_float(float *y, int ystride,
                                       const float *const *xs, int batch,
                                       const unsigned char *packed,
                                       const unsigned char *scales,
                                       int in, int rows, int group)
{
    k3_matmul_mxfp4_batch(y, ystride, xs, batch,
                          packed, scales, in, rows, group);
}
#endif
'''
s = s.replace(anchor2, wrap + anchor2)
p.write_text(s)
