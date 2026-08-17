from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text()
anchor = '\nvoid k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)\n'
assert s.count(anchor) == 1, s.count(anchor)

helper = r'''
#if defined(__AVX2__) && defined(K3_TEST_INTERNALS)
/* Test-only row-pair form of k3_matmul_bf16_batch. Two independent output rows share
 * each float->double activation conversion. Per row, weight widening, FMA lane
 * assignment, i-order, reduction tree, scalar tail and final float rounding are exactly
 * the production helper's. */
static void k3_matmul_bf16_batch_rowpair(float *y, int ystride,
                                          const float *const *xs, int batch,
                                          const uint16_t *W, int in, int out)
{
    if (batch <= 0) return;
    if (batch > K3_PREFILL_BATCH_MAX || out < 2) {
        k3_matmul_bf16_batch(y, ystride, xs, batch, W, in, out);
        return;
    }

    const int npair = out / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (npair > 32)
#endif
    for (int rp = 0; rp < npair; rp++) {
        const int o0 = 2 * rp, o1 = o0 + 1;
        const uint16_t *r0 = W + (size_t)o0 * in;
        const uint16_t *r1 = W + (size_t)o1 * in;
        int i = 0;
        __m256d a0[K3_PREFILL_BATCH_MAX], a1[K3_PREFILL_BATCH_MAX];
        __m256d a2[K3_PREFILL_BATCH_MAX], a3[K3_PREFILL_BATCH_MAX];
        __m256d b0[K3_PREFILL_BATCH_MAX], b1[K3_PREFILL_BATCH_MAX];
        __m256d b2[K3_PREFILL_BATCH_MAX], b3[K3_PREFILL_BATCH_MAX];
        for (int b = 0; b < batch; b++) {
            a0[b] = _mm256_setzero_pd(); a1[b] = _mm256_setzero_pd();
            a2[b] = _mm256_setzero_pd(); a3[b] = _mm256_setzero_pd();
            b0[b] = _mm256_setzero_pd(); b1[b] = _mm256_setzero_pd();
            b2[b] = _mm256_setzero_pd(); b3[b] = _mm256_setzero_pd();
        }

        for (; i + 15 < in; i += 16) {
#define K3_BF16_W4(ROW, OFF) _mm256_cvtps_pd(                              \
                _mm_castsi128_ps(_mm_slli_epi32(                           \
                    _mm_cvtepu16_epi32(_mm_loadl_epi64(                    \
                        (const __m128i *)((ROW) + i + (OFF)))), 16)))
            const __m256d aw0 = K3_BF16_W4(r0, 0);
            const __m256d aw1 = K3_BF16_W4(r0, 4);
            const __m256d aw2 = K3_BF16_W4(r0, 8);
            const __m256d aw3 = K3_BF16_W4(r0, 12);
            const __m256d bw0 = K3_BF16_W4(r1, 0);
            const __m256d bw1 = K3_BF16_W4(r1, 4);
            const __m256d bw2 = K3_BF16_W4(r1, 8);
            const __m256d bw3 = K3_BF16_W4(r1, 12);
#undef K3_BF16_W4
            for (int b = 0; b < batch; b++) {
                const float *xb = xs[b];
                const __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(xb + i));
                const __m256d x1 = _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 4));
                const __m256d x2 = _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 8));
                const __m256d x3 = _mm256_cvtps_pd(_mm_loadu_ps(xb + i + 12));
                a0[b] = _mm256_fmadd_pd(aw0, x0, a0[b]);
                a1[b] = _mm256_fmadd_pd(aw1, x1, a1[b]);
                a2[b] = _mm256_fmadd_pd(aw2, x2, a2[b]);
                a3[b] = _mm256_fmadd_pd(aw3, x3, a3[b]);
                b0[b] = _mm256_fmadd_pd(bw0, x0, b0[b]);
                b1[b] = _mm256_fmadd_pd(bw1, x1, b1[b]);
                b2[b] = _mm256_fmadd_pd(bw2, x2, b2[b]);
                b3[b] = _mm256_fmadd_pd(bw3, x3, b3[b]);
            }
        }

        for (int b = 0; b < batch; b++) {
            double aa[4], bb[4];
            _mm256_storeu_pd(aa, _mm256_add_pd(_mm256_add_pd(a0[b], a1[b]),
                                               _mm256_add_pd(a2[b], a3[b])));
            _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(b0[b], b1[b]),
                                               _mm256_add_pd(b2[b], b3[b])));
            double acc0 = (aa[0] + aa[1]) + (aa[2] + aa[3]);
            double acc1 = (bb[0] + bb[1]) + (bb[2] + bb[3]);
            const float *xb = xs[b];
            for (int j = i; j < in; j++) {
                const double xj = (double)xb[j];
                acc0 = fma((double)k3_bf16f(r0[j]), xj, acc0);
                acc1 = fma((double)k3_bf16f(r1[j]), xj, acc1);
            }
            y[(size_t)b * ystride + o0] = (float)acc0;
            y[(size_t)b * ystride + o1] = (float)acc1;
        }
    }

    if (out & 1) {
        const int o = out - 1;
        k3_matmul_bf16_batch(y + o, ystride, xs, batch,
                             W + (size_t)o * in, in, 1);
    }
}

void k3_test_matmul_bf16_batch_rowpair(float *y, int ystride,
                                        const float *const *xs, int batch,
                                        const uint16_t *W, int in, int out)
{
    k3_matmul_bf16_batch_rowpair(y, ystride, xs, batch, W, in, out);
}

void k3_test_matmul_bf16_batch_float(float *y, int ystride,
                                      const float *const *xs, int batch,
                                      const uint16_t *W, int in, int out)
{
    k3_matmul_bf16_batch(y, ystride, xs, batch, W, in, out);
}
#endif
'''
p.write_text(s.replace(anchor, helper + anchor, 1))
