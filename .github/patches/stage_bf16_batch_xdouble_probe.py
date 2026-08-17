from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text()

anchor = '\nvoid k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)\n'
assert s.count(anchor) == 1, s.count(anchor)

helper = r'''
#if defined(__AVX2__) && defined(K3_TEST_INTERNALS)
/* Test-only candidate for prompt-prefill BF16 batches. `xs` are exact widenings of
 * the float activations consumed by k3_matmul_bf16_batch. The weight widening,
 * accumulator lanes, FMA order, reduction tree, scalar tail and final float rounding
 * are intentionally identical to the production float-input batch helper. */
static void k3_matmul_bf16_batch_xd(float *y, int ystride,
                                     const double *const *xs, int batch,
                                     const uint16_t *W, int in, int out)
{
    if (batch <= 0) return;
    if (batch > K3_PREFILL_BATCH_MAX) batch = K3_PREFILL_BATCH_MAX;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int i = 0;
        __m256d v0[K3_PREFILL_BATCH_MAX], v1[K3_PREFILL_BATCH_MAX];
        __m256d v2[K3_PREFILL_BATCH_MAX], v3[K3_PREFILL_BATCH_MAX];
        for (int b = 0; b < batch; b++) {
            v0[b] = _mm256_setzero_pd(); v1[b] = _mm256_setzero_pd();
            v2[b] = _mm256_setzero_pd(); v3[b] = _mm256_setzero_pd();
        }
        for (; i + 15 < in; i += 16) {
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
                const double *xb = xs[b];
                v0[b] = _mm256_fmadd_pd(w0, _mm256_loadu_pd(xb + i),      v0[b]);
                v1[b] = _mm256_fmadd_pd(w1, _mm256_loadu_pd(xb + i + 4),  v1[b]);
                v2[b] = _mm256_fmadd_pd(w2, _mm256_loadu_pd(xb + i + 8),  v2[b]);
                v3[b] = _mm256_fmadd_pd(w3, _mm256_loadu_pd(xb + i + 12), v3[b]);
            }
        }
        for (int b = 0; b < batch; b++) {
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0[b], v1[b]),
                                             _mm256_add_pd(v2[b], v3[b]));
            double a[4];
            _mm256_storeu_pd(a, vt);
            double acc = (a[0] + a[1]) + (a[2] + a[3]);
            const double *xb = xs[b];
            for (int j = i; j < in; j++)
                acc = fma((double)k3_bf16f(row[j]), xb[j], acc);
            y[(size_t)b * ystride + o] = (float)acc;
        }
    }
}

void k3_test_matmul_bf16_batch_xd(float *y, int ystride,
                                   const double *const *xs, int batch,
                                   const uint16_t *W, int in, int out)
{
    k3_matmul_bf16_batch_xd(y, ystride, xs, batch, W, in, out);
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
