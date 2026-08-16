#!/usr/bin/env python3
from pathlib import Path

p=Path('src/core/k3_ops.c')
s=p.read_text()

old='''static void k3_matmul_mxfp4_xd(float *y, const double *x,\n                               const unsigned char *packed,\n                               const unsigned char *scales, int in, int rows, int group);\n'''
new='''static void k3_matmul_mxfp4_xd(float *y, const double *x,\n                               const unsigned char *packed,\n                               const unsigned char *scales, int in, int rows, int group,\n                               int rowpair);\n'''
if s.count(old)!=1: raise SystemExit(f'declaration anchor={s.count(old)}')
s=s.replace(old,new,1)

old='''    static int no_mx_xdouble = -1;\n    if (no_mx_xdouble < 0) no_mx_xdouble = getenv("K3_NO_MX_XDOUBLE") ? 1 : 0;\n#endif\n'''
new='''    static int no_mx_xdouble = -1, no_mx_rowpair = -1;\n    if (no_mx_xdouble < 0) no_mx_xdouble = getenv("K3_NO_MX_XDOUBLE") ? 1 : 0;\n    if (no_mx_rowpair < 0) no_mx_rowpair = getenv("K3_NO_MX_ROWPAIR") ? 1 : 0;\n#endif\n'''
if s.count(old)!=1: raise SystemExit(f'env anchor={s.count(old)}')
s=s.replace(old,new,1)

old='''        const int use_mx_xdouble = w->src && !no_mx_xdouble;\n        if (use_mx_xdouble)\n'''
new='''        const int use_mx_xdouble = w->src && !no_mx_xdouble;\n        const int use_mx_rowpair = use_mx_xdouble && !no_mx_rowpair;\n        if (use_mx_xdouble)\n'''
if s.count(old)!=1: raise SystemExit(f'use anchor={s.count(old)}')
s=s.replace(old,new,1)

for old,new in [
('''k3_matmul_mxfp4_xd(gu,     zxd, q.p1, q.s1, L, I, K3_MXFP4_GROUP);''',
 '''k3_matmul_mxfp4_xd(gu,     zxd, q.p1, q.s1, L, I, K3_MXFP4_GROUP, use_mx_rowpair);'''),
('''k3_matmul_mxfp4_xd(gu + I, zxd, q.p3, q.s3, L, I, K3_MXFP4_GROUP);''',
 '''k3_matmul_mxfp4_xd(gu + I, zxd, q.p3, q.s3, L, I, K3_MXFP4_GROUP, use_mx_rowpair);'''),
('''k3_matmul_mxfp4_xd(edn, actxd, q.p2, q.s2, I, L, K3_MXFP4_GROUP);''',
 '''k3_matmul_mxfp4_xd(edn, actxd, q.p2, q.s2, I, L, K3_MXFP4_GROUP, use_mx_rowpair);''')]:
    if s.count(old)!=1: raise SystemExit(f'call anchor missing: {old[:35]} count={s.count(old)}')
    s=s.replace(old,new,1)

start='''static void k3_matmul_mxfp4_xd(float *y, const double *x,\n                               const unsigned char *packed,\n                               const unsigned char *scales, int in, int rows, int group)\n{\n'''
end='''}\n#endif\n\nvoid k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,\n'''
a=s.find(start); b=s.find(end,a)
if a<0 or b<0: raise SystemExit('xd implementation anchors missing')
b += len('''}\n#endif\n''')
replacement=r'''static void k3_matmul_mxfp4_xd(float *y, const double *x,
                               const unsigned char *packed,
                               const unsigned char *scales, int in, int rows, int group,
                               int rowpair)
{
    if (group != 32 || (in & 31)) {
        /* No production K3 expert takes this branch. Keep a defensive exact fallback by
         * converting back to float once; callers only use xd after an exact float->double
         * widening, so this round-trip is lossless. */
        float *xf = (float *)malloc((size_t)in * sizeof(float));
        if (!xf) k3_fatal_oom("MXFP4 xd fallback", (size_t)in * sizeof(float));
        for (int i = 0; i < in; i++) xf[i] = (float)x[i];
        k3_matmul_mxfp4(y, xf, packed, scales, in, rows, group);
        free(xf);
        return;
    }

    const int pcols = in / 2, ngrp = in / 32;
    if (!k3_e8m0_ready) k3_e8m0_init();
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i half_units = _mm_setr_epi8(
         0,  1,  2,  3,  4,  6,  8, 12,
         0, -1, -2, -3, -4, -6, -8,-12);

    /* Existing #22 single-row implementation is kept as the exact A/B fallback. */
    if (!rowpair || rows < 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 64)
#endif
        for (int r = 0; r < rows; r++) {
            const unsigned char *pr = packed + (size_t)r * pcols;
            const unsigned char *sr = scales + (size_t)r * ngrp;
            double acc = 0.0;
            for (int g = 0; g < ngrp; g++) {
                const unsigned char sb = sr[g];
                if (sb == 255) continue;
                const unsigned char *pb = pr + (size_t)g * 16;
                const double *xg = x + (size_t)g * 32;
                const __m128i b = _mm_loadu_si128((const __m128i *)pb);
                const __m128i lo = _mm_shuffle_epi8(half_units, _mm_and_si128(b, mask));
                const __m128i hi = _mm_shuffle_epi8(
                    half_units, _mm_and_si128(_mm_srli_epi16(b, 4), mask));
                const __m128i q0 = _mm_unpacklo_epi8(lo, hi);
                const __m128i q1 = _mm_unpackhi_epi8(lo, hi);
                const __m256i i0 = _mm256_cvtepi8_epi32(q0);
                const __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));
                const __m256i i2 = _mm256_cvtepi8_epi32(q1);
                const __m256i i3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));
                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
#define K3_XD_SINGLE_F4(V, I128, O) do {                                      \
                    (V) = _mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),          \
                                          _mm256_loadu_pd(xg + (O)), (V));      \
                } while (0)
                K3_XD_SINGLE_F4(v0, _mm256_castsi256_si128(i0),       0);
                K3_XD_SINGLE_F4(v1, _mm256_extracti128_si256(i0, 1),  4);
                K3_XD_SINGLE_F4(v0, _mm256_castsi256_si128(i1),       8);
                K3_XD_SINGLE_F4(v1, _mm256_extracti128_si256(i1, 1), 12);
                K3_XD_SINGLE_F4(v0, _mm256_castsi256_si128(i2),      16);
                K3_XD_SINGLE_F4(v1, _mm256_extracti128_si256(i2, 1), 20);
                K3_XD_SINGLE_F4(v0, _mm256_castsi256_si128(i3),      24);
                K3_XD_SINGLE_F4(v1, _mm256_extracti128_si256(i3, 1), 28);
#undef K3_XD_SINGLE_F4
                double a4[4];
                _mm256_storeu_pd(a4, _mm256_add_pd(v0, v1));
                const double sub2 = (a4[0] + a4[1]) + (a4[2] + a4[3]);
                acc += sub2 * ((double)K3_E8M0[sb] * 0.5);
            }
            y[r] = (float)acc;
        }
        return;
    }

    /* Two rows share the same 32 activation values. Load those eight 256-bit x vectors
     * once per group, then run two completely independent weight decodes/accumulators.
     * Each row retains exactly the #22 FMA lane assignment, reduction and scale order. */
    const int npair = rows / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (npair > 32)
#endif
    for (int rp = 0; rp < npair; rp++) {
        const int r0 = 2 * rp, r1 = r0 + 1;
        const unsigned char *p0 = packed + (size_t)r0 * pcols;
        const unsigned char *p1 = packed + (size_t)r1 * pcols;
        const unsigned char *s0 = scales + (size_t)r0 * ngrp;
        const unsigned char *s1 = scales + (size_t)r1 * ngrp;
        double acc0 = 0.0, acc1 = 0.0;

        for (int g = 0; g < ngrp; g++) {
            const unsigned char sb0 = s0[g], sb1 = s1[g];
            if (sb0 == 255 && sb1 == 255) continue;
            const double *xg = x + (size_t)g * 32;
            const __m256d x0 = _mm256_loadu_pd(xg + 0);
            const __m256d x1 = _mm256_loadu_pd(xg + 4);
            const __m256d x2 = _mm256_loadu_pd(xg + 8);
            const __m256d x3 = _mm256_loadu_pd(xg + 12);
            const __m256d x4 = _mm256_loadu_pd(xg + 16);
            const __m256d x5 = _mm256_loadu_pd(xg + 20);
            const __m256d x6 = _mm256_loadu_pd(xg + 24);
            const __m256d x7 = _mm256_loadu_pd(xg + 28);

#define K3_XD_PAIR_ROW(ACC, PB, SB) do {                                      \
                if ((SB) != 255) {                                            \
                    const __m128i rb = _mm_loadu_si128((const __m128i *)(PB)); \
                    const __m128i rlo = _mm_shuffle_epi8(half_units,           \
                        _mm_and_si128(rb, mask));                              \
                    const __m128i rhi = _mm_shuffle_epi8(half_units,           \
                        _mm_and_si128(_mm_srli_epi16(rb, 4), mask));           \
                    const __m128i rq0 = _mm_unpacklo_epi8(rlo, rhi);           \
                    const __m128i rq1 = _mm_unpackhi_epi8(rlo, rhi);           \
                    const __m256i ri0 = _mm256_cvtepi8_epi32(rq0);            \
                    const __m256i ri1 = _mm256_cvtepi8_epi32(                  \
                        _mm_srli_si128(rq0, 8));                               \
                    const __m256i ri2 = _mm256_cvtepi8_epi32(rq1);            \
                    const __m256i ri3 = _mm256_cvtepi8_epi32(                  \
                        _mm_srli_si128(rq1, 8));                               \
                    __m256d rv0 = _mm256_setzero_pd();                         \
                    __m256d rv1 = _mm256_setzero_pd();                         \
                    rv0 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_castsi256_si128(ri0)), x0, rv0);                \
                    rv1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_extracti128_si256(ri0, 1)), x1, rv1);           \
                    rv0 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_castsi256_si128(ri1)), x2, rv0);                \
                    rv1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_extracti128_si256(ri1, 1)), x3, rv1);           \
                    rv0 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_castsi256_si128(ri2)), x4, rv0);                \
                    rv1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_extracti128_si256(ri2, 1)), x5, rv1);           \
                    rv0 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_castsi256_si128(ri3)), x6, rv0);                \
                    rv1 = _mm256_fmadd_pd(_mm256_cvtepi32_pd(                 \
                        _mm256_extracti128_si256(ri3, 1)), x7, rv1);           \
                    double ra[4];                                              \
                    _mm256_storeu_pd(ra, _mm256_add_pd(rv0, rv1));             \
                    const double rsub = (ra[0] + ra[1]) + (ra[2] + ra[3]);    \
                    (ACC) += rsub * ((double)K3_E8M0[(SB)] * 0.5);             \
                }                                                              \
            } while (0)
            K3_XD_PAIR_ROW(acc0, p0 + (size_t)g * 16, sb0);
            K3_XD_PAIR_ROW(acc1, p1 + (size_t)g * 16, sb1);
#undef K3_XD_PAIR_ROW
        }
        y[r0] = (float)acc0;
        y[r1] = (float)acc1;
    }

    if (rows & 1) {
        const int r = rows - 1;
        k3_matmul_mxfp4_xd(y + r, x,
                           packed + (size_t)r * pcols,
                           scales + (size_t)r * ngrp,
                           in, 1, group, 0);
    }
}

#ifdef K3_TEST_INTERNALS
void k3_test_matmul_mxfp4_xd(float *y, const double *x,
                             const unsigned char *packed,
                             const unsigned char *scales,
                             int in, int rows, int group, int rowpair)
{
    k3_matmul_mxfp4_xd(y, x, packed, scales, in, rows, group, rowpair);
}
#endif
#endif
'''
s=s[:a]+replacement+s[b:]

p.write_text(s)
print('patched production exact row-pair MXFP4 xd path')
