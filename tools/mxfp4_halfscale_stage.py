#!/usr/bin/env python3
from pathlib import Path
p=Path('src/core/k3_ops.c')
s=p.read_text()

old='''    const __m128i mx_half_units = _mm_setr_epi8(\n         0,  1,  2,  3,  4,  6,  8, 12,\n         0, -1, -2, -3, -4, -6, -8,-12);\n    const __m256d mx_half = _mm256_set1_pd(0.5);\n'''
new='''    const __m128i mx_half_units = _mm_setr_epi8(\n         0,  1,  2,  3,  4,  6,  8, 12,\n         0, -1, -2, -3, -4, -6, -8,-12);\n'''
if s.count(old)!=1: raise SystemExit(f'constant anchor count={s.count(old)}')
s=s.replace(old,new,1)

start='''                const __m128i q0 = _mm_unpacklo_epi8(lo, hi);\n                const __m128i q1 = _mm_unpackhi_epi8(lo, hi);\n                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();\n\n                /* Same lane ownership as the old wf[] AVX2 loop:\n'''
end='''                continue;\n            }\n#endif\n'''
a=s.find(start); b=s.find(end,a)
if a<0 or b<0: raise SystemExit('fast block anchors missing')
b += len(end)
newblock='''                const __m128i q0 = _mm_unpacklo_epi8(lo, hi);\n                const __m128i q1 = _mm_unpackhi_epi8(lo, hi);\n\n                /* Widen eight signed half-units at once. Splitting each 256-bit int\n                 * vector into two 128-bit halves feeds the exact same four-double FMA\n                 * lanes as before, but halves the number of int8->int32 conversions. */\n                const __m256i i0 = _mm256_cvtepi8_epi32(q0);\n                const __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));\n                const __m256i i2 = _mm256_cvtepi8_epi32(q1);\n                const __m256i i3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));\n                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();\n\n                /* These integers are exactly 2*E2M1. Summing products at twice the\n                 * magnitude and multiplying the group scale by 0.5 is an exact binary\n                 * rescaling: all source floats convert exactly to double, the integer\n                 * products fit easily in double precision, and the FMA/reduction order\n                 * is unchanged. The scalar-vs-AVX2 parity gate covers both real shapes. */\n#define K3_MX_F4(V, I128, XOFF) do {                                          \\\n                    (V) = _mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),          \\\n                        _mm256_cvtps_pd(_mm_loadu_ps(xg + (XOFF))), (V));       \\\n                } while (0)\n                K3_MX_F4(v0, _mm256_castsi256_si128(i0),       0);\n                K3_MX_F4(v1, _mm256_extracti128_si256(i0, 1),  4);\n                K3_MX_F4(v0, _mm256_castsi256_si128(i1),       8);\n                K3_MX_F4(v1, _mm256_extracti128_si256(i1, 1), 12);\n                K3_MX_F4(v0, _mm256_castsi256_si128(i2),      16);\n                K3_MX_F4(v1, _mm256_extracti128_si256(i2, 1), 20);\n                K3_MX_F4(v0, _mm256_castsi256_si128(i3),      24);\n                K3_MX_F4(v1, _mm256_extracti128_si256(i3, 1), 28);\n#undef K3_MX_F4\n                double a[4];\n                _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));\n                const double sub2 = (a[0] + a[1]) + (a[2] + a[3]);\n                acc += sub2 * ((double)K3_E8M0[sb] * 0.5);\n                continue;\n            }\n#endif\n'''
s=s[:a]+newblock+s[b:]

old_comment='''            /* Generic/scalar fallback: expand the group to floats first, then take a\n             * plain dot product. Kept unchanged for non-K3 geometry and partial tails.\n             * split exists so the second loop can vectorise, which it cannot do while\n             * a table lookup sits in the middle of the accumulation. */\n'''
new_comment='''            /* Generic/scalar fallback: expand the group to floats first, then take a\n             * plain dot product. Kept unchanged for non-K3 geometry and partial tails;\n             * the split lets the second loop vectorise without a table lookup inside\n             * the accumulation. */\n'''
if s.count(old_comment)!=1: raise SystemExit(f'comment anchor count={s.count(old_comment)}')
s=s.replace(old_comment,new_comment,1)
p.write_text(s)
print('patched half-scale MXFP4 fast path')
