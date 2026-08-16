#!/usr/bin/env python3
from pathlib import Path

p=Path('src/core/k3_ops.c')
s=p.read_text()

# Private helper declaration before k3_moe, implementation lives beside public MXFP4 kernel.
anchor='''/* ------------------------------------------------------- Stable LatentMoE ---- */\n'''
insert='''#if defined(__AVX2__)\n/* Exact-K3 streamed-expert helper: x is already widened from float to double once by\n * the MoE caller, so thousands of output rows do not repeat the same cvtps2pd work. */\nstatic void k3_matmul_mxfp4_xd(float *y, const double *x,\n                               const unsigned char *packed,\n                               const unsigned char *scales, int in, int rows, int group);\n#endif\n\n'''
if s.count(anchor)!=1: raise SystemExit('MoE anchor mismatch')
s=s.replace(anchor,insert+anchor,1)

old='''    float *sdn  = sact + SI;            /* [E]    shared down-projection    */\n\n    for (int t = 0; t < T; t++) {\n'''
new='''    float *sdn  = sact + SI;            /* [E]    shared down-projection    */\n#if defined(__AVX2__)\n    /* Align the private double workspace inside the caller-owned float scratch. The\n     * scratch contract reserves one extra float for worst-case 8-byte alignment. */\n    uintptr_t xdp = ((uintptr_t)(sdn + E) + 7u) & ~(uintptr_t)7u;\n    double *zxd   = (double *)xdp;       /* [L] shared by w1+w3 across ALL routed experts */\n    double *actxd = zxd + L;             /* [I] refreshed once per expert for w2           */\n    static int no_mx_xdouble = -1;\n    if (no_mx_xdouble < 0) no_mx_xdouble = getenv("K3_NO_MX_XDOUBLE") ? 1 : 0;\n#endif\n\n    for (int t = 0; t < T; t++) {\n'''
if s.count(old)!=1: raise SystemExit('scratch layout anchor mismatch')
s=s.replace(old,new,1)

old='''        k3_mmw(z, xt, w->down, w->wdt, E, L);\n\n        /* The shared expert is also independent of the routed experts. Compute it early\n'''
new='''        k3_mmw(z, xt, w->down, w->wdt, E, L);\n#if defined(__AVX2__)\n        const int use_mx_xdouble = w->src && !no_mx_xdouble;\n        if (use_mx_xdouble)\n            for (int i = 0; i < L; i++) zxd[i] = (double)z[i];\n#endif\n\n        /* The shared expert is also independent of the routed experts. Compute it early\n'''
if s.count(old)!=1: raise SystemExit('down-project anchor mismatch')
s=s.replace(old,new,1)

old='''                k3_matmul_mxfp4(gu,     z, q.p1, q.s1, L, I, K3_MXFP4_GROUP);\n                k3_matmul_mxfp4(gu + I, z, q.p3, q.s3, L, I, K3_MXFP4_GROUP);\n                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);\n                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, K3_MXFP4_GROUP);\n'''
new='''#if defined(__AVX2__)\n                if (use_mx_xdouble) {\n                    /* z is identical for every selected expert in this token. Widen it\n                     * once above and reuse it for both gate/up matrices across top-k. */\n                    k3_matmul_mxfp4_xd(gu,     zxd, q.p1, q.s1, L, I, K3_MXFP4_GROUP);\n                    k3_matmul_mxfp4_xd(gu + I, zxd, q.p3, q.s3, L, I, K3_MXFP4_GROUP);\n                    k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);\n                    /* act changes per expert, but widening it ONCE still replaces one\n                     * conversion per output row in w2. */\n                    for (int i = 0; i < I; i++) actxd[i] = (double)act[i];\n                    k3_matmul_mxfp4_xd(edn, actxd, q.p2, q.s2, I, L, K3_MXFP4_GROUP);\n                } else\n#endif\n                {\n                    k3_matmul_mxfp4(gu,     z, q.p1, q.s1, L, I, K3_MXFP4_GROUP);\n                    k3_matmul_mxfp4(gu + I, z, q.p3, q.s3, L, I, K3_MXFP4_GROUP);\n                    k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);\n                    k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, K3_MXFP4_GROUP);\n                }\n'''
if s.count(old)!=1: raise SystemExit('streamed expert anchor mismatch')
s=s.replace(old,new,1)

old='''    return (size_t)2 * c->latent          /* z, accL            */\n         + (size_t)3 * c->moe_inter       /* gu (2*I) + act (I) */\n         + (size_t)c->latent              /* edn                */\n         + (size_t)3 * SI                 /* sgu (2*SI) + sact  */\n         + (size_t)c->hidden;             /* sdn                */\n'''
new='''    size_t n = (size_t)2 * c->latent     /* z, accL            */\n             + (size_t)3 * c->moe_inter   /* gu (2*I) + act (I) */\n             + (size_t)c->latent          /* edn                */\n             + (size_t)3 * SI             /* sgu (2*SI) + sact  */\n             + (size_t)c->hidden;         /* sdn                */\n#if defined(__AVX2__)\n    /* double z + double act, expressed in float units, plus one float for alignment. */\n    n += (size_t)2 * (c->latent + c->moe_inter) + 1u;\n#endif\n    return n;\n'''
if s.count(old)!=1: raise SystemExit('moe scratch anchor mismatch')
s=s.replace(old,new,1)

# Insert private xd kernel immediately before the public kernel. It is deliberately only
# the released K3 geometry fast path; unusual shapes fall back to the public implementation.
anchor='''void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,\n                     const unsigned char *scales, int in, int rows, int group)\n{\n'''
helper=r'''#if defined(__AVX2__)
static void k3_matmul_mxfp4_xd(float *y, const double *x,
                               const unsigned char *packed,
                               const unsigned char *scales, int in, int rows, int group)
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
#define K3_XD_F4(V, I128, O) do {                                              \
                (V) = _mm256_fmadd_pd(_mm256_cvtepi32_pd((I128)),              \
                                      _mm256_loadu_pd(xg + (O)), (V));          \
            } while (0)
            K3_XD_F4(v0, _mm256_castsi256_si128(i0),       0);
            K3_XD_F4(v1, _mm256_extracti128_si256(i0, 1),  4);
            K3_XD_F4(v0, _mm256_castsi256_si128(i1),       8);
            K3_XD_F4(v1, _mm256_extracti128_si256(i1, 1), 12);
            K3_XD_F4(v0, _mm256_castsi256_si128(i2),      16);
            K3_XD_F4(v1, _mm256_extracti128_si256(i2, 1), 20);
            K3_XD_F4(v0, _mm256_castsi256_si128(i3),      24);
            K3_XD_F4(v1, _mm256_extracti128_si256(i3, 1), 28);
#undef K3_XD_F4
            double a[4];
            _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
            const double sub2 = (a[0] + a[1]) + (a[2] + a[3]);
            acc += sub2 * ((double)K3_E8M0[sb] * 0.5);
        }
        y[r] = (float)acc;
    }
}
#endif

'''
if s.count(anchor)!=1: raise SystemExit('public MXFP4 anchor mismatch')
s=s.replace(anchor,helper+anchor,1)

p.write_text(s)
print('patched shared-xdouble streamed MXFP4 path')
