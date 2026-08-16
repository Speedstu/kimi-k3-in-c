/* bench_bf16_dict7_literal.c - lossless 3-bit BF16 high-byte codec with direct escapes.
 *
 * Real K3 sample: top-7 high bytes cover 94.154286%. Codes 0..6 select those values;
 * code 7 consumes one literal high byte. Three bitplanes hold the 3-bit code stream.
 * The low BF16 byte remains verbatim. Predicted ratio on that sample is ~0.716729.
 *
 * Unlike the tiered prototype, escape patching is vectorized eight words at a time.
 * A 2 KiB mask->shuffle table scatters the next packed literal bytes into exactly the
 * lanes whose primary code is 7. No per-escape loop is required.
 */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint32_t rngs = 0x243f6a88u;
static uint32_t rng32(void)
{
    uint32_t x = rngs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rngs = x;
    return x;
}

static uint64_t EXPAND8[256];
static unsigned char SCATTER8[256][16];
static void init_tables(void)
{
    for (unsigned x = 0; x < 256; x++) {
        uint64_t v = 0;
        for (unsigned b = 0; b < 8; b++)
            v |= (uint64_t)((x >> b) & 1u) << (8u * b);
        EXPAND8[x] = v;

        unsigned rank = 0;
        for (unsigned b = 0; b < 8; b++)
            SCATTER8[x][b] = (x & (1u << b)) ? (unsigned char)rank++ : 0x80u;
        for (unsigned b = 8; b < 16; b++) SCATTER8[x][b] = 0x80u;
    }
}

static size_t decode_dict7_literal(unsigned char *dst, size_t n,
                                   const unsigned char *low,
                                   const unsigned char *p0,
                                   const unsigned char *p1,
                                   const unsigned char *p2,
                                   const unsigned char *lit,
                                   const unsigned char d7[7])
{
    unsigned char d16[16] = {0};
    memcpy(d16, d7, 7);
    const __m128i vd = _mm_loadu_si128((const __m128i *)d16);
    const __m128i vesc = _mm_set1_epi8(7);
    size_t ne = 0, i = 0;

#if defined(__SSSE3__) && defined(__SSE4_1__)
    for (; i + 7 < n; i += 8) {
        const size_t j = i >> 3;
        const uint64_t qq = EXPAND8[p0[j]] | (EXPAND8[p1[j]] << 1) | (EXPAND8[p2[j]] << 2);
        const __m128i q = _mm_cvtsi64_si128((long long)qq);
        __m128i hi = _mm_shuffle_epi8(vd, q);
        const __m128i isesc = _mm_cmpeq_epi8(q, vesc);
        const unsigned mask = (unsigned)_mm_movemask_epi8(isesc) & 255u;
        if (mask) {
            /* lit has at least 8 bytes of allocation padding in this benchmark. A
             * production decoder will validate stored_nbytes before permitting this load. */
            const __m128i packed = _mm_loadl_epi64((const __m128i *)(lit + ne));
            const __m128i ctl = _mm_loadu_si128((const __m128i *)SCATTER8[mask]);
            const __m128i scattered = _mm_shuffle_epi8(packed, ctl);
            hi = _mm_blendv_epi8(hi, scattered, isesc);
#if defined(__GNUC__) || defined(__clang__)
            ne += (size_t)__builtin_popcount(mask);
#else
            unsigned m = mask; while (m) { ne++; m &= m - 1u; }
#endif
        }
        const __m128i lo = _mm_loadl_epi64((const __m128i *)(low + i));
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm_unpacklo_epi8(lo, hi));
    }
#endif

    for (; i < n; i++) {
        const size_t j = i >> 3;
        const unsigned bit = (unsigned)(i & 7u);
        const unsigned q = ((p0[j] >> bit) & 1u)
                         | (((p1[j] >> bit) & 1u) << 1)
                         | (((p2[j] >> bit) & 1u) << 2);
        dst[2 * i] = low[i];
        dst[2 * i + 1] = q < 7 ? d7[q] : lit[ne++];
    }
    return ne;
}

int main(int argc, char **argv)
{
    size_t raw_mb = 256;
    int reps = 5;
    if (argc > 1) raw_mb = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) reps = atoi(argv[2]);
    const size_t raw_bytes = raw_mb << 20;
    const size_t n = raw_bytes / 2;
    const size_t plane_bytes = (n + 7) / 8;
    const unsigned char d7[7] = {0x3c,0xbc,0x3b,0xbb,0xbd,0x3d,0x3a};

    init_tables();
    unsigned char *raw=(unsigned char *)malloc(raw_bytes);
    unsigned char *out=(unsigned char *)malloc(raw_bytes);
    unsigned char *low=(unsigned char *)malloc(n);
    unsigned char *p0=(unsigned char *)calloc(plane_bytes,1);
    unsigned char *p1=(unsigned char *)calloc(plane_bytes,1);
    unsigned char *p2=(unsigned char *)calloc(plane_bytes,1);
    unsigned char *lit=(unsigned char *)malloc(n/8 + 32);
    if (!raw || !out || !low || !p0 || !p1 || !p2 || !lit) {
        fprintf(stderr,"allocation failed for %zu MiB benchmark\n",raw_mb); return 2;
    }

    size_t ne=0;
    for (size_t i=0;i<n;i++) {
        const uint32_t r=rng32();
        const unsigned bucket=r%1000000u;
        unsigned q; unsigned char hb;
        if (bucket < 941543u) { q=(r>>8)%7u; hb=d7[q]; }
        else { q=7; hb=(unsigned char)(r>>16); lit[ne++]=hb; }
        const unsigned char lb=(unsigned char)(r>>24);
        raw[2*i]=lb; raw[2*i+1]=hb; low[i]=lb;
        const unsigned char bit=(unsigned char)(1u<<(i&7u));
        if (q&1u) p0[i>>3]|=bit;
        if (q&2u) p1[i>>3]|=bit;
        if (q&4u) p2[i>>3]|=bit;
    }
    memset(lit + ne, 0, 16); /* safe final SIMD over-read padding */

    const size_t packed=n+3*plane_bytes+ne+7;
    printf("raw %.1f MiB words %zu escapes %zu (%.4f%%) ratio %.6f reduction %.2f%%\n",
           raw_bytes/1048576.0,n,ne,100.0*ne/n,(double)packed/raw_bytes,
           100.0*(1.0-(double)packed/raw_bytes));

    size_t used=decode_dict7_literal(out,n,low,p0,p1,p2,lit,d7);
    if (used!=ne || memcmp(raw,out,raw_bytes)!=0) {
        fprintf(stderr,"ROUNDTRIP FAIL escapes=%zu/%zu\n",used,ne); return 1;
    }
    puts("byte-identical roundtrip: PASS");

    double best=1e99,sum=0.0; uint64_t checksum=0;
    for (int r=0;r<reps;r++) {
        const double t0=now_s();
        used=decode_dict7_literal(out,n,low,p0,p1,p2,lit,d7);
        const double dt=now_s()-t0;
        if (dt<best) best=dt;
        sum+=dt;
        checksum+=out[(size_t)r*4099u%raw_bytes]+used;
        printf("run %d: %.3f s %.2f GB/s reconstructed %.2f GB/s packed-input-equiv\n",
               r+1,dt,(double)raw_bytes/1e9/dt,(double)packed/1e9/dt);
    }
    printf("best %.2f GB/s reconstructed; mean %.2f GB/s; checksum %llu\n",
           (double)raw_bytes/1e9/best,(double)raw_bytes/1e9/(sum/reps),
           (unsigned long long)checksum);

    free(raw);free(out);free(low);free(p0);free(p1);free(p2);free(lit);
    return 0;
}
