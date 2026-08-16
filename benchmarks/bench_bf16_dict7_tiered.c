/* bench_bf16_dict7_tiered.c - lossless 3-bit primary BF16 high-byte codec prototype.
 *
 * Real K3 sample (16 MiB across layers 14/25/34/37): top-7 high bytes cover 94.1543%,
 * top-22 cover 99.99937%. A 3-bit primary dictionary therefore has a useful size win
 * over dict15 if its decoder remains fast enough.
 *
 * Layout for N BF16 words:
 *   low[N]
 *   primary bitplane0[ceil(N/8)] | bitplane1[...] | bitplane2[...]
 *   secondary[ceil(primary_escapes/2)]    (4-bit dict15 codes)
 *   literals[secondary escapes]
 *
 * Primary codes 0..6 select dict7; code 7 consumes one secondary nibble. Secondary
 * codes 0..14 select dict15; code 15 consumes one literal high byte. The low byte is
 * always stored verbatim. This is storage-only and byte-exact, never model quantisation.
 *
 * Build: cc -O3 -mavx2 -Wall -Wextra -o /tmp/bench benchmarks/bench_bf16_dict7_tiered.c
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

static uint32_t rngs = 0x9e3779b9u;
static uint32_t rng32(void)
{
    uint32_t x = rngs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rngs = x;
    return x;
}

/* Expand the eight bits of one byte into eight byte lanes containing 0 or 1.
 * Example: 0b10000101 -> bytes {1,0,1,0,0,0,0,1}. This lets three bitplanes become
 * eight 3-bit dictionary codes with three multiplies instead of scalar shifts. */
static inline uint64_t expand8(unsigned x)
{
    return ((uint64_t)(x & 255u) * UINT64_C(0x0002040810204081))
           & UINT64_C(0x0101010101010101);
}

static inline int pop_lsb(unsigned *m)
{
#if defined(__GNUC__) || defined(__clang__)
    const int bit = __builtin_ctz(*m);
#else
    int bit = 0; unsigned v = *m; while (!(v & 1u)) { v >>= 1; bit++; }
#endif
    *m &= *m - 1u;
    return bit;
}

static size_t decode_tiered(unsigned char *dst, size_t n,
                            const unsigned char *low,
                            const unsigned char *p0,
                            const unsigned char *p1,
                            const unsigned char *p2,
                            const unsigned char *sec,
                            const unsigned char *lit,
                            const unsigned char d7[7],
                            const unsigned char d15[15],
                            size_t *nlit_out)
{
    unsigned char d16[16] = {0};
    memcpy(d16, d7, 7);
    const __m128i vd = _mm_loadu_si128((const __m128i *)d16);
    const __m128i vesc = _mm_set1_epi8(7);
    size_t ne = 0, nl = 0, i = 0;

#if defined(__SSSE3__)
    for (; i + 15 < n; i += 16) {
        const size_t j = i >> 3;
        const uint64_t qa = expand8(p0[j]) | (expand8(p1[j]) << 1) | (expand8(p2[j]) << 2);
        const uint64_t qb = expand8(p0[j + 1]) | (expand8(p1[j + 1]) << 1) | (expand8(p2[j + 1]) << 2);
        const __m128i q = _mm_set_epi64x((long long)qb, (long long)qa);
        const __m128i hi = _mm_shuffle_epi8(vd, q);
        const __m128i lo = _mm_loadu_si128((const __m128i *)(low + i));
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm_unpacklo_epi8(lo, hi));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 16), _mm_unpackhi_epi8(lo, hi));

        unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(q, vesc));
        while (m) {
            const int pos = pop_lsb(&m);
            const unsigned char sb = sec[ne >> 1];
            const unsigned q2 = (ne & 1u) ? (sb >> 4) : (sb & 15u);
            ne++;
            dst[2 * (i + (size_t)pos) + 1] = q2 < 15 ? d15[q2] : lit[nl++];
        }
    }
#endif

    for (; i < n; i++) {
        const size_t j = i >> 3;
        const unsigned bit = (unsigned)(i & 7u);
        const unsigned q = ((p0[j] >> bit) & 1u)
                         | (((p1[j] >> bit) & 1u) << 1)
                         | (((p2[j] >> bit) & 1u) << 2);
        dst[2 * i] = low[i];
        if (q < 7) {
            dst[2 * i + 1] = d7[q];
        } else {
            const unsigned char sb = sec[ne >> 1];
            const unsigned q2 = (ne & 1u) ? (sb >> 4) : (sb & 15u);
            ne++;
            dst[2 * i + 1] = q2 < 15 ? d15[q2] : lit[nl++];
        }
    }
    if (nlit_out) *nlit_out = nl;
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

    /* Ordered from the real K3 BF16 sample recorded by the research workflow. */
    const unsigned char d7[7] = {0x3c,0xbc,0x3b,0xbb,0xbd,0x3d,0x3a};
    const unsigned char d15[15] = {
        0xba,0x39,0xb9,0xb8,0x38,0x37,0xb7,0xb6,0x36,0x3e,0xbe,0xb5,0x35,0x34,0xb4
    };

    unsigned char *raw=(unsigned char *)malloc(raw_bytes);
    unsigned char *out=(unsigned char *)malloc(raw_bytes);
    unsigned char *low=(unsigned char *)malloc(n);
    unsigned char *p0=(unsigned char *)calloc(plane_bytes,1);
    unsigned char *p1=(unsigned char *)calloc(plane_bytes,1);
    unsigned char *p2=(unsigned char *)calloc(plane_bytes,1);
    /* 5.85% primary escapes on real K3. Allocate generous capacity. */
    unsigned char *sec=(unsigned char *)calloc(n/16 + 4096,1);
    unsigned char *lit=(unsigned char *)malloc(n/1000 + 4096);
    if (!raw || !out || !low || !p0 || !p1 || !p2 || !sec || !lit) {
        fprintf(stderr,"allocation failed for %zu MiB benchmark\n",raw_mb); return 2;
    }

    size_t ne=0,nl=0;
    for (size_t i=0;i<n;i++) {
        const uint32_t r=rng32();
        const unsigned bucket=r%1000000u;
        unsigned q1,q2=0; unsigned char hb;
        if (bucket < 941543u) {
            q1=(r>>8)%7u; hb=d7[q1];
        } else {
            q1=7;
            /* Of all words, only ~0.000632% fall outside top22. */
            if (bucket < 999994u) { q2=(r>>16)%15u; hb=d15[q2]; }
            else { q2=15; hb=(unsigned char)(0x70u+((r>>20)&15u)); lit[nl++]=hb; }
            if (ne & 1u) sec[ne>>1] |= (unsigned char)(q2<<4);
            else sec[ne>>1]=(unsigned char)q2;
            ne++;
        }
        const unsigned char lb=(unsigned char)(r>>24);
        raw[2*i]=lb; raw[2*i+1]=hb; low[i]=lb;
        const unsigned char bit=(unsigned char)(1u<<(i&7u));
        if (q1&1u) p0[i>>3]|=bit;
        if (q1&2u) p1[i>>3]|=bit;
        if (q1&4u) p2[i>>3]|=bit;
    }

    const size_t sec_bytes=(ne+1)/2;
    const size_t packed=n+3*plane_bytes+sec_bytes+nl+22;
    printf("raw %.1f MiB words %zu primary_esc %zu (%.4f%%) literals %zu (%.6f%%)\n",
           raw_bytes/1048576.0,n,ne,100.0*ne/n,nl,100.0*nl/n);
    printf("ratio %.6f reduction %.2f%% packed %.1f MiB\n",
           (double)packed/raw_bytes,100.0*(1.0-(double)packed/raw_bytes),packed/1048576.0);

    size_t got_lit=0;
    size_t got_esc=decode_tiered(out,n,low,p0,p1,p2,sec,lit,d7,d15,&got_lit);
    if (got_esc!=ne || got_lit!=nl || memcmp(raw,out,raw_bytes)!=0) {
        fprintf(stderr,"ROUNDTRIP FAIL esc=%zu/%zu lit=%zu/%zu\n",got_esc,ne,got_lit,nl);
        return 1;
    }
    puts("byte-identical roundtrip: PASS");

    double best=1e99,sum=0.0; uint64_t checksum=0;
    for (int r=0;r<reps;r++) {
        const double t0=now_s();
        got_esc=decode_tiered(out,n,low,p0,p1,p2,sec,lit,d7,d15,&got_lit);
        const double dt=now_s()-t0;
        if (dt<best) best=dt; sum+=dt;
        checksum+=out[(size_t)r*4099u%raw_bytes]+got_esc+got_lit;
        printf("run %d: %.3f s %.2f GB/s reconstructed %.2f GB/s packed-input-equiv\n",
               r+1,dt,(double)raw_bytes/1e9/dt,(double)packed/1e9/dt);
    }
    printf("best %.2f GB/s reconstructed; mean %.2f GB/s; checksum %llu\n",
           (double)raw_bytes/1e9/best,(double)raw_bytes/1e9/(sum/reps),
           (unsigned long long)checksum);

    free(raw);free(out);free(low);free(p0);free(p1);free(p2);free(sec);free(lit);
    return 0;
}
