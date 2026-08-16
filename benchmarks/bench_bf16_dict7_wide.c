/* 32-wide lossless BF16 dict7 decoder research prototype.
 * Three bitplanes carry primary codes 0..7. Codes 0..6 select the top-7 K3 BF16 high
 * bytes; code 7 consumes one literal high byte. The hot path handles 32 words at once;
 * only the sparse (~5.85%) escape lanes are patched scalar after the vector store.
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

static uint32_t rngs = 0xa4093822u;
static uint32_t rng32(void)
{
    uint32_t x=rngs; x^=x<<13; x^=x>>17; x^=x<<5; rngs=x; return x;
}

static uint64_t EXPAND8[256];
static void init_expand8(void)
{
    for (unsigned x=0;x<256;x++) {
        uint64_t v=0;
        for (unsigned b=0;b<8;b++) v|=(uint64_t)((x>>b)&1u)<<(8u*b);
        EXPAND8[x]=v;
    }
}

static inline unsigned pop_lsb32(uint32_t *m)
{
#if defined(__GNUC__) || defined(__clang__)
    const unsigned b=(unsigned)__builtin_ctz(*m);
#else
    unsigned b=0,v=*m; while (!(v&1u)) { v>>=1; b++; }
#endif
    *m&=*m-1u;
    return b;
}

static size_t decode_wide(unsigned char *dst,size_t n,const unsigned char *low,
                          const unsigned char *p0,const unsigned char *p1,
                          const unsigned char *p2,const unsigned char *lit,
                          const unsigned char d7[7])
{
    unsigned char d16[16]={0}; memcpy(d16,d7,7);
    const __m128i d128=_mm_loadu_si128((const __m128i *)d16);
    const __m256i vd=_mm256_broadcastsi128_si256(d128);
    const __m256i vesc=_mm256_set1_epi8(7);
    size_t ne=0,i=0;
#if defined(__AVX2__)
    for (;i+31<n;i+=32) {
        const size_t j=i>>3;
#define Q8(k) (EXPAND8[p0[(k)]] | (EXPAND8[p1[(k)]]<<1) | (EXPAND8[p2[(k)]]<<2))
        const uint64_t q0=Q8(j),q1=Q8(j+1),q2=Q8(j+2),q3=Q8(j+3);
#undef Q8
        const __m256i q=_mm256_set_epi64x((long long)q3,(long long)q2,
                                          (long long)q1,(long long)q0);
        const __m256i hi=_mm256_shuffle_epi8(vd,q);
        const __m256i lo=_mm256_loadu_si256((const __m256i *)(low+i));

        /* AVX2 unpack is 128-bit lane-local. Store each 16-word half explicitly so the
         * destination remains ordinary [low,high] BF16 byte order. */
        const __m128i lo0=_mm256_castsi256_si128(lo);
        const __m128i lo1=_mm256_extracti128_si256(lo,1);
        const __m128i hi0=_mm256_castsi256_si128(hi);
        const __m128i hi1=_mm256_extracti128_si256(hi,1);
        _mm_storeu_si128((__m128i *)(dst+2*i),_mm_unpacklo_epi8(lo0,hi0));
        _mm_storeu_si128((__m128i *)(dst+2*i+16),_mm_unpackhi_epi8(lo0,hi0));
        _mm_storeu_si128((__m128i *)(dst+2*i+32),_mm_unpacklo_epi8(lo1,hi1));
        _mm_storeu_si128((__m128i *)(dst+2*i+48),_mm_unpackhi_epi8(lo1,hi1));

        uint32_t mask=(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(q,vesc));
        while (mask) {
            const unsigned pos=pop_lsb32(&mask);
            dst[2*(i+(size_t)pos)+1]=lit[ne++];
        }
    }
#endif
    for (;i<n;i++) {
        const size_t j=i>>3; const unsigned b=(unsigned)(i&7u);
        const unsigned q=((p0[j]>>b)&1u)|(((p1[j]>>b)&1u)<<1)|(((p2[j]>>b)&1u)<<2);
        dst[2*i]=low[i]; dst[2*i+1]=q<7?d7[q]:lit[ne++];
    }
    return ne;
}

int main(int argc,char **argv)
{
    size_t raw_mb=256; int reps=5;
    if (argc>1) raw_mb=(size_t)strtoull(argv[1],NULL,10);
    if (argc>2) reps=atoi(argv[2]);
    const size_t raw_bytes=raw_mb<<20,n=raw_bytes/2,pb=(n+7)/8;
    const unsigned char d7[7]={0x3c,0xbc,0x3b,0xbb,0xbd,0x3d,0x3a};
    init_expand8();

    unsigned char *raw=malloc(raw_bytes),*out=malloc(raw_bytes),*low=malloc(n);
    unsigned char *p0=calloc(pb,1),*p1=calloc(pb,1),*p2=calloc(pb,1),*lit=malloc(n/8+32);
    if (!raw||!out||!low||!p0||!p1||!p2||!lit) return 2;
    size_t ne=0;
    for (size_t i=0;i<n;i++) {
        const uint32_t r=rng32(); const unsigned bucket=r%1000000u;
        unsigned q; unsigned char hb;
        if (bucket<941543u) { q=(r>>8)%7u; hb=d7[q]; }
        else { q=7; hb=(unsigned char)(r>>16); lit[ne++]=hb; }
        const unsigned char lb=(unsigned char)(r>>24); raw[2*i]=lb; raw[2*i+1]=hb; low[i]=lb;
        const unsigned char bit=(unsigned char)(1u<<(i&7u));
        if(q&1u)p0[i>>3]|=bit; if(q&2u)p1[i>>3]|=bit; if(q&4u)p2[i>>3]|=bit;
    }
    const size_t packed=n+3*pb+ne+7;
    printf("raw %.1f MiB words %zu escapes %zu (%.4f%%) ratio %.6f reduction %.2f%%\n",
           raw_bytes/1048576.0,n,ne,100.0*ne/n,(double)packed/raw_bytes,
           100.0*(1.0-(double)packed/raw_bytes));
    size_t used=decode_wide(out,n,low,p0,p1,p2,lit,d7);
    if(used!=ne||memcmp(raw,out,raw_bytes)){fprintf(stderr,"ROUNDTRIP FAIL %zu/%zu\n",used,ne);return 1;}
    puts("byte-identical roundtrip: PASS");
    double best=1e99,sum=0; uint64_t checksum=0;
    for(int r=0;r<reps;r++){
        double t0=now_s(); used=decode_wide(out,n,low,p0,p1,p2,lit,d7); double dt=now_s()-t0;
        if(dt<best)best=dt; sum+=dt; checksum+=out[(size_t)r*4099u%raw_bytes]+used;
        printf("run %d: %.3f s %.2f GB/s reconstructed %.2f GB/s packed-input-equiv\n",
               r+1,dt,(double)raw_bytes/1e9/dt,(double)packed/1e9/dt);
    }
    printf("best %.2f GB/s reconstructed; mean %.2f GB/s; checksum %llu\n",
           (double)raw_bytes/1e9/best,(double)raw_bytes/1e9/(sum/reps),(unsigned long long)checksum);
    free(raw);free(out);free(low);free(p0);free(p1);free(p2);free(lit);return 0;
}
