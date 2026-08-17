/* Throughput smoke for the exact trunk storage codecs. Not a model benchmark. */
#define _POSIX_C_SOURCE 200809L
#include "k3_codec.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void) {
    const size_t n = (size_t)16 << 20; /* 32 MiB reconstructed BF16 */
    const size_t raw_n = 2u * n;
    const size_t cb7 = (3u * n + 7u) / 8u;
    const size_t cb15 = (n + 1u) / 2u;
    const unsigned char d7[7] = {0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e};
    const unsigned char d15[15] = {0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e,0xbe,0x3a,0xba,0x39,0xb9,0x3f,0xbf,0x00};
    unsigned char *low = malloc(n), *c7 = calloc(cb7,1), *c15 = calloc(cb15,1);
    unsigned char *esc7 = malloc(n/100u+64u), *src7 = NULL, *src15 = NULL;
    unsigned char *out7 = malloc(raw_n), *out15 = malloc(raw_n), *ref = malloc(raw_n);
    if (!low || !c7 || !c15 || !esc7 || !out7 || !out15 || !ref) return 2;

    size_t ne7 = 0;
    for (size_t i=0;i<n;i++) {
        const unsigned char lo = (unsigned char)(i * 131u + (i >> 9));
        unsigned char q = (unsigned char)((i * 5u + (i >> 7)) % 7u);
        unsigned char hi = d7[q];
        if ((i % 257u) == 0u) { q = 7; hi = (unsigned char)(0x70u + (i & 31u)); esc7[ne7++] = hi; }
        low[i]=lo; ref[2u*i]=lo; ref[2u*i+1u]=hi;
        const size_t bit=3u*i, bo=bit>>3; const unsigned sh=(unsigned)(bit&7u);
        c7[bo] |= (unsigned char)(q << sh);
        if (sh>5u && bo+1u<cb7) c7[bo+1u] |= (unsigned char)(q >> (8u-sh));
        /* Same source distribution is representable without escapes in dict15 except
         * the deliberately rare literal, which becomes code 15. */
        unsigned char q15 = q < 7 ? q : 15;
        if (i & 1u) c15[i>>1] |= (unsigned char)(q15 << 4); else c15[i>>1] = q15;
    }
    const size_t e7n = n + cb7 + ne7;
    const size_t e15n = n + cb15 + ne7;
    src7=malloc(e7n); src15=malloc(e15n);
    if (!src7 || !src15) return 2;
    memcpy(src7,low,n); memcpy(src7+n,c7,cb7); memcpy(src7+n+cb7,esc7,ne7);
    memcpy(src15,low,n); memcpy(src15+n,c15,cb15); memcpy(src15+n+cb15,esc7,ne7);
    if (k3_dict7_decode(out7,raw_n,src7,e7n,d7)!=ne7 || memcmp(out7,ref,raw_n)) return 1;
    if (k3_dict15_decode(out15,raw_n,src15,e15n,d15)!=ne7 || memcmp(out15,ref,raw_n)) return 1;

    const int rounds=12;
    volatile unsigned checksum=0;
    double t0=now_s();
    for (int r=0;r<rounds;r++) { if(k3_dict7_decode(out7,raw_n,src7,e7n,d7)==SIZE_MAX)return 1; checksum+=out7[(size_t)r*997u%raw_n]; }
    double t7=now_s()-t0;
    t0=now_s();
    for (int r=0;r<rounds;r++) { if(k3_dict15_decode(out15,raw_n,src15,e15n,d15)==SIZE_MAX)return 1; checksum+=out15[(size_t)r*991u%raw_n]; }
    double t15=now_s()-t0;
    const double gb=(double)raw_n*rounds/1e9;
    printf("dict7  ratio %.4f decode %.2f GB/s\n",(double)e7n/raw_n,gb/t7);
    printf("dict15 ratio %.4f decode %.2f GB/s\n",(double)e15n/raw_n,gb/t15);
    printf("dict7 physical-byte saving vs dict15 %.1f%% checksum %u\n",100.0*(1.0-(double)e7n/e15n),checksum);
    free(low);free(c7);free(c15);free(esc7);free(src7);free(src15);free(out7);free(out15);free(ref);
    return 0;
}
