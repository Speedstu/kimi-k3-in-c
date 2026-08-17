/* test_trunk_codec.c - weightless byte-exact gate for the lossless trunk codec. */
#include "k3_codec.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rs = 0x9e3779b9u;
static uint32_t rnd(void)
{
    uint32_t x = rs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rs = x;
    return x;
}

int main(void)
{
    const size_t n = 65539;              /* deliberately not a SIMD multiple */
    const size_t raw_n = n * 2;
    const size_t cb = (n + 1) / 2;
    const unsigned char dict[15] = {
        0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e,0xbe,0x3a,0xba,0x39,0xb9,0x3f,0xbf,0x00
    };
    unsigned char *raw = (unsigned char *)malloc(raw_n);
    unsigned char *low = (unsigned char *)malloc(n);
    unsigned char *code = (unsigned char *)calloc(cb, 1);
    unsigned char *esc = (unsigned char *)malloc(n);
    unsigned char *out = (unsigned char *)malloc(raw_n);
    if (!raw || !low || !code || !esc || !out) return 2;

    size_t ne = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = rnd();
        unsigned char q, hi;
        if ((r % 97u) == 0u) {          /* enough escapes to exercise every path */
            q = 15;
            hi = (unsigned char)(0x70u + ((r >> 16) & 15u));
            esc[ne++] = hi;
        } else {
            q = (unsigned char)((r >> 8) % 15u);
            hi = dict[q];
        }
        const unsigned char lo = (unsigned char)r;
        raw[2 * i] = lo; raw[2 * i + 1] = hi; low[i] = lo;
        if (i & 1) code[i >> 1] |= (unsigned char)(q << 4);
        else       code[i >> 1]  = q;
    }

    const size_t encoded = n + cb + ne;
    unsigned char *src = (unsigned char *)malloc(encoded);
    if (!src) return 2;
    memcpy(src, low, n);
    memcpy(src + n, code, cb);
    memcpy(src + n + cb, esc, ne);

    const size_t used = k3_dict15_decode(out, raw_n, src, encoded, dict);
    if (used != ne || memcmp(out, raw, raw_n) != 0) {
        fprintf(stderr, "FAIL dict15 roundtrip: escapes %zu/%zu, bytes_equal=%d\n",
                used, ne, memcmp(out, raw, raw_n) == 0);
        return 1;
    }
    /* Truncating the escape tail must be rejected rather than reading past it. */
    if (ne && k3_dict15_decode(out, raw_n, src, encoded - 1, dict) != SIZE_MAX) {
        fprintf(stderr, "FAIL dict15 accepted truncated escape payload\n");
        return 1;
    }
    /* dict7 uses an independent 3-bit packing and must survive a non-multiple-of-eight
     * tail plus frequent escapes. */
    const unsigned char d7[7] = {0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e};
    const size_t cb7 = (3u * n + 7u) / 8u;
    unsigned char *c7 = (unsigned char *)calloc(cb7, 1);
    unsigned char *e7 = (unsigned char *)malloc(n);
    unsigned char *r7 = (unsigned char *)malloc(raw_n);
    unsigned char *o7 = (unsigned char *)malloc(raw_n);
    if (!c7 || !e7 || !r7 || !o7) return 2;
    size_t ne7 = 0;
    rs = 0x51f15e11u;
    for (size_t j = 0; j < n; j++) {
        const uint32_t r = rnd();
        unsigned char q, hi;
        if ((r % 29u) == 0u) { q = 7; hi = (unsigned char)(0x70u + ((r >> 16) & 31u)); e7[ne7++] = hi; }
        else { q = (unsigned char)((r >> 8) % 7u); hi = d7[q]; }
        r7[2u*j] = low[j]; r7[2u*j+1u] = hi;
        const size_t bit = 3u*j, bo = bit >> 3;
        const unsigned sh = (unsigned)(bit & 7u);
        c7[bo] |= (unsigned char)(q << sh);
        if (sh > 5u && bo + 1u < cb7) c7[bo+1u] |= (unsigned char)(q >> (8u-sh));
    }
    const size_t enc7 = n + cb7 + ne7;
    unsigned char *s7 = (unsigned char *)malloc(enc7);
    if (!s7) return 2;
    memcpy(s7, low, n); memcpy(s7+n, c7, cb7); memcpy(s7+n+cb7, e7, ne7);
    const size_t used7 = k3_dict7_decode(o7, raw_n, s7, enc7, d7);
    if (used7 != ne7 || memcmp(o7, r7, raw_n) != 0) {
        fprintf(stderr, "FAIL dict7 roundtrip: escapes %zu/%zu, bytes_equal=%d\n",
                used7, ne7, memcmp(o7, r7, raw_n) == 0);
        return 1;
    }
    if (ne7 && k3_dict7_decode(o7, raw_n, s7, enc7 - 1u, d7) != SIZE_MAX) {
        fprintf(stderr, "FAIL dict7 accepted truncated escape payload\n");
        return 1;
    }
    printf("LOSSLESS TRUNK CODECS PASSED: dict15 %zu escapes, dict7 %zu escapes, byte-identical\n",
           ne, ne7);
    free(raw); free(low); free(code); free(esc); free(out); free(src);
    free(c7); free(e7); free(r7); free(o7); free(s7);
    return 0;
}
