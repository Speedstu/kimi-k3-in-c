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
    printf("LOSSLESS TRUNK CODEC PASSED: %zu bytes, %zu escapes, byte-identical\n",
           raw_n, ne);
    free(raw); free(low); free(code); free(esc); free(out); free(src);
    return 0;
}
