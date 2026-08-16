/* bench_bf16_dict_codec.c - prototype a deliberately simple lossless BF16 codec.
 *
 * A BF16 value is two bytes little-endian: low mantissa byte + high sign/exponent byte.
 * On the released K3 trunk the high-byte alphabet is extremely concentrated (the prior
 * Huffman experiment found roughly a dozen values cover 99.9%). Instead of entropy
 * coding it, encode 15 common high bytes as a 4-bit dictionary index and reserve nibble
 * 15 as an escape. The low byte is stored verbatim.
 *
 * Per N words, excluding a tiny header:
 *   N low bytes + ceil(N/2) code bytes + escapes
 * so at 0.1% escapes the ratio is ~0.7505 of raw BF16.
 *
 * This benchmark measures DECODE because that is the gating property for K3: a codec
 * which saves SSD bytes but cannot reconstruct them faster than the SSD is useless.
 * It also memcmp-gates byte-exact roundtrip before printing throughput.
 *
 * Build: cc -O3 -mavx2 -o bench_bf16_dict_codec benchmarks/bench_bf16_dict_codec.c
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

static uint32_t rngs = 0x6d2b79f5u;
static uint32_t rng32(void)
{
    uint32_t x = rngs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rngs = x;
    return x;
}

/* Decode the fixed-layout payload. dict[15] are literal high bytes. `esc` contains
 * literal high bytes for code 15, in code-stream order. Returns escapes consumed. */
static size_t decode_dict15(unsigned char *dst, size_t n,
                            const unsigned char *low,
                            const unsigned char *code,
                            const unsigned char *esc,
                            const unsigned char dict[15])
{
    unsigned char d16[16] = {0};
    memcpy(d16, dict, 15);
#if defined(__AVX2__)
    const __m128i vd = _mm_loadu_si128((const __m128i *)d16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    size_t i = 0;
    for (; i + 31 < n; i += 32) {
        const __m128i c = _mm_loadu_si128((const __m128i *)(code + (i >> 1)));
        const __m128i loq = _mm_and_si128(c, mask);
        const __m128i hiq = _mm_and_si128(_mm_srli_epi16(c, 4), mask);
        const __m128i q0 = _mm_unpacklo_epi8(loq, hiq);  /* codes i..i+15 */
        const __m128i q1 = _mm_unpackhi_epi8(loq, hiq);  /* codes i+16..i+31 */
        const __m128i h0 = _mm_shuffle_epi8(vd, q0);
        const __m128i h1 = _mm_shuffle_epi8(vd, q1);
        const __m128i l0 = _mm_loadu_si128((const __m128i *)(low + i));
        const __m128i l1 = _mm_loadu_si128((const __m128i *)(low + i + 16));
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm_unpacklo_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 16), _mm_unpackhi_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 32), _mm_unpacklo_epi8(l1, h1));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 48), _mm_unpackhi_epi8(l1, h1));
    }
    for (; i < n; i++) {
        const unsigned char b = code[i >> 1];
        const unsigned char q = (i & 1) ? (b >> 4) : (b & 15);
        dst[2 * i] = low[i];
        dst[2 * i + 1] = q < 15 ? dict[q] : 0;
    }
#else
    for (size_t i = 0; i < n; i++) {
        const unsigned char b = code[i >> 1];
        const unsigned char q = (i & 1) ? (b >> 4) : (b & 15);
        dst[2 * i] = low[i];
        dst[2 * i + 1] = q < 15 ? dict[q] : 0;
    }
#endif

    /* Escapes are deliberately rare. A second linear scan is simpler than introducing
     * variable-length state into the SIMD loop and costs only half a byte read per word. */
    size_t ne = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char b = code[i >> 1];
        const unsigned char q = (i & 1) ? (b >> 4) : (b & 15);
        if (q == 15) dst[2 * i + 1] = esc[ne++];
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
    const size_t code_bytes = (n + 1) / 2;

    /* Representative K3-ish BF16 high bytes; exact values do not affect decoder speed,
     * only dictionary-hit/escape rate does. */
    const unsigned char dict[15] = {
        0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e,0xbe,0x3a,0xba,0x39,0xb9,0x3f,0xbf,0x00
    };

    unsigned char *raw = (unsigned char *)malloc(raw_bytes);
    unsigned char *out = (unsigned char *)malloc(raw_bytes);
    unsigned char *low = (unsigned char *)malloc(n);
    unsigned char *code = (unsigned char *)calloc(code_bytes, 1);
    unsigned char *esc = (unsigned char *)malloc(n / 100 + 1024);
    if (!raw || !out || !low || !code || !esc) {
        fprintf(stderr, "allocation failed for %zu MiB benchmark\n", raw_mb);
        return 2;
    }

    size_t ne = 0;
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = rng32();
        const unsigned char lb = (unsigned char)r;
        unsigned char q;
        unsigned char hb;
        /* Exactly about 0.1% escapes. */
        if ((r % 1000u) == 0u) {
            q = 15;
            hb = (unsigned char)(0x70u + ((r >> 16) & 15u));
            esc[ne++] = hb;
        } else {
            q = (unsigned char)((r >> 8) % 15u);
            hb = dict[q];
        }
        raw[2 * i] = lb;
        raw[2 * i + 1] = hb;
        low[i] = lb;
        if (i & 1) code[i >> 1] |= (unsigned char)(q << 4);
        else       code[i >> 1]  = q;
    }

    const size_t packed = n + code_bytes + ne + 15 + 32;
    const double ratio = (double)packed / (double)raw_bytes;
    printf("raw %.1f MiB  words %zu  escapes %zu (%.4f%%)  ratio %.4f  reduction %.1f%%\n",
           raw_bytes / 1048576.0, n, ne, 100.0 * ne / n, ratio, 100.0 * (1.0 - ratio));

    size_t used = decode_dict15(out, n, low, code, esc, dict);
    if (used != ne || memcmp(raw, out, raw_bytes) != 0) {
        fprintf(stderr, "ROUNDTRIP FAIL: consumed %zu/%zu escapes\n", used, ne);
        return 1;
    }
    puts("byte-identical roundtrip: PASS");

    double best = 1e99, sum = 0.0;
    uint64_t checksum = 0;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_s();
        used = decode_dict15(out, n, low, code, esc, dict);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
        sum += dt;
        checksum += out[(size_t)r * 4099u % raw_bytes] + used;
        printf("run %d: %.3f s  %.2f GB/s reconstructed  %.2f GB/s packed-input-equiv\n",
               r + 1, dt, (double)raw_bytes / 1e9 / dt,
               (double)packed / 1e9 / dt);
    }
    printf("best %.2f GB/s reconstructed; mean %.2f GB/s; checksum %llu\n",
           (double)raw_bytes / 1e9 / best,
           (double)raw_bytes / 1e9 / (sum / reps),
           (unsigned long long)checksum);

    free(raw); free(out); free(low); free(code); free(esc);
    return 0;
}
