/* k3_codec.h - tiny lossless byte codec used only by compressed trunk storage.
 *
 * BF16 is little-endian [mantissa-low, sign/exponent-high]. K3 trunk high bytes are
 * extremely concentrated. A block stores all low bytes verbatim, two 4-bit dictionary
 * codes per byte for the high plane, then literal high bytes for code 15 escapes.
 *
 * This is a STORAGE codec, not quantisation. The output bytes are exactly the input
 * bytes and k3_bind_layer_mem sees the original bf16/f32 run with original offsets.
 */
#ifndef K3_CODEC_H
#define K3_CODEC_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define K3_DICT15_SIZE 15
#define K3_DICT15_ESCAPE 15

#define K3_DICT7_SIZE 7
#define K3_DICT7_ESCAPE 7

static inline int k3_codec_pop_lsb(unsigned *m)
{
#if defined(__GNUC__) || defined(__clang__)
    const int bit = __builtin_ctz(*m);
#else
    int bit = 0;
    unsigned v = *m;
    while (!(v & 1u)) { v >>= 1; bit++; }
#endif
    *m &= *m - 1u;
    return bit;
}

/* Decode one independently encoded block.
 *
 * src layout for raw_nbytes = 2*N:
 *   low[N] | codes[ceil(N/2)] | escape_literals[...]
 *
 * encoded_nbytes excludes any 4096-byte O_DIRECT padding. Returns the number of escape
 * bytes consumed, or SIZE_MAX for malformed input. */
static inline size_t k3_dict15_decode(unsigned char *dst, size_t raw_nbytes,
                                      const unsigned char *src, size_t encoded_nbytes,
                                      const unsigned char dict[K3_DICT15_SIZE])
{
    if (!dst || !src || !dict || (raw_nbytes & 1u)) return SIZE_MAX;
    const size_t n = raw_nbytes / 2u;
    const size_t cb = (n + 1u) / 2u;
    if (encoded_nbytes < n + cb) return SIZE_MAX;
    const unsigned char *low = src;
    const unsigned char *code = src + n;
    const unsigned char *esc = code + cb;
    const size_t esc_cap = encoded_nbytes - n - cb;
    size_t ne = 0, i = 0;

#if defined(__AVX2__)
    unsigned char d16[16] = {0};
    memcpy(d16, dict, K3_DICT15_SIZE);
    const __m128i vd = _mm_loadu_si128((const __m128i *)d16);
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i vesc = _mm_set1_epi8(K3_DICT15_ESCAPE);
    for (; i + 31u < n; i += 32u) {
        const __m128i c = _mm_loadu_si128((const __m128i *)(code + (i >> 1)));
        const __m128i loq = _mm_and_si128(c, mask);
        const __m128i hiq = _mm_and_si128(_mm_srli_epi16(c, 4), mask);
        const __m128i q0 = _mm_unpacklo_epi8(loq, hiq);
        const __m128i q1 = _mm_unpackhi_epi8(loq, hiq);
        const __m128i h0 = _mm_shuffle_epi8(vd, q0);
        const __m128i h1 = _mm_shuffle_epi8(vd, q1);
        const __m128i l0 = _mm_loadu_si128((const __m128i *)(low + i));
        const __m128i l1 = _mm_loadu_si128((const __m128i *)(low + i + 16));
        _mm_storeu_si128((__m128i *)(dst + 2u * i),
                         _mm_unpacklo_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(dst + 2u * i + 16),
                         _mm_unpackhi_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(dst + 2u * i + 32),
                         _mm_unpacklo_epi8(l1, h1));
        _mm_storeu_si128((__m128i *)(dst + 2u * i + 48),
                         _mm_unpackhi_epi8(l1, h1));

        unsigned m0 = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(q0, vesc));
        unsigned m1 = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(q1, vesc));
        if ((size_t)__builtin_popcount(m0) + (size_t)__builtin_popcount(m1)
            > esc_cap - ne)
            return SIZE_MAX;
        while (m0) {
            const int p = k3_codec_pop_lsb(&m0);
            dst[2u * (i + (size_t)p) + 1u] = esc[ne++];
        }
        while (m1) {
            const int p = k3_codec_pop_lsb(&m1);
            dst[2u * (i + 16u + (size_t)p) + 1u] = esc[ne++];
        }
    }
#endif

    for (; i < n; i++) {
        const unsigned char b = code[i >> 1];
        const unsigned char q = (i & 1u) ? (b >> 4) : (b & 15u);
        dst[2u * i] = low[i];
        if (q < K3_DICT15_SIZE) {
            dst[2u * i + 1u] = dict[q];
        } else {
            if (ne >= esc_cap) return SIZE_MAX;
            dst[2u * i + 1u] = esc[ne++];
        }
    }
    return ne;
}


/* Fixed-3-bit companion to dict15.
 *
 * src layout for raw_nbytes = 2*N:
 *   low[N] | codes[ceil(3*N/8)] | escape_literals[...]
 *
 * Codes 0..6 index the block dictionary and code 7 is a literal high-byte escape.
 * Eight codes occupy exactly three bytes.  The 4096-entry 4-code table is rebuilt once
 * per block from its seven-byte dictionary, then the hot loop expands eight high bytes
 * from two table lookups.  On AVX2 builds the eight low/high bytes are interleaved with
 * one SSE-width unpack; the model still receives the original BF16 byte stream.
 */
static inline size_t k3_dict7_decode(unsigned char *dst, size_t raw_nbytes,
                                     const unsigned char *src, size_t encoded_nbytes,
                                     const unsigned char dict[K3_DICT7_SIZE])
{
    if (!dst || !src || !dict || (raw_nbytes & 1u)) return SIZE_MAX;
    const size_t n = raw_nbytes / 2u;
    const size_t cb = (3u * n + 7u) / 8u;
    if (encoded_nbytes < n + cb) return SIZE_MAX;
    const unsigned char *low = src;
    const unsigned char *code = src + n;
    const unsigned char *esc = code + cb;
    const size_t esc_cap = encoded_nbytes - n - cb;

    uint32_t high4[4096];
    unsigned char emask4[4096];
    for (unsigned v = 0; v < 4096u; v++) {
        uint32_t hv = 0;
        unsigned char em = 0;
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned q = (v >> (3u * j)) & 7u;
            if (q < K3_DICT7_SIZE)
                hv |= (uint32_t)dict[q] << (8u * j);
            else
                em |= (unsigned char)(1u << j);
        }
        high4[v] = hv;
        emask4[v] = em;
    }

    size_t ne = 0, i = 0, ci = 0;
    for (; i + 8u <= n; i += 8u, ci += 3u) {
        const uint32_t bits = (uint32_t)code[ci]
                            | ((uint32_t)code[ci + 1u] << 8)
                            | ((uint32_t)code[ci + 2u] << 16);
        const unsigned a = bits & 0xfffu;
        const unsigned b = (bits >> 12) & 0xfffu;
        const uint64_t hp = (uint64_t)high4[a] | ((uint64_t)high4[b] << 32);
#if defined(__AVX2__)
        const __m128i vl = _mm_loadl_epi64((const __m128i *)(low + i));
        const __m128i vh = _mm_cvtsi64_si128((long long)hp);
        _mm_storeu_si128((__m128i *)(dst + 2u * i), _mm_unpacklo_epi8(vl, vh));
#else
        for (unsigned j = 0; j < 8u; j++) {
            dst[2u * (i + j)] = low[i + j];
            dst[2u * (i + j) + 1u] = (unsigned char)(hp >> (8u * j));
        }
#endif
        unsigned m = (unsigned)emask4[a] | ((unsigned)emask4[b] << 4);
#if defined(__GNUC__) || defined(__clang__)
        if ((size_t)__builtin_popcount(m) > esc_cap - ne) return SIZE_MAX;
#else
        unsigned mc = m, pc = 0; while (mc) { pc += mc & 1u; mc >>= 1; }
        if ((size_t)pc > esc_cap - ne) return SIZE_MAX;
#endif
        while (m) {
            const int bit = k3_codec_pop_lsb(&m);
            dst[2u * (i + (size_t)bit) + 1u] = esc[ne++];
        }
    }

    /* Generic tail. Packed trunks are 4096-byte aligned, hence N is divisible by 8;
     * this path exists for unit tests and defensive format handling. */
    for (; i < n; i++) {
        const size_t bit = 3u * i;
        const size_t bo = bit >> 3;
        const unsigned sh = (unsigned)(bit & 7u);
        uint32_t w = code[bo];
        if (bo + 1u < cb) w |= (uint32_t)code[bo + 1u] << 8;
        if (bo + 2u < cb) w |= (uint32_t)code[bo + 2u] << 16;
        const unsigned q = (w >> sh) & 7u;
        dst[2u * i] = low[i];
        if (q < K3_DICT7_SIZE) dst[2u * i + 1u] = dict[q];
        else {
            if (ne >= esc_cap) return SIZE_MAX;
            dst[2u * i + 1u] = esc[ne++];
        }
    }
    return ne;
}

#endif /* K3_CODEC_H */
