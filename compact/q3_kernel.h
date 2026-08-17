/* SPDX-License-Identifier: Apache-2.0
 * K3-Compact Q3 runtime primitive.
 *
 * This is deliberately isolated from the exact K3 weight tags. K3-Compact is a new
 * distilled/QAT model, so enabling Q3 here must never silently reinterpret the original
 * K3 checkpoint. Integration happens only for a checkpoint explicitly tagged with the
 * k3compact-q3-bf16scale-v1 format.
 *
 * Per 128 values:
 *   uint16 little-endian BF16 scale
 *   128 signed 3-bit two's-complement codes, little-endian bitstream (48 bytes)
 * Full group = 50 bytes = 3.125 bits/weight including scale.
 */
#ifndef K3_COMPACT_Q3_KERNEL_H
#define K3_COMPACT_Q3_KERNEL_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define K3C_Q3_GROUP 128

static inline size_t k3c_q3_code_bytes(int n)
{
    return ((size_t)3 * (size_t)n + 7u) / 8u;
}

static inline size_t k3c_q3_row_bytes(int in)
{
    if (in <= 0) return 0;
    const int full = in / K3C_Q3_GROUP;
    const int rem = in % K3C_Q3_GROUP;
    return (size_t)full * (2u + k3c_q3_code_bytes(K3C_Q3_GROUP))
         + (rem ? 2u + k3c_q3_code_bytes(rem) : 0u);
}

static inline float k3c_bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

static inline int k3c_q3_code(const unsigned char *packed, int i)
{
    const unsigned bit = (unsigned)(3 * i);
    const unsigned byte = bit >> 3;
    const unsigned shift = bit & 7u;
    uint16_t word = packed[byte];
    /* Only shifts 6/7 cross a byte boundary. A caller passes the exact packed span,
     * and for those positions the second byte necessarily exists. */
    if (shift > 5u) word |= (uint16_t)packed[byte + 1] << 8;
    const int code = (int)((word >> shift) & 7u);
    return (code & 4) ? code - 8 : code;
}

static inline void k3c_matmul_q3(float *y, const float *x, const void *W,
                                 int in, int out)
{
    const unsigned char *base = (const unsigned char *)W;
    const size_t rowb = k3c_q3_row_bytes(in);
    for (int o = 0; o < out; o++) {
        const unsigned char *p = base + (size_t)o * rowb;
        double acc = 0.0;
        int col = 0;
        while (col < in) {
            const int n = in - col < K3C_Q3_GROUP ? in - col : K3C_Q3_GROUP;
            const uint16_t sb = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
            p += 2;
            const float scale = k3c_bf16f(sb);
            const unsigned char *codes = p;
            p += k3c_q3_code_bytes(n);
            for (int i = 0; i < n; i++) {
                const float w = scale * (float)k3c_q3_code(codes, i);
                acc = fma((double)w, (double)x[col + i], acc);
            }
            col += n;
        }
        y[o] = (float)acc;
    }
}

#endif /* K3_COMPACT_Q3_KERNEL_H */
