#include "../../compact/q3_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_code(unsigned char *p, int i, int q)
{
    const unsigned code = (unsigned)q & 7u;
    const unsigned bit = (unsigned)(3 * i);
    const unsigned byte = bit >> 3;
    const unsigned shift = bit & 7u;
    const unsigned v = code << shift;
    p[byte] |= (unsigned char)(v & 0xffu);
    if (shift > 5u) p[byte + 1] |= (unsigned char)((v >> 8) & 0xffu);
}

static unsigned char *pack_row(const int *q, int n, const float *scales)
{
    const size_t nb = k3c_q3_row_bytes(n);
    unsigned char *row = (unsigned char *)calloc(nb, 1);
    if (!row) return NULL;
    size_t off = 0;
    int col = 0;
    int g = 0;
    while (col < n) {
        const int gn = n - col < K3C_Q3_GROUP ? n - col : K3C_Q3_GROUP;
        union { float f; uint32_t u; } s = {scales[g++]};
        const uint16_t bf = (uint16_t)(s.u >> 16);
        row[off + 0] = (unsigned char)(bf & 0xffu);
        row[off + 1] = (unsigned char)(bf >> 8);
        off += 2;
        for (int i = 0; i < gn; i++) put_code(row + off, i, q[col + i]);
        off += k3c_q3_code_bytes(gn);
        col += gn;
    }
    if (off != nb) {
        free(row);
        return NULL;
    }
    return row;
}

static int closef(float a, float b)
{
    const float d = fabsf(a - b);
    const float lim = 1e-6f + 2e-6f * fmaxf(fabsf(a), fabsf(b));
    return d <= lim;
}

int main(void)
{
    enum { IN = 131, OUT = 2 };
    if (k3c_q3_code_bytes(128) != 48u || k3c_q3_row_bytes(IN) != 54u) {
        fprintf(stderr, "Q3 physical-size contract failed\n");
        return 1;
    }

    int q[OUT][IN];
    float x[IN];
    for (int i = 0; i < IN; i++) {
        q[0][i] = (i % 7) - 3;
        q[1][i] = 3 - (i % 7);
        x[i] = (float)((i % 19) - 9) / 7.0f;
    }
    /* Exercise the otherwise-unused signed 3-bit -4 representation too. */
    q[1][17] = -4;
    q[1][130] = -4;

    const float scales[OUT][2] = {{0.5f, 0.25f}, {0.125f, 1.0f}};
    const size_t rowb = k3c_q3_row_bytes(IN);
    unsigned char *W = (unsigned char *)calloc((size_t)OUT * rowb, 1);
    if (!W) return 2;

    for (int o = 0; o < OUT; o++) {
        unsigned char *row = pack_row(q[o], IN, scales[o]);
        if (!row) { free(W); return 2; }
        memcpy(W + (size_t)o * rowb, row, rowb);
        free(row);
    }

    /* Check signed decode at positions that exercise every 3-bit alignment. */
    const unsigned char *codes0 = W + 2;
    for (int i = 0; i < 128; i++) {
        if (k3c_q3_code(codes0, i) != q[0][i]) {
            fprintf(stderr, "decode mismatch at %d: got %d expected %d\n",
                    i, k3c_q3_code(codes0, i), q[0][i]);
            free(W);
            return 1;
        }
    }

    float y[OUT] = {0};
    k3c_matmul_q3(y, x, W, IN, OUT);
    for (int o = 0; o < OUT; o++) {
        double ref = 0.0;
        for (int i = 0; i < IN; i++) {
            const float s = scales[o][i / K3C_Q3_GROUP];
            ref = fma((double)(s * (float)q[o][i]), (double)x[i], ref);
        }
        if (!closef(y[o], (float)ref)) {
            fprintf(stderr, "matmul row %d mismatch: %.9g vs %.9g\n", o, y[o], (float)ref);
            free(W);
            return 1;
        }
    }

    free(W);
    printf("K3-COMPACT Q3 C KERNEL PASS: %zu bytes/row, %.3f bits/weight at full groups\n",
           rowb, 50.0 * 8.0 / 128.0);
    return 0;
}
