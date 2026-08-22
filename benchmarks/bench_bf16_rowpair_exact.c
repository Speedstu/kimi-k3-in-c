#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXB 64

extern void k3_test_matmul_bf16_batch_single(float *y, int ystride,
                                              const float *const *xs, int batch,
                                              const uint16_t *W, int in, int out);
extern void k3_test_matmul_bf16_batch_rowpair(float *y, int ystride,
                                               const float *const *xs, int batch,
                                               const uint16_t *W, int in, int out);

static uint32_t rs = 0x9e3779b9u;
static uint32_t rnd32(void)
{
    uint32_t x = rs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rs = x;
}

static uint16_t bf16_from_float(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return (uint16_t)(v.u >> 16);
}

static int one(const char *name, int in, int out, int batch)
{
    const size_t nw = (size_t)in * out;
    uint16_t *w = (uint16_t *)malloc(nw * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)batch * in * sizeof(float));
    float *a = (float *)malloc((size_t)batch * out * sizeof(float));
    float *b = (float *)malloc((size_t)batch * out * sizeof(float));
    if (!w || !x || !a || !b) return 2;

    for (size_t i = 0; i < nw; i++) {
        const float f = ((int)(rnd32() & 0xffffu) - 32768) * (1.0f / 65536.0f);
        w[i] = bf16_from_float(f);
    }
    for (size_t i = 0; i < (size_t)batch * in; i++)
        x[i] = ((int)(rnd32() & 0xffffu) - 32768) * (1.0f / 32768.0f);

    const float *xp[MAXB];
    for (int t = 0; t < batch; t++) xp[t] = x + (size_t)t * in;

    k3_test_matmul_bf16_batch_single(a, out, xp, batch, w, in, out);
    k3_test_matmul_bf16_batch_rowpair(b, out, xp, batch, w, in, out);

    if (memcmp(a, b, (size_t)batch * out * sizeof(float)) != 0) {
        int bad = 0;
        for (size_t i = 0; i < (size_t)batch * out; i++)
            if (memcmp(a + i, b + i, sizeof(float)) != 0) {
                if (bad < 8)
                    fprintf(stderr, "%s mismatch[%zu] single=%.9g pair=%.9g\n",
                            name, i, a[i], b[i]);
                bad++;
            }
        fprintf(stderr, "%s PARITY FAIL: %d values\n", name, bad);
        free(w); free(x); free(a); free(b);
        return 1;
    }

    printf("%s in=%d out=%d batch=%d: PASS\n", name, in, out, batch);
    free(w); free(x); free(a); free(b);
    return 0;
}

int main(void)
{
    /* Covers production batch thresholds, tiny in, tiny out, odd out fallback and
     * released-K3 sized input widths without making CI allocate the full dense matrix. */
    if (one("large-in-small-out", 7168, 96, 32)) return 1;
    rs ^= 0x11111111u;
    if (one("small-in-large-out", 128, 1024, 64)) return 1;
    rs ^= 0x22222222u;
    if (one("odd-output", 512, 65, 32)) return 1;
    rs ^= 0x33333333u;
    if (one("latent", 3584, 512, 64)) return 1;
    puts("BF16 ROWPAIR EXACTNESS PASS");
    return 0;
}
