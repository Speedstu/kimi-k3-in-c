/* Compare the actual typed k3_router at released K3 dimensions.
 *
 * Resident mode: upstream widens the BF16 gate once at startup and reuses FP32, so
 * FP32 router timing is the right baseline and should remain the resident path.
 *
 * Streamed mode: upstream widens all 896*7168 BF16 gate values into a scratch FP32
 * matrix EVERY layer/token, then routes from it. The fair streamed baseline is therefore
 * widen + FP32 router. The optimized path routes straight from BF16.
 *
 * Both receive numerically identical BF16-derived values. The benchmark refuses to
 * report speed unless top-k indices AND combining weights match bit-for-bit. */
#define _POSIX_C_SOURCE 200809L
#include "k3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint32_t rs = 0x31415926u;
static uint32_t rnd(void)
{
    uint32_t x = rs;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rs = x;
}

static void widen_gate(float *dst, const uint16_t *src, size_t n)
{
    for (size_t i = 0; i < n; i++) dst[i] = k3_bf16f(src[i]);
}

int main(int argc, char **argv)
{
    enum { E = 896, H = 7168, K = 16 };
    int reps = argc > 1 ? atoi(argv[1]) : 8;
    if (reps < 2) reps = 2;

    const size_t n = (size_t)E * H;
    uint16_t *wb = (uint16_t *)malloc(n * sizeof(uint16_t));
    float *wf = (float *)malloc(n * sizeof(float));
    float *x = (float *)malloc((size_t)H * sizeof(float));
    float *bias = (float *)malloc((size_t)E * sizeof(float));
    if (!wb || !wf || !x || !bias) return 2;

    for (size_t i = 0; i < n; i++) {
        const float z = ((int)(rnd() & 0xffffu) - 32768) * (1.0f / 131072.0f);
        union {
            float f;
            uint32_t u;
        } v;
        v.f = z;
        wb[i] = (uint16_t)(v.u >> 16);
    }
    widen_gate(wf, wb, n);
    for (int i = 0; i < H; i++)
        x[i] = ((int)(rnd() & 0xffffu) - 32768) * (1.0f / 32768.0f);
    for (int e = 0; e < E; e++) bias[e] = (float)(e - E / 2) * 1e-7f;

    int ia[K], ib[K];
    float wa[K], wbw[K];
    k3_router(ia, wa, x, wf, K3_WF32, bias, H, E, K, 1, 1.0f);
    k3_router(ib, wbw, x, wb, K3_WBF16, bias, H, E, K, 1, 1.0f);
    if (memcmp(ia, ib, sizeof ia) || memcmp(wa, wbw, sizeof wa)) {
        fprintf(stderr, "router parity FAIL\n");
        return 1;
    }
    puts("router parity: BIT-IDENTICAL");
#ifdef _OPENMP
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
#endif
    printf("gate storage: fp32 %.2f MB, bf16 %.2f MB\n", n * 4.0 / 1e6, n * 2.0 / 1e6);

    double resident_fp32 = 0.0, direct_bf16 = 0.0, streamed_old = 0.0, widen_only = 0.0;
    for (int r = 0; r < reps; r++) {
        double t = now_s();
        k3_router(ia, wa, x, wf, K3_WF32, bias, H, E, K, 1, 1.0f);
        resident_fp32 += now_s() - t;

        t = now_s();
        k3_router(ib, wbw, x, wb, K3_WBF16, bias, H, E, K, 1, 1.0f);
        direct_bf16 += now_s() - t;

        t = now_s();
        widen_gate(wf, wb, n);
        widen_only += now_s() - t;
        k3_router(ia, wa, x, wf, K3_WF32, bias, H, E, K, 1, 1.0f);
        streamed_old += now_s() - t;
    }

    const double rf = resident_fp32 / reps;
    const double db = direct_bf16 / reps;
    const double so = streamed_old / reps;
    const double wo = widen_only / reps;
    printf("resident: fp32 %.3f ms/router; bf16 direct %.3f ms (%.3fx vs fp32)\n",
           rf * 1000.0, db * 1000.0, rf / db);
    printf("streamed old: widen %.3f ms + route = %.3f ms/layer\n", wo * 1000.0, so * 1000.0);
    printf("streamed new: bf16 direct %.3f ms/layer; speedup %.3fx\n", db * 1000.0, so / db);
    printf("92 routed layers: old %.3f s, new %.3f s, saved %.3f s/token on this runner\n",
           92.0 * so, 92.0 * db, 92.0 * (so - db));

    free(wb);
    free(wf);
    free(x);
    free(bias);
    return 0;
}
