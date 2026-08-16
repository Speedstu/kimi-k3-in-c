/* Compare the actual typed k3_router at released K3 dimensions.
 * Both paths receive numerically identical BF16-derived values. The benchmark refuses
 * to report speed unless top-k indices AND combining weights match bit-for-bit. */
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
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rs = x;
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
        union { float f; uint32_t u; } v;
        v.f = z;
        wb[i] = (uint16_t)(v.u >> 16);
        wf[i] = k3_bf16f(wb[i]);
    }
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

    double tf = 0.0, tb = 0.0;
    for (int r = 0; r < reps; r++) {
        double t = now_s();
        k3_router(ia, wa, x, wf, K3_WF32, bias, H, E, K, 1, 1.0f);
        tf += now_s() - t;
        t = now_s();
        k3_router(ib, wbw, x, wb, K3_WBF16, bias, H, E, K, 1, 1.0f);
        tb += now_s() - t;
    }
    printf("fp32 %.3f ms/router, bf16 %.3f ms/router, speedup %.3fx\n",
           1000.0 * tf / reps, 1000.0 * tb / reps, tf / tb);
    printf("92 routed layers extrapolated router wall: fp32 %.3f s, bf16 %.3f s\n",
           92.0 * tf / reps, 92.0 * tb / reps);

    free(wb); free(wf); free(x); free(bias);
    return 0;
}
