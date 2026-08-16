/* Exactness/perf gate for the PRIVATE streamed-expert xdouble row-pair helper.
 * k3_ops.c is compiled with K3_TEST_INTERNALS to expose only a test wrapper. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern void k3_test_matmul_mxfp4_xd(float *y, const double *x,
                                    const unsigned char *packed,
                                    const unsigned char *scales,
                                    int in, int rows, int group, int rowpair);

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint32_t rs = 0x8badf00du;
static uint32_t rnd(void)
{
    uint32_t x = rs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rs = x;
}

static unsigned long long hashf(const float *v, int n)
{
    unsigned long long h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } b;
        b.f = v[i];
        for (int k = 0; k < 4; k++) {
            h ^= (b.u >> (8 * k)) & 255u;
            h *= 1099511628211ull;
        }
    }
    return h;
}

static int one(const char *name, int in, int rows, int reps, int dense_255)
{
    const int pcols = in / 2, ngrp = in / 32;
    const size_t np = (size_t)rows * pcols;
    const size_t ns = (size_t)rows * ngrp;
    unsigned char *pk = (unsigned char *)malloc(np);
    unsigned char *sc = (unsigned char *)malloc(ns);
    double *x = (double *)malloc((size_t)in * sizeof(double));
    float *a = (float *)malloc((size_t)rows * sizeof(float));
    float *b = (float *)malloc((size_t)rows * sizeof(float));
    if (!pk || !sc || !x || !a || !b) return 2;

    for (size_t i = 0; i < np; i++) pk[i] = (unsigned char)rnd();
    for (size_t i = 0; i < ns; i++) {
        sc[i] = (unsigned char)(120 + rnd() % 15);
        if (dense_255 ? ((i % 17u) == 0u) : ((rnd() & 2047u) == 0u)) sc[i] = 255;
    }
    for (int i = 0; i < in; i++) {
        const float f = ((int)(rnd() & 0xffffu) - 32768) * (1.0f / 65536.0f);
        x[i] = (double)f;              /* exactly the production float->double widening */
    }

    k3_test_matmul_mxfp4_xd(a, x, pk, sc, in, rows, 32, 0);
    k3_test_matmul_mxfp4_xd(b, x, pk, sc, in, rows, 32, 1);
    const unsigned long long ha = hashf(a, rows), hb = hashf(b, rows);
    printf("%s in=%d rows=%d single=%016llx pair=%016llx\n",
           name, in, rows, ha, hb);
    if (memcmp(a, b, (size_t)rows * sizeof(float)) != 0) {
        int bad = 0;
        for (int i = 0; i < rows; i++)
            if (memcmp(a + i, b + i, sizeof(float)) != 0) {
                if (bad < 8) printf("mismatch row %d single=%.9g pair=%.9g\n", i, a[i], b[i]);
                bad++;
            }
        printf("PARITY FAIL %s: %d rows\n", name, bad);
        return 1;
    }

    double ta = 0.0, tb = 0.0;
    for (int r = 0; r < reps; r++) {
        double t = now_s();
        k3_test_matmul_mxfp4_xd(a, x, pk, sc, in, rows, 32, 0);
        ta += now_s() - t;
        t = now_s();
        k3_test_matmul_mxfp4_xd(b, x, pk, sc, in, rows, 32, 1);
        tb += now_s() - t;
    }
    ta /= reps; tb /= reps;
    printf("%s single=%.3fms pair=%.3fms speedup=%.3fx\n",
           name, ta * 1e3, tb * 1e3, ta / tb);

    free(pk); free(sc); free(x); free(a); free(b);
    return 0;
}

int main(void)
{
#ifdef _OPENMP
    printf("threads=%d\n", omp_get_max_threads());
#endif
    if (one("w1w3", 3584, 3072, 6, 0)) return 1;
    rs = 0x13579bdfu;
    if (one("w2", 3072, 3584, 6, 0)) return 1;
    rs = 0x2468ace0u;
    if (one("odd+scale255", 64, 65, 40, 1)) return 1;
    puts("ROWPAIR EXACTNESS PASS");
    return 0;
}
