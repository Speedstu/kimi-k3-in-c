/* Production-shape probe for reusing one widened latent across routed experts.
 * The private xd helper is exposed only under K3_TEST_INTERNALS. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                     const unsigned char *scales, int in, int rows, int group);
extern void k3_test_matmul_mxfp4_xd(float *y, const double *x,
                                    const unsigned char *packed,
                                    const unsigned char *scales,
                                    int in, int rows, int group, int rowpair);

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint32_t rng = 0x6f2a91d3u;
static uint32_t rnd(void)
{
    uint32_t x = rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng = x;
}

static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static unsigned long long hashf(const float *v, size_t n)
{
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        union { float f; uint32_t u; } b;
        b.f = v[i];
        for (int k = 0; k < 4; k++) {
            h ^= (b.u >> (8 * k)) & 255u;
            h *= 1099511628211ull;
        }
    }
    return h;
}

int main(void)
{
    enum { IN = 3584, ROWS = 3072, EXPERTS = 16, REPS = 7, GROUP = 32 };
    const size_t pcols = IN / 2u, ngrp = IN / GROUP;
    const size_t one_pk = (size_t)ROWS * pcols;
    const size_t one_sc = (size_t)ROWS * ngrp;
    const size_t np = (size_t)EXPERTS * one_pk;
    const size_t ns = (size_t)EXPERTS * one_sc;

    unsigned char *pk = (unsigned char *)malloc(np);
    unsigned char *sc = (unsigned char *)malloc(ns);
    float *x = (float *)malloc((size_t)IN * sizeof(float));
    double *xd = (double *)malloc((size_t)IN * sizeof(double));
    float *a = (float *)malloc((size_t)ROWS * sizeof(float));
    float *b = (float *)malloc((size_t)ROWS * sizeof(float));
    if (!pk || !sc || !x || !xd || !a || !b) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    for (size_t i = 0; i < np; i++) pk[i] = (unsigned char)rnd();
    for (size_t i = 0; i < ns; i++) {
        sc[i] = (unsigned char)(120u + rnd() % 15u);
        if ((rnd() & 4095u) == 0u) sc[i] = 255u;
    }
    for (int i = 0; i < IN; i++)
        x[i] = ((int)(rnd() & 0xffffu) - 32768) * (1.0f / 65536.0f);
    for (int i = 0; i < IN; i++) xd[i] = (double)x[i];

    /* Every expert must agree bit-for-bit before timing. */
    unsigned long long exact_hash = 0;
    for (int e = 0; e < EXPERTS; e++) {
        const unsigned char *pe = pk + (size_t)e * one_pk;
        const unsigned char *se = sc + (size_t)e * one_sc;
        k3_matmul_mxfp4(a, x, pe, se, IN, ROWS, GROUP);
        k3_test_matmul_mxfp4_xd(b, xd, pe, se, IN, ROWS, GROUP, 0);
        if (memcmp(a, b, (size_t)ROWS * sizeof(float)) != 0) {
            fprintf(stderr, "PARITY FAIL expert=%d\n", e);
            return 1;
        }
        exact_hash ^= hashf(a, ROWS);
    }

    double old_t[REPS], xd_t[REPS];
    volatile unsigned long long sink = exact_hash;
    for (int r = 0; r < REPS; r++) {
        /* Alternate ordering so the same implementation is not always second. */
        const int xd_first = r & 1;
        for (int phase = 0; phase < 2; phase++) {
            const int do_xd = (phase == 0) ? xd_first : !xd_first;
            const double t0 = now_s();
            if (do_xd) {
                /* Production cost: widen the shared latent ONCE per token/chunk slot. */
                for (int i = 0; i < IN; i++) xd[i] = (double)x[i];
                for (int e = 0; e < EXPERTS; e++) {
                    const unsigned char *pe = pk + (size_t)e * one_pk;
                    const unsigned char *se = sc + (size_t)e * one_sc;
                    k3_test_matmul_mxfp4_xd(b, xd, pe, se, IN, ROWS, GROUP, 0);
                    sink ^= hashf(b, ROWS);
                }
                xd_t[r] = now_s() - t0;
            } else {
                for (int e = 0; e < EXPERTS; e++) {
                    const unsigned char *pe = pk + (size_t)e * one_pk;
                    const unsigned char *se = sc + (size_t)e * one_sc;
                    k3_matmul_mxfp4(a, x, pe, se, IN, ROWS, GROUP);
                    sink ^= hashf(a, ROWS);
                }
                old_t[r] = now_s() - t0;
            }
        }
    }

    qsort(old_t, REPS, sizeof(double), cmpd);
    qsort(xd_t, REPS, sizeof(double), cmpd);
    const double old_med = old_t[REPS / 2];
    const double xd_med = xd_t[REPS / 2];
    printf("shape=%dx%d experts=%d bytes=%.1fMiB hash=%016llx sink=%016llx\n",
           IN, ROWS, EXPERTS, (double)(np + ns) / 1048576.0,
           exact_hash, (unsigned long long)sink);
    printf("median old=%.3fms xdouble=%.3fms speedup=%.4fx\n",
           old_med * 1e3, xd_med * 1e3, old_med / xd_med);
    puts("PREFILL XDOUBLE EXACTNESS PASS");

    free(pk); free(sc); free(x); free(xd); free(a); free(b);
    return 0;
}
