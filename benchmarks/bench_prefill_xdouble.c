#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "k3/k3.h"

/* Test-only wrapper emitted by k3_ops.c under K3_TEST_INTERNALS. */
void k3_test_matmul_mxfp4_xd(float *y, const double *x,
                             const unsigned char *packed,
                             const unsigned char *scales,
                             int in, int rows, int group, int rowpair);

static uint32_t rng_state = 0x12345678u;
static uint32_t xrnd(void)
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

static double now_s(void)
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

static int dcmp(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *a, int n)
{
    qsort(a, (size_t)n, sizeof(*a), dcmp);
    return (n & 1) ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

int main(void)
{
    enum { IN = 3584, ROWS = 3072, GROUP = 32, TOKENS = 8, FANOUT = 8, REPS = 7 };
    const size_t pbytes = (size_t)ROWS * (IN / 2);
    const size_t sbytes = (size_t)ROWS * (IN / GROUP);
    float *x = (float *)malloc((size_t)TOKENS * IN * sizeof(float));
    double *xd = (double *)malloc((size_t)TOKENS * IN * sizeof(double));
    unsigned char *packed = (unsigned char *)malloc(pbytes);
    unsigned char *scales = (unsigned char *)malloc(sbytes);
    float *a = (float *)malloc((size_t)ROWS * sizeof(float));
    float *b = (float *)malloc((size_t)ROWS * sizeof(float));
    if (!x || !xd || !packed || !scales || !a || !b) {
        fputs("allocation failed\n", stderr);
        return 2;
    }

    for (size_t i = 0; i < (size_t)TOKENS * IN; i++) {
        const int v = (int)(xrnd() % 20001u) - 10000;
        x[i] = (float)v / 8192.0f;
        xd[i] = (double)x[i];
    }
    for (size_t i = 0; i < pbytes; i++) packed[i] = (unsigned char)xrnd();
    for (size_t i = 0; i < sbytes; i++) {
        /* Keep exponents in a sane finite range; inject sparse 255 to cover skip semantics. */
        scales[i] = ((i % 997u) == 0u) ? 255u : (unsigned char)(120u + (xrnd() % 15u));
    }

    /* Exactness: every token must match public float-input kernel vs pre-widened helper. */
    for (int t = 0; t < TOKENS; t++) {
        k3_matmul_mxfp4(a, x + (size_t)t * IN, packed, scales, IN, ROWS, GROUP);
        k3_test_matmul_mxfp4_xd(b, xd + (size_t)t * IN, packed, scales,
                                IN, ROWS, GROUP, 0);
        if (memcmp(a, b, (size_t)ROWS * sizeof(float)) != 0) {
            fprintf(stderr, "PARITY FAIL token=%d\n", t);
            return 1;
        }
    }
    puts("PARITY PASS");

    double base[REPS], wide[REPS];
    volatile float sink = 0.0f;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_s();
        for (int t = 0; t < TOKENS; t++) {
            const float *xt = x + (size_t)t * IN;
            for (int e = 0; e < FANOUT; e++) {
                k3_matmul_mxfp4(a, xt, packed, scales, IN, ROWS, GROUP);
                sink += a[(t + e) % ROWS];
            }
        }
        base[r] = now_s() - t0;

        t0 = now_s();
        /* Production prefill proposal: widen each token ONCE, then reuse across all
         * experts that selected that token. Include this one-time widening in timing. */
        for (int t = 0; t < TOKENS; t++) {
            const float *xt = x + (size_t)t * IN;
            double *xdt = xd + (size_t)t * IN;
            for (int i = 0; i < IN; i++) xdt[i] = (double)xt[i];
        }
        for (int t = 0; t < TOKENS; t++) {
            const double *xdt = xd + (size_t)t * IN;
            for (int e = 0; e < FANOUT; e++) {
                k3_test_matmul_mxfp4_xd(b, xdt, packed, scales, IN, ROWS, GROUP, 0);
                sink += b[(t + e) % ROWS];
            }
        }
        wide[r] = now_s() - t0;
    }

    const double mb = median(base, REPS), mw = median(wide, REPS);
#ifdef _OPENMP
    printf("threads=%d ", omp_get_max_threads());
#endif
    printf("tokens=%d fanout=%d base=%.3fms xdouble=%.3fms speedup=%.3fx sink=%g\n",
           TOKENS, FANOUT, 1000.0 * mb, 1000.0 * mw, mb / mw, (double)sink);

    free(x); free(xd); free(packed); free(scales); free(a); free(b);
    return 0;
}
