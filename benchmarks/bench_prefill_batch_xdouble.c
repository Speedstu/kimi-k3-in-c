#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAXB 8
#define REPS 7

extern void k3_test_matmul_mxfp4_batch_float(float *y, int ystride,
                                              const float *const *xs, int batch,
                                              const unsigned char *packed,
                                              const unsigned char *scales,
                                              int in, int rows, int group);
extern void k3_test_matmul_mxfp4_batch_xd(float *y, int ystride,
                                           const double *const *xs, int batch,
                                           const unsigned char *packed,
                                           const unsigned char *scales,
                                           int in, int rows, int group);

static uint32_t rng_state = 0x31415926u;
static uint32_t rnd32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v)
{
    qsort(v, REPS, sizeof(*v), cmpd);
    return v[REPS / 2];
}

static void fill_weights(unsigned char *pk, unsigned char *sc,
                         size_t np, size_t ns)
{
    for (size_t i = 0; i < np; i++) pk[i] = (unsigned char)rnd32();
    for (size_t i = 0; i < ns; i++) {
        sc[i] = (unsigned char)(119u + rnd32() % 17u);
        if ((rnd32() & 4095u) == 0u) sc[i] = 255;
    }
}

static void fill_x(float *x, int batch, int in)
{
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < in; i++)
            x[(size_t)b * in + i] =
                ((int)(rnd32() & 0xffffu) - 32768) * (1.0f / 32768.0f);
}

static void widen(const float *x, double *xd, int batch, int in)
{
    for (int b = 0; b < batch; b++)
        for (int i = 0; i < in; i++)
            xd[(size_t)b * in + i] = (double)x[(size_t)b * in + i];
}

static void ptrs_f(const float **p, const float *x, int batch, int stride)
{
    for (int b = 0; b < batch; b++) p[b] = x + (size_t)b * stride;
}

static void ptrs_d(const double **p, const double *x, int batch, int stride)
{
    for (int b = 0; b < batch; b++) p[b] = x + (size_t)b * stride;
}

static int parity(const float *a, const float *b, size_t n, const char *name)
{
    if (memcmp(a, b, n * sizeof(float)) == 0) return 0;
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            if (bad < 4) fprintf(stderr, "%s mismatch[%zu] %.9g %.9g\n", name, i, a[i], b[i]);
            bad++;
        }
    fprintf(stderr, "%s parity FAIL: %zu/%zu\n", name, bad, n);
    return 1;
}

static int bench_pair(int batch)
{
    const int in = 3584, rows = 3072, group = 32;
    const int pcols = in / 2, ngrp = in / group;
    const size_t np = (size_t)rows * pcols, ns = (size_t)rows * ngrp;
    unsigned char *p1 = (unsigned char *)malloc(np), *s1 = (unsigned char *)malloc(ns);
    unsigned char *p3 = (unsigned char *)malloc(np), *s3 = (unsigned char *)malloc(ns);
    float *x = (float *)malloc((size_t)batch * in * sizeof(float));
    double *xd = (double *)malloc((size_t)batch * in * sizeof(double));
    float *fa = (float *)malloc((size_t)batch * rows * sizeof(float));
    float *fb = (float *)malloc((size_t)batch * rows * sizeof(float));
    float *da = (float *)malloc((size_t)batch * rows * sizeof(float));
    float *db = (float *)malloc((size_t)batch * rows * sizeof(float));
    if (!p1 || !s1 || !p3 || !s3 || !x || !xd || !fa || !fb || !da || !db) return 2;

    fill_weights(p1, s1, np, ns); fill_weights(p3, s3, np, ns); fill_x(x, batch, in);
    const float *xf[MAXB]; const double *dx[MAXB];
    ptrs_f(xf, x, batch, in); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);

    k3_test_matmul_mxfp4_batch_float(fa, rows, xf, batch, p1, s1, in, rows, group);
    k3_test_matmul_mxfp4_batch_float(fb, rows, xf, batch, p3, s3, in, rows, group);
    k3_test_matmul_mxfp4_batch_xd(da, rows, dx, batch, p1, s1, in, rows, group);
    k3_test_matmul_mxfp4_batch_xd(db, rows, dx, batch, p3, s3, in, rows, group);
    if (parity(fa, da, (size_t)batch * rows, "w1") ||
        parity(fb, db, (size_t)batch * rows, "w3")) return 1;

    double tf[REPS], td[REPS];
    for (int r = 0; r < REPS; r++) {
        double t;
        if (r & 1) {
            t = now_s(); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);
            k3_test_matmul_mxfp4_batch_xd(da, rows, dx, batch, p1, s1, in, rows, group);
            k3_test_matmul_mxfp4_batch_xd(db, rows, dx, batch, p3, s3, in, rows, group);
            td[r] = now_s() - t;
            t = now_s();
            k3_test_matmul_mxfp4_batch_float(fa, rows, xf, batch, p1, s1, in, rows, group);
            k3_test_matmul_mxfp4_batch_float(fb, rows, xf, batch, p3, s3, in, rows, group);
            tf[r] = now_s() - t;
        } else {
            t = now_s();
            k3_test_matmul_mxfp4_batch_float(fa, rows, xf, batch, p1, s1, in, rows, group);
            k3_test_matmul_mxfp4_batch_float(fb, rows, xf, batch, p3, s3, in, rows, group);
            tf[r] = now_s() - t;
            t = now_s(); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);
            k3_test_matmul_mxfp4_batch_xd(da, rows, dx, batch, p1, s1, in, rows, group);
            k3_test_matmul_mxfp4_batch_xd(db, rows, dx, batch, p3, s3, in, rows, group);
            td[r] = now_s() - t;
        }
    }
    const double mf = median(tf), md = median(td);
    printf("w1+w3 batch=%d float=%.3fms xdouble=%.3fms speedup=%.4fx\n",
           batch, mf * 1e3, md * 1e3, mf / md);

    free(p1); free(s1); free(p3); free(s3); free(x); free(xd);
    free(fa); free(fb); free(da); free(db);
    return 0;
}

static int bench_w2(int batch)
{
    const int in = 3072, rows = 3584, group = 32;
    const int pcols = in / 2, ngrp = in / group;
    const size_t np = (size_t)rows * pcols, ns = (size_t)rows * ngrp;
    unsigned char *pk = (unsigned char *)malloc(np), *sc = (unsigned char *)malloc(ns);
    float *x = (float *)malloc((size_t)batch * in * sizeof(float));
    double *xd = (double *)malloc((size_t)batch * in * sizeof(double));
    float *yf = (float *)malloc((size_t)batch * rows * sizeof(float));
    float *yd = (float *)malloc((size_t)batch * rows * sizeof(float));
    if (!pk || !sc || !x || !xd || !yf || !yd) return 2;

    fill_weights(pk, sc, np, ns); fill_x(x, batch, in);
    const float *xf[MAXB]; const double *dx[MAXB];
    ptrs_f(xf, x, batch, in); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);
    k3_test_matmul_mxfp4_batch_float(yf, rows, xf, batch, pk, sc, in, rows, group);
    k3_test_matmul_mxfp4_batch_xd(yd, rows, dx, batch, pk, sc, in, rows, group);
    if (parity(yf, yd, (size_t)batch * rows, "w2")) return 1;

    double tf[REPS], td[REPS];
    for (int r = 0; r < REPS; r++) {
        double t;
        if (r & 1) {
            t = now_s(); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);
            k3_test_matmul_mxfp4_batch_xd(yd, rows, dx, batch, pk, sc, in, rows, group);
            td[r] = now_s() - t;
            t = now_s(); k3_test_matmul_mxfp4_batch_float(yf, rows, xf, batch, pk, sc, in, rows, group);
            tf[r] = now_s() - t;
        } else {
            t = now_s(); k3_test_matmul_mxfp4_batch_float(yf, rows, xf, batch, pk, sc, in, rows, group);
            tf[r] = now_s() - t;
            t = now_s(); widen(x, xd, batch, in); ptrs_d(dx, xd, batch, in);
            k3_test_matmul_mxfp4_batch_xd(yd, rows, dx, batch, pk, sc, in, rows, group);
            td[r] = now_s() - t;
        }
    }
    const double mf = median(tf), md = median(td);
    printf("w2 batch=%d float=%.3fms xdouble=%.3fms speedup=%.4fx\n",
           batch, mf * 1e3, md * 1e3, mf / md);

    free(pk); free(sc); free(x); free(xd); free(yf); free(yd);
    return 0;
}

int main(void)
{
#ifdef _OPENMP
    printf("threads=%d\n", omp_get_max_threads());
#endif
    for (int b = 2; b <= 8; b *= 2) {
        rng_state = 0x31415926u + (uint32_t)b;
        if (bench_pair(b)) return 1;
        rng_state = 0x27182818u + (uint32_t)b;
        if (bench_w2(b)) return 1;
    }
    puts("PREFILL BATCH XDOUBLE EXACTNESS PASS");
    return 0;
}
