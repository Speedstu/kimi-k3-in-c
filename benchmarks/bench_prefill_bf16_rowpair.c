#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAXB 64
#define REPS 3

extern void k3_test_matmul_bf16_batch_float(float *y, int ystride,
                                             const float *const *xs, int batch,
                                             const uint16_t *W, int in, int out);
extern void k3_test_matmul_bf16_batch_rowpair(float *y, int ystride,
                                               const float *const *xs, int batch,
                                               const uint16_t *W, int in, int out);

static uint32_t rs = 0xa53c9e17u;
static uint32_t rnd32(void)
{
    uint32_t x = rs;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rs = x;
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v)
{
    qsort(v, REPS, sizeof(*v), cmpd);
    return v[REPS / 2];
}

static uint16_t bf16_from_float(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return (uint16_t)(v.u >> 16);
}

static int run_case(const char *name, int in, int out, int batch)
{
    const size_t nw = (size_t)in * out;
    uint16_t *w = (uint16_t *)malloc(nw * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)batch * in * sizeof(float));
    float *ref = (float *)malloc((size_t)batch * out * sizeof(float));
    float *pair = (float *)malloc((size_t)batch * out * sizeof(float));
    if (!w || !x || !ref || !pair) {
        fprintf(stderr, "OOM %s\n", name);
        free(w); free(x); free(ref); free(pair);
        return 2;
    }

    for (size_t i = 0; i < nw; i++) {
        float f = ((int)(rnd32() & 0xffffu) - 32768) * (1.0f / 65536.0f);
        w[i] = bf16_from_float(f);
    }
    for (size_t i = 0; i < (size_t)batch * in; i++)
        x[i] = ((int)(rnd32() & 0xffffu) - 32768) * (1.0f / 32768.0f);

    const float *xp[MAXB];
    for (int b = 0; b < batch; b++) xp[b] = x + (size_t)b * in;

    k3_test_matmul_bf16_batch_float(ref, out, xp, batch, w, in, out);
    k3_test_matmul_bf16_batch_rowpair(pair, out, xp, batch, w, in, out);
    if (memcmp(ref, pair, (size_t)batch * out * sizeof(float)) != 0) {
        int bad = 0;
        for (size_t i = 0; i < (size_t)batch * out; i++)
            if (memcmp(ref + i, pair + i, sizeof(float)) != 0) {
                if (bad < 4)
                    fprintf(stderr, "%s mismatch[%zu] ref=%.9g pair=%.9g\n",
                            name, i, ref[i], pair[i]);
                bad++;
            }
        fprintf(stderr, "%s PARITY FAIL: %d\n", name, bad);
        free(w); free(x); free(ref); free(pair);
        return 1;
    }

    double tr[REPS], tp[REPS];
    for (int r = 0; r < REPS; r++) {
        double t;
        if (r & 1) {
            t = now_s();
            k3_test_matmul_bf16_batch_rowpair(pair, out, xp, batch, w, in, out);
            tp[r] = now_s() - t;
            t = now_s();
            k3_test_matmul_bf16_batch_float(ref, out, xp, batch, w, in, out);
            tr[r] = now_s() - t;
        } else {
            t = now_s();
            k3_test_matmul_bf16_batch_float(ref, out, xp, batch, w, in, out);
            tr[r] = now_s() - t;
            t = now_s();
            k3_test_matmul_bf16_batch_rowpair(pair, out, xp, batch, w, in, out);
            tp[r] = now_s() - t;
        }
    }

    const double mr = median(tr), mp = median(tp);
    printf("%s in=%d out=%d batch=%d single=%.3fms pair=%.3fms speedup=%.4fx\n",
           name, in, out, batch, mr * 1e3, mp * 1e3, mr / mp);

    free(w); free(x); free(ref); free(pair);
    return 0;
}

struct Shape { const char *name; int in, out; };

int main(void)
{
#ifdef _OPENMP
    printf("threads=%d\n", omp_get_max_threads());
#endif
    static const struct Shape shapes[] = {
        {"moe-down",    7168,  3584},
        {"moe-up",      3584,  7168},
        {"shared-up",   7168,  6144},
        {"mla-qb",      1536, 18432},
        {"mla-kvb",      512, 24576},
        {"kda-main",    7168, 12288},
        {"dense-down", 33792,  7168},
    };
    const int batches[] = {16, 32, 64};
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        for (size_t bi = 0; bi < sizeof(batches) / sizeof(batches[0]); bi++) {
            rs = 0xa53c9e17u ^ (uint32_t)(si * 0x9e3779b9u) ^ (uint32_t)batches[bi];
            if (run_case(shapes[si].name, shapes[si].in, shapes[si].out, batches[bi]))
                return 1;
        }
    }
    puts("PREFILL BF16 ROWPAIR EXACTNESS PASS");
    return 0;
}
