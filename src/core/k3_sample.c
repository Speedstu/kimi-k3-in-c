#include "k3_sample.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* splitmix64 is used only to initialise the stream. The generated sequence is local to
 * the sampler and therefore independent of libc rand(), thread count and OpenMP state. */
static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t rng64(K3Sampler *s)
{
    /* xorshift64*: compact, deterministic and more than adequate for categorical
     * sampling. A zero state is forbidden by init(). */
    uint64_t x = s->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s->state = x;
    return x * UINT64_C(2685821657736338717);
}

double k3_sampler_uniform(K3Sampler *s)
{
    if (!s) return NAN;
    /* 53 random bits, exactly representable in a double mantissa. The interval is
     * [0,1), so a cumulative scan cannot step past the final bucket. */
    return (double)(rng64(s) >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    int id;
    double weight;
} K3Candidate;

static int candidate_desc(const void *ap, const void *bp)
{
    const K3Candidate *a = (const K3Candidate *)ap;
    const K3Candidate *b = (const K3Candidate *)bp;
    if (a->weight > b->weight) return -1;
    if (a->weight < b->weight) return 1;
    /* Deterministic tie break. */
    return (a->id > b->id) - (a->id < b->id);
}

static int argmax_(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static int sampler_valid(const K3Sampler *s)
{
    if (!s) return 0;
    if (!isfinite(s->temperature) || s->temperature < 0.0) return 0;
    if (!isfinite(s->top_p) || !(s->top_p > 0.0 && s->top_p <= 1.0)) return 0;
    return 1;
}

void k3_sampler_init(K3Sampler *s, double temperature, double top_p, uint64_t seed)
{
    if (!s) return;
    s->temperature = temperature;
    s->top_p = top_p;
    uint64_t x = seed ? seed : UINT64_C(0x4b334d41584c4f43); /* "K3MAXLOC" */
    s->state = splitmix64(&x);
    if (s->state == 0) s->state = UINT64_C(0x2545f4914f6cdd1d);
}

int k3_sampler_distribution(const K3Sampler *s, const float *logits, int n, double *probs)
{
    if (!sampler_valid(s) || !logits || !probs || n <= 0) return -1;
    memset(probs, 0, (size_t)n * sizeof(*probs));

    if (s->temperature <= 0.0) {
        probs[argmax_(logits, n)] = 1.0;
        return 0;
    }

    double max_scaled = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (!isfinite((double)logits[i])) return -1;
        const double x = (double)logits[i] / s->temperature;
        if (x > max_scaled) max_scaled = x;
    }

    if (s->top_p >= 1.0) {
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            probs[i] = exp((double)logits[i] / s->temperature - max_scaled);
            total += probs[i];
        }
        if (!(total > 0.0) || !isfinite(total)) return -1;
        const double inv = 1.0 / total;
        for (int i = 0; i < n; i++) probs[i] *= inv;
        return 0;
    }

    K3Candidate *c = (K3Candidate *)malloc((size_t)n * sizeof(*c));
    if (!c) return -1;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        c[i].id = i;
        c[i].weight = exp((double)logits[i] / s->temperature - max_scaled);
        total += c[i].weight;
    }
    if (!(total > 0.0) || !isfinite(total)) {
        free(c);
        return -1;
    }

    qsort(c, (size_t)n, sizeof(*c), candidate_desc);
    const double cutoff = s->top_p * total;
    double kept = 0.0;
    int nkeep = 0;
    do {
        kept += c[nkeep].weight;
        nkeep++;
    } while (nkeep < n && kept < cutoff);
    if (!(kept > 0.0) || !isfinite(kept)) {
        free(c);
        return -1;
    }
    const double inv = 1.0 / kept;
    for (int i = 0; i < nkeep; i++) probs[c[i].id] = c[i].weight * inv;
    free(c);
    return 0;
}

int k3_sample_probs(K3Sampler *s, const double *probs, int n)
{
    if (!s || !probs || n <= 0) return -1;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(probs[i]) || probs[i] < 0.0) return -1;
        total += probs[i];
    }
    if (!(total > 0.0) || !isfinite(total)) return -1;

    const double target = k3_sampler_uniform(s) * total;
    double acc = 0.0;
    int last = -1;
    for (int i = 0; i < n; i++) {
        if (probs[i] > 0.0) last = i;
        acc += probs[i];
        if (target < acc) return i;
    }
    /* Rounding can leave target infinitesimally above the cumulative sum. */
    return last;
}

int k3_sample_residual(K3Sampler *s, const double *p, const double *q, int n)
{
    if (!s || !p || !q || n <= 0) return -1;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(p[i]) || !isfinite(q[i]) || p[i] < 0.0 || q[i] < 0.0) return -1;
        const double r = p[i] > q[i] ? p[i] - q[i] : 0.0;
        total += r;
    }
    if (!(total > 0.0) || !isfinite(total)) return -1;
    const double target = k3_sampler_uniform(s) * total;
    double acc = 0.0;
    int last = -1;
    for (int i = 0; i < n; i++) {
        const double r = p[i] > q[i] ? p[i] - q[i] : 0.0;
        if (r > 0.0) last = i;
        acc += r;
        if (target < acc) return i;
    }
    return last;
}

int k3_sample_token(K3Sampler *s, const float *logits, int n)
{
    if (!sampler_valid(s) || !logits || n <= 0) return -1;
    if (s->temperature <= 0.0) return argmax_(logits, n);

    /* Agentic K3 benchmark parity uses top_p=1.0. Keep that important serial path O(V)
     * with no probability array and no 160k-element sort. */
    if (s->top_p >= 1.0) {
        double max_scaled = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (!isfinite((double)logits[i])) return -1;
            const double x = (double)logits[i] / s->temperature;
            if (x > max_scaled) max_scaled = x;
        }
        double total = 0.0;
        for (int i = 0; i < n; i++)
            total += exp((double)logits[i] / s->temperature - max_scaled);
        if (!(total > 0.0) || !isfinite(total)) return -1;
        const double target = k3_sampler_uniform(s) * total;
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += exp((double)logits[i] / s->temperature - max_scaled);
            if (target < acc) return i;
        }
        return n - 1;
    }

    double *probs = (double *)malloc((size_t)n * sizeof(*probs));
    if (!probs) return -1;
    const int rc = k3_sampler_distribution(s, logits, n, probs);
    const int answer = rc == 0 ? k3_sample_probs(s, probs, n) : -1;
    free(probs);
    return answer;
}
