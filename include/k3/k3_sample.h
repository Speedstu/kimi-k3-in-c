#ifndef K3_SAMPLE_H
#define K3_SAMPLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t state;
    double temperature;
    double top_p;
} K3Sampler;

/* Initialise a reproducible categorical sampler. temperature <= 0 selects greedy
 * argmax. top_p is the usual nucleus cutoff in (0, 1]. */
void k3_sampler_init(K3Sampler *s, double temperature, double top_p, uint64_t seed);

/* Return one vocabulary id, or -1 on invalid input / allocation failure.
 * For temperature <= 0 this is exactly argmax and allocates nothing. */
int k3_sample_token(K3Sampler *s, const float *logits, int n);

/* Build the exact categorical distribution implied by this sampler's temperature and
 * top-p settings without consuming RNG state. `probs` must hold n doubles. Returns 0 on
 * success. Greedy mode is represented as a one-hot distribution at argmax. */
int k3_sampler_distribution(const K3Sampler *s, const float *logits, int n, double *probs);

/* Sample from an already-normalised probability vector. This consumes exactly one
 * uniform draw from `s`; entries may be zero. Returns -1 for an invalid distribution. */
int k3_sample_probs(K3Sampler *s, const double *probs, int n);

/* One reproducible U[0,1) draw. Exposed for speculative accept/reject decisions so the
 * algorithm never falls back to libc rand() or thread-dependent randomness. */
double k3_sampler_uniform(K3Sampler *s);

/* Sample from normalised max(p-q, 0). This is the residual distribution required by
 * probability-correct speculative decoding after a rejected draft token. */
int k3_sample_residual(K3Sampler *s, const double *p, const double *q, int n);

#ifdef __cplusplus
}
#endif

#endif
