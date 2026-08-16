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

#ifdef __cplusplus
}
#endif

#endif
