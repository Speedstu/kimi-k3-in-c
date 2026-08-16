#include <math.h>
#include <stdio.h>
#include <string.h>

#include "k3_sample.h"

static int fail = 0;

static void check(int ok, const char *name)
{
    if (ok) printf("  PASS  %s\n", name);
    else { printf("  FAIL  %s\n", name); fail++; }
}

int main(void)
{
    const float logits[] = {-4.0f, 1.0f, 7.0f, 2.0f};
    K3Sampler a, b;

    k3_sampler_init(&a, 0.0, 1.0, 123);
    check(k3_sample_token(&a, logits, 4) == 2, "temperature=0 is exact argmax");

    k3_sampler_init(&a, 1.0, 1.0, 424242);
    k3_sampler_init(&b, 1.0, 1.0, 424242);
    int xa[64], xb[64];
    for (int i = 0; i < 64; i++) {
        xa[i] = k3_sample_token(&a, logits, 4);
        xb[i] = k3_sample_token(&b, logits, 4);
    }
    check(memcmp(xa, xb, sizeof xa) == 0, "same seed reproduces token stream");

    /* With these logits the highest token alone exceeds p=0.10 by an enormous margin,
     * so nucleus truncation must leave exactly that token for every RNG state. */
    const float sharp[] = {-10.0f, 10.0f, -8.0f, -7.0f};
    k3_sampler_init(&a, 1.0, 0.10, 7);
    int nucleus_ok = 1;
    for (int i = 0; i < 100; i++)
        if (k3_sample_token(&a, sharp, 4) != 1) nucleus_ok = 0;
    check(nucleus_ok, "top-p nucleus truncates to dominant token");

    k3_sampler_init(&a, 1.0, 1.1, 1);
    check(k3_sample_token(&a, logits, 4) == -1, "invalid top-p is rejected");

    /* Distribution construction is a separate API because sampled speculative decode
     * needs p(y), q(y) and the residual (p-q)+ without consuming a random number. */
    double p[4];
    k3_sampler_init(&a, 1.0, 1.0, 9);
    int drc = k3_sampler_distribution(&a, logits, 4, p);
    double psum = 0.0;
    for (int i = 0; i < 4; i++) psum += p[i];
    check(drc == 0 && fabs(psum - 1.0) < 1e-12 && p[2] > p[3],
          "probability vector is normalised without consuming RNG");

    const uint64_t state_before = a.state;
    drc = k3_sampler_distribution(&a, logits, 4, p);
    check(drc == 0 && a.state == state_before,
          "distribution construction does not advance sampler state");

    const double pp[] = {0.60, 0.40};
    const double qq[] = {0.20, 0.80};
    k3_sampler_init(&a, 1.0, 1.0, 1234);
    int residual_ok = 1;
    for (int i = 0; i < 100; i++)
        if (k3_sample_residual(&a, pp, qq, 2) != 0) residual_ok = 0;
    check(residual_ok, "residual sampler uses normalised max(p-q,0)");

    /* Tiny Monte-Carlo proof of the accept/reject identity used by the engine. q is a
     * deliberately poor draft for p. The output should still converge to p. Separate
     * deterministic RNG streams model the draft proposal and target accept/residual
     * draws; no libc randomness enters the test. */
    const double target_p[] = {0.10, 0.20, 0.70};
    const double draft_q[]  = {0.55, 0.35, 0.10};
    K3Sampler draft_rng, verify_rng;
    k3_sampler_init(&draft_rng, 1.0, 1.0, 0x1111);
    k3_sampler_init(&verify_rng, 1.0, 1.0, 0x2222);
    int count[3] = {0, 0, 0};
    const int N = 200000;
    for (int r = 0; r < N; r++) {
        const int y = k3_sample_probs(&draft_rng, draft_q, 3);
        const double ratio = draft_q[y] > 0.0 ? target_p[y] / draft_q[y] : 1.0;
        int z;
        if (k3_sampler_uniform(&verify_rng) < (ratio < 1.0 ? ratio : 1.0))
            z = y;
        else
            z = k3_sample_residual(&verify_rng, target_p, draft_q, 3);
        if (z >= 0 && z < 3) count[z]++;
    }
    int mc_ok = 1;
    for (int i = 0; i < 3; i++) {
        const double observed = (double)count[i] / N;
        if (fabs(observed - target_p[i]) > 0.006) mc_ok = 0;
    }
    check(mc_ok, "speculative accept/residual Monte Carlo preserves target distribution");

    if (fail) {
        printf("SAMPLER TEST FAILED: %d checks\n", fail);
        return 1;
    }
    puts("SAMPLER TEST PASSED");
    return 0;
}
