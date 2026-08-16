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

    if (fail) {
        printf("SAMPLER TEST FAILED: %d checks\n", fail);
        return 1;
    }
    puts("SAMPLER TEST PASSED");
    return 0;
}
