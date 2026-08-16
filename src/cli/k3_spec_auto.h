#ifndef K3_SPEC_AUTO_H
#define K3_SPEC_AUTO_H

/* Cost-aware controller for greedy speculative decode.
 *
 * Exactness never depends on this code: it only chooses how many draft tokens are
 * proposed before the exact K3 model verifies them.  The cost signal is end-to-end
 * wall time for a completed speculative round divided by the number of exact tokens
 * that round committed (accepted prefix + correction).
 *
 * Two observations are required before comparing a horizon.  A candidate that is more
 * than 5% slower than the best stable horizon is blocked for the rest of the run.  Poor
 * acceptance remains an immediate independent shrink signal, because wasted proposals
 * are expensive even before enough timing samples exist.
 */

#define K3_SPEC_AUTO_COST_MAX 8
#define K3_SPEC_AUTO_COST_MIN_SAMPLES 2
#define K3_SPEC_AUTO_COST_REGRET 1.05

typedef struct {
    double ema_spt[K3_SPEC_AUTO_COST_MAX + 1];
    unsigned samples[K3_SPEC_AUTO_COST_MAX + 1];
    unsigned blocked_mask;
    int probes;
    int backoffs;
} K3SpecAutoCost;

static inline void k3_spec_auto_cost_init(K3SpecAutoCost *s)
{
    int i;
    for (i = 0; i <= K3_SPEC_AUTO_COST_MAX; i++) {
        s->ema_spt[i] = 0.0;
        s->samples[i] = 0;
    }
    s->blocked_mask = 0u;
    s->probes = 0;
    s->backoffs = 0;
}

static inline void k3_spec_auto_cost_observe(K3SpecAutoCost *s, int horizon,
                                              double seconds, int emitted)
{
    if (!s || horizon < 1 || horizon > K3_SPEC_AUTO_COST_MAX ||
        !(seconds >= 0.0) || emitted <= 0)
        return;
    const double v = seconds / (double)emitted;
    if (s->samples[horizon] == 0)
        s->ema_spt[horizon] = v;
    else
        s->ema_spt[horizon] = 0.75 * s->ema_spt[horizon] + 0.25 * v;
    s->samples[horizon]++;
}

static inline int k3_spec_auto_cost_best(const K3SpecAutoCost *s, int limit,
                                          int stable_only)
{
    int h, best = 0;
    if (!s) return 0;
    if (limit > K3_SPEC_AUTO_COST_MAX) limit = K3_SPEC_AUTO_COST_MAX;
    for (h = 1; h <= limit; h++) {
        const unsigned need = stable_only ? K3_SPEC_AUTO_COST_MIN_SAMPLES : 1u;
        if (s->samples[h] < need) continue;
        if (!best || s->ema_spt[h] < s->ema_spt[best]) best = h;
    }
    return best;
}

static inline int k3_spec_auto_cost_choose(K3SpecAutoCost *s, int current, int limit,
                                            int proposed, int accepted)
{
    int best, next;
    if (!s || current < 1) return current;
    if (limit > K3_SPEC_AUTO_COST_MAX) limit = K3_SPEC_AUTO_COST_MAX;
    if (current > limit) current = limit;

    /* Acceptance is a hard signal: if more than half the draft was wasted, do not wait
     * for the timing estimator to become statistically useful. */
    if (proposed > 0 && accepted * 2 < proposed) {
        next = (current + 1) / 2;
        return next < 1 ? 1 : next;
    }

    /* Do not explore upward after a partial acceptance.  The current width may still be
     * fine for the next part of the sequence, so unlike the <50% case we simply hold. */
    if (proposed <= 0 || accepted != proposed) return current;
    if (s->samples[current] < K3_SPEC_AUTO_COST_MIN_SAMPLES) return current;

    best = k3_spec_auto_cost_best(s, limit, 1);
    if (!best) return current;

    /* A measured regression is enough to abandon this candidate.  Marking it blocked
     * avoids ping-ponging back to the same known-bad horizon a few rounds later. */
    if (current != best &&
        s->ema_spt[current] > s->ema_spt[best] * K3_SPEC_AUTO_COST_REGRET) {
        s->blocked_mask |= 1u << current;
        s->backoffs++;
        return best;
    }

    /* Explore the next not-yet-rejected horizon.  This intentionally allows a jump over
     * a blocked width: batching can make h+2 cheaper even when h+1 was a local loser. */
    for (next = current + 1; next <= limit; next++) {
        if (!(s->blocked_mask & (1u << next))) {
            s->probes++;
            return next;
        }
    }

    /* Exploration is exhausted.  Settle on the measured winner instead of remaining on
     * a merely "within 5%" candidate forever. */
    return best;
}

#endif
