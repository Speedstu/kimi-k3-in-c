#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
p = root / 'src/cli/k3_run.c'
s = p.read_text(encoding='utf-8')
old = '''    float *nrm = scratch;
    if (arg_all) {
        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            arg_all[t] = argmax_(logits_last, c->vocab);
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        return 0;
    }
'''
new = '''    float *nrm = scratch;
    if (arg_all) {
        /* Speculative verification asks for argmax at several adjacent positions. The
         * old path ran the full lm_head matvec once per position, re-reading the same
         * ~2.35 GB bf16 matrix T times. For the short speculative batches (<=9), walk
         * vocabulary rows OUTER and positions INNER instead: each weight row is fetched
         * once from memory and reused immediately for every position while it is hot.
         *
         * Each individual dot product still goes through k3_mmw with out=1, so its
         * arithmetic/reduction order is exactly the same as the corresponding row of a
         * normal full lm_head matvec. Only the order in which independent rows/positions
         * are evaluated changes. The result is therefore logit-identical.
         *
         * tf-check can pass a long sequence, so bound this layout to speculative-size
         * batches. Longer arg_all calls retain the old low-memory path below. */
        if (T > 1 && T <= K3_SPEC_MAX + 1) {
            const size_t nv = (size_t)c->vocab;
            float *all = (float *)malloc((size_t)T * nv * sizeof(float));
            if (all) {
                /* h is dead after logits are produced, so normalise in place and avoid
                 * another T*hidden scratch allocation. k3_rmsnorm supports y == x. */
                for (int t = 0; t < T; t++)
                    k3_rmsnorm(h + (size_t)t * E, h + (size_t)t * E,
                               w->mb.norm, E, c->rms_eps);

                const unsigned char *W = (const unsigned char *)w->mb.lm_head;
                const size_t rowb = k3_row_bytes(w->mb.wdt, E);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int o = 0; o < c->vocab; o++) {
                    const void *row = W + (size_t)o * rowb;
                    for (int t = 0; t < T; t++)
                        k3_mmw(all + (size_t)t * nv + o,
                               h + (size_t)t * E, row, w->mb.wdt, E, 1);
                }
                for (int t = 0; t < T; t++)
                    arg_all[t] = argmax_(all + (size_t)t * nv, c->vocab);
                memcpy(logits_last, all + (size_t)(T - 1) * nv, nv * sizeof(float));
                free(all);
                return 0;
            }
            /* Allocation failure is a performance miss, not a correctness failure:
             * fall back to the original one-position-at-a-time path. */
        }
        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            arg_all[t] = argmax_(logits_last, c->vocab);
        }
        /* logits_last now holds the FINAL position's vector, same as the plain path. */
        return 0;
    }
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one forward arg_all block, found {s.count(old)}')
p.write_text(s.replace(old, new, 1), encoding='utf-8', newline='\n')
print('patched batched lm_head speculative verification')
