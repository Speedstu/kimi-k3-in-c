#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

if "K3_NO_PREFILL_BATCH_DENSE_BF16" in s:
    print("exact dense BF16 prefill batching already applied")
    raise SystemExit(0)

old = '''    if (w->moe) {
        int   idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];
        /* Prefill batches (T > 1, streamed source) fetch each unique expert once for
         * the whole chunk; decode (T == 1) and the resident path fall straight through
         * to k3_moe inside, byte-identical. */
        k3_moe_prefill(tmp, hin, w->moe, c, T, idx, wt, sub);
    } else {
        for (int t = 0; t < T; t++) {
            k3_mmw(dgu, hin + (size_t)t * E, w->dense_gate, w->wdt, E, c->dense_inter);
            k3_mmw(dgu + c->dense_inter, hin + (size_t)t * E, w->dense_up, w->wdt,
                      E, c->dense_inter);
            k3_situ_glu(sub, dgu, c->dense_inter, c->situ_b1, c->situ_b2);
            k3_mmw(tmp + (size_t)t * E, sub, w->dense_down, w->wdt, c->dense_inter, E);
        }
    }
'''
new = '''    if (w->moe) {
        int   idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];
        /* Prefill batches (T > 1, streamed source) fetch each unique expert once for
         * the whole chunk; decode (T == 1) and the resident path fall straight through
         * to k3_moe inside, byte-identical. */
        k3_moe_prefill(tmp, hin, w->moe, c, T, idx, wt, sub);
    } else {
        /* Layer 0 is a dense BF16 MLP. At released K3 dimensions its gate/up/down
         * matrices contain 726,663,168 bf16 values (~1.45 GB), and the old prompt path
         * reread all three matrices once for every token. Share only the weight
         * widening/cache traffic across up to 64 prompt tokens; every token keeps the
         * exact k3_matmul_bf16 FMA lanes, SiTU arithmetic and reduction order. */
        static int no_prefill_batch_dense_bf16 = -1;
        if (no_prefill_batch_dense_bf16 < 0)
            no_prefill_batch_dense_bf16 = getenv("K3_NO_PREFILL_BATCH_DENSE_BF16") ? 1 : 0;
        const int batch_dense = !no_prefill_batch_dense_bf16 && T > 1
                             && w->wdt == K3_WBF16;

        if (batch_dense) {
            const int I = c->dense_inter;
            const int B = T < K3_PREFILL_BATCH_MAX ? T : K3_PREFILL_BATCH_MAX;
            const size_t bn = (size_t)B * 3 * I;
            float *dbuf = (float *)malloc(bn * sizeof(float));
            if (!dbuf) k3_fatal_oom("dense prefill batched BF16", bn * sizeof(float));
            float *bgu  = dbuf;                        /* [B][2*I] */
            float *bact = bgu + (size_t)B * 2 * I;    /* [B][I]   */

            for (int t0 = 0; t0 < T; t0 += K3_PREFILL_BATCH_MAX) {
                const int n = (T - t0) < K3_PREFILL_BATCH_MAX
                            ? (T - t0) : K3_PREFILL_BATCH_MAX;
                const float *xp[K3_PREFILL_BATCH_MAX];
                const float *ap[K3_PREFILL_BATCH_MAX];
                for (int b = 0; b < n; b++)
                    xp[b] = hin + (size_t)(t0 + b) * E;

                k3_matmul_bf16_batch(bgu, 2 * I, xp, n,
                                     (const uint16_t *)w->dense_gate, E, I);
                k3_matmul_bf16_batch(bgu + I, 2 * I, xp, n,
                                     (const uint16_t *)w->dense_up, E, I);
                for (int b = 0; b < n; b++) {
                    float *ab = bact + (size_t)b * I;
                    k3_situ_glu(ab, bgu + (size_t)b * 2 * I,
                                I, c->situ_b1, c->situ_b2);
                    ap[b] = ab;
                }
                k3_matmul_bf16_batch(tmp + (size_t)t0 * E, E, ap, n,
                                     (const uint16_t *)w->dense_down, I, E);
            }
            free(dbuf);
        } else {
            for (int t = 0; t < T; t++) {
                k3_mmw(dgu, hin + (size_t)t * E, w->dense_gate, w->wdt,
                       E, c->dense_inter);
                k3_mmw(dgu + c->dense_inter, hin + (size_t)t * E,
                       w->dense_up, w->wdt, E, c->dense_inter);
                k3_situ_glu(sub, dgu, c->dense_inter, c->situ_b1, c->situ_b2);
                k3_mmw(tmp + (size_t)t * E, sub, w->dense_down, w->wdt,
                       c->dense_inter, E);
            }
        }
    }
'''
if old not in s:
    raise SystemExit("dense decoder-layer anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("applied exact dense BF16 prefill batching")
