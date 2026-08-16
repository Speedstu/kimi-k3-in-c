#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

if "K3_NO_PREFILL_BATCH_MLA_BF16" in s:
    print("exact MLA BF16 prefill batching already applied")
    raise SystemExit(0)

# MLA appears before the existing prefill batch helper declaration, so move the shared
# exact-batch width/prototype above MLA and remove the later duplicate declaration.
anchor = '''size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)\n'''
insert = '''#define K3_PREFILL_BATCH_MAX 64\nstatic void k3_matmul_bf16_batch(float *y, int ystride,\n                                  const float *const *xs, int batch,\n                                  const uint16_t *W, int in, int out);\n\nsize_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)\n'''
if anchor not in s:
    raise SystemExit("MLA scratch declaration anchor not found")
s = s.replace(anchor, insert, 1)

later = '''#define K3_PREFILL_BATCH_MAX 64\nstatic void k3_matmul_bf16_batch(float *y, int ystride,\n                                  const float *const *xs, int batch,\n                                  const uint16_t *W, int in, int out);\nstatic void k3_matmul_mxfp4_batch(float *y, int ystride,\n'''
later_new = '''static void k3_matmul_mxfp4_batch(float *y, int ystride,\n'''
if later not in s:
    raise SystemExit("later batch prototype anchor not found")
s = s.replace(later, later_new, 1)

old = '''    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;\n    const size_t kvd = (size_t)(c->qk_nope + vh);\n    size_t n = (size_t)T * H * qh                      /* q            */\n             + (size_t)(c->kv_lora + c->qk_rope)       /* ct transient */\n             + (size_t)c->q_lora\n             + (size_t)2 * H * vh                      /* acc, gbuf    */\n             + (size_t)(cap > T ? cap : T);            /* scores       */\n'''
new = '''    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;\n    const size_t kvd = (size_t)(c->qk_nope + vh);\n    /* Exact BF16 projection batching is intentionally bounded to the same 64-token\n     * prefill width as MoE. Above that width MLA takes the proven per-token path, so\n     * there is no reason to make long-context scratch grow by another O(T) gate block. */\n    const int B = (T > 1 && T <= K3_PREFILL_BATCH_MAX) ? T : 1;\n    size_t n = (size_t)T * H * qh                      /* q            */\n             + (size_t)B * (c->kv_lora + c->qk_rope)  /* batched ct   */\n             + (size_t)B * c->q_lora                  /* batched ql   */\n             + (size_t)H * vh                         /* acc          */\n             + (size_t)B * H * vh                     /* batched gate */\n             + (size_t)(cap > T ? cap : T);           /* scores       */\n'''
if old not in s:
    raise SystemExit("MLA scratch body anchor not found")
s = s.replace(old, new, 1)

old = '''    const float scale = 1.0f / sqrtf((float)qh);  /* :359, over qh not qn           */\n    if (!kvc) cached = 0;\n    const int last = cached + T - 1;              /* highest absolute position      */\n    if (kvc && last >= cap)\n        k3_fatal_bound("MLA KV cache position", (long)last, (long)cap - 1);\n\n    /* Scratch layout. Every region below is DISJOINT and must stay so. Overlapping\n'''
new = '''    const float scale = 1.0f / sqrtf((float)qh);  /* :359, over qh not qn           */\n    if (!kvc) cached = 0;\n    const int last = cached + T - 1;              /* highest absolute position      */\n    if (kvc && last >= cap)\n        k3_fatal_bound("MLA KV cache position", (long)last, (long)cap - 1);\n\n    static int no_prefill_batch_mla_bf16 = -1;\n    if (no_prefill_batch_mla_bf16 < 0)\n        no_prefill_batch_mla_bf16 = getenv("K3_NO_PREFILL_BATCH_MLA_BF16") ? 1 : 0;\n    const int B = (T > 1 && T <= K3_PREFILL_BATCH_MAX) ? T : 1;\n    const int batch_mla = !no_prefill_batch_mla_bf16 && B > 1 && w->wdt == K3_WBF16;\n\n    /* Scratch layout. Every region below is DISJOINT and must stay so. Overlapping\n'''
if old not in s:
    raise SystemExit("MLA batch flag anchor not found")
s = s.replace(old, new, 1)

old = '''    float *q    = scratch;                          /* [T][H][qh]     */\n    float *ct   = q    + (size_t)T * H * qh;        /* [kvw] transient, one token */\n    float *ql   = ct   + (size_t)kvw;               /* [q_lora]       */\n    float *acc  = ql   + (size_t)c->q_lora;         /* [H][vh]        */\n    float *gbuf = acc  + (size_t)H * vh;            /* [H][vh] gate   */\n    float *sc   = gbuf + (size_t)H * vh;            /* [last+1] scores */\n'''
new = '''    float *q    = scratch;                          /* [T][H][qh]     */\n    float *ct   = q    + (size_t)T * H * qh;        /* [B][kvw]       */\n    float *ql   = ct   + (size_t)B * kvw;           /* [B][q_lora]    */\n    float *acc  = ql   + (size_t)B * c->q_lora;     /* [H][vh]        */\n    float *gbuf = acc  + (size_t)H * vh;            /* [B][H][vh] gate */\n    float *sc   = gbuf + (size_t)B * H * vh;        /* [last+1] scores */\n'''
if old not in s:
    raise SystemExit("MLA scratch layout anchor not found")
s = s.replace(old, new, 1)

old = '''    /* ---- per-token projections ---- */\n    for (int t = 0; t < T; t++) {\n        const int p = cached + t;\n        const float *xt = x + (size_t)t * E;\n        k3_mmw(ql, xt, w->q_a, w->wdt, E, c->q_lora);\n        k3_rmsnorm(ql, ql, w->q_a_norm, c->q_lora, c->rms_eps);\n        k3_mmw(q + (size_t)t * H * qh, ql, w->q_b, w->wdt, c->q_lora, H * qh);\n\n        /* ONE projection emits the compressed latent AND the shared rope slot */\n        k3_mmw(ct, xt, w->kv_a, w->wdt, E, kvw);\n        /* the norm covers the latent only, never the rope slot */\n        k3_rmsnorm(ct, ct, w->kv_a_norm, c->kv_lora, c->rms_eps);\n        memcpy(K3_ROPE_AT(p), ct + c->kv_lora, (size_t)qr * sizeof(float));\n        k3_mmw(K3_KV_AT(p), ct, w->kv_b, w->wdt, c->kv_lora, H * kvd);\n    }\n'''
new = '''    /* ---- per-token projections ---- */\n    if (batch_mla) {\n        const float *xp[K3_PREFILL_BATCH_MAX];\n        const float *qp[K3_PREFILL_BATCH_MAX];\n        const float *cp[K3_PREFILL_BATCH_MAX];\n        for (int t = 0; t < T; t++) xp[t] = x + (size_t)t * E;\n\n        /* Stage 1 has no token dependency: share BF16 row widening/cache traffic. */\n        k3_matmul_bf16_batch(ql, c->q_lora, xp, T,\n                             (const uint16_t *)w->q_a, E, c->q_lora);\n        k3_matmul_bf16_batch(ct, kvw, xp, T,\n                             (const uint16_t *)w->kv_a, E, kvw);\n\n        /* Norms and rope-slot publication keep their original token order. */\n        for (int t = 0; t < T; t++) {\n            const int p = cached + t;\n            float *qlt = ql + (size_t)t * c->q_lora;\n            float *ctt = ct + (size_t)t * kvw;\n            k3_rmsnorm(qlt, qlt, w->q_a_norm, c->q_lora, c->rms_eps);\n            k3_rmsnorm(ctt, ctt, w->kv_a_norm, c->kv_lora, c->rms_eps);\n            memcpy(K3_ROPE_AT(p), ctt + c->kv_lora, (size_t)qr * sizeof(float));\n            qp[t] = qlt;\n            cp[t] = ctt;\n        }\n\n        /* Stage 2 consumes the exact same normalised latents. The T cache positions\n         * are contiguous whether the cache is external or scratch-backed. */\n        k3_matmul_bf16_batch(q, H * qh, qp, T,\n                             (const uint16_t *)w->q_b, c->q_lora, H * qh);\n        k3_matmul_bf16_batch(K3_KV_AT(cached), H * kvd, cp, T,\n                             (const uint16_t *)w->kv_b, c->kv_lora, H * kvd);\n    } else {\n        for (int t = 0; t < T; t++) {\n            const int p = cached + t;\n            const float *xt = x + (size_t)t * E;\n            k3_mmw(ql, xt, w->q_a, w->wdt, E, c->q_lora);\n            k3_rmsnorm(ql, ql, w->q_a_norm, c->q_lora, c->rms_eps);\n            k3_mmw(q + (size_t)t * H * qh, ql, w->q_b, w->wdt, c->q_lora, H * qh);\n\n            /* ONE projection emits the compressed latent AND the shared rope slot */\n            k3_mmw(ct, xt, w->kv_a, w->wdt, E, kvw);\n            /* the norm covers the latent only, never the rope slot */\n            k3_rmsnorm(ct, ct, w->kv_a_norm, c->kv_lora, c->rms_eps);\n            memcpy(K3_ROPE_AT(p), ct + c->kv_lora, (size_t)qr * sizeof(float));\n            k3_mmw(K3_KV_AT(p), ct, w->kv_b, w->wdt, c->kv_lora, H * kvd);\n        }\n    }\n'''
if old not in s:
    raise SystemExit("MLA projection anchor not found")
s = s.replace(old, new, 1)

old = '''        /* ---- output gate then projection. Gate BEFORE o_proj, and no norm on it,\n         * unlike KDA which norms first. :470-473 ---- */\n        if (w->g) {\n            k3_mmw(gbuf, x + (size_t)t * E, w->g, w->wdt, E, H * vh);\n            for (int i = 0; i < H * vh; i++)\n                acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));\n        }\n        k3_mmw(out + (size_t)t * E, acc, w->o, w->wdt, H * vh, E);\n    }\n    #undef K3_KV_AT\n'''
new = '''        /* ---- output gate then projection. Gate BEFORE o_proj, and no norm on it,\n         * unlike KDA which norms first. :470-473 ---- */\n        if (batch_mla) {\n            /* q[t] is dead after this token's causal attention. Its 18,432-float row\n             * is wider than the 12,288-float attention output, so retain the exact\n             * attention result there until gate/o_proj can share BF16 weights. */\n            memcpy(q + (size_t)t * H * qh, acc, (size_t)H * vh * sizeof(float));\n        } else {\n            if (w->g) {\n                k3_mmw(gbuf, x + (size_t)t * E, w->g, w->wdt, E, H * vh);\n                for (int i = 0; i < H * vh; i++)\n                    acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));\n            }\n            k3_mmw(out + (size_t)t * E, acc, w->o, w->wdt, H * vh, E);\n        }\n    }\n\n    if (batch_mla) {\n        const float *xp[K3_PREFILL_BATCH_MAX];\n        const float *op[K3_PREFILL_BATCH_MAX];\n        for (int t = 0; t < T; t++) xp[t] = x + (size_t)t * E;\n\n        if (w->g)\n            k3_matmul_bf16_batch(gbuf, H * vh, xp, T,\n                                 (const uint16_t *)w->g, E, H * vh);\n\n        for (int t = 0; t < T; t++) {\n            float *at = q + (size_t)t * H * qh;\n            if (w->g) {\n                const float *gt = gbuf + (size_t)t * H * vh;\n                for (int i = 0; i < H * vh; i++)\n                    at[i] *= 1.0f / (1.0f + expf(-gt[i]));\n            }\n            op[t] = at;\n        }\n        k3_matmul_bf16_batch(out, E, op, T,\n                             (const uint16_t *)w->o, H * vh, E);\n    }\n    #undef K3_KV_AT\n'''
if old not in s:
    raise SystemExit("MLA output anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("applied exact MLA BF16 prefill batching")
