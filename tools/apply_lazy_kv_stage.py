#!/usr/bin/env python3
from pathlib import Path


def once(s: str, old: str, new: str, label: str) -> str:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1 match, got {n}")
    return s.replace(old, new, 1)

p = Path(__file__).resolve().parents[1] / "src/cli/k3_worker.c"
s = p.read_text()

s = once(s,
'''#undef main

#define K3_WORKER_DEFAULT_CONTEXT 1024
''',
'''#undef main

#include <sys/mman.h>

#define K3_WORKER_DEFAULT_CONTEXT 1024
#define K3_WORKER_PREFILL_CHUNK 64

/* A resident worker may reserve many gigabytes of expanded MLA KV address space while
 * touching only a short prefix. Anonymous mmap gives that capacity a lazy physical
 * footprint: pages are zero-backed until first write. This is an allocation policy only;
 * the float representation and every arithmetic operation stay unchanged. */
typedef struct {
    float *ptr;
    size_t bytes;
    int mapped;
} WorkerVM;

static WorkerVM worker_vm_alloc(size_t nfloats)
{
    WorkerVM m; memset(&m, 0, sizeof m);
    if (nfloats == 0 || nfloats > SIZE_MAX / sizeof(float)) return m;
    m.bytes = nfloats * sizeof(float);
#if defined(MAP_ANONYMOUS) || defined(MAP_ANON)
    int flags = MAP_PRIVATE;
#  if defined(MAP_ANONYMOUS)
    flags |= MAP_ANONYMOUS;
#  else
    flags |= MAP_ANON;
#  endif
#  if defined(MAP_NORESERVE)
    flags |= MAP_NORESERVE;
#  endif
    void *q = mmap(NULL, m.bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (q != MAP_FAILED) {
        m.ptr = (float *)q;
        m.mapped = 1;
        return m;
    }
#endif
    /* Current worker contexts are still bounded conservatively. Fall back to calloc on
     * platforms without anonymous mmap; correctness is identical, only reservation
     * behaviour may be less lazy. A later large-context gate can require mmap. */
    m.ptr = (float *)calloc(nfloats, sizeof(float));
    return m;
}

static void worker_vm_free(WorkerVM *m)
{
    if (!m || !m->ptr) return;
    if (m->mapped) munmap(m->ptr, m->bytes);
    else free(m->ptr);
    memset(m, 0, sizeof *m);
}
''', "worker VM helpers")

s = once(s,
'''static void worker_reset_state(Weights *w, float *ks, size_t kper, int nl,
                               const K3Cfg *c)
{
    if (!w || !ks) return;
    memset(ks, 0, kper * (size_t)nl * sizeof(float));
    if (w->kvc) {
        const size_t kvper = (size_t)w->kv_cap * c->n_heads * (c->qk_nope + c->v_head);
        memset(w->kvc, 0, kvper * (size_t)w->n_mla * sizeof(float));
    }
    if (w->ropec) {
        const size_t rpper = (size_t)w->kv_cap * c->qk_rope;
        memset(w->ropec, 0, rpper * (size_t)w->n_mla * sizeof(float));
    }
    w->cached = 0;
}
''',
'''static void worker_reset_state(Weights *w, float *ks, size_t kper, int nl,
                               const K3Cfg *c)
{
    (void)c;
    if (!w || !ks) return;
    /* KDA recurrence/ShortConv history is true recurrent state and MUST reset. MLA KV is
     * position-addressed: cached=0 makes every old row unreachable, and each newly used
     * row is fully overwritten before it can be read. Zeroing the whole capacity here
     * used to write ~2.37 MB per configured position on released K3 (twice with a draft),
     * turning a branch/reset into a multi-GB memory sweep for no numerical reason. */
    memset(ks, 0, kper * (size_t)nl * sizeof(float));
    w->cached = 0;
}
''', "O(KDA) reset")

insert_after = '''static int worker_replay_prefix(Weights *w, const K3Cfg *c, K3Cache *cache,
                                const int *seq, int base, int n,
                                float *lg, float *sc, float *h, float *br, float *ks)
{
    w->cached = base;
    if (n <= 0) return 0;
    if (forward(w, c, cache, seq + base, n, lg, sc, h, br, ks, NULL, NULL) != 0)
        return -1;
    w->cached = base + n;
    return 0;
}
'''
replacement = insert_after + '''

/* Feed a potentially long prompt suffix with a bounded T. KDA already carries recurrent
 * state across calls, MLA's absolute cache positions are `w->cached`, and AttnRes/MoE
 * arithmetic is per-token; splitting only bounds transient hidden/residual/scratch RAM.
 * The permanent >64-token parity gate compares this path against one full one-shot sweep. */
static int worker_prefill_to(Weights *w, const K3Cfg *c, K3Cache *cache,
                             const int *seq, int target, int chunk,
                             float *lg, float *sc, float *h, float *br, float *ks)
{
    if (!w || w->cached < 0 || w->cached >= target || chunk < 1) return -1;
    while (w->cached < target) {
        const int base = w->cached;
        int n = target - base;
        if (n > chunk) n = chunk;
        if (forward(w, c, cache, seq + base, n, lg, sc, h, br, ks, NULL, NULL) != 0)
            return -1;
        w->cached = base + n;
    }
    return 0;
}
'''
s = once(s, insert_after, replacement, "chunked prefill helper")

s = once(s,
'''    const size_t kvper = (size_t)context * c.n_heads * (c.qk_nope + c.v_head);
    const size_t rpper = (size_t)context * c.qk_rope;
    w.kvc = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
    w.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));

    const int E = c.hidden;
''',
'''    const size_t kvper = (size_t)context * c.n_heads * (c.qk_nope + c.v_head);
    const size_t rpper = (size_t)context * c.qk_rope;
    WorkerVM w_kv_mem = worker_vm_alloc(kvper * (size_t)w.n_mla);
    WorkerVM w_rope_mem = worker_vm_alloc(rpper * (size_t)w.n_mla);
    w.kvc = w_kv_mem.ptr;
    w.ropec = w_rope_mem.ptr;

    const int E = c.hidden;
''', "exact lazy KV allocation")

s = once(s,
'''    float *ks = (float *)calloc(kper * (size_t)NL, sizeof(float));
    float *spec_snap = (float *)malloc(kper * (size_t)NL * sizeof(float));
    float *h = (float *)malloc((size_t)context * E * sizeof(float));
    float *br = (float *)malloc((size_t)context * maxb * E * sizeof(float));
    size_t sc_need = k3_layer_scratch(&c, context);
    const size_t cached_need = k3_mla_scratch_cached(&c, context, context, 1);
    if (cached_need > sc_need) sc_need = cached_need;
''',
'''    float *ks = (float *)calloc(kper * (size_t)NL, sizeof(float));
    float *spec_snap = (float *)malloc(kper * (size_t)NL * sizeof(float));
    const int prefill_cap = context < K3_WORKER_PREFILL_CHUNK
                          ? context : K3_WORKER_PREFILL_CHUNK;
    float *h = (float *)malloc((size_t)prefill_cap * E * sizeof(float));
    float *br = (float *)malloc((size_t)prefill_cap * maxb * E * sizeof(float));
    size_t sc_need = k3_layer_scratch(&c, prefill_cap);
    /* Cached MLA scores must still address the full resident prefix, but the dominant q/
     * hidden/residual temporaries scale only with prefill_cap, not configured context. */
    const size_t cached_need = k3_mla_scratch_cached(&c, prefill_cap, context, 1);
    if (cached_need > sc_need) sc_need = cached_need;
''', "bounded transient allocation")

s = once(s,
'''        dw.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
        dw.kvc = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
        dw.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));
        dks = (float *)calloc(kper * (size_t)NL, sizeof(float));
''',
'''        dw.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
        WorkerVM d_kv_mem_tmp = worker_vm_alloc(kvper * (size_t)w.n_mla);
        WorkerVM d_rope_mem_tmp = worker_vm_alloc(rpper * (size_t)w.n_mla);
        dw.kvc = d_kv_mem_tmp.ptr;
        dw.ropec = d_rope_mem_tmp.ptr;
        /* Ownership moves to the function-scope records declared below. */
        d_kv_mem = d_kv_mem_tmp;
        d_rope_mem = d_rope_mem_tmp;
        dks = (float *)calloc(kper * (size_t)NL, sizeof(float));
''', "draft lazy KV allocation")

# Declare draft WorkerVM records before the optional-draft block.
s = once(s,
'''    K3Trunk trunk_d; memset(&trunk_d, 0, sizeof trunk_d);
    Weights dw; memset(&dw, 0, sizeof dw);
    float *dks = NULL, *dsnap = NULL;
''',
'''    K3Trunk trunk_d; memset(&trunk_d, 0, sizeof trunk_d);
    Weights dw; memset(&dw, 0, sizeof dw);
    WorkerVM d_kv_mem; memset(&d_kv_mem, 0, sizeof d_kv_mem);
    WorkerVM d_rope_mem; memset(&d_rope_mem, 0, sizeof d_rope_mem);
    float *dks = NULL, *dsnap = NULL;
''', "draft VM ownership declarations")

s = once(s,
'''        if (w.cached >= np || forward(&w, &c, &cache, seq + w.cached, np - w.cached,
                                      lg, sc, h, br, ks, NULL, NULL) != 0) {
            failed = 1;
        } else {
            w.cached = np;
            /* IMPORTANT: sample from exact logits NOW. Draft prefill reuses `lg` and
''',
'''        if (w.cached >= np || worker_prefill_to(&w, &c, &cache, seq, np, prefill_cap,
                                                  lg, sc, h, br, ks) != 0) {
            failed = 1;
        } else {
            /* IMPORTANT: sample from exact logits NOW. Draft prefill reuses `lg` and
''', "exact chunked prefill")

s = once(s,
'''        if (draft_dir && !failed) {
            if (dw.cached >= np || forward(&dw, &c, &cache, seq + dw.cached, np - dw.cached,
                                           lg, sc, h, br, dks, NULL, NULL) != 0) {
                failed = 1;
            } else {
                dw.cached = np;
            }
        }
''',
'''        if (draft_dir && !failed) {
            if (dw.cached >= np || worker_prefill_to(&dw, &c, &cache, seq, np, prefill_cap,
                                                     lg, sc, h, br, dks) != 0) {
                failed = 1;
            }
        }
''', "draft chunked prefill")

s = once(s,
'''    if (draft_dir) {
        free(dw.kvc); free(dw.ropec);
        for (int L = 0; L < dw.n_bound; L++) k3_bind_free(&dw.lay[L]);
''',
'''    if (draft_dir) {
        worker_vm_free(&d_kv_mem); worker_vm_free(&d_rope_mem);
        for (int L = 0; L < dw.n_bound; L++) k3_bind_free(&dw.lay[L]);
''', "draft VM cleanup")

s = once(s,
'''    free(w.kvc); free(w.ropec); free(w.mla_slot);
    k3_cache_free(&cache);
''',
'''    worker_vm_free(&w_kv_mem); worker_vm_free(&w_rope_mem); free(w.mla_slot);
    k3_cache_free(&cache);
''', "exact VM cleanup")

p.write_text(s)
print("lazy KV + O(KDA) reset + bounded prefill materialized")
