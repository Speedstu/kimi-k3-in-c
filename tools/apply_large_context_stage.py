#!/usr/bin/env python3
from pathlib import Path


def once(s: str, old: str, new: str, label: str) -> str:
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1 match, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'src/cli/k3_worker.c'
s=p.read_text()

s=once(s,
'''#include <sys/mman.h>

#define K3_WORKER_DEFAULT_CONTEXT 1024
#define K3_WORKER_PREFILL_CHUNK 64
''',
'''#include <sys/mman.h>
#include <unistd.h>

#define K3_WORKER_DEFAULT_CONTEXT 1024
#define K3_WORKER_MAX_CONTEXT 1048576
#define K3_WORKER_PREFILL_CHUNK 64
/* If anonymous mmap is unavailable/fails, small allocations may safely use calloc.
 * Never ask libc for a multi-terabyte zeroed fallback when a large virtual KV reservation
 * was explicitly requested: fail cleanly instead. */
#define K3_WORKER_CALLOC_FALLBACK_MAX ((size_t)1 << 30)
''','context constants')

s=once(s,
'''    /* Current worker contexts are still bounded conservatively. Fall back to calloc on
     * platforms without anonymous mmap; correctness is identical, only reservation
     * behaviour may be less lazy. A later large-context gate can require mmap. */
    m.ptr = (float *)calloc(nfloats, sizeof(float));
    return m;
}

static void worker_vm_free(WorkerVM *m)
''',
'''    /* Keep a portable fallback for ordinary contexts, but never turn an mmap failure
     * for a TB-scale reservation into a giant calloc. That failure mode can appear to
     * succeed under overcommit and then kill the machine when pages are touched. */
    if (m.bytes <= K3_WORKER_CALLOC_FALLBACK_MAX)
        m.ptr = (float *)calloc(nfloats, sizeof(float));
    return m;
}

/* Best-effort physical-page reclamation after a conversation branch/reset. Numerical
 * correctness does NOT rely on this: cached=0 already makes all old rows unreachable.
 * The advice merely gives anonymous pages back without writing zeros through gigabytes.
 * Operate on only the prefix that was actually touched, per MLA layer, so a 1M-token
 * virtual reservation does not require a kernel walk over untouched address space. */
static void worker_vm_discard_rows(WorkerVM *m, int nlayers, int cap,
                                   size_t row_floats, int used)
{
#if defined(MADV_DONTNEED)
    if (!m || !m->mapped || !m->ptr || nlayers <= 0 || cap <= 0 ||
        row_floats == 0 || used <= 0) return;
    if (used > cap) used = cap;
    long psl = sysconf(_SC_PAGESIZE);
    if (psl <= 0) return;
    const uintptr_t ps = (uintptr_t)psl;
    const uintptr_t map_lo = (uintptr_t)m->ptr;
    const uintptr_t map_hi = map_lo + m->bytes;
    const size_t stride = (size_t)cap * row_floats * sizeof(float);
    const size_t live = (size_t)used * row_floats * sizeof(float);
    for (int L = 0; L < nlayers; L++) {
        uintptr_t a = map_lo + (size_t)L * stride;
        uintptr_t b = a + live;
        uintptr_t lo = (a / ps) * ps;
        uintptr_t hi = ((b + ps - 1) / ps) * ps;
        if (lo < map_lo) lo = map_lo;
        if (hi > map_hi) hi = map_hi;
        if (hi > lo) (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
    }
#else
    (void)m; (void)nlayers; (void)cap; (void)row_floats; (void)used;
#endif
}

static void worker_discard_model_kv(const Weights *w, WorkerVM *kv, WorkerVM *rope,
                                    const K3Cfg *c)
{
    if (!w || w->cached <= 0) return;
    worker_vm_discard_rows(kv, w->n_mla, w->kv_cap,
                           (size_t)c->n_heads * (c->qk_nope + c->v_head), w->cached);
    worker_vm_discard_rows(rope, w->n_mla, w->kv_cap,
                           (size_t)c->qk_rope, w->cached);
}

static void worker_vm_free(WorkerVM *m)
''','safe fallback and discard helpers')

s=once(s,
'''        "  --context N          resident conversation capacity (default 1024)\\n"
''',
'''        "  --context N          resident capacity, up to 1,048,576 (default 1024)\\n"
''','usage context')

s=once(s,
'''    if (context < 2 || context > K3_MAX_PROMPT + K3_MAX_GEN) {
        fprintf(stderr, "--context must be in [2,%d]\\n", K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
''',
'''    if (context < 2 || context > K3_WORKER_MAX_CONTEXT) {
        fprintf(stderr, "--context must be in [2,%d]\\n", K3_WORKER_MAX_CONTEXT);
        return 2;
    }
''','worker context ceiling')

# Add an explicit 64-bit guard after config, before sizes are multiplied.
s=once(s,
'''    K3Cfg c; static int fa[128];
    if (!real_cfg(&c, fa, 128, dir, cfg_path)) return 2;
    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {
''',
'''    K3Cfg c; static int fa[128];
    if (!real_cfg(&c, fa, 128, dir, cfg_path)) return 2;
    if (sizeof(size_t) < 8 && context > K3_MAX_PROMPT + K3_MAX_GEN) {
        fprintf(stderr, "large resident contexts require a 64-bit build\\n");
        return 2;
    }
    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {
''','64-bit context guard')

# Emit a transparent capacity cost after the exact lazy mappings are created.
s=once(s,
'''    w.kvc = w_kv_mem.ptr;
    w.ropec = w_rope_mem.ptr;

    const int E = c.hidden;
''',
'''    w.kvc = w_kv_mem.ptr;
    w.ropec = w_rope_mem.ptr;

    const double kv_bytes_per_pos = (double)w.n_mla * sizeof(float) *
        ((double)c.n_heads * (c.qk_nope + c.v_head) + c.qk_rope);
    printf("resident MLA KV: %.2f MiB/used position/model; context %d reserves %.2f GiB "
           "of virtual address space%s\\n",
           kv_bytes_per_pos / (1024.0 * 1024.0), context,
           kv_bytes_per_pos * context / (1024.0 * 1024.0 * 1024.0),
           w_kv_mem.mapped && w_rope_mem.mapped ? " lazily" : "");

    const int E = c.hidden;
''','resident KV diagnostics')

# Explicit RESET: reclaim exact + draft used pages before dropping cached counters.
s=once(s,
'''        if (!strcmp(op, "RESET")) {
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
''',
'''        if (!strcmp(op, "RESET")) {
            worker_discard_model_kv(&w, &w_kv_mem, &w_rope_mem, &c);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) {
                worker_discard_model_kv(&dw, &d_kv_mem, &d_rope_mem, &c);
                worker_reset_state(&dw, dks, kper, NL, &c);
            }
''','RESET page discard')

# Branch/reset caused by a non-prefix prompt.
s=once(s,
'''        if (!reuse_tokens) {
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
            history_len = 0;
        }
''',
'''        if (!reuse_tokens) {
            worker_discard_model_kv(&w, &w_kv_mem, &w_rope_mem, &c);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) {
                worker_discard_model_kv(&dw, &d_kv_mem, &d_rope_mem, &c);
                worker_reset_state(&dw, dks, kper, NL, &c);
            }
            history_len = 0;
        }
''','branch page discard')

# Failed request cleanup.
s=once(s,
'''        if (failed || k3_expert_drops) {
            printf("@K3ERROR %llu 1\\n", rid);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
            history_len = 0;
''',
'''        if (failed || k3_expert_drops) {
            printf("@K3ERROR %llu 1\\n", rid);
            worker_discard_model_kv(&w, &w_kv_mem, &w_rope_mem, &c);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) {
                worker_discard_model_kv(&dw, &d_kv_mem, &d_rope_mem, &c);
                worker_reset_state(&dw, dks, kper, NL, &c);
            }
            history_len = 0;
''','failure page discard')

p.write_text(s)
print('large-context resident worker materialized')
