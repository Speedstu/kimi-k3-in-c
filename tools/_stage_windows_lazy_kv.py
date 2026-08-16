from pathlib import Path
import re

p = Path("src/cli/k3_worker.c")
s = p.read_text()

ALLOC_START = "typedef struct {\n    float *ptr;\n    size_t bytes;\n    int mapped;\n} WorkerVM;\n"
ALLOC_END = "/* Best-effort physical-page reclamation after a conversation branch/reset."

new_alloc = r'''typedef struct {
    float *ptr;
    size_t bytes;
    int mapped;
    int committed_rows;   /* high-water position committed/touched in every MLA slice */
} WorkerVM;

static WorkerVM worker_vm_alloc(size_t nfloats)
{
    WorkerVM m; memset(&m, 0, sizeof m);
    if (nfloats == 0 || nfloats > SIZE_MAX / sizeof(float)) return m;
    m.bytes = nfloats * sizeof(float);
#if defined(_WIN32)
    /* Reserve the address range only. The first forward that reaches a row commits it.
     * This is the Windows equivalent of the POSIX MAP_NORESERVE policy below, but unlike
     * the original compatibility mmap it consumes no pagefile commit for untouched KV. */
    void *q = k3_vm_reserve(m.bytes);
    if (q) {
        m.ptr = (float *)q;
        m.mapped = 1;
        return m;
    }
#elif defined(MAP_ANONYMOUS) || defined(MAP_ANON)
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
    /* Keep a portable fallback for ordinary contexts, but never turn a virtual-reserve
     * failure for a TB-scale mapping into a giant calloc. */
    if (m.bytes <= K3_WORKER_CALLOC_FALLBACK_MAX)
        m.ptr = (float *)calloc(nfloats, sizeof(float));
    return m;
}

/* Ensure rows [start,start+count) are writable in every MLA-layer slice. POSIX anonymous
 * mappings are already demand-paged, so tracking the high-water mark is enough there.
 * On Windows the address range is reserve-only and each layer's exact row span is
 * committed just before the model can touch it. */
static int worker_vm_commit_rows(WorkerVM *m, int nlayers, int cap,
                                 size_t row_floats, int start, int count)
{
    if (!m || !m->ptr || nlayers <= 0 || cap <= 0 || row_floats == 0 ||
        start < 0 || count < 0 || start > cap || count > cap - start)
        return -1;
    if (count == 0) return 0;
    const int target = start + count;
    if (target > m->committed_rows) m->committed_rows = target;
#if defined(_WIN32)
    if (!m->mapped) return 0; /* small calloc fallback is already committed */
    const size_t stride = (size_t)cap * row_floats * sizeof(float);
    const size_t live = (size_t)count * row_floats * sizeof(float);
    for (int L = 0; L < nlayers; L++) {
        void *a = (char *)m->ptr + (size_t)L * stride +
                  (size_t)start * row_floats * sizeof(float);
        if (k3_vm_commit_span(m->ptr, m->bytes, a, live) != 0) return -1;
    }
#else
    (void)m; (void)nlayers; (void)cap; (void)row_floats; (void)start; (void)count;
#endif
    return 0;
}

static int worker_prepare_model_kv(const Weights *w, WorkerVM *kv, WorkerVM *rope,
                                   const K3Cfg *c, int T)
{
    if (!w || !c || T < 0 || w->cached < 0 || w->cached > w->kv_cap ||
        T > w->kv_cap - w->cached)
        return -1;
    if (T == 0 || w->n_mla <= 0) return 0;
    if (worker_vm_commit_rows(kv, w->n_mla, w->kv_cap,
            (size_t)c->n_heads * (c->qk_nope + c->v_head), w->cached, T) != 0)
        return -1;
    if (worker_vm_commit_rows(rope, w->n_mla, w->kv_cap,
            (size_t)c->qk_rope, w->cached, T) != 0)
        return -1;
    return 0;
}

/* The ONLY worker entry to the shared model forward. Keeping VM preparation here means
 * speculative, replay, prefill and fallback paths cannot accidentally touch a reserved
 * but uncommitted Windows KV row. */
static int worker_forward(Weights *w, WorkerVM *kv, WorkerVM *rope,
                          const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                          float *logits_last, float *scratch, float *h, float *br,
                          float *kstate, int *arg_all, float *logits_all)
{
    if (worker_prepare_model_kv(w, kv, rope, c, T) != 0) {
        fprintf(stderr, "worker: KV commit failed for rows [%d,%d)\n",
                w ? w->cached : -1, w ? w->cached + T : -1);
        return -1;
    }
    return forward(w, c, cache, ids, T, logits_last, scratch, h, br, kstate,
                   arg_all, logits_all);
}

'''

if "static int worker_vm_commit_rows" not in s:
    a = s.find(ALLOC_START)
    b = s.find(ALLOC_END, a)
    if a < 0 or b < 0:
        raise SystemExit("WorkerVM allocation region not found")
    s = s[:a] + new_alloc + s[b:]

DISCARD_START = "static void worker_vm_discard_rows(WorkerVM *m, int nlayers, int cap,"
DISCARD_END = "static void worker_vm_free(WorkerVM *m)"
new_discard = r'''static void worker_vm_discard_rows(WorkerVM *m, int nlayers, int cap,
                                   size_t row_floats, int used)
{
    if (!m || !m->mapped || !m->ptr || nlayers <= 0 || cap <= 0 ||
        row_floats == 0 || used <= 0) {
        if (m) m->committed_rows = 0;
        return;
    }
    if (used > cap) used = cap;
    const size_t stride = (size_t)cap * row_floats * sizeof(float);
    const size_t live = (size_t)used * row_floats * sizeof(float);
#if defined(_WIN32)
    /* Return PAGEFILE commit while retaining the giant address reservation. */
    for (int L = 0; L < nlayers; L++) {
        void *a = (char *)m->ptr + (size_t)L * stride;
        (void)k3_vm_decommit_span(m->ptr, m->bytes, a, live);
    }
#elif defined(MADV_DONTNEED)
    long psl = sysconf(_SC_PAGESIZE);
    if (psl > 0) {
        const uintptr_t ps = (uintptr_t)psl;
        const uintptr_t map_lo = (uintptr_t)m->ptr;
        const uintptr_t map_hi = map_lo + m->bytes;
        for (int L = 0; L < nlayers; L++) {
            uintptr_t a = map_lo + (size_t)L * stride;
            uintptr_t b = a + live;
            uintptr_t lo = (a / ps) * ps;
            uintptr_t hi = ((b + ps - 1) / ps) * ps;
            if (lo < map_lo) lo = map_lo;
            if (hi > map_hi) hi = map_hi;
            if (hi > lo) (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
        }
    }
#else
    (void)stride; (void)live;
#endif
    m->committed_rows = 0;
}

static void worker_discard_model_kv(const Weights *w, WorkerVM *kv, WorkerVM *rope,
                                    const K3Cfg *c)
{
    if (!w || !c) return;
    worker_vm_discard_rows(kv, w->n_mla, w->kv_cap,
                           (size_t)c->n_heads * (c->qk_nope + c->v_head),
                           kv ? kv->committed_rows : 0);
    worker_vm_discard_rows(rope, w->n_mla, w->kv_cap,
                           (size_t)c->qk_rope,
                           rope ? rope->committed_rows : 0);
}

'''
if "Return PAGEFILE commit" not in s:
    a = s.find(DISCARD_START)
    b = s.find(DISCARD_END, a)
    if a < 0 or b < 0:
        raise SystemExit("discard region not found")
    s = s[:a] + new_discard + s[b:]

# Helper declarations and bodies.
s = s.replace(
    "static int worker_replay_prefix(Weights *w, const K3Cfg *c, K3Cache *cache,\n"
    "                                const int *seq, int base, int n,\n"
    "                                float *lg, float *sc, float *h, float *br, float *ks)\n",
    "static int worker_replay_prefix(Weights *w, WorkerVM *kv, WorkerVM *rope,\n"
    "                                const K3Cfg *c, K3Cache *cache,\n"
    "                                const int *seq, int base, int n,\n"
    "                                float *lg, float *sc, float *h, float *br, float *ks)\n",
)
s = s.replace(
    "    if (forward(w, c, cache, seq + base, n, lg, sc, h, br, ks, NULL, NULL) != 0)\n"
    "        return -1;\n",
    "    if (worker_forward(w, kv, rope, c, cache, seq + base, n,\n"
    "                       lg, sc, h, br, ks, NULL, NULL) != 0)\n"
    "        return -1;\n",
    1,
)
s = s.replace(
    "static int worker_prefill_to(Weights *w, const K3Cfg *c, K3Cache *cache,\n"
    "                             const int *seq, int target, int chunk,\n"
    "                             float *lg, float *sc, float *h, float *br, float *ks)\n",
    "static int worker_prefill_to(Weights *w, WorkerVM *kv, WorkerVM *rope,\n"
    "                             const K3Cfg *c, K3Cache *cache,\n"
    "                             const int *seq, int target, int chunk,\n"
    "                             float *lg, float *sc, float *h, float *br, float *ks)\n",
)
s = s.replace(
    "        if (forward(w, c, cache, seq + base, n, lg, sc, h, br, ks, NULL, NULL) != 0)\n"
    "            return -1;\n",
    "        if (worker_forward(w, kv, rope, c, cache, seq + base, n,\n"
    "                           lg, sc, h, br, ks, NULL, NULL) != 0)\n"
    "            return -1;\n",
    1,
)

# Initial prefills.
s = s.replace(
    "worker_prefill_to(&w, &c, &cache, seq, np, prefill_cap,\n"
    "                                                  lg, sc, h, br, ks)",
    "worker_prefill_to(&w, &w_kv_mem, &w_rope_mem, &c, &cache,\n"
    "                                                  seq, np, prefill_cap, lg, sc, h, br, ks)",
)
s = s.replace(
    "worker_prefill_to(&dw, &c, &cache, seq, np, prefill_cap,\n"
    "                                                     lg, sc, h, br, dks)",
    "worker_prefill_to(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n"
    "                                                     seq, np, prefill_cap, lg, sc, h, br, dks)",
)

# Direct worker forward paths.
s = s.replace(
    "if (forward(&dw, &c, &cache, &prev, 1, lg, sc, h, br,\n"
    "                                dks, NULL, NULL) != 0) break;",
    "if (worker_forward(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n"
    "                                       &prev, 1, lg, sc, h, br, dks, NULL, NULL) != 0) break;",
)
s = s.replace(
    "int frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks,\n"
    "                                  arg, temperature > 0.0 ? spec_target_logits : NULL);",
    "int frc = worker_forward(&w, &w_kv_mem, &w_rope_mem, &c, &cache,\n"
    "                                  seq + base, nd + 1, lg, sc, h, br, ks, arg,\n"
    "                                  temperature > 0.0 ? spec_target_logits : NULL);",
)
s = s.replace(
    "if (forward(&dw, &c, &cache, &last, 1, lg, sc, h, br,\n"
    "                                dks, NULL, NULL) != 0) {",
    "if (worker_forward(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n"
    "                                       &last, 1, lg, sc, h, br, dks, NULL, NULL) != 0) {",
)
s = s.replace(
    "if (forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br,\n"
    "                            ks, NULL, NULL) != 0) {",
    "if (worker_forward(&w, &w_kv_mem, &w_rope_mem, &c, &cache,\n"
    "                                   seq + base, 1, lg, sc, h, br, ks, NULL, NULL) != 0) {",
)
s = s.replace(
    "if (forward(&dw, &c, &cache, seq + base, 1, lg, sc, h, br,\n"
    "                                dks, NULL, NULL) != 0) {",
    "if (worker_forward(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n"
    "                                       seq + base, 1, lg, sc, h, br, dks, NULL, NULL) != 0) {",
)

# Replay paths.
for old, new in [
    ("worker_replay_prefix(&w, &c, &cache, seq, base, m + 1,",
     "worker_replay_prefix(&w, &w_kv_mem, &w_rope_mem, &c, &cache,\n                                             seq, base, m + 1,"),
    ("worker_replay_prefix(&dw, &c, &cache, seq, base, m + 1,",
     "worker_replay_prefix(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n                                             seq, base, m + 1,"),
    ("worker_replay_prefix(&w, &c, &cache, seq, round_base, commit_n,",
     "worker_replay_prefix(&w, &w_kv_mem, &w_rope_mem, &c, &cache,\n                                         seq, round_base, commit_n,"),
    ("worker_replay_prefix(&dw, &c, &cache, seq, round_base, commit_n,",
     "worker_replay_prefix(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,\n                                         seq, round_base, commit_n,"),
]:
    s = s.replace(old, new)

# Clean repeated Windows protocol buffering guard.
old_guard = """#if defined(_WIN32)
    setvbuf(stdout, NULL, _IONBF, 0);
#else
#if defined(_WIN32)
    setvbuf(stdout, NULL, _IONBF, 0);
#else
#if defined(_WIN32)
    setvbuf(stdout, NULL, _IONBF, 0);
#else
    setvbuf(stdout, NULL, _IOLBF, 0);
#endif
#endif
#endif"""
new_guard = """#if defined(_WIN32)
    setvbuf(stdout, NULL, _IONBF, 0);
#else
    setvbuf(stdout, NULL, _IOLBF, 0);
#endif"""
s = s.replace(old_guard, new_guard, 1)

# Hard gates before writing.
if "static int worker_forward" not in s or "k3_vm_reserve" not in s:
    raise SystemExit("worker VM wrapper missing")
if s.count("#if defined(_WIN32)\n    setvbuf(stdout, NULL, _IONBF, 0);") != 1:
    raise SystemExit("stdout guard was not normalized")

# Ignore the shared forward() inside the wrapper itself; no later worker path may call it.
wrapper_pos = s.index("static int worker_forward")
replay_pos = s.index("static int worker_replay_prefix", wrapper_pos)
post = s[replay_pos:]
residual = post.replace("worker_forward(", "")
if re.search(r"(?<![A-Za-z0-9_])forward\s*\(", residual):
    raise SystemExit("direct forward() remains after worker wrapper")

p.write_text(s)
print("worker lazy-KV transform: PASS")
