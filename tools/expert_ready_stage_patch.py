#!/usr/bin/env python3
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    if new in s:
        print(path, "already patched")
        return
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected exactly one match, got {n}")
    p.write_text(s.replace(old, new, 1))
    print(path, "patched")


def replace_between(path: str, start: str, end: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    a = s.find(start)
    b = s.find(end, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        raise SystemExit(f"{path}: replacement markers missing")
    if s.find(start, a + len(start)) >= 0:
        raise SystemExit(f"{path}: start marker is ambiguous")
    p.write_text(s[:a] + new + s[b:])
    print(path, "block patched")


# Public optional callback: consume one expert as soon as its own background read is ready.
replace_once(
    "include/k3/k3.h",
    """    int (*prefetch_begin)(struct K3ExpertSrc *self, int layer,\n                          const int *experts, int n);\n    int (*prefetch_wait)(struct K3ExpertSrc *self);\n""",
    """    int (*prefetch_begin)(struct K3ExpertSrc *self, int layer,\n                          const int *experts, int n);\n    /* OPTIONAL pipelined consumer. Valid only after begin() returned >0. It waits for\n     * THIS expert, not the whole batch, then returns the same exact bytes get() would.\n     * The caller must still call prefetch_wait() once after consuming the batch. This\n     * lets routed expert j compute while j+1..K are still loading, without changing\n     * the top-k accumulation order. NULL preserves the old whole-batch wait. */\n    int (*prefetch_get)(struct K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out);\n    int (*prefetch_wait)(struct K3ExpertSrc *self);\n""",
)

replace_once(
    "src/cache/k3_cache.h",
    """    uint64_t     async_batches;\n    double       async_wait_seconds;\n    uint32_t    *hist;            /* [n_layers*n_experts] request counts       */\n""",
    """    uint64_t     async_batches;\n    double       async_wait_seconds;       /* final whole-batch join wait             */\n    double       async_ready_wait_seconds; /* waits for the next top-k expert only    */\n    uint32_t    *hist;            /* [n_layers*n_experts] request counts       */\n""",
)

# Replace only the async implementation. Synchronous getmany remains the reference path.
start = "typedef struct {\n    K3Cache *cache;\n    pthread_t thread;\n    int active;\n"
end = "/* Is this expert already resident, i.e. would get() serve it with no disk read? Used by\n"
new_block = r'''typedef struct {
    int slot, expert;
    K3ExpertRef r;
    int64_t got, pad;
    int state;                         /* 0 reading, 1 published, -1 failed */
} K3AsyncWork;

typedef struct {
    K3Cache *cache;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int sync_init;
    int active;
    int result;
    int finished;
    int layer, nw, ok;
    K3AsyncWork w[K3_MAX_TOPK];
} K3AsyncPrefetch;

static int cache_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out);
static int cache_prefetch_wait(K3ExpertSrc *self);

/* Background side of the expert pipeline.
 *
 * Slots are RESERVED serially by begin() before this thread starts. Each worker therefore
 * owns a distinct destination. As soon as one read completes, publish JUST that slot under
 * the metadata mutex and wake the model thread. The model still consumes top-k in j order;
 * only storage completion is out of order. */
static void *cache_async_main(void *arg)
{
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)arg;
    K3Cache *c = a->cache;
#ifdef _OPENMP
    /* I/O workers mostly sleep in pread. Keep their team separate from compute. */
    omp_set_num_threads(c->async_io_threads);
#endif
    const double t0 = now_s();
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < a->nw; i++) {
        K3AsyncWork *w = &a->w[i];
        int64_t pad = 0;
        const int64_t got = k3_expert_load_direct(
            c->st, &w->r, c->arena + (size_t)w->slot * c->slot_bytes,
            c->slot_bytes, &pad);

        pthread_mutex_lock(&a->mu);
        w->got = got;
        w->pad = pad;
        if (got == w->r.nbytes) {
            const int32_t key = a->layer * c->n_experts + w->expert;
            c->ref[w->slot] = w->r;
            c->pad[w->slot] = (int32_t)pad;
            c->key_of[w->slot] = key;
            c->slot_of[key] = w->slot;
            c->used_at[w->slot] = ++c->clock;
            c->bytes_read += (uint64_t)got;
            c->prefetch_reads++;
            w->state = 1;
            a->ok++;
        } else {
            fprintf(stderr, "k3_cache: short async prefetch of L%d expert %d (%lld of %lld); "
                            "releasing its slot\n",
                    a->layer, w->expert, (long long)got, (long long)w->r.nbytes);
            c->key_of[w->slot] = K3_SLOT_EMPTY;
            w->state = -1;
        }
        pthread_cond_broadcast(&a->cv);
        pthread_mutex_unlock(&a->mu);
    }

    pthread_mutex_lock(&a->mu);
    c->load_seconds += now_s() - t0;
    a->result = a->ok;
    a->finished = 1;
    pthread_cond_broadcast(&a->cv);
    pthread_mutex_unlock(&a->mu);
    return NULL;
}

static int cache_prefetch_begin(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)c->async_ctx;
    if (!a || n <= 0) return -1;
    if (a->active) {
        fprintf(stderr, "k3_cache: async prefetch begin found a previous batch active; waiting\n");
        cache_prefetch_wait(self);
    }
    if (n > K3_MAX_TOPK) n = K3_MAX_TOPK;

    a->cache = c;
    a->layer = layer;
    a->nw = 0;
    a->ok = 0;
    a->result = -1;
    a->finished = 0;

    /* Reserve every miss BEFORE launching readers. This is the same serial victim phase
     * as cache_getmany(), and it guarantees that no ready expert can be evicted by a
     * later miss from this top-k while the model is multiplying it. */
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->slot_of[key] >= 0) continue;

        int dup = 0;
        for (int j = 0; j < a->nw; j++) if (a->w[j].expert == e) { dup = 1; break; }
        if (dup) continue;

        K3ExpertRef r;
        if (k3_expert_ref(c->st, layer, e, &r) != 0 || r.nbytes > c->slot_bytes) continue;
        const int slot = pick_victim(c);
        if (slot < 0) break;
        if (c->key_of[slot] >= 0) {
            c->slot_of[c->key_of[slot]] = -1;
            c->evictions++;
        }
        c->key_of[slot] = K3_SLOT_INFLIGHT;
        c->used_at[slot] = ++c->clock;

        K3AsyncWork *w = &a->w[a->nw++];
        memset(w, 0, sizeof *w);
        w->slot = slot;
        w->expert = e;
        w->r = r;
        w->got = -1;
    }
    if (a->nw == 0) return 0;             /* all selected experts were resident */

    /* Submit in physical order for the same mostly-forward disk sweep as getmany(). */
    for (int i = 1; i < a->nw; i++) {
        K3AsyncWork t = a->w[i];
        int j = i - 1;
        while (j >= 0 && (a->w[j].r.shard > t.r.shard ||
               (a->w[j].r.shard == t.r.shard && a->w[j].r.off > t.r.off))) {
            a->w[j + 1] = a->w[j];
            j--;
        }
        a->w[j + 1] = t;
    }

    if (pthread_create(&a->thread, NULL, cache_async_main, a) != 0) {
        /* Correctness-first fallback: the evicted old cache entries are gone, but every
         * reservation is released so synchronous getmany/get can refill safely. */
        for (int i = 0; i < a->nw; i++)
            if (c->key_of[a->w[i].slot] == K3_SLOT_INFLIGHT)
                c->key_of[a->w[i].slot] = K3_SLOT_EMPTY;
        return -1;
    }
    a->active = 1;
    c->async_batches++;
    return 1;
}

/* Wait for exactly one routed expert, then serve it through cache_get while holding the
 * same metadata mutex used by async publication. cache_get therefore records the SAME
 * histogram/trace/hit accounting as the old post-join path, with no slot_of/clock race. */
static int cache_prefetch_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)c->async_ctx;
    if (!a || !a->active || layer != a->layer)
        return cache_get(self, layer, expert, out);

    pthread_mutex_lock(&a->mu);
    int wi = -1;
    for (int i = 0; i < a->nw; i++)
        if (a->w[i].expert == expert) { wi = i; break; }

    if (wi >= 0) {
        const double t0 = now_s();
        while (a->w[wi].state == 0 && !a->finished)
            pthread_cond_wait(&a->cv, &a->mu);
        c->async_ready_wait_seconds += now_s() - t0;
    }

    /* On success this is an immediate cache hit. On a failed read it performs the normal
     * exact synchronous fallback. Holding mu during that rare fallback blocks metadata
     * publication, never the already-issued disk reads, and avoids a data race. */
    const int rc = cache_get(self, layer, expert, out);
    pthread_mutex_unlock(&a->mu);
    return rc;
}

static int cache_prefetch_wait(K3ExpertSrc *self)
{
    K3Cache *c = (K3Cache *)self;
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)c->async_ctx;
    if (!a || !a->active) return 0;
    const double t0 = now_s();
    const int rc = pthread_join(a->thread, NULL);
    c->async_wait_seconds += now_s() - t0;
    if (rc != 0) {
        fprintf(stderr, "k3_cache: FATAL pthread_join failed for expert prefetch (%d)\n", rc);
        abort();
    }
    a->active = 0;
    return a->result;
}

'''
replace_between("src/cache/k3_cache.c", start, end, new_block)

# Initialise/destroy the publication mutex+condition and expose the optional callback.
replace_once(
    "src/cache/k3_cache.c",
    """        K3AsyncPrefetch *a = (K3AsyncPrefetch *)calloc(1, sizeof *a);\n        if (!a) { k3_cache_free(c); return -1; }\n        c->async_ctx = a;\n        c->async_io_threads = 4;\n""",
    """        K3AsyncPrefetch *a = (K3AsyncPrefetch *)calloc(1, sizeof *a);\n        if (!a) { k3_cache_free(c); return -1; }\n        if (pthread_mutex_init(&a->mu, NULL) != 0) { free(a); k3_cache_free(c); return -1; }\n        if (pthread_cond_init(&a->cv, NULL) != 0) {\n            pthread_mutex_destroy(&a->mu); free(a); k3_cache_free(c); return -1;\n        }\n        a->sync_init = 1;\n        c->async_ctx = a;\n        c->async_io_threads = 4;\n""",
)
replace_once(
    "src/cache/k3_cache.c",
    """        c->src.prefetch_begin = cache_prefetch_begin;\n        c->src.prefetch_wait = cache_prefetch_wait;\n""",
    """        c->src.prefetch_begin = cache_prefetch_begin;\n        c->src.prefetch_wait = cache_prefetch_wait;\n        if (!getenv("K3_NO_EXPERT_PIPELINE")) c->src.prefetch_get = cache_prefetch_get;\n        else fprintf(stderr, "k3_cache: per-expert pipeline DISABLED by K3_NO_EXPERT_PIPELINE\\n");\n""",
)
replace_once(
    "src/cache/k3_cache.c",
    """    if (c->src.prefetch_wait) c->src.prefetch_wait(&c->src);\n    free(c->async_ctx);\n""",
    """    if (c->src.prefetch_wait) c->src.prefetch_wait(&c->src);\n    if (c->async_ctx) {\n        K3AsyncPrefetch *a = (K3AsyncPrefetch *)c->async_ctx;\n        if (a->sync_init) { pthread_cond_destroy(&a->cv); pthread_mutex_destroy(&a->mu); }\n    }\n    free(c->async_ctx);\n""",
)
replace_once(
    "src/cache/k3_cache.c",
    """    c->async_batches = 0;\n    c->async_wait_seconds = 0.0;\n""",
    """    c->async_batches = 0;\n    c->async_wait_seconds = 0.0;\n    c->async_ready_wait_seconds = 0.0;\n""",
)
replace_once(
    "src/cache/k3_cache.c",
    """        printf("  async overlap : %llu batches, caller waited %.2f s after independent compute "\n               "(I/O worker threads %d)\\n",\n               (unsigned long long)c->async_batches, c->async_wait_seconds,\n               c->async_io_threads);\n""",
    """        printf("  async overlap : %llu batches, final join wait %.2f s, next-expert waits %.2f s "\n               "(I/O worker threads %d)\\n",\n               (unsigned long long)c->async_batches, c->async_wait_seconds,\n               c->async_ready_wait_seconds, c->async_io_threads);\n""",
)

# Do not globally join before routed compute when per-expert readiness is available.
replace_once(
    "src/core/k3_ops.c",
    """            k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);\n            w->src->prefetch_wait(w->src);\n""",
    """            k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);\n            /* Old sources still require a whole-batch barrier. A pipeline-capable cache\n             * lets the routed loop below wait only for expert j while j+1..K keep loading. */\n            if (!w->src->prefetch_get) w->src->prefetch_wait(w->src);\n""",
)
replace_once(
    "src/core/k3_ops.c",
    """        /* 3. the selected experts, in latent space, weighted and summed. At this point\n         * an async batch is fully joined, so get() can never observe an inflight slot. */\n        for (int i = 0; i < L; i++) accL[i] = 0.0f;\n""",
    """        /* 3. Selected experts stay in the ORIGINAL top-k order. With prefetch_get,\n         * only the current expert must have landed; later reads continue while its three\n         * MXFP4 matmuls run. This changes wall-clock overlap, never accumulation order. */\n        const int expert_pipeline = async_prefetch && w->src && w->src->prefetch_get;\n        for (int i = 0; i < L; i++) accL[i] = 0.0f;\n""",
)
replace_once(
    "src/core/k3_ops.c",
    """                int miss = w->cache_only\n                    ? !w->src->resident(w->src, w->layer, idx[j], &q)\n                    : (w->src->get(w->src, w->layer, idx[j], &q) != 0);\n""",
    """                int miss = w->cache_only\n                    ? !w->src->resident(w->src, w->layer, idx[j], &q)\n                    : (expert_pipeline\n                       ? w->src->prefetch_get(w->src, w->layer, idx[j], &q) != 0\n                       : w->src->get(w->src, w->layer, idx[j], &q) != 0);\n""",
)
replace_once(
    "src/core/k3_ops.c",
    """            const float wj = wt[j];\n            for (int i = 0; i < L; i++) accL[i] += wj * edn[i];\n        }\n\n        /* 4. RMSNorm the AGGREGATE (not per expert), then 5. up-project */\n""",
    """            const float wj = wt[j];\n            for (int i = 0; i < L; i++) accL[i] += wj * edn[i];\n        }\n        if (expert_pipeline) w->src->prefetch_wait(w->src);\n\n        /* 4. RMSNorm the AGGREGATE (not per expert), then 5. up-project */\n""",
)

# Cache-level stress gate: consume async experts one-by-one before the whole batch joins.
replace_once(
    "tests/unit/test_cache.c",
    """    /* ---- 2b: interleaving the two paths must not corrupt either ---- */\n""",
    r'''    /* ---- 2aa: pipeline consumption before whole-batch join must be byte-exact ---- */
    if (!cache.src.prefetch_begin || !cache.src.prefetch_get || !cache.src.prefetch_wait) {
        ck(0, "per-expert pipeline present", "callbacks are NULL");
    } else {
        k3_cache_reset_stats(&cache);
        int badp = 0, launchedp = 0;
        for (int pass = 0; pass < 4; pass++) {
            for (int start = 0; start + c.topk <= NE; start += c.topk) {
                int ids[16];
                for (int j = 0; j < c.topk; j++) ids[j] = (start + j + pass) % NE;
                const int ar = cache.src.prefetch_begin(&cache.src, 0, ids, c.topk);
                if (ar > 0) {
                    launchedp++;
                    for (int j = 0; j < c.topk; j++) {
                        K3ExpertQ q;
                        if (cache.src.prefetch_get(&cache.src, 0, ids[j], &q) != 0) {
                            badp++; continue;
                        }
                        if (!same_expert(&st, 0, ids[j], &q)) badp++;
                    }
                    cache.src.prefetch_wait(&cache.src);
                } else {
                    for (int j = 0; j < c.topk; j++) {
                        K3ExpertQ q;
                        if (cache.src.get(&cache.src, 0, ids[j], &q) != 0 ||
                            !same_expert(&st, 0, ids[j], &q)) badp++;
                    }
                }
            }
        }
        char b[112];
        snprintf(b, sizeof b, "%d pipelined batches launched, %d wrong", launchedp, badp);
        ck(badp == 0 && launchedp > 0, "per-expert pipeline is byte-exact", b);
    }

    /* ---- 2b: interleaving the two paths must not corrupt either ---- */
''',
)

print("expert-ready pipeline source patch complete")
