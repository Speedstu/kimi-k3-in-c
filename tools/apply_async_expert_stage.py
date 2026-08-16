#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {n}')
    return s.replace(old, new, 1)

# ---- expert source API ---------------------------------------------------------------
p = root / 'include/k3/k3.h'
s = p.read_text()
s = one(s,
'''    int (*getmany)(struct K3ExpertSrc *self, int layer, const int *experts, int n);
    /* OPTIONAL: 1 if the expert is already resident (get() would read no disk), filling''',
'''    int (*getmany)(struct K3ExpertSrc *self, int layer, const int *experts, int n);
    /* OPTIONAL asynchronous form of getmany. begin returns:
     *   >0  a background batch was launched and wait MUST be called before get/resident;
     *    0  all requested experts were already resident, nothing was launched;
     *   <0  async unavailable/failed, caller should fall back to getmany/get.
     * wait returns the background getmany result. These callbacks exist so independent
     * trunk/shared-expert compute can overlap storage latency without letting any model
     * arithmetic observe a half-loaded expert. */
    int (*prefetch_begin)(struct K3ExpertSrc *self, int layer,
                          const int *experts, int n);
    int (*prefetch_wait)(struct K3ExpertSrc *self);
    /* OPTIONAL: 1 if the expert is already resident (get() would read no disk), filling''',
'expert async API')
p.write_text(s, newline='\n')

# ---- cache struct --------------------------------------------------------------------
p = root / 'src/cache/k3_cache.h'
s = p.read_text()
s = one(s,
'''    uint64_t     prefetch_reads;
    double       load_seconds;''',
'''    uint64_t     prefetch_reads;
    double       load_seconds;
    /* Async top-k prefetch overlaps this storage work with independent MoE compute.
     * The implementation state is opaque here so public users do not need pthread.h. */
    void        *async_ctx;
    int          async_io_threads;
    uint64_t     async_batches;
    double       async_wait_seconds;''',
'cache async fields')
p.write_text(s, newline='\n')

# ---- cache implementation ------------------------------------------------------------
p = root / 'src/cache/k3_cache.c'
s = p.read_text()
s = one(s,
'''#include <time.h>
#include <sys/mman.h>

#include "k3_cache.h"''',
'''#include <time.h>
#include <sys/mman.h>
#include <pthread.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "k3_cache.h"''',
'pthread includes')

# Insert async state/functions after synchronous getmany, before resident().
marker = '''/* Is this expert already resident, i.e. would get() serve it with no disk read? Used by
 * the draft model's cache-only routing to propose tokens without any expert I/O; if it
 * is resident, fill_q hands back the same bytes get() would. */'''
async_code = r'''typedef struct {
    K3Cache *cache;
    pthread_t thread;
    int active;
    int result;
    int layer, n;
    int ids[K3_MAX_TOPK];
} K3AsyncPrefetch;

static void *cache_async_main(void *arg)
{
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)arg;
#ifdef _OPENMP
    /* I/O threads mostly sleep in pread. Give the device enough queue depth without
     * spawning a second full compute-sized OpenMP team beside the model matmuls. */
    omp_set_num_threads(a->cache->async_io_threads);
#endif
    a->result = cache_getmany(&a->cache->src, a->layer, a->ids, a->n);
    return NULL;
}

static int cache_prefetch_wait(K3ExpertSrc *self);

static int cache_prefetch_begin(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    K3AsyncPrefetch *a = (K3AsyncPrefetch *)c->async_ctx;
    if (!a || n <= 0) return -1;
    if (a->active) {
        /* A previous caller forgot to wait. Do not race two batches through the same
         * LRU/cache state: finish it first, loudly but safely. */
        fprintf(stderr, "k3_cache: async prefetch begin found a previous batch active; waiting\n");
        cache_prefetch_wait(self);
    }

    int any_miss = 0;
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (key >= 0 && key < c->n_layers * c->n_experts && c->slot_of[key] < 0) {
            any_miss = 1;
            break;
        }
    }
    if (!any_miss) return 0;

    if (n > K3_MAX_TOPK) n = K3_MAX_TOPK;
    a->cache = c; a->layer = layer; a->n = n; a->result = -1;
    memcpy(a->ids, ids, (size_t)n * sizeof(int));
    if (pthread_create(&a->thread, NULL, cache_async_main, a) != 0) return -1;
    a->active = 1;
    c->async_batches++;
    return 1;
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
        /* Continuing would allow get() to race a possibly-live writer into the cache.
         * Abort instead of risking a plausible token from partially loaded weights. */
        fprintf(stderr, "k3_cache: FATAL pthread_join failed for expert prefetch (%d)\n", rc);
        abort();
    }
    a->active = 0;
    return a->result;
}

'''
if s.count(marker) != 1:
    raise RuntimeError('async insertion marker')
s = s.replace(marker, async_code + marker, 1)

# Init callbacks and async state.
s = one(s,
'''    c->src.getmany = getenv("K3_NOPREFETCH") ? NULL : cache_getmany;
    if (!c->src.getmany)
        fprintf(stderr, "k3_cache: batch prefetch DISABLED by K3_NOPREFETCH\\n");
    c->src.ctx = c;''',
'''    c->src.getmany = getenv("K3_NOPREFETCH") ? NULL : cache_getmany;
    if (!c->src.getmany)
        fprintf(stderr, "k3_cache: batch prefetch DISABLED by K3_NOPREFETCH\\n");
    c->src.ctx = c;''',
'init callback anchor')
# Allocate after all base arrays succeed, immediately before return 0.
s = one(s,
'''    for (size_t i = 0; i < nkey; i++) c->slot_of[i] = -1;
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = -1;
    return 0;''',
'''    for (size_t i = 0; i < nkey; i++) c->slot_of[i] = -1;
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = -1;

    if (c->src.getmany && !getenv("K3_NOASYNC_PREFETCH")) {
        K3AsyncPrefetch *a = (K3AsyncPrefetch *)calloc(1, sizeof *a);
        if (!a) { k3_cache_free(c); return -1; }
        c->async_ctx = a;
        c->async_io_threads = 4;
        const char *et = getenv("K3_ASYNC_IO_THREADS");
        if (et) {
            const int v = atoi(et);
            if (v >= 1 && v <= K3_MAX_TOPK) c->async_io_threads = v;
        }
        c->src.prefetch_begin = cache_prefetch_begin;
        c->src.prefetch_wait = cache_prefetch_wait;
    } else if (c->src.getmany) {
        fprintf(stderr, "k3_cache: async expert overlap DISABLED by K3_NOASYNC_PREFETCH\\n");
    }
    return 0;''',
'init async state')
# Free waits and releases ctx first.
s = one(s,
'''void k3_cache_free(K3Cache *c)
{
    free(c->arena); free(c->slot_of); free(c->key_of);''',
'''void k3_cache_free(K3Cache *c)
{
    if (c->src.prefetch_wait) c->src.prefetch_wait(&c->src);
    free(c->async_ctx);
    free(c->arena); free(c->slot_of); free(c->key_of);''',
'free async')
# Reset + report overlap stats.
s = one(s,
'''    c->prefetch_reads = 0;
}''',
'''    c->prefetch_reads = 0;
    c->async_batches = 0;
    c->async_wait_seconds = 0.0;
}''',
'reset async stats')
s = one(s,
'''    printf("  read from disk: %.2f GB in %.2f s (%.0f MB/s while loading)\\n",
           (double)c->bytes_read / 1e9, c->load_seconds,
           c->load_seconds > 0 ? (double)c->bytes_read / 1e6 / c->load_seconds : 0.0);
}''',
'''    printf("  read from disk: %.2f GB in %.2f s (%.0f MB/s while loading)\\n",
           (double)c->bytes_read / 1e9, c->load_seconds,
           c->load_seconds > 0 ? (double)c->bytes_read / 1e6 / c->load_seconds : 0.0);
    if (c->async_batches)
        printf("  async overlap : %llu batches, caller waited %.2f s after independent compute "
               "(I/O worker threads %d)\\n",
               (unsigned long long)c->async_batches, c->async_wait_seconds,
               c->async_io_threads);
}''',
'report async stats')
p.write_text(s, newline='\n')

# ---- MoE overlap ---------------------------------------------------------------------
p = root / 'src/core/k3_ops.c'
s = p.read_text()
old = '''        /* 2. down-project into the latent space */
        k3_mmw(z, xt, w->down, w->wdt, E, L);

        /* 3. the selected experts, in latent space, weighted and summed */
        for (int i = 0; i < L; i++) accL[i] = 0.0f;
        /* Hand the WHOLE top-k to the source first, so its reads can overlap. Without
         * this the loop below misses, blocks on a 17.55 MB read, computes, misses
         * again: a queue depth of one against a drive that needs depth to reach its
         * rated bandwidth. getmany is optional and may be NULL, in which case nothing
         * changes and the loop reads them one at a time exactly as before. */
        if (!w->cache_only && w->src && w->src->getmany)
            w->src->getmany(w->src, w->layer, idx, nk);
        for (int j = 0; j < nk; j++) {'''
new = '''        /* If the cache supports async top-k prefetch, start it as soon as routing has
         * identified the experts. Nothing below touches the cache until prefetch_wait. */
        int async_prefetch = 0, prefetch_known_ready = 0;
        if (!w->cache_only && w->src && w->src->prefetch_begin) {
            const int ar = w->src->prefetch_begin(w->src, w->layer, idx, nk);
            async_prefetch = ar > 0;
            prefetch_known_ready = ar == 0;
        }

        /* 2. down-project into the latent space. This is independent of expert bytes
         * and therefore overlaps the background reads when async_prefetch is active. */
        k3_mmw(z, xt, w->down, w->wdt, E, L);

        /* The shared expert is also independent of the routed experts. Compute it early
         * ONLY on the async path, into its normal disjoint scratch, then still add it at
         * the original step 6 after the routed up-projection. Arithmetic/output order is
         * unchanged; only when these independent multiplies happen in wall-clock time moves. */
        if (async_prefetch) {
            k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
            k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
            k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
            k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
            w->src->prefetch_wait(w->src);
        } else if (!w->cache_only && !prefetch_known_ready && w->src && w->src->getmany) {
            /* Async unsupported/failed: preserve the old synchronous batch path. */
            w->src->getmany(w->src, w->layer, idx, nk);
        }

        /* 3. the selected experts, in latent space, weighted and summed. At this point
         * an async batch is fully joined, so get() can never observe an inflight slot. */
        for (int i = 0; i < L; i++) accL[i] = 0.0f;
        for (int j = 0; j < nk; j++) {'''
s = one(s, old, new, 'MoE async overlap block')
# Shared expert tail: skip recompute if already done, but final add remains in same place.
s = one(s,
'''        /* 6. shared expert on the ORIGINAL full-width input, added UNWEIGHTED */
        k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
        k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
        k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
        k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        for (int i = 0; i < E; i++) ot[i] += sdn[i];''',
'''        /* 6. shared expert on the ORIGINAL full-width input, added UNWEIGHTED.
         * On async_prefetch its value was computed early to hide I/O, but this ADD stays
         * exactly here, after the routed up-projection, preserving model arithmetic. */
        if (!async_prefetch) {
            k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
            k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
            k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
            k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        }
        for (int i = 0; i < E; i++) ot[i] += sdn[i];''',
'MoE shared tail')
p.write_text(s, newline='\n')

# ---- cache test async equivalence ----------------------------------------------------
p = root / 'tests/unit/test_cache.c'
s = p.read_text()
insert_before = '''    /* ---- 2b: interleaving the two paths must not corrupt either ---- */'''
block = r'''    /* ---- 2a: async batch must publish the same bytes as synchronous getmany ---- */
    if (!cache.src.prefetch_begin || !cache.src.prefetch_wait) {
        ck(0, "async batch prefetch present", "callbacks are NULL");
    } else {
        k3_cache_reset_stats(&cache);
        int bada = 0, launched = 0;
        for (int start = 0; start + c.topk <= NE; start += c.topk) {
            int ids[16];
            for (int j = 0; j < c.topk; j++) ids[j] = start + j;
            const int ar = cache.src.prefetch_begin(&cache.src, 0, ids, c.topk);
            if (ar > 0) { launched++; cache.src.prefetch_wait(&cache.src); }
            else if (ar < 0 && cache.src.getmany) cache.src.getmany(&cache.src, 0, ids, c.topk);
            for (int j = 0; j < c.topk; j++) {
                K3ExpertQ q;
                if (cache.src.get(&cache.src, 0, ids[j], &q) != 0) { bada++; continue; }
                if (!same_expert(&st, 0, ids[j], &q)) bada++;
            }
        }
        char b[96];
        snprintf(b, sizeof b, "%d async batches launched, %d wrong", launched, bada);
        ck(bada == 0 && launched > 0, "async prefetch is byte-exact", b);
    }

'''
if s.count(insert_before) != 1:
    raise RuntimeError('cache async test marker')
s = s.replace(insert_before, block + insert_before, 1)
p.write_text(s, newline='\n')

print('staged exact async expert I/O overlap')
