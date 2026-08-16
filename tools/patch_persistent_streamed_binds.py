#!/usr/bin/env python3
from pathlib import Path
import re

hp = Path('src/io/k3_trunk.h')
h = hp.read_text()
cp = Path('src/io/k3_trunk.c')
c = cp.read_text()

if 'persistent_reuses' in h and 'trunk_slot_for_layer' in c:
    print('persistent streamed bind cache already applied')
    raise SystemExit(0)

# Header: persistent per-layer prepared bind + exact widened bytes cache. The current
# released K3 config was measured with k3_layer_mem_extra_bytes itself: all 93 worst-case
# widen arenas together are below 16 MiB. Still cap the feature at 64 MiB for arbitrary
# future configs and fall back cleanly above it.
anchor = '''    K3LayerBind  *prepared;      /* [nslot] */
    int          *prepared_layer;/* [nslot], -1 until preparation completed */
'''
repl = anchor + '''
    /* Streamed-layer persistent preparation. Raw matrices still live in ring slots;
     * widened small vectors are immutable model data, so cache their exact float bytes
     * once and restore them into the deterministic slot on later token passes. The full
     * prepared bind can then be reused without rerunning tensor lookup or widening. */
    K3LayerBind  *layer_prepared;       /* [n_layers], cached bind template */
    unsigned char *layer_prepared_valid;/* [n_layers] */
    unsigned char **layer_base;         /* raw slot base used by cached template */
    unsigned char *widen_cache;         /* n_layers * widen_bytes, if feature enabled */
    size_t        *widen_used;           /* exact used bytes per cached layer */
    size_t         widen_cache_bytes;
'''
if h.count(anchor) != 1:
    raise SystemExit(f'header prepared anchor mismatch: {h.count(anchor)}')
h = h.replace(anchor, repl, 1)

anchor = '''    uint64_t     async_prepared_builds;    /* subset prepared on reader thread     */
    double       prepare_seconds;           /* bind/widen wall across both threads  */
'''
repl = anchor + '''    uint64_t     persistent_cached_layers; /* streamed layer templates populated    */
    uint64_t     persistent_reuses;        /* binder/widen operations fully avoided */
    uint64_t     widen_restore_bytes;      /* tiny memcpy bytes restoring cache      */
'''
if h.count(anchor) != 1:
    raise SystemExit(f'header stats anchor mismatch: {h.count(anchor)}')
h = h.replace(anchor, repl, 1)
hp.write_text(h)

# C: prepare helper currently precedes trunk_copy_prepared. Add forward declaration and
# replace it with cache-aware logic. The base-address check is a correctness backstop: if
# any nonstandard caller ever maps a layer to another slot, we rebuild rather than reuse
# a template whose raw tensor pointers target the previous slot.
needle = '''static int trunk_prepare_bind(K3Trunk *tr, int L, unsigned char *base,
                              K3LayerBind *out, int async)
'''
if c.count(needle) != 1:
    raise SystemExit(f'trunk_prepare_bind signature mismatch: {c.count(needle)}')
c = c.replace(needle,
'''static void trunk_copy_prepared(const K3Cfg *c, int L, K3LayerBind *dst,
                                const K3LayerBind *src);

static int trunk_prepare_bind(K3Trunk *tr, int L, unsigned char *base,
                              K3LayerBind *out, int async)
''', 1)

m = re.search(r'''static int trunk_prepare_bind\(K3Trunk \*tr, int L, unsigned char \*base,\n                              K3LayerBind \*out, int async\)\n\{.*?\n\}\n\n/\* K3LayerW''', c, re.S)
if not m:
    raise SystemExit('cannot locate trunk_prepare_bind body')
new_body = '''static int trunk_prepare_bind(K3Trunk *tr, int L, unsigned char *base,
                              K3LayerBind *out, int async)
{
    unsigned char *widen = trunk_widen_ptr(tr, L, base);
    if (tr->widen_cache && tr->layer_prepared_valid[L] && tr->layer_base[L] == base) {
        const size_t used = tr->widen_used[L];
        if (used > (size_t)tr->widen_bytes) return -1;
        if (used) {
            memcpy(widen, tr->widen_cache + (size_t)L * (size_t)tr->widen_bytes, used);
            tr->widen_restore_bytes += used;
        }
        trunk_copy_prepared(tr->cfg, L, out, &tr->layer_prepared[L]);
        tr->persistent_reuses++;
        return 0;
    }

    Finder f; f.L = &tr->lay[L];
    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;
    size_t used = 0;
    const double t0 = now_s();
    const int rc = k3_bind_layer_mem(tr->cfg, L, out, base, &src,
                                     widen, (size_t)tr->widen_bytes, &used);
    tr->prepare_seconds += now_s() - t0;
    if (rc == 0) {
        tr->prepared_builds++;
        if (async) tr->async_prepared_builds++;
        if (tr->widen_cache && used <= (size_t)tr->widen_bytes) {
            if (used)
                memcpy(tr->widen_cache + (size_t)L * (size_t)tr->widen_bytes, widen, used);
            trunk_copy_prepared(tr->cfg, L, &tr->layer_prepared[L], out);
            tr->layer_base[L] = base;
            tr->widen_used[L] = used;
            if (!tr->layer_prepared_valid[L]) tr->persistent_cached_layers++;
            tr->layer_prepared_valid[L] = 1;
        }
    }
    return rc;
}

/* K3LayerW'''
c = c[:m.start()] + new_body + c[m.end():]

# Insert deterministic streaming-slot mapping immediately after trunk_copy_prepared.
m = re.search(r'''static void trunk_copy_prepared\(const K3Cfg \*c, int L, K3LayerBind \*dst,\n                                const K3LayerBind \*src\)\n\{.*?\n\}\n''', c, re.S)
if not m:
    raise SystemExit('cannot locate trunk_copy_prepared')
helper = '''

static int trunk_slot_for_layer(const K3Trunk *tr, int L)
{
    if (tr->nslot <= 1) return 0;
    if (tr->split_first && L == 0) return 0;
    int first = tr->npin;
    if (tr->split_first && first == 0) first = 1;
    const int rel = L - first;
    return rel > 0 ? rel % tr->nslot : 0;
}
'''
c = c[:m.end()] + helper + c[m.end():]

# Make ring selection deterministic. The engine's trunk walk is sequential; adjacent
# layers therefore ping-pong across distinct slots exactly like the former rotating ring,
# while each layer returns to the same virtual slot on the next token.
subs = 0
for pat in [r'\bint slot = tr->ring;', r'\bconst int slot = tr->ring;']:
    c, n = re.subn(pat, 'int slot = trunk_slot_for_layer(tr, L);', c)
    subs += n
if subs < 1:
    raise SystemExit('no streaming ring slot selection found')

# Allocate the tiny persistent arena after widen_bytes is known. Do not enable on future
# configs whose worst-case total exceeds 64 MiB; K3_NO_PERSIST_WIDEN is a benchmark/
# escape hatch and leaves the #40 async prebind path intact.
anchor = '''    tr->widen_bytes = (int64_t)widen;
'''
if c.count(anchor) != 1:
    raise SystemExit(f'widen_bytes anchor mismatch: {c.count(anchor)}')
alloc = anchor + '''    {
        const size_t persist = (size_t)tr->n_layers * (size_t)tr->widen_bytes;
        const size_t cap = (size_t)64 * 1024 * 1024;
        if (!getenv("K3_NO_PERSIST_WIDEN") && persist > 0 && persist <= cap) {
            tr->layer_prepared = (K3LayerBind *)calloc((size_t)tr->n_layers, sizeof(K3LayerBind));
            tr->layer_prepared_valid = (unsigned char *)calloc((size_t)tr->n_layers, 1);
            tr->layer_base = (unsigned char **)calloc((size_t)tr->n_layers, sizeof(unsigned char *));
            tr->widen_used = (size_t *)calloc((size_t)tr->n_layers, sizeof(size_t));
            tr->widen_cache = (unsigned char *)malloc(persist);
            if (!tr->layer_prepared || !tr->layer_prepared_valid || !tr->layer_base ||
                !tr->widen_used || !tr->widen_cache) return -1;
            tr->widen_cache_bytes = persist;
            printf("persistent streamed-bind cache: %.2f MiB for %d layer widen templates\\n",
                   (double)persist / (1024.0 * 1024.0), tr->n_layers);
        } else if (persist > cap) {
            printf("persistent streamed-bind cache disabled: %.2f MiB exceeds 64 MiB safety cap\\n",
                   (double)persist / (1024.0 * 1024.0));
        }
    }
'''
c = c.replace(anchor, alloc, 1)

# Free persistent storage.
anchor = '''    free(tr->prepared); free(tr->prepared_layer);
'''
if c.count(anchor) != 1:
    raise SystemExit(f'free anchor mismatch: {c.count(anchor)}')
c = c.replace(anchor, anchor + '''    free(tr->layer_prepared); free(tr->layer_prepared_valid); free(tr->layer_base);
    free(tr->widen_cache); free(tr->widen_used);
''', 1)

# Report path exercise and actual restored bytes. Add after existing prepared-bind report.
pat = re.compile(r'''(printf\("  prepared binds %llu built, %llu copied \(%llu prepared on reader\), %.3f s total\\n",\n\s*\(unsigned long long\)tr->prepared_builds,\n\s*\(unsigned long long\)tr->prepared_hits,\n\s*\(unsigned long long\)tr->async_prepared_builds, tr->prepare_seconds\);)''')
m = pat.search(c)
if not m:
    raise SystemExit('prepared bind report anchor missing')
extra = m.group(1) + '''
        if (tr->widen_cache) {
            printf("  persistent streamed binds: %llu layer templates, %llu reuses, %.3f MiB widen restored\\n",
                   (unsigned long long)tr->persistent_cached_layers,
                   (unsigned long long)tr->persistent_reuses,
                   (double)tr->widen_restore_bytes / (1024.0 * 1024.0));
        }
'''
c = c[:m.start()] + extra + c[m.end():]

cp.write_text(c)
print(f'applied persistent streamed bind cache; deterministic slot substitutions={subs}')
