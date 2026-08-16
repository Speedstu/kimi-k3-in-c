#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'src/cli/k3_worker.c'
s=p.read_text()

s=once(s,
'''#define K3_WORKER_DEFAULT_CONTEXT 1024
#define K3_WORKER_MAX_CONTEXT 1048576
#define K3_WORKER_PREFILL_CHUNK 64
''',
'''#define K3_WORKER_DEFAULT_CONTEXT 1024
#define K3_WORKER_MAX_CONTEXT 1048576
#define K3_WORKER_PREFILL_MB 256.0
#define K3_WORKER_PREFILL_MAX 8192
''','prefill constants')

# Insert sizing helpers before worker_usage.
marker='''static void worker_usage(FILE *f)
'''
helpers=r'''/* Transient RAM used by one forward prefill batch. Persistent exact/draft KV is not
 * included; that grows with positions actually used and is reported separately. Keeping
 * this formula next to the selector lets the worker trade RAM for fewer whole-trunk
 * sweeps without guessing from token count alone. */
static size_t worker_prefill_bytes(const K3Cfg *c, int context, int maxb, int T)
{
    if (!c || T < 1 || context < 1 || maxb < 1) return SIZE_MAX;
    const size_t t = (size_t)T, E = (size_t)c->hidden;
    if (t > SIZE_MAX / E) return SIZE_MAX;
    size_t hbr = t * E;
    if ((size_t)(maxb + 1) > SIZE_MAX / hbr) return SIZE_MAX;
    hbr *= (size_t)(maxb + 1);              /* h + AttnRes residual bank */
    size_t sc = k3_layer_scratch(c, T);
    const size_t mla = k3_mla_scratch_cached(c, T, context, 1);
    if (mla > sc) sc = mla;
    if (hbr > SIZE_MAX - sc || hbr + sc > SIZE_MAX / sizeof(float)) return SIZE_MAX;
    return (hbr + sc) * sizeof(float);
}

static int worker_pick_prefill_cap(const K3Cfg *c, int context, int maxb,
                                   double budget_mb, int override)
{
    int hi = context < K3_WORKER_PREFILL_MAX ? context : K3_WORKER_PREFILL_MAX;
    if (override > 0) return override < hi ? override : hi;
    if (!(budget_mb > 0.0) || !isfinite(budget_mb)) return 1;
    const double raw = budget_mb * 1024.0 * 1024.0;
    const size_t budget = raw >= (double)SIZE_MAX ? SIZE_MAX : (size_t)raw;
    int lo = 1, best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        const size_t need = worker_prefill_bytes(c, context, maxb, mid);
        if (need <= budget) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return best > 0 ? best : 1;
}

'''
if s.count(marker)!=1: raise SystemExit(f'helper marker: {s.count(marker)}')
s=s.replace(marker,helpers+marker,1)

s=once(s,
'''        "  --context N          resident capacity, up to 1,048,576 (default 1024)\\n"
        "  --trunk-gb X         exact packed-trunk memory budget (default 3)\\n"
''',
'''        "  --context N          resident capacity, up to 1,048,576 (default 1024)\\n"
        "  --prefill-mb X       transient prefill RAM budget (default 256 MiB)\\n"
        "  --prefill-chunk N    override automatic prefill chunk (max 8192)\\n"
        "  --trunk-gb X         exact packed-trunk memory budget (default 3)\\n"
''','usage prefill options')

s=once(s,
'''    double trunk_gb = 3.0, cache_gb = 1.0, draft_gb = 32.0;
    int context = K3_WORKER_DEFAULT_CONTEXT, threads = 0, want_layers = -1;
    int draft_topk = 4, spec_n = 4, draft_cache_only = 0;
''',
'''    double trunk_gb = 3.0, cache_gb = 1.0, draft_gb = 32.0;
    double prefill_mb = K3_WORKER_PREFILL_MB;
    int context = K3_WORKER_DEFAULT_CONTEXT, threads = 0, want_layers = -1;
    int prefill_override = 0;
    int draft_topk = 4, spec_n = 4, draft_cache_only = 0;
''','prefill vars')

s=once(s,
'''        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
''',
'''        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-mb") && i + 1 < argc) prefill_mb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) prefill_override = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
''','parse prefill options')

s=once(s,
'''    if (!(trunk_gb > 0.0) || !(cache_gb > 0.0) || (draft_dir && !(draft_gb > 0.0))) {
        fprintf(stderr, "trunk/cache budgets must be > 0\\n");
        return 2;
    }
''',
'''    if (!(trunk_gb > 0.0) || !(cache_gb > 0.0) || (draft_dir && !(draft_gb > 0.0))) {
        fprintf(stderr, "trunk/cache budgets must be > 0\\n");
        return 2;
    }
    if (!(prefill_mb > 0.0) || !isfinite(prefill_mb) ||
        prefill_override < 0 || prefill_override > K3_WORKER_PREFILL_MAX) {
        fprintf(stderr, "--prefill-mb must be > 0 and --prefill-chunk in [1,%d] when set\\n",
                K3_WORKER_PREFILL_MAX);
        return 2;
    }
''','validate prefill options')

old='''    const int E = c.hidden;
    const int maxb = c.n_layers / c.attn_res_block + 2;
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * c.kda_head_dim + (size_t)3 * P * (c.conv_k - 1);
    float *ks = (float *)calloc(kper * (size_t)NL, sizeof(float));
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
    float *sc = (float *)malloc(sc_need * sizeof(float));
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
'''
new='''    const int E = c.hidden;
    const int maxb = c.n_layers / c.attn_res_block + 2;
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * c.kda_head_dim + (size_t)3 * P * (c.conv_k - 1);
    float *ks = (float *)calloc(kper * (size_t)NL, sizeof(float));
    float *spec_snap = (float *)malloc(kper * (size_t)NL * sizeof(float));

    int prefill_cap = worker_pick_prefill_cap(&c, context, maxb, prefill_mb, prefill_override);
    float *h = NULL, *br = NULL, *sc = NULL;
    size_t sc_need = 0;
    /* Allocation reality beats the estimate. If fragmentation/rlimits make the selected
     * chunk unavailable, halve it until the exact same worker can start instead of OOMing. */
    for (;;) {
        sc_need = k3_layer_scratch(&c, prefill_cap);
        const size_t cached_need = k3_mla_scratch_cached(&c, prefill_cap, context, 1);
        if (cached_need > sc_need) sc_need = cached_need;
        h = (float *)malloc((size_t)prefill_cap * E * sizeof(float));
        br = (float *)malloc((size_t)prefill_cap * maxb * E * sizeof(float));
        sc = (float *)malloc(sc_need * sizeof(float));
        if (h && br && sc) break;
        free(h); free(br); free(sc); h = br = sc = NULL;
        if (prefill_cap <= 1) break;
        prefill_cap /= 2;
    }
    const size_t prefill_bytes = worker_prefill_bytes(&c, context, maxb, prefill_cap);
    printf("resident prefill: chunk %d tokens, %.1f MiB transient (budget %.1f MiB%s)\\n",
           prefill_cap, prefill_bytes / (1024.0 * 1024.0), prefill_mb,
           prefill_override > 0 ? ", manual override" : ", auto");
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
'''
if s.count(old)!=1: raise SystemExit(f'allocation block: {s.count(old)}')
s=s.replace(old,new,1)

p.write_text(s)
print('adaptive prefill budgeting materialized')
