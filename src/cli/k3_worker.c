/* k3_worker.c - resident local Kimi K3 inference worker.
 *
 * This is deliberately a separate binary from the human-oriented CLI.  It keeps the
 * safetensors index, packed trunk, model head and expert cache alive across requests and
 * carries the exact recurrent/KV state when the next prompt extends the previous one.
 * No network code lives here: localhost HTTP/OpenAI compatibility stays in
 * local/k3_local.py and talks to this process over stdin/stdout.
 *
 * The implementation includes k3_run.c with its main renamed so it can reuse the exact
 * loader/forward/state types.  That keeps one numerical implementation; a later source
 * split can move those helpers into a library without changing this protocol.
 */
#define main k3_legacy_cli_main
#include "k3_run.c"
#undef main

#include <errno.h>

#define K3_WORKER_DEFAULT_CONTEXT 1024

static void worker_usage(FILE *f)
{
    fprintf(f,
        "usage: k3-worker MODEL_DIR --trunk DIR [options]\n"
        "\n"
        "options:\n"
        "  --context N       resident conversation capacity (default 1024 positions)\n"
        "  --trunk-gb X      packed trunk memory budget (default 3)\n"
        "  --cache-gb X      expert-cache budget (default 1)\n"
        "  --threads N       OpenMP compute threads\n"
        "  --config PATH     explicit config.json\n"
        "  --layers N        partial-stack testing only\n"
        "\n"
        "stdin protocol (ASCII whitespace separated):\n"
        "  REQ id n_prompt max_tokens temperature top_p seed stop_id <n_prompt ids...>\n"
        "  RESET\n  PING\n  QUIT\n"
        "\n"
        "stdout protocol:\n"
        "  @K3READY context vocab\n"
        "  @K3TOKEN request_id token_id\n"
        "  @K3DONE request_id nout cached reused seconds\n"
        "  @K3ERROR request_id code\n");
}

static void worker_reset_state(Weights *w, float *ks, size_t kper, int nl,
                               const K3Cfg *c)
{
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

int main(int argc, char **argv)
{
    /* stdout is the machine protocol. Configure it BEFORE any helper can print: when
     * connected to Python it is a pipe, and the default full buffering can otherwise
     * hold @K3READY forever while the client waits for exactly that line. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { worker_usage(stderr); return 2; }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            worker_usage(stdout); return 0;
        }
        if (!strcmp(argv[i], "--version")) {
            printf("k3-worker %s\n", K3_VERSION); return 0;
        }
    }

    const char *dir = argv[1];
    const char *trunk_dir = NULL, *cfg_path = NULL;
    double trunk_gb = 3.0, cache_gb = 1.0;
    int context = K3_WORKER_DEFAULT_CONTEXT, threads = 0, want_layers = -1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_dir = argv[++i];
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) trunk_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) want_layers = atoi(argv[++i]);
        else { fprintf(stderr, "unknown worker option %s\n", argv[i]); return 2; }
    }
    if (!trunk_dir) {
        fprintf(stderr, "k3-worker requires --trunk DIR; resident-worker mode is designed "
                        "for the packed local trunk\n");
        return 2;
    }
    if (context < 2 || context > K3_MAX_PROMPT + K3_MAX_GEN) {
        fprintf(stderr, "--context must be in [2,%d]\n", K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    if (!(trunk_gb > 0.0) || !(cache_gb > 0.0)) {
        fprintf(stderr, "--trunk-gb and --cache-gb must be > 0\n");
        return 2;
    }
    if (threads < 0 || threads > 4096) {
        fprintf(stderr, "--threads must be in [1,4096] when supplied\n");
        return 2;
    }
#ifdef _OPENMP
    if (threads > 0) { omp_set_dynamic(0); omp_set_num_threads(threads); }
#else
    if (threads > 1) {
        fprintf(stderr, "this worker was built without OpenMP\n");
        return 2;
    }
#endif

    K3Cfg c; static int fa[128];
    if (!real_cfg(&c, fa, 128, dir, cfg_path)) return 2;
    const int NL = (want_layers > 0 && want_layers < c.n_layers) ? want_layers : c.n_layers;

    K3St st;
    if (k3_st_open(&st, dir) != 0) return 1;
    for (int L = 0; L < NL; L++) {
        if (k3_bind_layer_bytes(&st, &c, L) < 0) {
            fprintf(stderr, "worker: model is missing tensors for layer %d\n", L);
            k3_st_close(&st);
            return 1;
        }
    }

    Weights w; memset(&w, 0, sizeof w);
    w.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
    if (!w.lay) { k3_st_close(&st); return 1; }

    K3Trunk trunk;
    if (k3_trunk_open(&trunk, trunk_dir, &c, (int64_t)(trunk_gb * 1e9)) != 0) {
        free(w.lay); k3_st_close(&st); return 1;
    }
    if (trunk.n_layers < NL) {
        fprintf(stderr, "worker: packed trunk has %d layers, need %d\n", trunk.n_layers, NL);
        k3_trunk_close(&trunk); free(w.lay); k3_st_close(&st); return 1;
    }
    w.trunk = &trunk;
    w.n_bound = NL;
    if (k3_bind_model(&st, &c, 1, &w.mb) != 0) {
        k3_trunk_close(&trunk); free(w.lay); k3_st_close(&st); return 1;
    }

    K3Cache cache;
    if (k3_cache_init(&cache, &st, &c, (int64_t)(cache_gb * 1e9)) != 0) {
        k3_bind_model_free(&w.mb); k3_trunk_close(&trunk); free(w.lay); k3_st_close(&st);
        return 1;
    }

    w.mla_slot = (int *)malloc((size_t)NL * sizeof(int));
    if (!w.mla_slot) return 1;
    w.n_mla = 0;
    for (int L = 0; L < NL; L++) w.mla_slot[L] = k3_is_mla(&c, L) ? w.n_mla++ : -1;
    w.kv_cap = context;
    const size_t kvper = (size_t)context * c.n_heads * (c.qk_nope + c.v_head);
    const size_t rpper = (size_t)context * c.qk_rope;
    w.kvc = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
    w.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));

    const int E = c.hidden;
    const int maxb = c.n_layers / c.attn_res_block + 2;
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * c.kda_head_dim + (size_t)3 * P * (c.conv_k - 1);
    float *ks = (float *)calloc(kper * (size_t)NL, sizeof(float));
    float *h = (float *)malloc((size_t)context * E * sizeof(float));
    float *br = (float *)malloc((size_t)context * maxb * E * sizeof(float));
    size_t sc_need = k3_layer_scratch(&c, context);
    const size_t cached_need = k3_mla_scratch_cached(&c, context, context, 1);
    if (cached_need > sc_need) sc_need = cached_need;
    float *sc = (float *)malloc(sc_need * sizeof(float));
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
    int *seq = (int *)malloc((size_t)context * sizeof(int));
    int *req = (int *)malloc((size_t)context * sizeof(int));
    if (!w.kvc || !w.ropec || !ks || !h || !br || !sc || !lg || !seq || !req) {
        fprintf(stderr, "worker: buffer/KV allocation failed for context %d\n", context);
        return 1;
    }
    w.cached = 0;
    int history_len = 0;

    printf("@K3READY %d %d\n", context, c.vocab);

    char op[16];
    while (scanf("%15s", op) == 1) {
        if (!strcmp(op, "PING")) { printf("@K3PONG\n"); continue; }
        if (!strcmp(op, "RESET")) {
            worker_reset_state(&w, ks, kper, NL, &c);
            history_len = 0;
            printf("@K3RESET\n");
            continue;
        }
        if (!strcmp(op, "QUIT")) { printf("@K3BYE\n"); break; }
        if (strcmp(op, "REQ")) {
            fprintf(stderr, "worker: unknown protocol opcode '%s'\n", op);
            break;
        }

        unsigned long long rid = 0, seed = 1;
        int np = 0, gen = 0, stop_id = -1;
        double temperature = 0.0, top_p = 1.0;
        if (scanf("%llu%d%d%lf%lf%llu%d", &rid, &np, &gen, &temperature,
                  &top_p, &seed, &stop_id) != 7) {
            fprintf(stderr, "worker: truncated REQ header\n");
            break;
        }
        int bad = 0;
        if (np <= 0 || gen <= 0 || np + gen > context || gen > K3_MAX_GEN) bad = 1;
        if (!isfinite(temperature) || temperature < 0.0 ||
            !isfinite(top_p) || top_p <= 0.0 || top_p > 1.0) bad = 1;
        if (stop_id < -1 || stop_id >= c.vocab) bad = 1;
        if (np > context) bad = 1;
        for (int i = 0; i < np; i++) {
            long id;
            if (scanf("%ld", &id) != 1) { fprintf(stderr, "worker: truncated prompt ids\n"); goto done; }
            if (id < 0 || id >= c.vocab) bad = 1;
            req[i] = (int)id;
        }
        if (bad) { printf("@K3ERROR %llu 2\n", rid); continue; }

        const int reuse_tokens = (history_len > 0 && np >= history_len &&
                                  memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0)
                                 ? history_len : 0;
        if (!reuse_tokens) {
            worker_reset_state(&w, ks, kper, NL, &c);
            history_len = 0;
        }
        memcpy(seq, req, (size_t)np * sizeof(int));
        int T = np, nout = 0, failed = 0;
        K3Sampler sampler;
        k3_sampler_init(&sampler, temperature, top_p, (uint64_t)seed);
        k3_cache_reset_stats(&cache);
        const double started = now_s();

        /* `cached` intentionally trails history by one emitted token at request end.
         * Feeding [cached,np) therefore handles both a cold prompt and, on reuse, the
         * previous pending output token plus only the newly appended XTML/tool suffix. */
        if (w.cached >= np || forward(&w, &c, &cache, seq + w.cached, np - w.cached,
                                      lg, sc, h, br, ks, NULL, NULL) != 0) {
            failed = 1;
        } else {
            w.cached = np;
        }

        while (!failed && nout < gen && T < context) {
            int tok = k3_sample_token(&sampler, lg, c.vocab);
            if (tok < 0) { failed = 1; break; }
            seq[T++] = tok;
            nout++;
            printf("@K3TOKEN %llu %d\n", rid, tok);
            if (stop_id >= 0 && tok == stop_id) break;
            if (nout >= gen || T >= context) break;
            if (forward(&w, &c, &cache, seq + w.cached, 1,
                        lg, sc, h, br, ks, NULL, NULL) != 0) {
                failed = 1;
                break;
            }
            w.cached += 1;
        }

        if (failed || k3_expert_drops) {
            printf("@K3ERROR %llu 1\n", rid);
            worker_reset_state(&w, ks, kper, NL, &c);
            history_len = 0;
            k3_expert_drops = 0;
            continue;
        }
        history_len = T;
        printf("@K3DONE %llu %d %d %d %.6f\n", rid, nout, w.cached, reuse_tokens,
               now_s() - started);
    }

done:
    free(req); free(seq); free(lg); free(sc); free(br); free(h); free(ks);
    free(w.kvc); free(w.ropec); free(w.mla_slot);
    k3_cache_free(&cache);
    for (int L = 0; L < w.n_bound; L++) k3_bind_free(&w.lay[L]);
    free(w.lay);
    k3_bind_model_free(&w.mb);
    k3_trunk_close(&trunk);
    k3_st_close(&st);
    return 0;
}
