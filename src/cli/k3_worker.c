/* k3_worker.c - resident local Kimi K3 inference worker.
 *
 * The worker keeps the safetensors index, exact packed trunk, model head, expert cache,
 * KDA recurrence and MLA KV alive across requests.  When --draft-trunk is supplied it
 * also keeps a second (typically Q4/I8) draft trunk and its independent recurrent/KV
 * state resident.  Draft tokens are NEVER emitted directly: the exact model verifies
 * them, using the same probability-correct p/q speculative sampler as k3_run.c.
 *
 * The implementation includes k3_run.c with its main renamed so both CLIs share one
 * loader/forward/sampler implementation.  localhost HTTP/OpenAI compatibility remains
 * in local/k3_local.py and talks to this worker over stdin/stdout only.
 */
#define main k3_legacy_cli_main
#include "k3_run.c"
#undef main

#define K3_WORKER_DEFAULT_CONTEXT 1024

static void worker_usage(FILE *f)
{
    fprintf(f,
        "usage: k3-worker MODEL_DIR --trunk DIR [options]\n"
        "\n"
        "options:\n"
        "  --context N          resident conversation capacity (default 1024)\n"
        "  --trunk-gb X         exact packed-trunk memory budget (default 3)\n"
        "  --cache-gb X         shared expert-cache budget (default 1)\n"
        "  --threads N          OpenMP compute threads\n"
        "  --config PATH        explicit config.json\n"
        "  --layers N           partial-stack testing only\n"
        "  --draft-trunk DIR    resident approximate proposal trunk\n"
        "  --draft-trunk-gb X   draft trunk memory budget (default 32)\n"
        "  --draft-topk K       draft-only routed experts (default 4)\n"
        "  --spec N             proposals per exact verification sweep (default 4)\n"
        "  --draft-cache-only   draft routes only through resident cached experts\n"
        "\n"
        "stdin protocol (ASCII whitespace separated):\n"
        "  REQ id n_prompt max_tokens temperature top_p seed stop_id <n_prompt ids...>\n"
        "  RESET\n  PING\n  QUIT\n"
        "\n"
        "stdout protocol:\n"
        "  @K3READY context vocab\n"
        "  @K3TOKEN request_id token_id\n"
        "  @K3DRAFT request_id rounds proposed accepted draft_s verify_s\n"
        "  @K3DONE request_id nout cached reused_tokens seconds\n"
        "  @K3ERROR request_id code\n");
}

static void worker_reset_state(Weights *w, float *ks, size_t kper, int nl,
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

static int worker_replay_prefix(Weights *w, const K3Cfg *c, K3Cache *cache,
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

int main(int argc, char **argv)
{
    /* stdout is the machine protocol. Configure it BEFORE any loader can print: a pipe
     * is fully buffered by default, which can otherwise hold @K3READY indefinitely. */
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
    const char *trunk_dir = NULL, *cfg_path = NULL, *draft_dir = NULL;
    double trunk_gb = 3.0, cache_gb = 1.0, draft_gb = 32.0;
    int context = K3_WORKER_DEFAULT_CONTEXT, threads = 0, want_layers = -1;
    int draft_topk = 4, spec_n = 4, draft_cache_only = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_dir = argv[++i];
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) trunk_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) want_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft-trunk") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-trunk-gb") && i + 1 < argc) draft_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--draft-topk") && i + 1 < argc) draft_topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft-cache-only")) draft_cache_only = 1;
        else { fprintf(stderr, "unknown worker option %s\n", argv[i]); return 2; }
    }
    if (!trunk_dir) {
        fprintf(stderr, "k3-worker requires --trunk DIR\n");
        return 2;
    }
    if (context < 2 || context > K3_MAX_PROMPT + K3_MAX_GEN) {
        fprintf(stderr, "--context must be in [2,%d]\n", K3_MAX_PROMPT + K3_MAX_GEN);
        return 2;
    }
    if (!(trunk_gb > 0.0) || !(cache_gb > 0.0) || (draft_dir && !(draft_gb > 0.0))) {
        fprintf(stderr, "trunk/cache budgets must be > 0\n");
        return 2;
    }
    if (threads < 0 || threads > 4096) {
        fprintf(stderr, "--threads must be in [1,4096] when supplied\n");
        return 2;
    }
    if (spec_n < 1 || spec_n > K3_SPEC_MAX) {
        fprintf(stderr, "--spec must be in [1,%d]\n", K3_SPEC_MAX);
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
    if (draft_dir && (draft_topk < 1 || draft_topk > c.topk)) {
        fprintf(stderr, "--draft-topk must be in [1,%d], got %d\n", c.topk, draft_topk);
        return 2;
    }
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
    float *spec_snap = (float *)malloc(kper * (size_t)NL * sizeof(float));
    float *h = (float *)malloc((size_t)context * E * sizeof(float));
    float *br = (float *)malloc((size_t)context * maxb * E * sizeof(float));
    size_t sc_need = k3_layer_scratch(&c, context);
    const size_t cached_need = k3_mla_scratch_cached(&c, context, context, 1);
    if (cached_need > sc_need) sc_need = cached_need;
    float *sc = (float *)malloc(sc_need * sizeof(float));
    float *lg = (float *)malloc((size_t)c.vocab * sizeof(float));
    int *seq = (int *)malloc((size_t)context * sizeof(int));
    int *req = (int *)malloc((size_t)context * sizeof(int));
    if (!w.kvc || !w.ropec || !ks || !spec_snap || !h || !br || !sc || !lg || !seq || !req) {
        fprintf(stderr, "worker: buffer/KV allocation failed for context %d\n", context);
        return 1;
    }
    w.cached = 0;

    /* Optional resident proposal model. The exact and draft heads/embeddings/experts are
     * shared; only the packed trunk plus recurrent/KV state differ. */
    K3Trunk trunk_d; memset(&trunk_d, 0, sizeof trunk_d);
    Weights dw; memset(&dw, 0, sizeof dw);
    float *dks = NULL, *dsnap = NULL;
    float *spec_target_logits = NULL;
    double *spec_q_probs = NULL, *spec_p_probs = NULL;
    if (draft_dir) {
        if (k3_trunk_open(&trunk_d, draft_dir, &c, (int64_t)(draft_gb * 1e9)) != 0)
            return 1;
        if (trunk_d.n_layers < NL) {
            fprintf(stderr, "worker: draft trunk has %d layers, need %d\n", trunk_d.n_layers, NL);
            return 1;
        }
        dw.lay = (K3LayerBind *)calloc((size_t)NL, sizeof(K3LayerBind));
        dw.kvc = (float *)calloc(kvper * (size_t)w.n_mla, sizeof(float));
        dw.ropec = (float *)calloc(rpper * (size_t)w.n_mla, sizeof(float));
        dks = (float *)calloc(kper * (size_t)NL, sizeof(float));
        dsnap = (float *)malloc(kper * (size_t)NL * sizeof(float));
        spec_target_logits = (float *)malloc((size_t)(spec_n + 1) * c.vocab * sizeof(float));
        spec_q_probs = (double *)malloc((size_t)spec_n * c.vocab * sizeof(double));
        spec_p_probs = (double *)malloc((size_t)c.vocab * sizeof(double));
        if (!dw.lay || !dw.kvc || !dw.ropec || !dks || !dsnap ||
            !spec_target_logits || !spec_q_probs || !spec_p_probs) {
            fprintf(stderr, "worker: resident draft allocation failed\n");
            return 1;
        }
        dw.mb = w.mb;
        dw.trunk = &trunk_d;
        dw.n_bound = NL;
        dw.mla_slot = w.mla_slot;
        dw.n_mla = w.n_mla;
        dw.kv_cap = context;
        dw.cached = 0;
        dw.draft_mode = draft_cache_only;
        dw.draft_topk = draft_topk;
    }

    int history_len = 0;
    printf("@K3READY %d %d\n", context, c.vocab);

    char op[16];
    while (scanf("%15s", op) == 1) {
        if (!strcmp(op, "PING")) { printf("@K3PONG\n"); continue; }
        if (!strcmp(op, "RESET")) {
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
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
        if (stop_id < -1 || stop_id >= c.vocab || np > context) bad = 1;
        for (int i = 0; i < np; i++) {
            long id;
            if (scanf("%ld", &id) != 1) {
                fprintf(stderr, "worker: truncated prompt ids\n");
                goto done;
            }
            if (id < 0 || id >= c.vocab) bad = 1;
            req[i] = (int)id;
        }
        if (bad) { printf("@K3ERROR %llu 2\n", rid); continue; }

        const int reuse_tokens = (history_len > 0 && np >= history_len &&
                                  memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0)
                                 ? history_len : 0;
        if (!reuse_tokens) {
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
            history_len = 0;
        }
        memcpy(seq, req, (size_t)np * sizeof(int));
        int T = np, nout = 0, failed = 0, stop_hit = 0;
        K3Sampler sampler, draft_sampler, accept_sampler;
        k3_sampler_init(&sampler, temperature, top_p, (uint64_t)seed);
        k3_sampler_init(&draft_sampler, temperature, top_p,
                        (uint64_t)seed ^ UINT64_C(0xd6e8feb86659fd93));
        k3_sampler_init(&accept_sampler, temperature, top_p,
                        (uint64_t)seed ^ UINT64_C(0xa5a3564e27f8862b));
        long draft_rounds = 0, draft_proposed = 0, draft_accepted = 0;
        double draft_seconds = 0.0, verify_seconds = 0.0;
        k3_cache_reset_stats(&cache);
        const double started = now_s();

        /* Prefill only what is not already represented by the warm state. `cached`
         * intentionally trails the visible history by one emitted token between calls,
         * so this includes that pending token plus the new XTML/tool suffix. */
        int first_tok = -1;
        if (w.cached >= np || forward(&w, &c, &cache, seq + w.cached, np - w.cached,
                                      lg, sc, h, br, ks, NULL, NULL) != 0) {
            failed = 1;
        } else {
            w.cached = np;
            /* IMPORTANT: sample from exact logits NOW. Draft prefill reuses `lg` and
             * overwrites it. The one-shot decoder samples the target token before draft
             * prefill, so moving this below the draft forward silently changes the RNG
             * path and samples from q instead of p. */
            first_tok = k3_sample_token(&sampler, lg, c.vocab);
            if (first_tok < 0) failed = 1;
        }
        if (draft_dir && !failed) {
            if (dw.cached >= np || forward(&dw, &c, &cache, seq + dw.cached, np - dw.cached,
                                           lg, sc, h, br, dks, NULL, NULL) != 0) {
                failed = 1;
            } else {
                dw.cached = np;
            }
        }

        /* Commit only after both prefills succeed, but the token itself was sampled from
         * the exact logits before the draft was allowed to overwrite the shared buffer. */
        if (!failed) {
            seq[T++] = first_tok;
            nout++;
            printf("@K3TOKEN %llu %d\n", rid, first_tok);
            if (stop_id >= 0 && first_tok == stop_id) stop_hit = 1;
        }

        while (!failed && !stop_hit && nout < gen && T < context) {
            const int base = w.cached;
            if (base != T - 1 || (draft_dir && dw.cached != base)) {
                fprintf(stderr, "worker: internal pending-token invariant failed\n");
                failed = 1;
                break;
            }

            int emit[K3_SPEC_MAX + 1], emitn = 0;
            int round_base = base;
            int used_spec = 0;
            int d[K3_SPEC_MAX], nd = 0;
            /* Mirror k3_run.c's scheduling exactly, not just its p/q math. The one-shot
             * decoder has Tmax=np+gen+1 and only opens a speculative sweep when
             *     T + spec_n + 1 < Tmax
             * i.e. when MORE than spec_n output slots remain. It never shrinks a final
             * draft block. Continuing to speculate near max_tokens consumes additional
             * proposal/accept RNG draws and breaks same-seed parity even though the
             * marginal target distribution stays correct. Use the request-local horizon
             * here too; the worker's larger resident context is not generation budget. */
            const int request_tmax = np + gen + 1;
            const int can_full_spec = draft_dir &&
                T + spec_n + 1 < request_tmax &&
                base + spec_n + 1 <= w.kv_cap;
            const int want_drafts = can_full_spec ? spec_n : 0;

            if (want_drafts > 0) {
                used_spec = 1;
                memcpy(spec_snap, ks, kper * (size_t)NL * sizeof(float));
                memcpy(dsnap, dks, kper * (size_t)NL * sizeof(float));

                const double td = now_s();
                int prev = seq[base];
                while (nd < want_drafts) {
                    if (forward(&dw, &c, &cache, &prev, 1, lg, sc, h, br,
                                dks, NULL, NULL) != 0) break;
                    dw.cached += 1;
                    if (temperature > 0.0) {
                        double *qrow = spec_q_probs + (size_t)nd * c.vocab;
                        if (k3_sampler_distribution(&draft_sampler, lg, c.vocab, qrow) != 0)
                            break;
                        prev = k3_sample_probs(&draft_sampler, qrow, c.vocab);
                        if (prev < 0) break;
                    } else {
                        prev = argmax_(lg, c.vocab);
                    }
                    d[nd++] = prev;
                }
                draft_rounds++;
                draft_proposed += nd;
                draft_seconds += now_s() - td;
            }

            if (used_spec && nd > 0) {
                int arg[K3_SPEC_MAX + 1];
                for (int i = 0; i < nd; i++) seq[T + i] = d[i];
                const double tv = now_s();
                int frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks,
                                  arg, temperature > 0.0 ? spec_target_logits : NULL);
                if (frc != 0) {
                    failed = 1;
                    break;
                }

                int m = 0, correction = -1;
                if (temperature <= 0.0) {
                    while (m < nd && arg[m] == d[m]) m++;
                    correction = arg[m];
                } else {
                    for (; m < nd; m++) {
                        const float *plog = spec_target_logits + (size_t)m * c.vocab;
                        const double *qrow = spec_q_probs + (size_t)m * c.vocab;
                        if (k3_sampler_distribution(&sampler, plog, c.vocab,
                                                    spec_p_probs) != 0) {
                            failed = 1;
                            break;
                        }
                        const double qy = qrow[d[m]], py = spec_p_probs[d[m]];
                        if (!(qy > 0.0)) { failed = 1; break; }
                        double accept = py / qy;
                        if (accept > 1.0) accept = 1.0;
                        if (k3_sampler_uniform(&accept_sampler) >= accept) {
                            correction = k3_sample_residual(&sampler, spec_p_probs,
                                                            qrow, c.vocab);
                            break;
                        }
                    }
                    if (!failed && m == nd) {
                        const float *extra = spec_target_logits + (size_t)nd * c.vocab;
                        if (k3_sampler_distribution(&sampler, extra, c.vocab,
                                                    spec_p_probs) != 0)
                            failed = 1;
                        else
                            correction = k3_sample_probs(&sampler, spec_p_probs, c.vocab);
                    }
                }
                if (failed || correction < 0) {
                    failed = 1;
                    break;
                }

                if (m == nd) {
                    w.cached = base + nd + 1;
                } else {
                    memcpy(ks, spec_snap, kper * (size_t)NL * sizeof(float));
                    if (worker_replay_prefix(&w, &c, &cache, seq, base, m + 1,
                                             lg, sc, h, br, ks) != 0) {
                        failed = 1;
                        break;
                    }
                }
                verify_seconds += now_s() - tv;
                draft_accepted += m;

                if (m == nd) {
                    const int last = d[nd - 1];
                    if (forward(&dw, &c, &cache, &last, 1, lg, sc, h, br,
                                dks, NULL, NULL) != 0) {
                        failed = 1;
                        break;
                    }
                    dw.cached += 1;
                } else {
                    memcpy(dks, dsnap, kper * (size_t)NL * sizeof(float));
                    if (worker_replay_prefix(&dw, &c, &cache, seq, base, m + 1,
                                             lg, sc, h, br, dks) != 0) {
                        failed = 1;
                        break;
                    }
                }
                for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                emit[emitn++] = correction;
            } else {
                /* No usable draft: consume the pending token exactly and keep the draft
                 * in lockstep so a later round can resume speculation immediately. */
                if (forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br,
                            ks, NULL, NULL) != 0) {
                    failed = 1;
                    break;
                }
                w.cached = base + 1;
                const int tok = k3_sample_token(&sampler, lg, c.vocab);
                if (tok < 0) { failed = 1; break; }
                emit[emitn++] = tok;
                if (draft_dir) {
                    if (forward(&dw, &c, &cache, seq + base, 1, lg, sc, h, br,
                                dks, NULL, NULL) != 0) {
                        failed = 1;
                        break;
                    }
                    dw.cached = base + 1;
                }
            }

            int commit_n = emitn;
            const int remaining = gen - nout;
            if (commit_n > remaining) commit_n = remaining;
            for (int i = 0; i < commit_n; i++) {
                if (stop_id >= 0 && emit[i] == stop_id) {
                    commit_n = i + 1;
                    stop_hit = 1;
                    break;
                }
            }

            /* A block can compute farther than max_tokens or a stop token. If the cut is
             * inside accepted drafts, rewind both models to the round snapshot and replay
             * only the pending token + committed prefix. This is the key resident-worker
             * invariant: state after DONE corresponds exactly to the transcript the
             * client actually saw, never to uncommitted speculative work. */
            if (used_spec && nd > 0 && commit_n < emitn) {
                memcpy(ks, spec_snap, kper * (size_t)NL * sizeof(float));
                if (worker_replay_prefix(&w, &c, &cache, seq, round_base, commit_n,
                                         lg, sc, h, br, ks) != 0) {
                    failed = 1;
                    break;
                }
                memcpy(dks, dsnap, kper * (size_t)NL * sizeof(float));
                if (worker_replay_prefix(&dw, &c, &cache, seq, round_base, commit_n,
                                         lg, sc, h, br, dks) != 0) {
                    failed = 1;
                    break;
                }
            }

            for (int i = 0; i < commit_n; i++) {
                seq[T++] = emit[i];
                nout++;
                printf("@K3TOKEN %llu %d\n", rid, emit[i]);
            }
        }

        if (failed || k3_expert_drops) {
            printf("@K3ERROR %llu 1\n", rid);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) worker_reset_state(&dw, dks, kper, NL, &c);
            history_len = 0;
            k3_expert_drops = 0;
            continue;
        }
        history_len = T;
        if (draft_dir) {
            printf("@K3DRAFT %llu %ld %ld %ld %.6f %.6f\n", rid,
                   draft_rounds, draft_proposed, draft_accepted,
                   draft_seconds, verify_seconds);
        }
        printf("@K3DONE %llu %d %d %d %.6f\n", rid, nout, w.cached, reuse_tokens,
               now_s() - started);
    }

done:
    free(spec_p_probs); free(spec_q_probs); free(spec_target_logits);
    free(dsnap); free(dks);
    if (draft_dir) {
        free(dw.kvc); free(dw.ropec);
        for (int L = 0; L < dw.n_bound; L++) k3_bind_free(&dw.lay[L]);
        free(dw.lay);
        k3_trunk_close(&trunk_d);
    }
    free(req); free(seq); free(lg); free(sc); free(br); free(h); free(spec_snap); free(ks);
    free(w.kvc); free(w.ropec); free(w.mla_slot);
    k3_cache_free(&cache);
    for (int L = 0; L < w.n_bound; L++) k3_bind_free(&w.lay[L]);
    free(w.lay);
    k3_bind_model_free(&w.mb);
    k3_trunk_close(&trunk);
    k3_st_close(&st);
    return 0;
}
