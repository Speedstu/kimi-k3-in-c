from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text(encoding='utf-8')
old = '''    /* 2. expert-major: fetch each unique expert ONCE, apply it to every (token, slot)
     * that selected it. gu/act/edn are reused per (expert, token). */
    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */
    if (w->src->getmany) w->src->getmany(w->src, w->layer, uniq, nu);
    for (int u = 0; u < nu; u++) {
        const int e = uniq[u];
        K3ExpertQ q;
        if (w->src->get(w->src, w->layer, e, &q) != 0) {
            k3_expert_drops++;
            fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                            "this chunk is CORRUPT\\n", w->layer, e);
            continue;
        }
        for (int t = 0; t < T; t++) {
            const int   *it = ridx + (size_t)t * K;
            const float *zt = zz  + (size_t)t * Ll;
            for (int j = 0; j < K; j++) {
                if (it[j] != e) continue;
                k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
            }
        }
    }
'''
new = '''    /* 2. expert-major: fetch each unique expert ONCE, apply it to every (token, slot)
     * that selected it. gu/act/edn are reused per (expert, token).
     *
     * A chunk can contain hundreds of unique experts, but cache_getmany is deliberately
     * bounded by K3_MAX_TOPK and a low-RAM cache is guaranteed to hold only topk+1 slots.
     * One giant getmany(uniq, nu) therefore batches only an initial prefix; the remaining
     * experts fall back to queue-depth-1 synchronous reads. Consume uniq in top-k-sized
     * waves instead. Every wave is guaranteed to fit the minimum cache, and the current
     * expert's MXFP4 work overlaps reads for the later experts in that wave. Arithmetic
     * order is unchanged: experts are still COMPUTED in uniq[0..nu) order and per-token
     * contributions are still accumulated later in original routing order. */
    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */
    static int no_prefill_pipeline = -1;
    if (no_prefill_pipeline < 0)
        no_prefill_pipeline = getenv("K3_NO_PREFILL_PIPELINE") ? 1 : 0;

    int wave_end = no_prefill_pipeline ? nu : 0;
    int wave_async = 0, wave_pipeline = 0;
    if (no_prefill_pipeline && w->src->getmany)
        w->src->getmany(w->src, w->layer, uniq, nu);   /* exact old I/O schedule for A/B */

    for (int u = 0; u < nu; u++) {
        if (!no_prefill_pipeline && u == wave_end) {
            wave_end = u + K;
            if (wave_end > nu) wave_end = nu;
            const int nwave = wave_end - u;
            int ar = -1;
            if (w->src->prefetch_begin)
                ar = w->src->prefetch_begin(w->src, w->layer, uniq + u, nwave);
            if (ar > 0) {
                wave_async = 1;
                wave_pipeline = w->src->prefetch_get != NULL;
                /* Older async sources publish only at whole-batch completion. Join now;
                 * pipeline-capable caches instead wait only for expert u below. */
                if (!wave_pipeline) {
                    if (w->src->prefetch_wait) w->src->prefetch_wait(w->src);
                    wave_async = 0;
                }
            } else if (ar < 0 && w->src->getmany) {
                /* Async unavailable/failed: retain concurrent synchronous batch reads,
                 * but only for this wave so later waves do not degrade to serial gets. */
                w->src->getmany(w->src, w->layer, uniq + u, nwave);
            }
        }

        const int e = uniq[u];
        K3ExpertQ q;
        const int miss = wave_pipeline
            ? w->src->prefetch_get(w->src, w->layer, e, &q) != 0
            : w->src->get(w->src, w->layer, e, &q) != 0;
        if (miss) {
            k3_expert_drops++;
            fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                            "this chunk is CORRUPT\\n", w->layer, e);
        } else {
            for (int t = 0; t < T; t++) {
                const int   *it = ridx + (size_t)t * K;
                const float *zt = zz  + (size_t)t * Ll;
                for (int j = 0; j < K; j++) {
                    if (it[j] != e) continue;
                    k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                    k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                    k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                    k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                    memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
                }
            }
        }

        if (!no_prefill_pipeline && u + 1 == wave_end && wave_async) {
            if (w->src->prefetch_wait) w->src->prefetch_wait(w->src);
            wave_async = 0;
            wave_pipeline = 0;
        }
    }
'''
if old not in s:
    raise SystemExit('prefill expert-major block not found')
s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
