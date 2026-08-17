#!/usr/bin/env python3
"""Stage the exact mixed text/media embedding prefill path for K3.

The released K3 multimodal model expands each media placeholder into projected vision
features, then feeds the resulting hidden-size embedding sequence into the SAME language
model.  This patch gives the resident C worker an equivalent REQMM protocol without
changing any language-model arithmetic or the existing REQ protocol.
"""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: anchor count {n}, expected 1\n--- anchor ---\n{old}")
    p.write_text(s.replace(old, new, 1))


# ---------------------------------------------------------------- language core
# The external rows are already projector outputs in the language model's embedding
# space.  Text-only callers leave this NULL and execute the byte-for-byte old path.
replace_once(
    "src/cli/k3_run.c",
    """    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
    int          draft_topk;   /* 0 exact; >0 proposal-only reduced expert top-k */
} Weights;
""",
    """    int          draft_mode;   /* 1 for the hybrid draft: cache-only expert routing */
    int          draft_topk;   /* 0 exact; >0 proposal-only reduced expert top-k */
    /* Optional already-embedded rows for the NEXT forward only.  The resident multimodal
     * prefill sets this to a chunk of the official merged text/image embedding sequence;
     * every text-only call leaves it NULL and follows the historical embedding lookup. */
    const float *input_embeds;
} Weights;
""",
)

replace_once(
    "src/cli/k3_run.c",
    """    for (int t = 0; t < T; t++)
        k3_embed_row(h + (size_t)t * E, w->mb.embed, w->mb.wdt, ids[t], E);

    memset(br, 0, (size_t)T * maxb * E * sizeof(float));
""",
    """    if (w->input_embeds) {
        memcpy(h, w->input_embeds, (size_t)T * E * sizeof(float));
    } else {
        if (!ids) return -1;
        for (int t = 0; t < T; t++)
            k3_embed_row(h + (size_t)t * E, w->mb.embed, w->mb.wdt, ids[t], E);
    }

    memset(br, 0, (size_t)T * maxb * E * sizeof(float));
""",
)

# ---------------------------------------------------------------- worker helpers
replace_once(
    "src/cli/k3_worker.c",
    """static int worker_prefill_to(Weights *w, WorkerVM *kv, WorkerVM *rope,
                             const K3Cfg *c, K3Cache *cache,
                             const int *seq, int target, int chunk,
                             float *lg, float *sc, float *h, float *br, float *ks)
{
    if (!w || w->cached < 0 || w->cached >= target || chunk < 1) return -1;
    while (w->cached < target) {
        const int base = w->cached;
        int n = target - base;
        if (n > chunk) n = chunk;
        if (worker_forward(w, kv, rope, c, cache, seq + base, n,
                           lg, sc, h, br, ks, NULL, NULL) != 0)
            return -1;
        w->cached = base + n;
    }
    return 0;
}
""",
    """static int worker_prefill_to(Weights *w, WorkerVM *kv, WorkerVM *rope,
                             const K3Cfg *c, K3Cache *cache,
                             const int *seq, int target, int chunk,
                             float *lg, float *sc, float *h, float *br, float *ks)
{
    if (!w || w->cached < 0 || w->cached >= target || chunk < 1) return -1;
    while (w->cached < target) {
        const int base = w->cached;
        int n = target - base;
        if (n > chunk) n = chunk;
        if (worker_forward(w, kv, rope, c, cache, seq + base, n,
                           lg, sc, h, br, ks, NULL, NULL) != 0)
            return -1;
        w->cached = base + n;
    }
    return 0;
}

/* Same recurrent/KV path as worker_prefill_to, but the first-layer rows are already in
 * K3 hidden space.  This is precisely where the official K3 implementation enters the
 * language model after MoonViT + mm_projector + placeholder expansion. */
static int worker_prefill_embeds_to(Weights *w, WorkerVM *kv, WorkerVM *rope,
                                    const K3Cfg *c, K3Cache *cache,
                                    const float *embeds, int target, int chunk,
                                    float *lg, float *sc, float *h, float *br, float *ks)
{
    if (!w || !embeds || w->cached < 0 || w->cached >= target || chunk < 1) return -1;
    while (w->cached < target) {
        const int base = w->cached;
        int n = target - base;
        if (n > chunk) n = chunk;
        w->input_embeds = embeds + (size_t)base * c->hidden;
        const int rc = worker_forward(w, kv, rope, c, cache, NULL, n,
                                      lg, sc, h, br, ks, NULL, NULL);
        w->input_embeds = NULL;
        if (rc != 0) return -1;
        w->cached = base + n;
    }
    return 0;
}

/* REQMM feature sidecar.  Text token embeddings remain owned by C; the sidecar contains
 * only projected image rows, already rounded to the exact dtype used by the official
 * vision/projector path and serialized as their exact float32 values. */
typedef struct {
    float **image;
    int *length;
    int nimage;
} WorkerMedia;

static void worker_media_free(WorkerMedia *m)
{
    if (!m) return;
    for (int i = 0; i < m->nimage; i++) free(m->image ? m->image[i] : NULL);
    free(m->image); free(m->length);
    memset(m, 0, sizeof *m);
}

static int worker_read_u32le(FILE *f, uint32_t *out)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}

static int worker_media_load(const char *path, int hidden, int context, WorkerMedia *m)
{
    memset(m, 0, sizeof *m);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned char magic[8]; uint32_t version = 0, h = 0, ni = 0, reserved = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "K3MMF1\\0", 7) != 0 ||
        worker_read_u32le(f, &version) || worker_read_u32le(f, &h) ||
        worker_read_u32le(f, &ni) || worker_read_u32le(f, &reserved) ||
        version != 1 || h != (uint32_t)hidden || ni == 0 || ni > (uint32_t)context) {
        fclose(f); return -1;
    }
    (void)reserved;
    m->nimage = (int)ni;
    m->image = (float **)calloc((size_t)m->nimage, sizeof(float *));
    m->length = (int *)calloc((size_t)m->nimage, sizeof(int));
    if (!m->image || !m->length) { fclose(f); worker_media_free(m); return -1; }
    size_t total_rows = 0;
    for (int i = 0; i < m->nimage; i++) {
        uint32_t nr = 0;
        if (worker_read_u32le(f, &nr) || nr == 0 || nr > (uint32_t)context ||
            total_rows > (size_t)context - nr ||
            (size_t)nr > SIZE_MAX / (size_t)hidden ||
            (size_t)nr * (size_t)hidden > SIZE_MAX / sizeof(float)) {
            fclose(f); worker_media_free(m); return -1;
        }
        total_rows += nr;
        m->length[i] = (int)nr;
        const size_t nf = (size_t)nr * (size_t)hidden;
        m->image[i] = (float *)malloc(nf * sizeof(float));
        if (!m->image[i] || fread(m->image[i], sizeof(float), nf, f) != nf) {
            fclose(f); worker_media_free(m); return -1;
        }
    }
    if (fgetc(f) != EOF) { fclose(f); worker_media_free(m); return -1; }
    fclose(f);
    return 0;
}

static float *worker_merge_media(const Weights *w, const K3Cfg *c,
                                 const int *ids, int np, int placeholder,
                                 const WorkerMedia *m, int context, int *nout)
{
    int found = 0; size_t rows = 0;
    for (int i = 0; i < np; i++) {
        if (ids[i] == placeholder) {
            if (found >= m->nimage) return NULL;
            if (rows > (size_t)context - (size_t)m->length[found]) return NULL;
            rows += (size_t)m->length[found++];
        } else {
            if (rows >= (size_t)context) return NULL;
            rows++;
        }
    }
    if (found != m->nimage || rows == 0 || rows > (size_t)context ||
        rows > SIZE_MAX / (size_t)c->hidden ||
        rows * (size_t)c->hidden > SIZE_MAX / sizeof(float)) return NULL;
    float *out = (float *)malloc(rows * (size_t)c->hidden * sizeof(float));
    if (!out) return NULL;
    size_t p = 0; int im = 0;
    for (int i = 0; i < np; i++) {
        if (ids[i] == placeholder) {
            const size_t nf = (size_t)m->length[im] * c->hidden;
            memcpy(out + p * (size_t)c->hidden, m->image[im], nf * sizeof(float));
            p += (size_t)m->length[im++];
        } else {
            k3_embed_row(out + p * (size_t)c->hidden, w->mb.embed, w->mb.wdt,
                         ids[i], c->hidden);
            p++;
        }
    }
    *nout = (int)rows;
    return out;
}
""",
)

# Protocol documentation.
replace_once(
    "src/cli/k3_worker.c",
    """        "  REQ id n_prompt max_tokens temperature top_p seed stop_id <n_prompt ids...>\\n"
        "  RESET\\n  PING\\n  QUIT\\n"
""",
    """        "  REQ id n_prompt max_tokens temperature top_p seed stop_id <n_prompt ids...>\\n"
        "  REQMM id n_prompt max_tokens temperature top_p seed stop_id placeholder "
        "FEATURE_FILE <n_prompt ids...>\\n"
        "    FEATURE_FILE: K3MMF1 little-endian projected media rows; REQMM always "
        "re-prefills the full merged sequence\\n"
        "  RESET\\n  PING\\n  QUIT\\n"
""",
)

# Accept REQMM and parse its two extra fields.
replace_once(
    "src/cli/k3_worker.c",
    """        if (strcmp(op, "REQ")) {
            fprintf(stderr, "worker: unknown protocol opcode '%s'\\n", op);
            break;
        }

        unsigned long long rid = 0, seed = 1;
        int np = 0, gen = 0, stop_id = -1;
        double temperature = 0.0, top_p = 1.0;
        if (scanf("%llu%d%d%lf%lf%llu%d", &rid, &np, &gen, &temperature,
                  &top_p, &seed, &stop_id) != 7) {
            fprintf(stderr, "worker: truncated REQ header\\n");
            break;
        }
        int bad = 0;
        if (np <= 0 || gen <= 0 || np + gen > context) bad = 1;
""",
    """        const int is_mm = !strcmp(op, "REQMM");
        if (strcmp(op, "REQ") && !is_mm) {
            fprintf(stderr, "worker: unknown protocol opcode '%s'\\n", op);
            break;
        }

        unsigned long long rid = 0, seed = 1;
        int np = 0, gen = 0, stop_id = -1, media_placeholder = -1;
        char media_path[4096]; media_path[0] = 0;
        double temperature = 0.0, top_p = 1.0;
        if (scanf("%llu%d%d%lf%lf%llu%d", &rid, &np, &gen, &temperature,
                  &top_p, &seed, &stop_id) != 7) {
            fprintf(stderr, "worker: truncated request header\\n");
            break;
        }
        if (is_mm && scanf("%d%4095s", &media_placeholder, media_path) != 2) {
            fprintf(stderr, "worker: truncated REQMM media header\\n");
            break;
        }
        int bad = 0;
        if (np <= 0 || gen <= 0 || (!is_mm && np + gen > context)) bad = 1;
""",
)

# Add MM validation and construct the exact merged embedding sequence.
replace_once(
    "src/cli/k3_worker.c",
    """        if (stop_id < -1 || stop_id >= c.vocab || np > context) bad = 1;
        for (int i = 0; i < np; i++) {
            long id;
            if (scanf("%ld", &id) != 1) {
                fprintf(stderr, "worker: truncated prompt ids\\n");
                goto done;
            }
            if (id < 0 || id >= c.vocab) bad = 1;
            req[i] = (int)id;
        }
        if (bad) { printf("@K3ERROR %llu 2\\n", rid); continue; }

        const int reuse_tokens = (history_len > 0 && np >= history_len &&
                                  memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0)
                                 ? history_len : 0;
""",
    """        if (stop_id < -1 || stop_id >= c.vocab || np > context) bad = 1;
        if (is_mm && (media_placeholder < 0 || media_placeholder >= c.vocab)) bad = 1;
        for (int i = 0; i < np; i++) {
            long id;
            if (scanf("%ld", &id) != 1) {
                fprintf(stderr, "worker: truncated prompt ids\\n");
                goto done;
            }
            if (id < 0 || id >= c.vocab) bad = 1;
            req[i] = (int)id;
        }

        float *mixed_embeds = NULL;
        int prompt_positions = np;
        if (is_mm && !bad) {
            WorkerMedia media;
            if (worker_media_load(media_path, c.hidden, context, &media) != 0) {
                bad = 1;
            } else {
                mixed_embeds = worker_merge_media(&w, &c, req, np, media_placeholder,
                                                  &media, context, &prompt_positions);
                worker_media_free(&media);
                if (!mixed_embeds || prompt_positions + gen > context) bad = 1;
            }
        }
        if (bad) {
            free(mixed_embeds);
            printf("@K3ERROR %llu 2\\n", rid);
            continue;
        }

        /* Media occupies a variable number of language-model positions, so token-prefix
         * identity is not sufficient to reuse recurrent/KV state.  Re-prefill REQMM
         * exactly; ordinary text REQ retains the existing warm-prefix optimization. */
        const int reuse_tokens = (!is_mm && history_len > 0 && np >= history_len &&
                                  memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0)
                                 ? history_len : 0;
""",
)

# Force reset for MM, and make seq/T represent language-model positions.
replace_once(
    "src/cli/k3_worker.c",
    """        if (!reuse_tokens) {
            worker_discard_model_kv(&w, &w_kv_mem, &w_rope_mem, &c);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) {
                worker_discard_model_kv(&dw, &d_kv_mem, &d_rope_mem, &c);
                worker_reset_state(&dw, dks, kper, NL, &c);
            }
            history_len = 0;
        }
        memcpy(seq, req, (size_t)np * sizeof(int));
        int T = np, nout = 0, failed = 0, stop_hit = 0;
""",
    """        if (!reuse_tokens) {
            worker_discard_model_kv(&w, &w_kv_mem, &w_rope_mem, &c);
            worker_reset_state(&w, ks, kper, NL, &c);
            if (draft_dir) {
                worker_discard_model_kv(&dw, &d_kv_mem, &d_rope_mem, &c);
                worker_reset_state(&dw, dks, kper, NL, &c);
            }
            history_len = 0;
        }
        if (is_mm) memset(seq, 0, (size_t)prompt_positions * sizeof(int));
        else memcpy(seq, req, (size_t)np * sizeof(int));
        int T = prompt_positions, nout = 0, failed = 0, stop_hit = 0;
""",
)

# Select ordinary token prefill vs exact merged embedding prefill for both target/draft.
replace_once(
    "src/cli/k3_worker.c",
    """        int first_tok = -1;
        if (w.cached >= np || worker_prefill_to(&w, &w_kv_mem, &w_rope_mem, &c, &cache,
                                                  seq, np, prefill_cap, lg, sc, h, br, ks) != 0) {
            failed = 1;
        } else {
""",
    """        int first_tok = -1;
        int prefill_rc;
        if (is_mm)
            prefill_rc = (w.cached >= prompt_positions) ? -1 :
                worker_prefill_embeds_to(&w, &w_kv_mem, &w_rope_mem, &c, &cache,
                                         mixed_embeds, prompt_positions, prefill_cap,
                                         lg, sc, h, br, ks);
        else
            prefill_rc = (w.cached >= np) ? -1 :
                worker_prefill_to(&w, &w_kv_mem, &w_rope_mem, &c, &cache,
                                  seq, np, prefill_cap, lg, sc, h, br, ks);
        if (prefill_rc != 0) {
            failed = 1;
        } else {
""",
)

replace_once(
    "src/cli/k3_worker.c",
    """        if (draft_dir && !failed) {
            if (dw.cached >= np || worker_prefill_to(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,
                                                     seq, np, prefill_cap, lg, sc, h, br, dks) != 0) {
                failed = 1;
            }
        }

        /* Commit only after both prefills succeed, but the token itself was sampled from
""",
    """        if (draft_dir && !failed) {
            int draft_prefill_rc;
            if (is_mm)
                draft_prefill_rc = (dw.cached >= prompt_positions) ? -1 :
                    worker_prefill_embeds_to(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,
                                             mixed_embeds, prompt_positions, prefill_cap,
                                             lg, sc, h, br, dks);
            else
                draft_prefill_rc = (dw.cached >= np) ? -1 :
                    worker_prefill_to(&dw, &d_kv_mem, &d_rope_mem, &c, &cache,
                                      seq, np, prefill_cap, lg, sc, h, br, dks);
            if (draft_prefill_rc != 0) failed = 1;
        }
        free(mixed_embeds);
        mixed_embeds = NULL;

        /* Commit only after both prefills succeed, but the token itself was sampled from
""",
)

# Speculation horizon is measured in model positions after expansion.
replace_once(
    "src/cli/k3_worker.c",
    """            const int request_tmax = np + gen + 1;
""",
    """            const int request_tmax = prompt_positions + gen + 1;
""",
)

Path("src/cli/k3_run.c").write_text(Path("src/cli/k3_run.c").read_text())
print("applied exact mixed text/media embedding prefill path")
