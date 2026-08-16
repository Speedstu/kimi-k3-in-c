#!/usr/bin/env python3
"""Guarded materialization of probability-correct sampled speculation.

The helper is deleted before merge. Every replacement must match the current source
exactly once so a source drift cannot produce a half-applied decoder.
"""
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {n}")
    return text.replace(old, new, 1)


p = Path(__file__).resolve().parents[1] / "src/cli/k3_run.c"
s = p.read_text()

# forward() can optionally return the full logit vector for every short verification
# position. Greedy callers still use arg_all; sampled speculation consumes logits_all.
s = once(
    s,
    '''static int forward(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                   float *logits_last, float *scratch, float *h, float *br, float *kstate,
                   int *arg_all)
''',
    '''static int forward(Weights *w, const K3Cfg *c, K3Cache *cache, const int *ids, int T,
                   float *logits_last, float *scratch, float *h, float *br, float *kstate,
                   int *arg_all, float *logits_all)
''',
    "forward signature",
)
s = once(
    s,
    '''    if (arg_all) {
        /* Speculative verification asks for argmax at several adjacent positions. The
''',
    '''    if (arg_all || logits_all) {
        /* Speculative verification asks for complete distributions at several adjacent
         * positions in sampled mode and argmaxes in greedy mode. The
''',
    "forward all-position branch",
)
s = once(
    s,
    '''            float *all = (float *)malloc((size_t)T * nv * sizeof(float));
            if (all) {
''',
    '''            int own_all = 0;
            float *all = logits_all;
            if (!all) {
                all = (float *)malloc((size_t)T * nv * sizeof(float));
                own_all = 1;
            }
            if (all) {
''',
    "forward external all-logits buffer",
)
s = once(
    s,
    '''                for (int t = 0; t < T; t++)
                    arg_all[t] = argmax_(all + (size_t)t * nv, c->vocab);
                memcpy(logits_last, all + (size_t)(T - 1) * nv, nv * sizeof(float));
                free(all);
                return 0;
''',
    '''                if (arg_all)
                    for (int t = 0; t < T; t++)
                        arg_all[t] = argmax_(all + (size_t)t * nv, c->vocab);
                memcpy(logits_last, all + (size_t)(T - 1) * nv, nv * sizeof(float));
                if (own_all) free(all);
                return 0;
''',
    "forward all-logits result",
)
s = once(
    s,
    '''        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            arg_all[t] = argmax_(logits_last, c->vocab);
        }
''',
    '''        for (int t = 0; t < T; t++) {
            k3_rmsnorm(nrm, h + (size_t)t * E, w->mb.norm, E, c->rms_eps);
            k3_mmw(logits_last, nrm, w->mb.lm_head, w->mb.wdt, E, c->vocab);
            if (arg_all) arg_all[t] = argmax_(logits_last, c->vocab);
            if (logits_all)
                memcpy(logits_all + (size_t)t * c->vocab, logits_last,
                       (size_t)c->vocab * sizeof(float));
        }
''',
    "forward fallback all-logits result",
)

# Update every non-verification call with the new trailing NULL parameter.
replacements = [
    ("forward(&w, &c, &cache, seq, np, lg, sc, h, br, ks, arg)",
     "forward(&w, &c, &cache, seq, np, lg, sc, h, br, ks, arg, NULL)", "tf forward"),
    ("forward(&w, &c, &cache, seq + base, nT0, lg, sc, h, br, ks, NULL)",
     "forward(&w, &c, &cache, seq + base, nT0, lg, sc, h, br, ks, NULL, NULL)", "initial exact"),
    ("dks, NULL) == 0)\n                    dw.cached = base + nT0;",
     "dks, NULL, NULL) == 0)\n                    dw.cached = base + nT0;", "initial draft"),
    ("dks, NULL) != 0) break;",
     "dks, NULL, NULL) != 0) break;", "draft proposal"),
    ("ks, NULL);\n                        if (frc == 0) w.cached = base + m + 1;",
     "ks, NULL, NULL);\n                        if (frc == 0) w.cached = base + m + 1;", "exact replay"),
    ("dks, NULL) == 0) dw.cached += 1;",
     "dks, NULL, NULL) == 0) dw.cached += 1;", "draft close gap"),
    ("h, br, dks, NULL) == 0) dw.cached = base + m + 1;",
     "h, br, dks, NULL, NULL) == 0) dw.cached = base + m + 1;", "draft replay"),
    ("forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br, ks, NULL);",
     "forward(&w, &c, &cache, seq + base, 1, lg, sc, h, br, ks, NULL, NULL);", "serial exact"),
    ("dks, NULL) == 0) dw.cached = base + 1;",
     "dks, NULL, NULL) == 0) dw.cached = base + 1;", "serial draft"),
    ("forward(&w, &c, &cache, seq, T, lg, sc, h, br, ks, NULL);",
     "forward(&w, &c, &cache, seq, T, lg, sc, h, br, ks, NULL, NULL);", "full recompute"),
]
for old, new, label in replacements:
    s = once(s, old, new, label)

# Sampled speculation is now supported; no more correctness-preserving refusal.
s = once(
    s,
    '''    if (temperature > 0.0 && (spec_n > 0 || draft_dir != NULL)) {
        fprintf(stderr, "sampled speculative decoding is not implemented yet; remove --spec/--draft-trunk\\n"
                        "rather than silently changing the requested sampling distribution.\\n");
        return 2;
    }

''',
    "",
    "remove sampled-spec refusal",
)

# Three deterministic RNG streams: target corrections/extras, draft proposals, and
# accept/reject uniforms. Keeping their roles separate makes the algorithm repeatable and
# avoids changing accept decisions if the target sampler implementation is refactored.
s = once(
    s,
    '''    K3Sampler sampler;
    k3_sampler_init(&sampler, temperature, top_p, (uint64_t)sample_seed);

''',
    '''    K3Sampler sampler, draft_sampler, accept_sampler;
    k3_sampler_init(&sampler, temperature, top_p, (uint64_t)sample_seed);
    k3_sampler_init(&draft_sampler, temperature, top_p,
                    (uint64_t)sample_seed ^ UINT64_C(0xd6e8feb86659fd93));
    k3_sampler_init(&accept_sampler, temperature, top_p,
                    (uint64_t)sample_seed ^ UINT64_C(0xa5a3564e27f8862b));

''',
    "three sampler streams",
)

# Allocate probability/logit workspaces only when sampling and speculation are both on.
anchor = '''    /* --tf-check: teacher-forced agreement over the whole --ids sequence in ONE sweep.
'''
workspace = '''    /* Sampled speculative verification needs q_i for every proposal and p_i for every
     * verified position. At K3's 163840-token vocabulary and spec=4 this is only about
     * 8 MB of doubles plus 3 MB of logits -- tiny beside the model/KV working set, and
     * allocated only for temperature > 0. */
    float *spec_target_logits = NULL;
    double *spec_q_probs = NULL, *spec_p_probs = NULL;
    if (temperature > 0.0 && spec_n > 0) {
        spec_target_logits = (float *)malloc((size_t)(spec_n + 1) * c.vocab * sizeof(float));
        spec_q_probs = (double *)malloc((size_t)spec_n * c.vocab * sizeof(double));
        spec_p_probs = (double *)malloc((size_t)c.vocab * sizeof(double));
        if (!spec_target_logits || !spec_q_probs || !spec_p_probs) {
            fprintf(stderr, "OOM for sampled speculative probability buffers\\n");
            return 1;
        }
        printf("sampled speculation: probability-correct p/q accept + (p-q)+ residual; "
               "target distribution preserved\\n\\n");
    }

'''
if s.count(anchor) != 1:
    raise SystemExit("workspace insertion anchor not unique")
s = s.replace(anchor, workspace + anchor, 1)

# Draft proposal: sample q instead of argmax when temperature > 0 and retain the entire
# q distribution for exact residual sampling if the target rejects that proposal.
s = once(
    s,
    '''                        dw.cached += 1;
                        prev = argmax_(lg, c.vocab);
                        d[nd++] = prev;
''',
    '''                        dw.cached += 1;
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
''',
    "sample draft q",
)

# Deterministic n-gram proposals are a valid q distribution with mass 1 on each proposed
# token. Materialise that q so the same acceptance/residual proof applies.
s = once(
    s,
    '''                } else {
                    nd = spec_draft(seq, T, spec_n, d);
                }
''',
    '''                } else {
                    nd = spec_draft(seq, T, spec_n, d);
                    if (temperature > 0.0)
                        for (int j = 0; j < nd; j++) {
                            double *qrow = spec_q_probs + (size_t)j * c.vocab;
                            memset(qrow, 0, (size_t)c.vocab * sizeof(double));
                            qrow[d[j]] = 1.0;
                        }
                }
''',
    "ngram q distribution",
)

# Verification forward gets all p logits in sampled mode. Greedy keeps the old zero-extra
# allocation path.
s = once(
    s,
    '''                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks, arg);
                if (frc == 0) {
                    int m = 0;
                    while (m < nd && arg[m] == d[m]) m++;
''',
    '''                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks,
                              arg, temperature > 0.0 ? spec_target_logits : NULL);
                if (frc == 0) {
                    int m = 0;
                    int correction = -1;
                    if (temperature <= 0.0) {
                        while (m < nd && arg[m] == d[m]) m++;
                        correction = arg[m];
                    } else {
                        /* Standard speculative sampling. Proposal y~q is accepted with
                         * min(1,p(y)/q(y)). On first rejection sample from normalised
                         * (p-q)+. If every proposal is accepted, sample one extra token
                         * from the target distribution after the final draft. */
                        for (; m < nd; m++) {
                            const float *plog = spec_target_logits + (size_t)m * c.vocab;
                            const double *qrow = spec_q_probs + (size_t)m * c.vocab;
                            if (k3_sampler_distribution(&sampler, plog, c.vocab,
                                                        spec_p_probs) != 0) {
                                frc = -1;
                                break;
                            }
                            const double qy = qrow[d[m]];
                            const double py = spec_p_probs[d[m]];
                            if (!(qy > 0.0)) { frc = -1; break; }
                            double accept = py / qy;
                            if (accept > 1.0) accept = 1.0;
                            if (k3_sampler_uniform(&accept_sampler) >= accept) {
                                correction = k3_sample_residual(&sampler, spec_p_probs,
                                                                qrow, c.vocab);
                                break;
                            }
                        }
                        if (frc == 0 && m == nd) {
                            const float *extra = spec_target_logits + (size_t)nd * c.vocab;
                            if (k3_sampler_distribution(&sampler, extra, c.vocab,
                                                        spec_p_probs) != 0)
                                frc = -1;
                            else
                                correction = k3_sample_probs(&sampler, spec_p_probs, c.vocab);
                        }
                        if (correction < 0) frc = -1;
                    }
''',
    "sampled verification",
)

# Emit the mathematically-correct correction rather than greedy arg[m].
s = once(
    s,
    '''                    if (frc == 0) {
                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                        emit[emitn++] = arg[m];
                    }
''',
    '''                    if (frc == 0) {
                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                        emit[emitn++] = correction;
                    }
''',
    "emit correction",
)

# Free sampled workspaces with the existing speculative snapshot.
s = once(
    s,
    '''    free(spec_snap);
    printf("--------------------------------------------------------------------\\n");
''',
    '''    free(spec_snap);
    free(spec_target_logits); free(spec_q_probs); free(spec_p_probs);
    printf("--------------------------------------------------------------------\\n");
''',
    "free sampled spec workspaces",
)

p.write_text(s)
print("probability-correct sampled speculation materialized")
