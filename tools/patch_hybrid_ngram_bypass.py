#!/usr/bin/env python3
from pathlib import Path

p = Path('src/cli/k3_run.c')
s = p.read_text()

if 'hyb_ngram_rounds' in s:
    print('hybrid ngram bypass already applied')
    raise SystemExit(0)


def one(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'expected one match, found {n}: {old[:160]!r}')
    s = s.replace(old, new, 1)

one(
'''    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0;
    double hyb_draft_s = 0.0, hyb_verify_s = 0.0;
''',
'''    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0;
    double hyb_draft_s = 0.0, hyb_verify_s = 0.0;
    /* Greedy hybrid decode first tries the free sequence-history n-gram proposer. Exact
     * K3 still verifies every proposal. The expensive draft model is only resynchronised
     * in one batch after an accepted n-gram prefix. One poor n-gram round disables this
     * bypass for the rest of the run, so a bad history pattern cannot keep wasting exact
     * verification work. K3_DRAFT_NO_NGRAM is an A/B/escape hatch. */
    int hyb_ngram_enabled = getenv("K3_DRAFT_NO_NGRAM") ? 0 : 1;
    int hyb_ngram_disabled = 0;
    long hyb_model_steps = 0;
    long hyb_ngram_rounds = 0, hyb_ngram_drafted = 0, hyb_ngram_accepted = 0;
    double hyb_ngram_s = 0.0, hyb_ngram_resync_s = 0.0;
''')

one(
'''            int d[K3_SPEC_MAX], nd = 0;
            const int spec_now = spec_auto ? spec_cur : spec_n;
            if (spec_snap && spec_now > 0 &&
                T + spec_now + 1 < Tmax && base + spec_now + 1 <= w.kv_cap) {
                if (dw.trunk) {
                    const double hyb_draft_t0 = now_s();
                    /* The draft model proposes: k sequential one-token steps through
                     * the draft trunk, chaining its own argmax. Its state is
                     * snapshotted first so a partial acceptance can rewind it the
                     * same way the exact side rewinds. */
                    memcpy(dsnap, dks, kper_f * (size_t)w.n_bound * sizeof(float));
                    int prev = seq[base];
                    while (nd < spec_now) {
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
                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                    hyb_draft_s += now_s() - hyb_draft_t0;
                } else {
                    nd = spec_draft(seq, T, spec_now, d);
                    if (temperature > 0.0)
                        for (int j = 0; j < nd; j++) {
                            double *qrow = spec_q_probs + (size_t)j * c.vocab;
                            memset(qrow, 0, (size_t)c.vocab * sizeof(double));
                            qrow[d[j]] = 1.0;
                        }
                }
            }
''',
'''            int d[K3_SPEC_MAX], nd = 0;
            int used_ngram = 0;
            const int spec_now = spec_auto ? spec_cur : spec_n;
            if (spec_snap && spec_now > 0 &&
                T + spec_now + 1 < Tmax && base + spec_now + 1 <= w.kv_cap) {
                /* In greedy hybrid mode, sequence-history drafting costs no model
                 * forward. If it can propose a continuation, let exact K3 judge it
                 * before paying k sequential draft-trunk passes. Sampling deliberately
                 * stays on the existing draft-model path so seeded RNG behaviour is
                 * unchanged. */
                if (dw.trunk && temperature <= 0.0 && hyb_ngram_enabled) {
                    const double ng0 = now_s();
                    nd = spec_draft(seq, T, spec_now, d);
                    hyb_ngram_s += now_s() - ng0;
                    if (nd > 0) {
                        used_ngram = 1;
                        hyb_ngram_rounds++;
                        hyb_ngram_drafted += nd;
                        hyb_rounds++;
                        hyb_drafted += nd;
                    }
                }
                if (dw.trunk && !used_ngram) {
                    const double hyb_draft_t0 = now_s();
                    /* The draft model proposes: k sequential one-token steps through
                     * the draft trunk, chaining its own argmax. Its state is
                     * snapshotted first so a partial acceptance can rewind it the
                     * same way the exact side rewinds. */
                    memcpy(dsnap, dks, kper_f * (size_t)w.n_bound * sizeof(float));
                    int prev = seq[base];
                    while (nd < spec_now) {
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
                        hyb_model_steps++;
                    }
                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                    hyb_draft_s += now_s() - hyb_draft_t0;
                } else if (!dw.trunk) {
                    nd = spec_draft(seq, T, spec_now, d);
                    if (temperature > 0.0)
                        for (int j = 0; j < nd; j++) {
                            double *qrow = spec_q_probs + (size_t)j * c.vocab;
                            memset(qrow, 0, (size_t)c.vocab * sizeof(double));
                            qrow[d[j]] = 1.0;
                        }
                }
            }
''')

one(
'''                    if (dw.trunk && frc == 0) {
                        hyb_accepted += m;
                        if (m == nd) {
                            int last = d[nd - 1];
                            if (forward(&dw, &c, &cache, &last, 1, lg, sc, h, br,
                                        dks, NULL, NULL) == 0) dw.cached += 1;
                            else frc = -1;
                        } else {
                            memcpy(dks, dsnap, kper_f * (size_t)w.n_bound * sizeof(float));
                            dw.cached = base;
                            if (forward(&dw, &c, &cache, seq + base, m + 1, lg, sc,
                                        h, br, dks, NULL, NULL) == 0) dw.cached = base + m + 1;
                            else frc = -1;
                        }
                    }
''',
'''                    if (dw.trunk && frc == 0) {
                        hyb_accepted += m;
                        if (used_ngram) {
                            /* The draft model did not advance during n-gram proposal.
                             * Bring it to the exact accepted prefix in ONE batched pass:
                             * pending token + m accepted drafts. The exact correction is
                             * intentionally still pending, matching the normal decode
                             * state convention for the next generation step. */
                            const double rs0 = now_s();
                            hyb_ngram_accepted += m;
                            if (dw.cached != base) {
                                fprintf(stderr, "hybrid n-gram draft state drift: cached=%d base=%d\n",
                                        dw.cached, base);
                                frc = -1;
                            } else if (forward(&dw, &c, &cache, seq + base, m + 1, lg, sc,
                                               h, br, dks, NULL, NULL) == 0) {
                                dw.cached = base + m + 1;
                            } else {
                                frc = -1;
                            }
                            hyb_ngram_resync_s += now_s() - rs0;
                            if (m * 2 < nd && hyb_ngram_enabled) {
                                hyb_ngram_enabled = 0;
                                hyb_ngram_disabled = 1;
                            }
                        } else if (m == nd) {
                            int last = d[nd - 1];
                            if (forward(&dw, &c, &cache, &last, 1, lg, sc, h, br,
                                        dks, NULL, NULL) == 0) dw.cached += 1;
                            else frc = -1;
                        } else {
                            memcpy(dks, dsnap, kper_f * (size_t)w.n_bound * sizeof(float));
                            dw.cached = base;
                            if (forward(&dw, &c, &cache, seq + base, m + 1, lg, sc,
                                        h, br, dks, NULL, NULL) == 0) dw.cached = base + m + 1;
                            else frc = -1;
                        }
                    }
''')

one(
'''        printf("  draft proposal %.3f s, exact verify/replay %.3f s\n",
               hyb_draft_s, hyb_verify_s);
''',
'''        printf("  draft proposal %.3f s, exact verify/replay %.3f s\n",
               hyb_draft_s, hyb_verify_s);
        printf("  draft-model proposal steps %ld; n-gram bypass %ld rounds, %ld/%ld accepted, "
               "proposal %.6f s, batch resync %.3f s%s\n",
               hyb_model_steps, hyb_ngram_rounds, hyb_ngram_accepted, hyb_ngram_drafted,
               hyb_ngram_s, hyb_ngram_resync_s,
               hyb_ngram_disabled ? ", disabled after poor acceptance" : "");
''')

one(
'''                   "\\\"draft_seconds\\\":%.6f,\\\"verify_seconds\\\":%.6f,"
                   "\\\"spec_auto\\\":%d,\\\"spec_auto_rounds\\\":%ld,"
''',
'''                   "\\\"draft_seconds\\\":%.6f,\\\"verify_seconds\\\":%.6f,"
                   "\\\"draft_model_steps\\\":%ld,\\\"draft_ngram_rounds\\\":%ld,"
                   "\\\"draft_ngram_proposed\\\":%ld,\\\"draft_ngram_accepted\\\":%ld,"
                   "\\\"draft_ngram_disabled\\\":%d,\\\"draft_ngram_seconds\\\":%.6f,"
                   "\\\"draft_ngram_resync_seconds\\\":%.6f,"
                   "\\\"spec_auto\\\":%d,\\\"spec_auto_rounds\\\":%ld,"
''')

one(
'''                hyb_drafted ? (double)hyb_accepted / hyb_drafted : 0.0,
                hyb_draft_s, hyb_verify_s, spec_auto, spec_auto_rounds,
''',
'''                hyb_drafted ? (double)hyb_accepted / hyb_drafted : 0.0,
                hyb_draft_s, hyb_verify_s, hyb_model_steps, hyb_ngram_rounds,
                hyb_ngram_drafted, hyb_ngram_accepted, hyb_ngram_disabled,
                hyb_ngram_s, hyb_ngram_resync_s, spec_auto, spec_auto_rounds,
''')

p.write_text(s)
print('applied greedy hybrid n-gram bypass')
