#!/usr/bin/env python3
from pathlib import Path

p=Path('src/cli/k3_run.c')
s=p.read_text()
if 'hyb_sync_tokens' in s:
    print('lazy draft resync already applied')
    raise SystemExit(0)

def one(old,new):
    global s
    n=s.count(old)
    if n!=1:
        raise SystemExit(f'expected one match, found {n}: {old[:180]!r}')
    s=s.replace(old,new,1)

one(
'''    int hyb_ngram_enabled = getenv("K3_DRAFT_NO_NGRAM") ? 0 : 1;
    int hyb_ngram_disabled = 0;
    long hyb_model_steps = 0;
    long hyb_ngram_rounds = 0, hyb_ngram_drafted = 0, hyb_ngram_accepted = 0;
    double hyb_ngram_s = 0.0, hyb_ngram_resync_s = 0.0;
''',
'''    int hyb_ngram_enabled = getenv("K3_DRAFT_NO_NGRAM") ? 0 : 1;
    int hyb_ngram_disabled = 0;
    /* A/B escape hatch for the #43 behaviour: eagerly resynchronise the draft after
     * every n-gram round. Default is lazy: the draft may lag while free n-grams work,
     * then catches up in one batch only when a real draft-model/direct step needs it. */
    int hyb_ngram_eager_sync = getenv("K3_DRAFT_EAGER_NGRAM_SYNC") ? 1 : 0;
    long hyb_model_steps = 0;
    long hyb_ngram_rounds = 0, hyb_ngram_drafted = 0, hyb_ngram_accepted = 0;
    long hyb_sync_rounds = 0, hyb_sync_tokens = 0;
    double hyb_ngram_s = 0.0, hyb_ngram_resync_s = 0.0;
''')

one(
'''                if (dw.trunk && !used_ngram) {
                    const double hyb_draft_t0 = now_s();
                    /* The draft model proposes: k sequential one-token steps through
''',
'''                if (dw.trunk && !used_ngram) {
                    /* Lazy n-gram mode is allowed to leave the draft state behind. A
                     * real draft proposal needs state exactly at `base`, so consume the
                     * committed exact history in one batch now. Leave seq[base] pending:
                     * the normal proposer feeds that token as its first step below. */
                    if (dw.cached > base) {
                        fprintf(stderr, "lazy draft state ahead of exact state: cached=%d base=%d\n",
                                dw.cached, base);
                        return 1;
                    }
                    if (dw.cached < base) {
                        const int lag = base - dw.cached;
                        const double rs0 = now_s();
                        if (forward(&dw, &c, &cache, seq + dw.cached, lag, lg, sc, h, br,
                                    dks, NULL, NULL) != 0) {
                            fprintf(stderr, "lazy draft catch-up failed for %d token(s)\n", lag);
                            return 1;
                        }
                        dw.cached = base;
                        hyb_sync_rounds++;
                        hyb_sync_tokens += lag;
                        hyb_ngram_resync_s += now_s() - rs0;
                    }
                    const double hyb_draft_t0 = now_s();
                    /* The draft model proposes: k sequential one-token steps through
''')

one(
'''                        if (used_ngram) {
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
''',
'''                        if (used_ngram) {
                            hyb_ngram_accepted += m;
                            if (dw.cached > base) {
                                fprintf(stderr, "hybrid n-gram draft state ahead: cached=%d base=%d\n",
                                        dw.cached, base);
                                frc = -1;
                            }
                            /* Default: do NOTHING to the draft here. Exact K3 has already
                             * verified the accepted sequence, and future n-gram rounds do
                             * not need draft state. The lag is repaired once, in batch,
                             * only when the draft model is actually needed. The eager
                             * escape hatch preserves the previous behaviour for A/B. */
                            if (frc == 0 && hyb_ngram_eager_sync) {
                                const double rs0 = now_s();
                                if (dw.cached < base) {
                                    const int lag = base - dw.cached;
                                    if (forward(&dw, &c, &cache, seq + dw.cached, lag, lg, sc,
                                                h, br, dks, NULL, NULL) != 0) {
                                        frc = -1;
                                    } else {
                                        dw.cached = base;
                                        hyb_sync_rounds++;
                                        hyb_sync_tokens += lag;
                                    }
                                }
                                if (frc == 0) {
                                    if (forward(&dw, &c, &cache, seq + base, m + 1, lg, sc,
                                                h, br, dks, NULL, NULL) == 0) {
                                        dw.cached = base + m + 1;
                                        hyb_sync_rounds++;
                                        hyb_sync_tokens += m + 1;
                                    } else {
                                        frc = -1;
                                    }
                                }
                                hyb_ngram_resync_s += now_s() - rs0;
                            }
                            if (m * 2 < nd && hyb_ngram_enabled) {
                                hyb_ngram_enabled = 0;
                                hyb_ngram_disabled = 1;
                            }
''')

one(
'''                if (dw.trunk && frc == 0) {
                    if (forward(&dw, &c, &cache, seq + base, 1, lg, sc, h, br,
                                dks, NULL, NULL) == 0) dw.cached = base + 1;
                    else frc = -1;
                }
''',
'''                if (dw.trunk && frc == 0) {
                    /* A previous lazy n-gram stretch may have left the draft behind even
                     * though this step cannot speculate. Catch up through committed
                     * history first, then consume the current pending token exactly once. */
                    if (dw.cached > base) {
                        fprintf(stderr, "direct-step draft state ahead: cached=%d base=%d\n",
                                dw.cached, base);
                        frc = -1;
                    }
                    if (frc == 0 && dw.cached < base) {
                        const int lag = base - dw.cached;
                        const double rs0 = now_s();
                        if (forward(&dw, &c, &cache, seq + dw.cached, lag, lg, sc, h, br,
                                    dks, NULL, NULL) == 0) {
                            dw.cached = base;
                            hyb_sync_rounds++;
                            hyb_sync_tokens += lag;
                        } else frc = -1;
                        hyb_ngram_resync_s += now_s() - rs0;
                    }
                    if (frc == 0) {
                        if (forward(&dw, &c, &cache, seq + base, 1, lg, sc, h, br,
                                    dks, NULL, NULL) == 0) dw.cached = base + 1;
                        else frc = -1;
                    }
                }
''')

one(
'''        printf("  draft-model proposal steps %ld; n-gram bypass %ld rounds, %ld/%ld accepted, "
               "proposal %.6f s, batch resync %.3f s%s\n",
               hyb_model_steps, hyb_ngram_rounds, hyb_ngram_accepted, hyb_ngram_drafted,
               hyb_ngram_s, hyb_ngram_resync_s,
               hyb_ngram_disabled ? ", disabled after poor acceptance" : "");
''',
'''        printf("  draft-model proposal steps %ld; n-gram bypass %ld rounds, %ld/%ld accepted, "
               "proposal %.6f s, draft sync %ld rounds/%ld tokens in %.3f s%s%s\n",
               hyb_model_steps, hyb_ngram_rounds, hyb_ngram_accepted, hyb_ngram_drafted,
               hyb_ngram_s, hyb_sync_rounds, hyb_sync_tokens, hyb_ngram_resync_s,
               hyb_ngram_eager_sync ? ", eager-sync A/B mode" : ", lazy-sync",
               hyb_ngram_disabled ? ", disabled after poor acceptance" : "");
''')

one(
'''                   "\\\"draft_ngram_disabled\\\":%d,\\\"draft_ngram_seconds\\\":%.6f,"
                   "\\\"draft_ngram_resync_seconds\\\":%.6f,"
''',
'''                   "\\\"draft_ngram_disabled\\\":%d,\\\"draft_ngram_seconds\\\":%.6f,"
                   "\\\"draft_ngram_resync_seconds\\\":%.6f,"
                   "\\\"draft_ngram_eager_sync\\\":%d,\\\"draft_sync_rounds\\\":%ld,"
                   "\\\"draft_sync_tokens\\\":%ld,"
''')

one(
'''                hyb_ngram_drafted, hyb_ngram_accepted, hyb_ngram_disabled,
                hyb_ngram_s, hyb_ngram_resync_s, spec_auto, spec_auto_rounds,
''',
'''                hyb_ngram_drafted, hyb_ngram_accepted, hyb_ngram_disabled,
                hyb_ngram_s, hyb_ngram_resync_s, hyb_ngram_eager_sync,
                hyb_sync_rounds, hyb_sync_tokens, spec_auto, spec_auto_rounds,
''')

p.write_text(s)
print('applied lazy draft resynchronization')
