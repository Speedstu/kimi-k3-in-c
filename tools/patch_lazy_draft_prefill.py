#!/usr/bin/env python3
from pathlib import Path

p=Path('src/cli/k3_run.c')
s=p.read_text()
if 'hyb_prefill_tokens' in s:
    print('lazy draft prefill already applied')
    raise SystemExit(0)

def one(old,new):
    global s
    n=s.count(old)
    if n!=1:
        raise SystemExit(f'expected one match, found {n}: {old[:180]!r}')
    s=s.replace(old,new,1)

one(
'''    int hyb_ngram_eager_sync = getenv("K3_DRAFT_EAGER_NGRAM_SYNC") ? 1 : 0;
    long hyb_model_steps = 0;
''',
'''    int hyb_ngram_eager_sync = getenv("K3_DRAFT_EAGER_NGRAM_SYNC") ? 1 : 0;
    /* With lazy n-gram sync, the draft model does not need prompt state until the first
     * real draft proposal/direct sync. Defer its full prompt replay by default in greedy
     * n-gram mode; K3_DRAFT_EAGER_PREFILL restores the previous eager behaviour. */
    int hyb_defer_prefill = (temperature <= 0.0 && hyb_ngram_enabled &&
                             !hyb_ngram_eager_sync && !getenv("K3_DRAFT_EAGER_PREFILL"));
    int hyb_prefill_deferred = 0;
    long hyb_prefill_tokens = 0;
    double hyb_prefill_s = 0.0;
    long hyb_model_steps = 0;
''')

one(
'''            if (dw.trunk && frc == 0) {
                const int db = load_state ? 0 : base;
                if (forward(&dw, &c, &cache, seq + db, base + nT0 - db, lg, sc, h, br,
                            dks, NULL, NULL) == 0)
                    dw.cached = base + nT0;
                else frc = -1;
            }
''',
'''            if (dw.trunk && frc == 0) {
                if (hyb_defer_prefill) {
                    /* Exact K3 has already consumed the prompt. Keep draft state at its
                     * initial cache position; the lazy catch-up path can reconstruct the
                     * exact committed prefix in one batch later if n-grams stop being
                     * sufficient. If they never do, the whole draft prompt pass vanishes. */
                    hyb_prefill_deferred = 1;
                } else {
                    const int db = load_state ? 0 : base;
                    const int dn = base + nT0 - db;
                    const double dp0 = now_s();
                    if (forward(&dw, &c, &cache, seq + db, dn, lg, sc, h, br,
                                dks, NULL, NULL) == 0) {
                        dw.cached = base + nT0;
                        hyb_prefill_tokens += dn;
                    } else frc = -1;
                    hyb_prefill_s += now_s() - dp0;
                }
            }
''')

one(
'''        printf("  draft-model proposal steps %ld; n-gram bypass %ld rounds, %ld/%ld accepted, "
               "proposal %.6f s, draft sync %ld rounds/%ld tokens in %.3f s%s%s\n",
''',
'''        printf("  draft prefill %ld tokens in %.3f s%s\n", hyb_prefill_tokens, hyb_prefill_s,
               hyb_prefill_deferred ? " (deferred; paid only if later needed)" : "");
        printf("  draft-model proposal steps %ld; n-gram bypass %ld rounds, %ld/%ld accepted, "
               "proposal %.6f s, draft sync %ld rounds/%ld tokens in %.3f s%s%s\n",
''')

one(
'''                   "\\\"draft_ngram_eager_sync\\\":%d,\\\"draft_sync_rounds\\\":%ld,"
                   "\\\"draft_sync_tokens\\\":%ld,"
''',
'''                   "\\\"draft_ngram_eager_sync\\\":%d,\\\"draft_sync_rounds\\\":%ld,"
                   "\\\"draft_sync_tokens\\\":%ld,\\\"draft_prefill_deferred\\\":%d,"
                   "\\\"draft_prefill_tokens\\\":%ld,\\\"draft_prefill_seconds\\\":%.6f,"
''')

one(
'''                hyb_ngram_s, hyb_ngram_resync_s, hyb_ngram_eager_sync,
                hyb_sync_rounds, hyb_sync_tokens, spec_auto, spec_auto_rounds,
''',
'''                hyb_ngram_s, hyb_ngram_resync_s, hyb_ngram_eager_sync,
                hyb_sync_rounds, hyb_sync_tokens, hyb_prefill_deferred,
                hyb_prefill_tokens, hyb_prefill_s, spec_auto, spec_auto_rounds,
''')

p.write_text(s)
print('applied lazy hybrid draft prefill')
