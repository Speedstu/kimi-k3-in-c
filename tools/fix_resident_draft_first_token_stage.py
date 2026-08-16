#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'src/cli/k3_worker.c'
s = p.read_text()
old = '''        if (w.cached >= np || forward(&w, &c, &cache, seq + w.cached, np - w.cached,
                                      lg, sc, h, br, ks, NULL, NULL) != 0) {
            failed = 1;
        } else {
            w.cached = np;
        }
        if (draft_dir && !failed) {
            if (dw.cached >= np || forward(&dw, &c, &cache, seq + dw.cached, np - dw.cached,
                                           lg, sc, h, br, dks, NULL, NULL) != 0) {
                failed = 1;
            } else {
                dw.cached = np;
            }
        }

        /* Match the one-shot decoder: the first token of every request is sampled from
         * exact K3 after prefill. It becomes the pending token consumed by the first
         * speculative block, which also keeps fresh and prefix-reused requests identical. */
        if (!failed) {
            int tok = k3_sample_token(&sampler, lg, c.vocab);
            if (tok < 0) failed = 1;
            else {
                seq[T++] = tok;
                nout++;
                printf("@K3TOKEN %llu %d\\n", rid, tok);
                if (stop_id >= 0 && tok == stop_id) stop_hit = 1;
            }
        }
'''
new = '''        int first_tok = -1;
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
            printf("@K3TOKEN %llu %d\\n", rid, first_tok);
            if (stop_id >= 0 && first_tok == stop_id) stop_hit = 1;
        }
'''
if s.count(old) != 1:
    raise SystemExit(f'first-token ordering anchor: expected 1, got {s.count(old)}')
s = s.replace(old, new, 1)
p.write_text(s)
print('resident sampled-draft first-token ordering fixed')
