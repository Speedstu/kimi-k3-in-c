#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'src/cli/k3_worker.c'
s = p.read_text()
old = '''            int d[K3_SPEC_MAX], nd = 0;
            const int room_drafts = context - T - 1;
            int want_drafts = spec_n;
            if (want_drafts > room_drafts) want_drafts = room_drafts;

            if (draft_dir && want_drafts > 0) {
'''
new = '''            int d[K3_SPEC_MAX], nd = 0;
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
'''
if s.count(old) != 1:
    raise SystemExit(f'scheduling anchor: expected 1, got {s.count(old)}')
s = s.replace(old, new, 1)
p.write_text(s)
print('resident speculative scheduling now mirrors one-shot request horizon')
