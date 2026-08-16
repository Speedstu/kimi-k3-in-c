#!/usr/bin/env python3
from pathlib import Path

p = Path("src/cli/k3_run.c")
s = p.read_text()

if '#include "k3_spec_auto.h"' in s:
    print("cost-aware speculative horizon already applied")
    raise SystemExit(0)


def one(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"expected exactly one match, got {n}: {old[:120]!r}")
    s = s.replace(old, new, 1)


one(
    '#include "k3_sample.h" /* benchmark-parity temperature/top-p sampling */\n',
    '#include "k3_sample.h" /* benchmark-parity temperature/top-p sampling */\n'
    '#include "k3_spec_auto.h" /* cost-aware greedy speculative horizon */\n',
)

one(
    '"  --spec-auto           greedy-only adaptive horizon: start at 4 drafts, grow after a\\n"\n'
    '"                        fully accepted sweep and shrink after poor acceptance. Exact K3\\n"\n'
    '"                        still verifies every emitted token. --spec N becomes the ceiling\\n"\n'
    '"                        (default ceiling 8). Sampling keeps fixed --spec for seeded parity\\n"\n',
    '"  --spec-auto           greedy-only adaptive horizon: start at 4 drafts. With a draft\\n"\n'
    '"                        trunk, measure end-to-end seconds/exact-token at each width,\\n"\n'
    '"                        probe wider batches and back off measured regressions; poor\\n"\n'
    '"                        acceptance still shrinks immediately. Without a draft trunk the\\n"\n'
    '"                        acceptance controller is used. --spec N is the ceiling (8 max).\\n"\n'
    '"                        Sampling keeps fixed --spec for seeded parity\\n"\n',
)

one(
    '    long spec_auto_rounds = 0, spec_auto_proposed = 0, spec_auto_accepted = 0;\n'
    '    int spec_auto_grows = 0, spec_auto_shrinks = 0;\n',
    '    long spec_auto_rounds = 0, spec_auto_proposed = 0, spec_auto_accepted = 0;\n'
    '    int spec_auto_grows = 0, spec_auto_shrinks = 0;\n'
    '    K3SpecAutoCost spec_cost;\n'
    '    k3_spec_auto_cost_init(&spec_cost);\n',
)

one(
    '            const int spec_now = spec_auto ? spec_cur : spec_n;\n',
    '            const int spec_now = spec_auto ? spec_cur : spec_n;\n'
    '            double spec_round_t0 = 0.0;\n',
)

one(
    '                    const double hyb_draft_t0 = now_s();\n',
    '                    const double hyb_draft_t0 = now_s();\n'
    '                    spec_round_t0 = hyb_draft_t0;\n',
)

one(
    '                    if (spec_auto) {\n'
    '                        spec_auto_rounds++;\n'
    '                        spec_auto_proposed += nd;\n'
    '                        spec_auto_accepted += m;\n'
    '                        if (m == nd && spec_cur < spec_limit) {\n'
    '                            spec_cur++;\n'
    '                            spec_auto_grows++;\n'
    '                        } else if (m * 2 < nd && spec_cur > 1) {\n'
    '                            int next = (spec_cur + 1) / 2;\n'
    '                            if (next < 1) next = 1;\n'
    '                            if (next < spec_cur) {\n'
    '                                spec_cur = next;\n'
    '                                spec_auto_shrinks++;\n'
    '                            }\n'
    '                        }\n'
    '                    }\n',
    '',
)

one(
    '                    if (frc == 0) {\n'
    '                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];\n',
    '                    if (spec_auto && frc == 0) {\n'
    '                        int next = spec_cur;\n'
    '                        spec_auto_rounds++;\n'
    '                        spec_auto_proposed += nd;\n'
    '                        spec_auto_accepted += m;\n'
    '                        if (dw.trunk && spec_round_t0 > 0.0) {\n'
    '                            /* Measure the whole speculative transaction: proposal, exact\n'
    '                             * verify/replay, and draft resync.  This is the wall-clock cost\n'
    '                             * the user actually pays for the m accepted drafts plus the\n'
    '                             * exact correction token. */\n'
    '                            const double round_s = now_s() - spec_round_t0;\n'
    '                            k3_spec_auto_cost_observe(&spec_cost, spec_now, round_s, m + 1);\n'
    '                            next = k3_spec_auto_cost_choose(&spec_cost, spec_cur, spec_limit,\n'
    '                                                            nd, m);\n'
    '                        } else {\n'
    '                            /* Free n-gram drafting has no meaningful proposal cost to learn;\n'
    '                             * keep the proven acceptance-only controller for that path. */\n'
    '                            if (m == nd && spec_cur < spec_limit)\n'
    '                                next = spec_cur + 1;\n'
    '                            else if (m * 2 < nd && spec_cur > 1) {\n'
    '                                next = (spec_cur + 1) / 2;\n'
    '                                if (next < 1) next = 1;\n'
    '                            }\n'
    '                        }\n'
    '                        if (next > spec_cur) spec_auto_grows++;\n'
    '                        else if (next < spec_cur) spec_auto_shrinks++;\n'
    '                        spec_cur = next;\n'
    '                    }\n'
    '                    if (frc == 0) {\n'
    '                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];\n',
)

one(
    '    if (dw.trunk && hyb_rounds > 0) {\n',
    '    {\n'
    '        const int cbest = k3_spec_auto_cost_best(&spec_cost, spec_limit, 0);\n'
    '        if (spec_auto && cbest)\n'
    '            printf("  cost-aware horizon: best observed %d at %.4f s/exact-token, "\n'
    '                   "probes %d, backoffs %d\\n", cbest, spec_cost.ema_spt[cbest],\n'
    '                   spec_cost.probes, spec_cost.backoffs);\n'
    '    }\n'
    '    if (dw.trunk && hyb_rounds > 0) {\n',
)

one(
    '    FILE *f = fopen(outp, "w");\n',
    '    const int spec_cost_best = k3_spec_auto_cost_best(&spec_cost, spec_limit, 0);\n'
    '    const double spec_cost_best_spt = spec_cost_best ? spec_cost.ema_spt[spec_cost_best] : 0.0;\n'
    '    FILE *f = fopen(outp, "w");\n',
)

one(
    '                   "\\\"spec_auto_grows\\\":%d,\\\"spec_auto_shrinks\\\":%d,"\n'
    '                   "\\\"seconds_per_token\\\":%.4f}\\n",\n',
    '                   "\\\"spec_auto_grows\\\":%d,\\\"spec_auto_shrinks\\\":%d,"\n'
    '                   "\\\"spec_auto_cost_best\\\":%d,\\\"spec_auto_cost_best_spt\\\":%.9g,"\n'
    '                   "\\\"spec_auto_cost_probes\\\":%d,\\\"spec_auto_cost_backoffs\\\":%d,"\n'
    '                   "\\\"seconds_per_token\\\":%.4f}\\n",\n',
)

one(
    '                spec_auto_proposed, spec_auto_accepted, spec_cur, spec_limit,\n'
    '                spec_auto_grows, spec_auto_shrinks, t_total / nout);\n',
    '                spec_auto_proposed, spec_auto_accepted, spec_cur, spec_limit,\n'
    '                spec_auto_grows, spec_auto_shrinks, spec_cost_best, spec_cost_best_spt,\n'
    '                spec_cost.probes, spec_cost.backoffs, t_total / nout);\n',
)

p.write_text(s)
print("applied cost-aware speculative horizon controller")
