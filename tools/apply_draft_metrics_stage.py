#!/usr/bin/env python3
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, got {n}")
    return text.replace(old, new, 1)


p = Path(__file__).resolve().parents[1] / "src/cli/k3_run.c"
s = p.read_text()

s = once(
    s,
    "    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0;\n",
    "    long hyb_rounds = 0, hyb_drafted = 0, hyb_accepted = 0;\n"
    "    double hyb_draft_s = 0.0, hyb_verify_s = 0.0;\n",
    "hybrid timing declarations",
)

s = once(
    s,
    '''                if (dw.trunk) {
                    /* The draft model proposes: k sequential one-token steps through
''',
    '''                if (dw.trunk) {
                    const double hyb_draft_t0 = now_s();
                    /* The draft model proposes: k sequential one-token steps through
''',
    "draft timer start",
)
s = once(
    s,
    '''                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                } else {
''',
    '''                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                    hyb_draft_s += now_s() - hyb_draft_t0;
                } else {
''',
    "draft timer end",
)

s = once(
    s,
    '''                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks,
                              arg, temperature > 0.0 ? spec_target_logits : NULL);
''',
    '''                const double hyb_verify_t0 = dw.trunk ? now_s() : 0.0;
                frc = forward(&w, &c, &cache, seq + base, nd + 1, lg, sc, h, br, ks,
                              arg, temperature > 0.0 ? spec_target_logits : NULL);
''',
    "verify timer start",
)
s = once(
    s,
    '''                    /* Resync the draft model to the ACCEPTED sequence. On full
                     * acceptance its state already contains every fed token except
''',
    '''                    if (dw.trunk) hyb_verify_s += now_s() - hyb_verify_t0;
                    /* Resync the draft model to the ACCEPTED sequence. On full
                     * acceptance its state already contains every fed token except
''',
    "verify timer end",
)

s = once(
    s,
    '''        printf("\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "
               "mean accepted run %.2f\n",
               hyb_rounds, hyb_drafted, hyb_accepted,
               hyb_drafted ? 100.0 * hyb_accepted / hyb_drafted : 0.0,
               (double)hyb_accepted / hyb_rounds);
''',
    '''        printf("\nhybrid decode: %ld rounds, %ld drafted, %ld accepted (%.1f%%), "
               "mean accepted run %.2f\n",
               hyb_rounds, hyb_drafted, hyb_accepted,
               hyb_drafted ? 100.0 * hyb_accepted / hyb_drafted : 0.0,
               (double)hyb_accepted / hyb_rounds);
        printf("  draft proposal %.3f s, exact verify/replay %.3f s\n",
               hyb_draft_s, hyb_verify_s);
''',
    "hybrid timing report",
)

s = once(
    s,
    '''        fprintf(f, "],\"layers\":%d,\"threads\":%d,\"temperature\":%.9g,"
                   "\"top_p\":%.9g,\"seed\":%llu,\"stop_id\":%d,"
                   "\"seconds_per_token\":%.4f}\n",
                NL, compute_threads, temperature, top_p, sample_seed, stop_id, t_total / nout);
''',
    '''        fprintf(f, "],\"layers\":%d,\"threads\":%d,\"temperature\":%.9g,"
                   "\"top_p\":%.9g,\"seed\":%llu,\"stop_id\":%d,"
                   "\"draft_rounds\":%ld,\"draft_proposed\":%ld,"
                   "\"draft_accepted\":%ld,\"draft_acceptance\":%.9g,"
                   "\"draft_seconds\":%.6f,\"verify_seconds\":%.6f,"
                   "\"seconds_per_token\":%.4f}\n",
                NL, compute_threads, temperature, top_p, sample_seed, stop_id,
                hyb_rounds, hyb_drafted, hyb_accepted,
                hyb_drafted ? (double)hyb_accepted / hyb_drafted : 0.0,
                hyb_draft_s, hyb_verify_s, t_total / nout);
''',
    "hybrid JSON metrics",
)

p.write_text(s)
print("draft timing metrics materialized")
