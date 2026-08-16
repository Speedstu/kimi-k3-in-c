#!/usr/bin/env python3
from pathlib import Path

root=Path(__file__).resolve().parents[1]

def once(p, old, new, label):
    s=p.read_text(); n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1 got {n}')
    p.write_text(s.replace(old,new,1))

run=root/'src/cli/k3_run.c'
once(run,
'''                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                    hyb_draft_s += now_s() - hyb_draft_t0;
''',
'''                    hyb_rounds  += 1;
                    hyb_drafted += nd;
                    hyb_draft_s += now_s() - hyb_draft_t0;
                    if (getenv("K3_TRACE_SPEC")) {
                        fprintf(stderr, "TRACE one round=%ld base=%d nd=%d d=", hyb_rounds, base, nd);
                        for (int ti=0; ti<nd; ti++) fprintf(stderr, "%s%d", ti?",":"", d[ti]);
                        fprintf(stderr, "\\n");
                    }
''','one proposal trace')
once(run,
'''                    if (frc == 0) {
                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                        emit[emitn++] = correction;
                    }
''',
'''                    if (frc == 0) {
                        if (getenv("K3_TRACE_SPEC"))
                            fprintf(stderr, "TRACE one verify base=%d m=%d correction=%d cached=%d\\n",
                                    base, m, correction, w.cached);
                        for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                        emit[emitn++] = correction;
                    }
''','one verify trace')

w=root/'src/cli/k3_worker.c'
once(w,
'''                draft_rounds++;
                draft_proposed += nd;
                draft_seconds += now_s() - td;
''',
'''                draft_rounds++;
                draft_proposed += nd;
                draft_seconds += now_s() - td;
                if (getenv("K3_TRACE_SPEC")) {
                    fprintf(stderr, "TRACE worker round=%ld base=%d nd=%d d=", draft_rounds, base, nd);
                    for (int ti=0; ti<nd; ti++) fprintf(stderr, "%s%d", ti?",":"", d[ti]);
                    fprintf(stderr, "\\n");
                }
''','worker proposal trace')
once(w,
'''                for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                emit[emitn++] = correction;
''',
'''                if (getenv("K3_TRACE_SPEC"))
                    fprintf(stderr, "TRACE worker verify base=%d m=%d correction=%d cached=%d\\n",
                            base, m, correction, w.cached);
                for (int i = 0; i < m; i++) emit[emitn++] = d[i];
                emit[emitn++] = correction;
''','worker verify trace')
print('temporary speculation traces injected')
