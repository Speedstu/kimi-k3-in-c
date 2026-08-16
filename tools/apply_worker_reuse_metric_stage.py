#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'src/cli/k3_worker.c'
s=p.read_text()
old='''        const int reused = history_len > 0 && np >= history_len &&
                           memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0;
        if (!reused) {
'''
new='''        const int reuse_tokens = (history_len > 0 && np >= history_len &&
                                  memcmp(req, seq, (size_t)history_len * sizeof(int)) == 0)
                                 ? history_len : 0;
        if (!reuse_tokens) {
'''
if s.count(old)!=1: raise SystemExit(f'reuse anchor: {s.count(old)}')
s=s.replace(old,new,1)
old='''        printf("@K3DONE %llu %d %d %d %.6f\\n", rid, nout, w.cached, reused,
               now_s() - started);
'''
new='''        printf("@K3DONE %llu %d %d %d %.6f\\n", rid, nout, w.cached, reuse_tokens,
               now_s() - started);
'''
if s.count(old)!=1: raise SystemExit(f'done anchor: {s.count(old)}')
s=s.replace(old,new,1)
p.write_text(s)
print('worker reuse token metric materialized')
