#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'local/k3_local.py'
s=p.read_text()
old='''                        if len(fields) != 8:
                            raise RuntimeError(f"malformed resident worker DRAFT line: {line!r}")
'''
new='''                        if len(fields) != 7:
                            raise RuntimeError(f"malformed resident worker DRAFT line: {line!r}")
'''
if s.count(old)!=1: raise SystemExit(f'DRAFT field-count anchor: {s.count(old)}')
s=s.replace(old,new,1)
p.write_text(s)
print('resident draft protocol field count fixed')
