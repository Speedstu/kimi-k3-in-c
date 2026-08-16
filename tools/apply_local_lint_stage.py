#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "local/k3_local.py"
s = p.read_text()

repls = {
    'O = "<|open|>"': 'OPEN = "<|open|>"',
    'think_open = O + "think" + S': 'think_open = OPEN + "think" + S',
    'response_open = O + "response" + S': 'response_open = OPEN + "response" + S',
    'tools_open = O + "tools" + S': 'tools_open = OPEN + "tools" + S',
    're.escape(O)': 're.escape(OPEN)',
    'def do_GET(self) -> None:  # noqa: N802': 'def do_GET(self) -> None:',
    'def do_POST(self) -> None:  # noqa: N802': 'def do_POST(self) -> None:',
    'except Exception as exc:  # noqa: BLE001 - HTTP API boundary': 'except Exception as exc:',
}
for old, new in repls.items():
    n = s.count(old)
    if n == 0:
        raise SystemExit(f"missing expected fragment: {old}")
    s = s.replace(old, new)

if ' O ' in s or 'O +' in s or 're.escape(O)' in s:
    raise SystemExit('ambiguous O constant still referenced')

p.write_text(s)
print('local bridge lint cleanup applied')
