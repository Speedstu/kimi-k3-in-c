#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'local/k3_local.py'
s = p.read_text()
old = 'from typing import Any\n'
new = 'from typing import Any, ClassVar\n'
if s.count(old) != 1:
    raise SystemExit(f'typing import anchor: expected 1, got {s.count(old)}')
s = s.replace(old, new, 1)
old = '    _PRESET_BUDGETS = {\n'
new = '    _PRESET_BUDGETS: ClassVar[dict[str, tuple[float, float]]] = {\n'
if s.count(old) != 1:
    raise SystemExit(f'preset ClassVar anchor: expected 1, got {s.count(old)}')
s = s.replace(old, new, 1)
p.write_text(s)
print('resident bridge lint typing fixed')
