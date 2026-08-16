#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'src/cli/k3_worker.c'
s = p.read_text()
old = '''int main(int argc, char **argv)\n{\n    if (argc < 2)'''
new = '''int main(int argc, char **argv)\n{\n    /* stdout is the machine protocol. Configure it BEFORE any helper can print: when\n     * connected to Python it is a pipe, and the default full buffering can otherwise\n     * hold @K3READY forever while the client waits for exactly that line. */\n    setvbuf(stdout, NULL, _IOLBF, 0);\n    if (argc < 2)'''
if s.count(old) != 1:
    raise SystemExit(f'worker main anchor: expected 1, got {s.count(old)}')
s = s.replace(old, new, 1)
late = '''    setvbuf(stdout, NULL, _IOLBF, 0);\n    printf("@K3READY %d %d\\n", context, c.vocab);'''
repl = '''    printf("@K3READY %d %d\\n", context, c.vocab);'''
if s.count(late) != 1:
    raise SystemExit(f'late buffering anchor: expected 1, got {s.count(late)}')
s = s.replace(late, repl, 1)
p.write_text(s)
print('worker protocol buffering fixed')
