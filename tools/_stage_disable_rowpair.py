from pathlib import Path

p = Path('src/core/k3_ops.c')
s = p.read_text(encoding='utf-8')

pairs = [
    (
        '    static int no_mx_xdouble = -1, no_mx_rowpair = -1;\n'
        '    if (no_mx_xdouble < 0) no_mx_xdouble = getenv("K3_NO_MX_XDOUBLE") ? 1 : 0;\n'
        '    if (no_mx_rowpair < 0) no_mx_rowpair = getenv("K3_NO_MX_ROWPAIR") ? 1 : 0;',
        '    static int no_mx_xdouble = -1, force_mx_rowpair = -1;\n'
        '    if (no_mx_xdouble < 0) no_mx_xdouble = getenv("K3_NO_MX_XDOUBLE") ? 1 : 0;\n'
        '    /* Repeated production-helper medians showed row-pair is ~3-6% slower on\n'
        '     * the AVX2 runner despite an earlier standalone prototype looking faster.\n'
        '     * Keep it only as an explicit experiment; the measured single-row path is\n'
        '     * the default. */\n'
        '    if (force_mx_rowpair < 0) force_mx_rowpair = getenv("K3_FORCE_MX_ROWPAIR") ? 1 : 0;'
    ),
    (
        '        const int use_mx_rowpair = use_mx_xdouble && !no_mx_rowpair;',
        '        const int use_mx_rowpair = use_mx_xdouble && force_mx_rowpair;'
    ),
]
for old, new in pairs:
    if old not in s:
        raise SystemExit('expected source fragment not found:\n' + old)
    s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8')
