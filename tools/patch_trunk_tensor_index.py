#!/usr/bin/env python3
from pathlib import Path

p = Path('src/io/k3_trunk.c')
s = p.read_text()

if 'trunk_tensor_name_cmp' in s:
    print('trunk tensor index already applied')
    raise SystemExit(0)

old = '''static int find_in_layer(void *ctx, const char *name,
                         int64_t *off, int64_t *nbytes, int *dtype)
{
    const K3TrunkLayer *L = ((Finder *)ctx)->L;
    for (int i = 0; i < L->nt; i++)
        if (!strcmp(L->t[i].name, name)) {
            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;
            return 0;
        }
    return -1;
}
'''
new = '''static int trunk_tensor_name_cmp(const void *a, const void *b)
{
    const K3TrunkTensor *x = (const K3TrunkTensor *)a;
    const K3TrunkTensor *y = (const K3TrunkTensor *)b;
    return strcmp(x->name, y->name);
}

static int find_in_layer(void *ctx, const char *name,
                         int64_t *off, int64_t *nbytes, int *dtype)
{
    K3TrunkLayer *L = (K3TrunkLayer *)((Finder *)ctx)->L;
    /* The manifest order is irrelevant to tensor offsets. Sort names once per layer so
     * every later prepared bind does O(log n) strcmp work instead of scanning the whole
     * layer table for every requested tensor. Preparation of one layer is serialized by
     * the trunk reader/current-layer path, so the one-time sort cannot race itself. */
    if (!L->names_sorted) {
        qsort(L->t, (size_t)L->nt, sizeof(*L->t), trunk_tensor_name_cmp);
        L->names_sorted = 1;
    }
    int lo = 0, hi = L->nt;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        const int cmp = strcmp(L->t[mid].name, name);
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo < L->nt && !strcmp(L->t[lo].name, name)) {
        *off = L->t[lo].off;
        *nbytes = L->t[lo].nbytes;
        *dtype = L->t[lo].dtype;
        return 0;
    }
    return -1;
}
'''
if s.count(old) != 1:
    raise SystemExit(f'find_in_layer anchor mismatch: {s.count(old)}')
s = s.replace(old, new, 1)
p.write_text(s)

h = Path('src/io/k3_trunk.h')
t = h.read_text()
anchor = '''    K3TrunkBlock *blocks;  /* NULL for the original raw one-pread format */
    int      nblocks;
} K3TrunkLayer;
'''
repl = '''    K3TrunkBlock *blocks;  /* NULL for the original raw one-pread format */
    int      nblocks;
    int      names_sorted; /* tensor manifest sorted once for binary name lookup */
} K3TrunkLayer;
'''
if t.count(anchor) != 1:
    raise SystemExit(f'K3TrunkLayer anchor mismatch: {t.count(anchor)}')
t = t.replace(anchor, repl, 1)
h.write_text(t)
print('applied sorted trunk tensor lookup')
