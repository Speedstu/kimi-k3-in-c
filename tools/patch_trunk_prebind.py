#!/usr/bin/env python3
from pathlib import Path

p = Path("src/io/k3_trunk.c")
s = p.read_text()

if "trunk_prepare_bind" in s:
    print("async trunk prebind already applied")
    raise SystemExit(0)


def one(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"expected one match, found {n}: {old[:120]!r}")
    s = s.replace(old, new, 1)


one(
    "static int find_in_layer(void *ctx, const char *name,\n"
    "                         int64_t *off, int64_t *nbytes, int *dtype)\n"
    "{\n"
    "    const K3TrunkLayer *L = ((Finder *)ctx)->L;\n"
    "    for (int i = 0; i < L->nt; i++)\n"
    "        if (!strcmp(L->t[i].name, name)) {\n"
    "            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;\n"
    "            return 0;\n"
    "        }\n"
    "    return -1;\n"
    "}\n",
    "static int find_in_layer(void *ctx, const char *name,\n"
    "                         int64_t *off, int64_t *nbytes, int *dtype)\n"
    "{\n"
    "    const K3TrunkLayer *L = ((Finder *)ctx)->L;\n"
    "    for (int i = 0; i < L->nt; i++)\n"
    "        if (!strcmp(L->t[i].name, name)) {\n"
    "            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;\n"
    "            return 0;\n"
    "        }\n"
    "    return -1;\n"
    "}\n\n"
    "static unsigned char *trunk_widen_ptr(const K3Trunk *tr, int L, unsigned char *base)\n"
    "{\n"
    "    const int64_t raw = (tr->lay[L].nbytes + K3_TRUNK_ALIGN - 1)\n"
    "                        & ~(int64_t)(K3_TRUNK_ALIGN - 1);\n"
    "    return base + raw;\n"
    "}\n\n"
    "/* A memory bind owns no allocation: its tensors point into base/widen. Preparing it\n"
    " * in the reader is therefore safe as long as the slot is not published until this\n"
    " * function has finished. */\n"
    "static int trunk_prepare_bind(K3Trunk *tr, int L, unsigned char *base,\n"
    "                              K3LayerBind *out, int async)\n"
    "{\n"
    "    Finder f; f.L = &tr->lay[L];\n"
    "    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;\n"
    "    const double t0 = now_s();\n"
    "    const int rc = k3_bind_layer_mem(tr->cfg, L, out, base, &src,\n"
    "                                     trunk_widen_ptr(tr, L, base),\n"
    "                                     (size_t)tr->widen_bytes, NULL);\n"
    "    tr->prepare_seconds += now_s() - t0;\n"
    "    if (rc == 0) {\n"
    "        tr->prepared_builds++;\n"
    "        if (async) tr->async_prepared_builds++;\n"
    "    }\n"
    "    return rc;\n"
    "}\n\n"
    "/* K3LayerW points at the kda/mla/moe members embedded in its containing bind. A\n"
    " * struct copy preserves tensor pointers into the trunk slot but would leave those\n"
    " * three self-pointers aimed at the source struct, so rebase exactly them. */\n"
    "static void trunk_copy_prepared(const K3Cfg *c, int L, K3LayerBind *dst,\n"
    "                                const K3LayerBind *src)\n"
    "{\n"
    "    *dst = *src;\n"
    "    dst->lay.kda = k3_is_mla(c, L) ? NULL : &dst->kda;\n"
    "    dst->lay.mla = k3_is_mla(c, L) ? &dst->mla : NULL;\n"
    "    dst->lay.moe = k3_is_dense(c, L) ? NULL : &dst->moe;\n"
    "    /* Defensive: memory binds never own a blob, and a copied prepared bind must\n"
    "     * remain a no-op for k3_bind_free(). */\n"
    "    dst->blob = NULL;\n"
    "    dst->nbytes = 0;\n"
    "}\n"
)

one(
    "    tr->fd = -1;\n\n"
    "    char p[1024];\n",
    "    tr->fd = -1;\n"
    "    tr->cfg = c;\n\n"
    "    char p[1024];\n"
)

one(
    "    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(unsigned char *));\n"
    "    if (!tr->pin) return -1;\n",
    "    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(unsigned char *));\n"
    "    tr->pin_prepared = (K3LayerBind *)calloc((size_t)(npin ? npin : 1), sizeof(K3LayerBind));\n"
    "    tr->pin_prepared_valid = (unsigned char *)calloc((size_t)(npin ? npin : 1), 1);\n"
    "    if (!tr->pin || !tr->pin_prepared || !tr->pin_prepared_valid) return -1;\n"
)

one(
    "    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));\n"
    "    for (int i = 0; i < RING; i++) tr->layer_of[i] = -1;\n"
    "    tr->widen_bytes = (int64_t)widen;\n",
    "    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));\n"
    "    tr->prepared = (K3LayerBind *)calloc((size_t)RING, sizeof(K3LayerBind));\n"
    "    tr->prepared_layer = (int *)malloc((size_t)RING * sizeof(int));\n"
    "    if (!tr->layer_of || !tr->prepared || !tr->prepared_layer) return -1;\n"
    "    for (int i = 0; i < RING; i++) { tr->layer_of[i] = -1; tr->prepared_layer[i] = -1; }\n"
    "    tr->widen_bytes = (int64_t)widen;\n"
)

one(
    "    if (tr->pin) { for (int i = 0; i < tr->npin; i++) free(tr->pin[i]); free(tr->pin); }\n"
    "    free(tr->arena); free(tr->codec_arena); free(tr->layer_of); free(tr->slot_of);\n",
    "    if (tr->pin) { for (int i = 0; i < tr->npin; i++) free(tr->pin[i]); free(tr->pin); }\n"
    "    free(tr->pin_prepared); free(tr->pin_prepared_valid);\n"
    "    free(tr->arena); free(tr->codec_arena); free(tr->layer_of); free(tr->slot_of);\n"
    "    free(tr->prepared); free(tr->prepared_layer);\n"
)

one(
    "        const int rc = load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes, slot);\n\n"
    "        pthread_mutex_lock(&io->mu);\n",
    "        unsigned char *base = tr->arena + (size_t)slot * tr->slot_bytes;\n"
    "        int rc = load_run(tr, L, base, slot);\n"
    "        if (rc == 0) {\n"
    "            rc = trunk_prepare_bind(tr, L, base, &tr->prepared[slot], 1);\n"
    "            if (rc == 0) tr->prepared_layer[slot] = L;\n"
    "        }\n\n"
    "        pthread_mutex_lock(&io->mu);\n"
)

one(
    "        for (int i = 0; i < tr->nslot; i++) {\n"
    "            if (tr->layer_of[i] >= 0) tr->slot_of[tr->layer_of[i]] = -1;\n"
    "            tr->layer_of[i] = -1;\n"
    "        }\n"
    "        tr->ring = 0;\n"
    "        if (load_run(tr, 0, tr->arena, 0) != 0) return -1;\n"
    "        tr->misses++;\n"
    "        base = tr->arena;\n",
    "        for (int i = 0; i < tr->nslot; i++) {\n"
    "            if (tr->layer_of[i] >= 0) tr->slot_of[tr->layer_of[i]] = -1;\n"
    "            tr->layer_of[i] = -1;\n"
    "            tr->prepared_layer[i] = -1;\n"
    "        }\n"
    "        tr->ring = 0;\n"
    "        if (load_run(tr, 0, tr->arena, 0) != 0) return -1;\n"
    "        if (trunk_prepare_bind(tr, 0, tr->arena, &tr->prepared[0], 0) != 0) return -1;\n"
    "        tr->prepared_layer[0] = 0;\n"
    "        tr->misses++;\n"
    "        base = tr->arena;\n"
)

one(
    "        if (tr->slot_of[L] < 0) {            /* first touch: load once, keep forever */\n"
    "            if (load_run(tr, L, base, 0) != 0) return -1;\n"
    "            tr->slot_of[L] = L;\n"
    "            tr->misses++;\n"
    "        } else {\n"
    "            tr->hits++;\n"
    "        }\n",
    "        if (tr->slot_of[L] < 0) {            /* first touch: load + bind once */\n"
    "            if (load_run(tr, L, base, 0) != 0) return -1;\n"
    "            if (trunk_prepare_bind(tr, L, base, &tr->pin_prepared[L], 0) != 0) return -1;\n"
    "            tr->pin_prepared_valid[L] = 1;\n"
    "            tr->slot_of[L] = L;\n"
    "            tr->misses++;\n"
    "        } else {\n"
    "            tr->hits++;\n"
    "        }\n"
)

one(
    "                if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;\n"
    "                /* Mark the slot EMPTY before reading into it, not after. */\n"
    "                tr->layer_of[slot] = -1;\n"
    "                if (load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes, slot) != 0) return -1;\n"
    "                tr->layer_of[slot] = L;\n"
    "                tr->misses++;\n",
    "                if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;\n"
    "                /* Mark both raw residency and prepared bind EMPTY before reuse. */\n"
    "                tr->layer_of[slot] = -1;\n"
    "                tr->prepared_layer[slot] = -1;\n"
    "                unsigned char *slot_base = tr->arena + (size_t)slot * tr->slot_bytes;\n"
    "                if (load_run(tr, L, slot_base, slot) != 0) return -1;\n"
    "                if (trunk_prepare_bind(tr, L, slot_base, &tr->prepared[slot], 0) != 0) return -1;\n"
    "                tr->prepared_layer[slot] = L;\n"
    "                tr->layer_of[slot] = L;\n"
    "                tr->misses++;\n"
)

old_tail = (
    "    Finder f; f.L = &tr->lay[L];\n"
    "    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;\n"
    "    unsigned char *widen = base + (((tr->lay[L].nbytes + K3_TRUNK_ALIGN - 1)\n"
    "                                    & ~(int64_t)(K3_TRUNK_ALIGN - 1)));\n"
    "    /* Pinned layers own exactly nbytes + widen; ring slots own slot_bytes. */\n"
    "    const size_t cap = (size_t)tr->widen_bytes;\n"
    "    const double tw = now_s();\n"
    "    const int rc = k3_bind_layer_mem(c, L, b, base, &src, widen, cap, NULL);\n"
    "    const double tnow = now_s();\n"
    "    k3_trunk_widen_wall += tnow - tw;\n"
    "    k3_trunk_bind_wall  += tnow - t_bind0;\n"
    "    return rc;\n"
)
new_tail = (
    "    const K3LayerBind *ready = NULL;\n"
    "    if (L < tr->npin) {\n"
    "        if (!tr->pin_prepared_valid[L]) return -1;\n"
    "        ready = &tr->pin_prepared[L];\n"
    "    } else {\n"
    "        int slot = -1;\n"
    "        if (tr->split_first && L == 0) slot = 0;\n"
    "        else for (int i = 0; i < tr->nslot; i++)\n"
    "            if (tr->layer_of[i] == L) { slot = i; break; }\n"
    "        if (slot < 0 || tr->prepared_layer[slot] != L) return -1;\n"
    "        ready = &tr->prepared[slot];\n"
    "    }\n"
    "    trunk_copy_prepared(c, L, b, ready);\n"
    "    tr->prepared_hits++;\n"
    "    k3_trunk_bind_wall += now_s() - t_bind0;\n"
    "    return 0;\n"
)
one(old_tail, new_tail)

one(
    "    if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;\n"
    "    tr->layer_of[slot] = -1;\n"
    "    io->layer = L;\n",
    "    if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;\n"
    "    tr->layer_of[slot] = -1;\n"
    "    tr->prepared_layer[slot] = -1;\n"
    "    io->layer = L;\n"
)

old_report = (
    "        const double serial = tr->load_seconds + k3_trunk_widen_wall;\n"
    "        const double overlapped = serial - k3_trunk_bind_wall;\n"
)
new_report = (
    "        const double serial = tr->load_seconds + tr->prepare_seconds;\n"
    "        const double overlapped = serial - k3_trunk_bind_wall;\n"
    "        printf(\"  prepared binds %llu built, %llu copied (%llu prepared on reader), %.3f s total\\n\",\n"
    "               (unsigned long long)tr->prepared_builds,\n"
    "               (unsigned long long)tr->prepared_hits,\n"
    "               (unsigned long long)tr->async_prepared_builds, tr->prepare_seconds);\n"
)
one(old_report, new_report)

s = s.replace("k3_trunk_widen_wall, serial, overlapped,", "tr->prepare_seconds, serial, overlapped,")
s = s.replace("k3_trunk_widen_wall, other);", "tr->prepare_seconds, other);")
s = s.replace("100.0 * k3_trunk_widen_wall / k3_trunk_bind_wall,", "100.0 * tr->prepare_seconds / k3_trunk_bind_wall,")
s = s.replace('"device work,\\n"', '"device+prepare work,\\n"')
s = s.replace('"  bind wall %.2f s over %ld binds  =  read %.2f + widen %.2f + other %.2f\\n",',
              '"  bind wall %.2f s over %ld binds  =  read %.2f + prepare %.2f + other %.2f\\n",')
s = s.replace('"                    shares:      read %.0f%%  widen %.0f%%  other %.0f%%\\n",',
              '"                    shares:      read %.0f%%  prepare %.0f%%  other %.0f%%\\n",')

p.write_text(s)
print("applied async trunk prebind")
