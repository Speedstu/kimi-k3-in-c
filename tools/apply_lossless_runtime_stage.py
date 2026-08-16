#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one match, found {n}")
    return s.replace(old, new, 1)

# ---- k3_trunk.h ---------------------------------------------------------------------
p = root / "src/io/k3_trunk.h"
s = p.read_text()
s = one(s,
'''typedef struct {
    int64_t  file_off;     /* offset in trunk.bin */
    int64_t  nbytes;
    K3TrunkTensor *t;
    int      nt;
} K3TrunkLayer;''',
'''typedef struct {
    int64_t file_off;       /* absolute offset in trunk.bin, 4096 aligned */
    int64_t stored_nbytes;  /* bytes read with O_DIRECT, includes tail padding */
    int64_t encoded_nbytes; /* payload bytes before O_DIRECT padding */
    int64_t raw_off;        /* destination offset in the reconstructed layer run */
    int64_t raw_nbytes;
    int     codec;          /* 0 raw, 1 dict15 */
    unsigned char dict[K3_DICT15_SIZE];
} K3TrunkBlock;

typedef struct {
    int64_t  file_off;     /* offset in trunk.bin (raw trunks) / first block */
    int64_t  nbytes;       /* RECONSTRUCTED raw bytes; tensor offsets refer to these */
    int64_t  stored_nbytes;/* physical bytes for compressed blocks, informational */
    K3TrunkTensor *t;
    int      nt;
    K3TrunkBlock *blocks;  /* NULL for the original raw one-pread format */
    int      nblocks;
} K3TrunkLayer;''',
"block types")
s = one(s,
'''    unsigned char *arena;       /* [nslot] uniform ring slots                   */
    int64_t      slot_bytes;    /* raw run + the widen area                     */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */''',
'''    unsigned char *arena;       /* [nslot] uniform RECONSTRUCTED ring slots     */
    unsigned char *codec_arena; /* [nslot] compressed-block O_DIRECT scratch     */
    int64_t      slot_bytes;    /* raw run + the widen area                     */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */
    int64_t      codec_slot_bytes; /* max stored block, counted inside budget    */''',
"codec arena fields")
s = one(s,
'''    uint64_t     bytes_read;
    double       load_seconds;''',
'''    uint64_t     bytes_read;             /* physical bytes read from trunk.bin */
    uint64_t     raw_bytes_reconstructed; /* raw bytes produced for the binder   */
    double       load_seconds;             /* pread only                          */
    double       decode_seconds;           /* dict15 reconstruction only          */''',
"codec stats")
s = one(s, '#include "k3_bind.h"\n', '#include "k3_bind.h"\n#include "k3_codec.h"\n', "codec header include")
p.write_text(s, newline="\n")

# ---- k3_trunk.c ---------------------------------------------------------------------
p = root / "src/io/k3_trunk.c"
s = p.read_text()
s = one(s, '#include "k3_st.h"\n#include "k3_trunk.h"\n',
        '#include "k3_st.h"\n#include "k3_trunk.h"\n', "trunk include anchor")

# Parse optional block directory after tensors.
anchor = '''        for (int k = 0; k < ts->len; k++) {
            K3TrunkTensor *t = &L->t[k];
            /* keys live in the parser arena, which is kept for the process lifetime */
            t->name = ts->keys[k];
            jval *o = ts->kids[k];
            if ((v = json_get(o, "off"))    && v->t == J_NUM) t->off    = (int64_t)v->num;
            if ((v = json_get(o, "nbytes")) && v->t == J_NUM) t->nbytes = (int64_t)v->num;
            if ((v = json_get(o, "dtype"))  && v->t == J_STR) t->dtype  = dt_of(v->str);
        }
'''
insert = anchor + '''        if ((v = json_get(e, "stored_nbytes")) && v->t == J_NUM)
            L->stored_nbytes = (int64_t)v->num;
        jval *bs = json_get(e, "blocks");
        if (bs) {
            if (bs->t != J_ARR || bs->len <= 0) {
                fprintf(stderr, "k3_trunk: layer %d has invalid blocks array\\n", i);
                goto bad;
            }
            L->nblocks = bs->len;
            L->blocks = (K3TrunkBlock *)calloc((size_t)L->nblocks, sizeof(K3TrunkBlock));
            if (!L->blocks) goto bad;
            int64_t expect_raw = 0;
            for (int bi = 0; bi < L->nblocks; bi++) {
                jval *bo = bs->kids[bi];
                K3TrunkBlock *b = &L->blocks[bi];
                if ((v = json_get(bo, "file_off")) && v->t == J_NUM) b->file_off = (int64_t)v->num;
                if ((v = json_get(bo, "stored_nbytes")) && v->t == J_NUM) b->stored_nbytes = (int64_t)v->num;
                if ((v = json_get(bo, "encoded_nbytes")) && v->t == J_NUM) b->encoded_nbytes = (int64_t)v->num;
                if ((v = json_get(bo, "raw_off")) && v->t == J_NUM) b->raw_off = (int64_t)v->num;
                if ((v = json_get(bo, "raw_nbytes")) && v->t == J_NUM) b->raw_nbytes = (int64_t)v->num;
                jval *co = json_get(bo, "codec");
                if (!co || co->t != J_STR) { fprintf(stderr, "k3_trunk: layer %d block %d has no codec\\n", i, bi); goto bad; }
                if (!strcmp(co->str, "raw")) b->codec = 0;
                else if (!strcmp(co->str, "dict15")) b->codec = 1;
                else { fprintf(stderr, "k3_trunk: layer %d block %d unknown codec %s\\n", i, bi, co->str); goto bad; }
                if (b->raw_off != expect_raw || b->raw_nbytes <= 0 || b->stored_nbytes <= 0 ||
                    b->encoded_nbytes <= 0 || b->encoded_nbytes > b->stored_nbytes ||
                    (b->file_off % K3_TRUNK_ALIGN) || (b->stored_nbytes % K3_TRUNK_ALIGN)) {
                    fprintf(stderr, "k3_trunk: layer %d block %d has invalid geometry\\n", i, bi);
                    goto bad;
                }
                if (b->codec == 1) {
                    jval *da = json_get(bo, "dict");
                    if (!da || da->t != J_ARR || da->len != K3_DICT15_SIZE) {
                        fprintf(stderr, "k3_trunk: layer %d block %d needs a 15-byte dictionary\\n", i, bi);
                        goto bad;
                    }
                    for (int di = 0; di < K3_DICT15_SIZE; di++) {
                        if (!da->kids[di] || da->kids[di]->t != J_NUM ||
                            da->kids[di]->num < 0 || da->kids[di]->num > 255) {
                            fprintf(stderr, "k3_trunk: layer %d block %d bad dictionary\\n", i, bi);
                            goto bad;
                        }
                        b->dict[di] = (unsigned char)da->kids[di]->num;
                    }
                } else if (b->encoded_nbytes != b->raw_nbytes) {
                    fprintf(stderr, "k3_trunk: raw block length mismatch at layer %d block %d\\n", i, bi);
                    goto bad;
                }
                expect_raw += b->raw_nbytes;
            }
            if (expect_raw != L->nbytes) {
                fprintf(stderr, "k3_trunk: layer %d blocks reconstruct %lld bytes, manifest says %lld\\n",
                        i, (long long)expect_raw, (long long)L->nbytes);
                goto bad;
            }
        }
'''
s = one(s, anchor, insert, "parse blocks")

# Totals + codec scratch geometry.
s = one(s,
'''    const size_t widen = k3_bind_widen_bytes(c);
    int64_t total = 0;
    for (int i = 0; i < tr->n_layers; i++) total += tr->lay[i].nbytes;''',
'''    const size_t widen = k3_bind_widen_bytes(c);
    int64_t total = 0, stored_total = 0, codec_slot = 0;
    for (int i = 0; i < tr->n_layers; i++) {
        total += tr->lay[i].nbytes;
        if (tr->lay[i].nblocks) {
            for (int bi = 0; bi < tr->lay[i].nblocks; bi++) {
                const K3TrunkBlock *b = &tr->lay[i].blocks[bi];
                stored_total += b->stored_nbytes;
                if (b->stored_nbytes > codec_slot) codec_slot = b->stored_nbytes;
            }
        } else {
            stored_total += tr->lay[i].nbytes;
        }
    }
    codec_slot = (codec_slot + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);''',
"codec totals")

# Budget includes one scratch per ring slot.
s = one(s,
'''        RING = RING_WANT;
        while (RING > 1 && (int64_t)RING * rs > budget_bytes) RING--;

        int64_t sp = (int64_t)RING * rs;''',
'''        RING = RING_WANT;
        while (RING > 1 && (int64_t)RING * (rs + codec_slot) > budget_bytes) RING--;

        int64_t sp = (int64_t)RING * (rs + codec_slot);''',
"codec scratch budget")

# Store field + allocate.
s = one(s,
'''    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;''',
'''    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;
    tr->codec_slot_bytes = codec_slot;''',
"codec slot field")
s = one(s,
'''    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\\n",
                (double)RING * ring_slot / 1e9);
        return -1;
    }
    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));''',
'''    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\\n",
                (double)RING * ring_slot / 1e9);
        return -1;
    }
    if (codec_slot > 0 &&
        k3_alloc_direct((void **)&tr->codec_arena, (size_t)RING * (size_t)codec_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate %.2f GB codec scratch\\n",
                (double)RING * codec_slot / 1e9);
        return -1;
    }
    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));''',
"codec scratch alloc")

# Opening report.
s = one(s,
'''    printf("trunk stream: %.2f GB packed, %d/%d layers PINNED (%.2f GB), "
           "ring %d x %.2f GB\\n",
           (double)total / 1e9, npin, tr->n_layers,
           (double)(spent - (int64_t)RING * ring_slot) / 1e9,
           RING, (double)ring_slot / 1e9);''',
'''    printf("trunk stream: %.2f GB raw / %.2f GB stored, %d/%d layers PINNED, "
           "ring %d x %.2f GB\\n",
           (double)total / 1e9, (double)stored_total / 1e9, npin, tr->n_layers,
           RING, (double)ring_slot / 1e9);
    if (codec_slot > 0)
        printf("              lossless dict15 blocks: %.2f GB codec scratch per ring slot "
               "(counted in trunk budget)\\n", (double)codec_slot / 1e9);''',
"open report")

# Close allocations / blocks.
s = one(s,
'''    free(tr->arena); free(tr->layer_of); free(tr->slot_of);
    if (tr->lay) { for (int i = 0; i < tr->n_layers; i++) free(tr->lay[i].t); free(tr->lay); }''',
'''    free(tr->arena); free(tr->codec_arena); free(tr->layer_of); free(tr->slot_of);
    if (tr->lay) {
        for (int i = 0; i < tr->n_layers; i++) { free(tr->lay[i].t); free(tr->lay[i].blocks); }
        free(tr->lay);
    }''',
"close codec")

# Replace load_run with block-aware implementation.
start = s.find('static int load_run(K3Trunk *tr, int L, unsigned char *dst)')
end = s.find('\nstatic void *trunk_io_main', start)
if start < 0 or end < 0:
    raise RuntimeError('load_run bounds')
new_load = r'''static int pread_all(K3Trunk *tr, unsigned char *dst, int64_t nbytes, int64_t off)
{
    int64_t got = 0;
    const double t0 = now_s();
    while (got < nbytes) {
        ssize_t r = pread(tr->fd, dst + got, (size_t)(nbytes - got), (off_t)(off + got));
        if (r <= 0) return -1;
        got += r;
    }
    tr->load_seconds += now_s() - t0;
    tr->bytes_read += (uint64_t)got;
    return 0;
}

/* Read and, when needed, byte-exactly reconstruct one layer into dst. scratch_slot is
 * the ring slot index, which gives async and synchronous readers disjoint codec buffers. */
static int load_run(K3Trunk *tr, int L, unsigned char *dst, int scratch_slot)
{
    const K3TrunkLayer *lay = &tr->lay[L];
    if (!lay->nblocks) {
        if (pread_all(tr, dst, lay->nbytes, lay->file_off) != 0) {
            fprintf(stderr, "k3_trunk: short read on layer %d\n", L);
            return -1;
        }
        tr->raw_bytes_reconstructed += (uint64_t)lay->nbytes;
        return 0;
    }
    if (!tr->codec_arena || scratch_slot < 0 || scratch_slot >= tr->nslot) return -1;
    unsigned char *scratch = tr->codec_arena + (size_t)scratch_slot * tr->codec_slot_bytes;
    for (int bi = 0; bi < lay->nblocks; bi++) {
        const K3TrunkBlock *b = &lay->blocks[bi];
        unsigned char *out = dst + b->raw_off;
        if (b->codec == 0) {
            if (pread_all(tr, out, b->stored_nbytes, b->file_off) != 0) {
                fprintf(stderr, "k3_trunk: short raw block read layer %d block %d\n", L, bi);
                return -1;
            }
        } else {
            if (b->stored_nbytes > tr->codec_slot_bytes ||
                pread_all(tr, scratch, b->stored_nbytes, b->file_off) != 0) {
                fprintf(stderr, "k3_trunk: short compressed read layer %d block %d\n", L, bi);
                return -1;
            }
            const double td = now_s();
            const size_t used = k3_dict15_decode(out, (size_t)b->raw_nbytes, scratch,
                                                  (size_t)b->encoded_nbytes, b->dict);
            tr->decode_seconds += now_s() - td;
            if (used == SIZE_MAX) {
                fprintf(stderr, "k3_trunk: corrupt dict15 block layer %d block %d\n", L, bi);
                return -1;
            }
        }
        tr->raw_bytes_reconstructed += (uint64_t)b->raw_nbytes;
    }
    return 0;
}
'''
s = s[:start] + new_load + s[end:]

# Call sites.
s = s.replace('load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes)',
              'load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes, slot)')
s = one(s, 'if (load_run(tr, L, base) != 0) return -1;',
        'if (load_run(tr, L, base, 0) != 0) return -1;', "pinned load call")
s = one(s, 'if (load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes) != 0) return -1;',
        'if (load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes, slot) != 0) return -1;',
        "sync load call")

# Report codec savings/decode.
needle = '''    printf("  read %.2f GB in %.2f s (%.0f MB/s)\\n",
           (double)tr->bytes_read / 1e9, tr->load_seconds,
           tr->load_seconds > 0 ? (double)tr->bytes_read / 1e6 / tr->load_seconds : 0.0);'''
replacement = needle + '''
    if (tr->raw_bytes_reconstructed > tr->bytes_read) {
        printf("  reconstructed %.2f GB raw from %.2f GB stored (%.1f%% fewer bytes); "
               "decode %.2f s (%.0f MB/s raw)\\n",
               (double)tr->raw_bytes_reconstructed / 1e9, (double)tr->bytes_read / 1e9,
               100.0 * (1.0 - (double)tr->bytes_read / tr->raw_bytes_reconstructed),
               tr->decode_seconds,
               tr->decode_seconds > 0 ? (double)tr->raw_bytes_reconstructed / 1e6 / tr->decode_seconds : 0.0);
    }'''
s = one(s, needle, replacement, "codec report")
p.write_text(s, newline="\n")

# ---- Makefile ------------------------------------------------------------------------
p = root / "Makefile"
s = p.read_text()
s = one(s,
'''UNIT_TESTS := test_ops test_cache test_st test_cfg test_tok scale_test k3_model''',
'''UNIT_TESTS := test_ops test_cache test_st test_cfg test_tok test_trunk_codec scale_test k3_model''',
"unit test list")
s = one(s,
'''$(BIN)/test_st: tests/unit/test_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)
''',
'''$(BIN)/test_st: tests/unit/test_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_trunk_codec: tests/unit/test_trunk_codec.c | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)
''',
"codec test target")
s = one(s,
'''\t@echo "== config reader ==";     ./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
''',
'''\t@echo "== config reader ==";     ./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
\t@echo "== lossless trunk codec =="; ./$(BIN)/test_trunk_codec
''',
"codec test run")
p.write_text(s, newline="\n")
print("staged lossless runtime integration")
