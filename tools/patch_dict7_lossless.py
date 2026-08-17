#!/usr/bin/env python3
"""Stage the adaptive fixed-3-bit dict7 lossless trunk codec.

This is a development transform used by CI. It never changes model bytes: dict7 is only a
storage representation and every decoded block is compared byte-for-byte in the gates.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# C decoder
# ---------------------------------------------------------------------------
p = Path("src/io/k3_codec.h")
s = p.read_text()
anchor = "#define K3_DICT15_ESCAPE 15\n"
insert = anchor + "\n#define K3_DICT7_SIZE 7\n#define K3_DICT7_ESCAPE 7\n"
if "K3_DICT7_SIZE" not in s:
    if anchor not in s:
        raise SystemExit("codec constants anchor not found")
    s = s.replace(anchor, insert, 1)

end = "\n#endif /* K3_CODEC_H */\n"
fn = r'''

/* Fixed-3-bit companion to dict15.
 *
 * src layout for raw_nbytes = 2*N:
 *   low[N] | codes[ceil(3*N/8)] | escape_literals[...]
 *
 * Codes 0..6 index the block dictionary and code 7 is a literal high-byte escape.
 * Eight codes occupy exactly three bytes.  The 4096-entry 4-code table is rebuilt once
 * per block from its seven-byte dictionary, then the hot loop expands eight high bytes
 * from two table lookups.  On AVX2 builds the eight low/high bytes are interleaved with
 * one SSE-width unpack; the model still receives the original BF16 byte stream.
 */
static inline size_t k3_dict7_decode(unsigned char *dst, size_t raw_nbytes,
                                     const unsigned char *src, size_t encoded_nbytes,
                                     const unsigned char dict[K3_DICT7_SIZE])
{
    if (!dst || !src || !dict || (raw_nbytes & 1u)) return SIZE_MAX;
    const size_t n = raw_nbytes / 2u;
    const size_t cb = (3u * n + 7u) / 8u;
    if (encoded_nbytes < n + cb) return SIZE_MAX;
    const unsigned char *low = src;
    const unsigned char *code = src + n;
    const unsigned char *esc = code + cb;
    const size_t esc_cap = encoded_nbytes - n - cb;

    uint32_t high4[4096];
    unsigned char emask4[4096];
    for (unsigned v = 0; v < 4096u; v++) {
        uint32_t hv = 0;
        unsigned char em = 0;
        for (unsigned j = 0; j < 4u; j++) {
            const unsigned q = (v >> (3u * j)) & 7u;
            if (q < K3_DICT7_SIZE)
                hv |= (uint32_t)dict[q] << (8u * j);
            else
                em |= (unsigned char)(1u << j);
        }
        high4[v] = hv;
        emask4[v] = em;
    }

    size_t ne = 0, i = 0, ci = 0;
    for (; i + 8u <= n; i += 8u, ci += 3u) {
        const uint32_t bits = (uint32_t)code[ci]
                            | ((uint32_t)code[ci + 1u] << 8)
                            | ((uint32_t)code[ci + 2u] << 16);
        const unsigned a = bits & 0xfffu;
        const unsigned b = (bits >> 12) & 0xfffu;
        const uint64_t hp = (uint64_t)high4[a] | ((uint64_t)high4[b] << 32);
#if defined(__AVX2__)
        const __m128i vl = _mm_loadl_epi64((const __m128i *)(low + i));
        const __m128i vh = _mm_cvtsi64_si128((long long)hp);
        _mm_storeu_si128((__m128i *)(dst + 2u * i), _mm_unpacklo_epi8(vl, vh));
#else
        for (unsigned j = 0; j < 8u; j++) {
            dst[2u * (i + j)] = low[i + j];
            dst[2u * (i + j) + 1u] = (unsigned char)(hp >> (8u * j));
        }
#endif
        unsigned m = (unsigned)emask4[a] | ((unsigned)emask4[b] << 4);
#if defined(__GNUC__) || defined(__clang__)
        if ((size_t)__builtin_popcount(m) > esc_cap - ne) return SIZE_MAX;
#else
        unsigned mc = m, pc = 0; while (mc) { pc += mc & 1u; mc >>= 1; }
        if ((size_t)pc > esc_cap - ne) return SIZE_MAX;
#endif
        while (m) {
            const int bit = k3_codec_pop_lsb(&m);
            dst[2u * (i + (size_t)bit) + 1u] = esc[ne++];
        }
    }

    /* Generic tail. Packed trunks are 4096-byte aligned, hence N is divisible by 8;
     * this path exists for unit tests and defensive format handling. */
    for (; i < n; i++) {
        const size_t bit = 3u * i;
        const size_t bo = bit >> 3;
        const unsigned sh = (unsigned)(bit & 7u);
        uint32_t w = code[bo];
        if (bo + 1u < cb) w |= (uint32_t)code[bo + 1u] << 8;
        if (bo + 2u < cb) w |= (uint32_t)code[bo + 2u] << 16;
        const unsigned q = (w >> sh) & 7u;
        dst[2u * i] = low[i];
        if (q < K3_DICT7_SIZE) dst[2u * i + 1u] = dict[q];
        else {
            if (ne >= esc_cap) return SIZE_MAX;
            dst[2u * i + 1u] = esc[ne++];
        }
    }
    return ne;
}
'''
if "static inline size_t k3_dict7_decode" not in s:
    if end not in s:
        raise SystemExit("codec footer not found")
    s = s.replace(end, fn + end, 1)
p.write_text(s)

# ---------------------------------------------------------------------------
# Trunk manifest/runtime dispatch
# ---------------------------------------------------------------------------
p = Path("src/io/k3_trunk.h")
s = p.read_text()
s = s.replace("int     codec;          /* 0 raw, 1 dict15 */",
              "int     codec;          /* 0 raw, 1 dict15, 2 dict7 */")
p.write_text(s)

p = Path("src/io/k3_trunk.c")
s = p.read_text()
old = '''                if (!strcmp(co->str, "raw")) b->codec = 0;\n                else if (!strcmp(co->str, "dict15")) b->codec = 1;\n                else { fprintf(stderr, "k3_trunk: layer %d block %d unknown codec %s\\n", i, bi, co->str); goto bad; }'''
new = '''                if (!strcmp(co->str, "raw")) b->codec = 0;\n                else if (!strcmp(co->str, "dict15")) b->codec = 1;\n                else if (!strcmp(co->str, "dict7")) b->codec = 2;\n                else { fprintf(stderr, "k3_trunk: layer %d block %d unknown codec %s\\n", i, bi, co->str); goto bad; }'''
if old not in s:
    raise SystemExit("codec parser anchor not found")
s = s.replace(old, new, 1)

old = '''                if (b->codec == 1) {\n                    jval *da = json_get(bo, "dict");\n                    if (!da || da->t != J_ARR || da->len != K3_DICT15_SIZE) {\n                        fprintf(stderr, "k3_trunk: layer %d block %d needs a 15-byte dictionary\\n", i, bi);\n                        goto bad;\n                    }\n                    for (int di = 0; di < K3_DICT15_SIZE; di++) {'''
new = '''                if (b->codec == 1 || b->codec == 2) {\n                    const int dict_n = b->codec == 1 ? K3_DICT15_SIZE : K3_DICT7_SIZE;\n                    jval *da = json_get(bo, "dict");\n                    if (!da || da->t != J_ARR || da->len != dict_n) {\n                        fprintf(stderr, "k3_trunk: layer %d block %d needs a %d-byte dictionary\\n",\n                                i, bi, dict_n);\n                        goto bad;\n                    }\n                    for (int di = 0; di < dict_n; di++) {'''
if old not in s:
    raise SystemExit("dictionary parser anchor not found")
s = s.replace(old, new, 1)

s = s.replace('printf("              lossless dict15 blocks: %.2f GB codec scratch per ring slot "',
              'printf("              adaptive lossless blocks: %.2f GB codec scratch per ring slot "')

old = '''            const double td = now_s();\n            const size_t used = k3_dict15_decode(out, (size_t)b->raw_nbytes, scratch,\n                                                  (size_t)b->encoded_nbytes, b->dict);\n            tr->decode_seconds += now_s() - td;\n            if (used == SIZE_MAX) {\n                fprintf(stderr, "k3_trunk: corrupt dict15 block layer %d block %d\\n", L, bi);\n                return -1;\n            }'''
new = '''            const double td = now_s();\n            size_t used;\n            if (b->codec == 1)\n                used = k3_dict15_decode(out, (size_t)b->raw_nbytes, scratch,\n                                        (size_t)b->encoded_nbytes, b->dict);\n            else if (b->codec == 2)\n                used = k3_dict7_decode(out, (size_t)b->raw_nbytes, scratch,\n                                       (size_t)b->encoded_nbytes, b->dict);\n            else\n                used = SIZE_MAX;\n            tr->decode_seconds += now_s() - td;\n            if (used == SIZE_MAX) {\n                fprintf(stderr, "k3_trunk: corrupt compressed block layer %d block %d codec %d\\n",\n                        L, bi, b->codec);\n                return -1;\n            }'''
if old not in s:
    raise SystemExit("decode dispatch anchor not found")
s = s.replace(old, new, 1)
p.write_text(s)

# ---------------------------------------------------------------------------
# Python encoder: adaptive raw/dict15/dict7 selection by physical O_DIRECT bytes.
# ---------------------------------------------------------------------------
p = Path("tools/lossless_trunk.py")
s = p.read_text()
s = s.replace('"""Convert a normal packed trunk into a byte-exact dict15-compressed trunk.',
              '"""Convert a normal packed trunk into a byte-exact adaptive compressed trunk.')
s = s.replace('The input is the output of tools/pack_trunk.py. Every layer is split into independent\n128 MiB RAW blocks. A block is encoded as:\n\n    low-byte plane | packed 4-bit high-byte dictionary codes | escape high bytes\n\nThe 15 most common high bytes are chosen independently per block. Code 15 is an escape.\nIf a block does not shrink, it is stored raw. Every stored block starts and ends on a',
'''The input is the output of tools/pack_trunk.py. Every layer is split into independent\n128 MiB RAW blocks. Each block is evaluated as raw, dict15, and dict7. dict15 uses a\n4-bit code with 15 dictionary entries; dict7 uses a 3-bit code with seven entries. The\nlast code in either format is an escape followed by the literal high byte. The encoder\nchooses the representation with the fewest physical 4096-aligned O_DIRECT bytes (raw\nwins ties, then dict15). Every stored block starts and ends on a''')

s = s.replace("def encode_block(raw: bytes):", "def encode_block_dict15(raw: bytes):", 1)
anchor = '''    payload = low.tobytes() + codes.tobytes() + escapes.tobytes()\n    return payload, dictionary.tolist(), len(escapes)\n\n\ndef main() -> int:\n'''
add = '''    payload = low.tobytes() + codes.tobytes() + escapes.tobytes()\n    return payload, dictionary.tolist(), len(escapes)\n\n\ndef encode_block_dict7(raw: bytes):\n    if len(raw) & 1:\n        raise ValueError("dict7 blocks must contain an even number of bytes")\n    a = np.frombuffer(raw, dtype=np.uint8)\n    low = a[0::2]\n    high = a[1::2]\n    hist = np.bincount(high, minlength=256)\n    dictionary = np.argsort(-hist, kind="stable")[:7].astype(np.uint8)\n    lut = np.full(256, 7, dtype=np.uint8)\n    lut[dictionary] = np.arange(7, dtype=np.uint8)\n    q = lut[high]\n    codes = np.zeros((3 * len(q) + 7) // 8, dtype=np.uint8)\n    groups = len(q) // 8\n    if groups:\n        x = q[: groups * 8].reshape(groups, 8)\n        codes[0 : 3 * groups : 3] = x[:, 0] | (x[:, 1] << 3) | ((x[:, 2] & 3) << 6)\n        codes[1 : 3 * groups : 3] = (x[:, 2] >> 2) | (x[:, 3] << 1) | (x[:, 4] << 4) | ((x[:, 5] & 1) << 7)\n        codes[2 : 3 * groups : 3] = (x[:, 5] >> 1) | (x[:, 6] << 2) | (x[:, 7] << 5)\n    for i in range(groups * 8, len(q)):\n        bit = 3 * i\n        v = int(q[i])\n        codes[bit >> 3] |= np.uint8((v << (bit & 7)) & 0xff)\n        if (bit & 7) > 5 and (bit >> 3) + 1 < len(codes):\n            codes[(bit >> 3) + 1] |= np.uint8(v >> (8 - (bit & 7)))\n    escapes = high[q == 7]\n    payload = low.tobytes() + codes.tobytes() + escapes.tobytes()\n    return payload, dictionary.tolist(), len(escapes)\n\n\ndef choose_block_codec(raw: bytes):\n    p15, d15, e15 = encode_block_dict15(raw)\n    p7, d7, e7 = encode_block_dict7(raw)\n    candidates = [\n        ("raw", raw, None, 0),\n        ("dict15", p15, d15, e15),\n        ("dict7", p7, d7, e7),\n    ]\n    priority = {"raw": 0, "dict15": 1, "dict7": 2}\n    return min(candidates, key=lambda x: (align_up(len(x[1])), priority[x[0]]))\n\n\ndef main() -> int:\n'''
if anchor not in s:
    raise SystemExit("encoder function anchor not found")
s = s.replace(anchor, add, 1)
s = s.replace('outman["storage_codec"] = "dict15-block-v1"',
              'outman["storage_codec"] = "adaptive-dict15-dict7-block-v2"')
s = s.replace("compressed_blocks = raw_blocks = 0", "dict15_blocks = dict7_blocks = raw_blocks = 0")
old = '''            payload, dictionary, nesc = encode_block(raw)\n            use_codec = len(payload) < raw_n\n            blob = payload if use_codec else raw'''
new = '''            codec, blob, dictionary, nesc = choose_block_codec(raw)'''
if old not in s:
    raise SystemExit("main codec selection anchor not found")
s = s.replace(old, new, 1)
old = '''                "codec": "dict15" if use_codec else "raw",'''
new = '''                "codec": codec,'''
s = s.replace(old, new, 1)
old = '''            if use_codec:\n                block["dict"] = dictionary\n                block["escapes"] = nesc\n                compressed_blocks += 1\n            else:\n                raw_blocks += 1'''
new = '''            if codec != "raw":\n                block["dict"] = dictionary\n                block["escapes"] = nesc\n                if codec == "dict7":\n                    dict7_blocks += 1\n                else:\n                    dict15_blocks += 1\n            else:\n                raw_blocks += 1'''
if old not in s:
    raise SystemExit("main manifest codec anchor not found")
s = s.replace(old, new, 1)
s = s.replace('print(f"blocks: {compressed_blocks} dict15, {raw_blocks} raw fallback")',
              'print(f"blocks: {dict7_blocks} dict7, {dict15_blocks} dict15, {raw_blocks} raw fallback")')
p.write_text(s)

# ---------------------------------------------------------------------------
# C unit gate: arbitrary tail, escapes, and truncation for dict7 as well.
# ---------------------------------------------------------------------------
p = Path("tests/unit/test_trunk_codec.c")
s = p.read_text()
needle = '''    printf("LOSSLESS TRUNK CODEC PASSED: %zu bytes, %zu escapes, byte-identical\\n",\n           raw_n, ne);\n    free(raw); free(low); free(code); free(esc); free(out); free(src);\n    return 0;\n}'''
replacement = '''    /* dict7 uses an independent 3-bit packing and must survive a non-multiple-of-eight\n     * tail plus frequent escapes. */\n    const unsigned char d7[7] = {0x3c,0xbc,0x3d,0xbd,0x3b,0xbb,0x3e};\n    const size_t cb7 = (3u * n + 7u) / 8u;\n    unsigned char *c7 = (unsigned char *)calloc(cb7, 1);\n    unsigned char *e7 = (unsigned char *)malloc(n);\n    unsigned char *r7 = (unsigned char *)malloc(raw_n);\n    unsigned char *o7 = (unsigned char *)malloc(raw_n);\n    if (!c7 || !e7 || !r7 || !o7) return 2;\n    size_t ne7 = 0;\n    rs = 0x51f15e11u;\n    for (size_t j = 0; j < n; j++) {\n        const uint32_t r = rnd();\n        unsigned char q, hi;\n        if ((r % 29u) == 0u) { q = 7; hi = (unsigned char)(0x70u + ((r >> 16) & 31u)); e7[ne7++] = hi; }\n        else { q = (unsigned char)((r >> 8) % 7u); hi = d7[q]; }\n        r7[2u*j] = low[j]; r7[2u*j+1u] = hi;\n        const size_t bit = 3u*j, bo = bit >> 3;\n        const unsigned sh = (unsigned)(bit & 7u);\n        c7[bo] |= (unsigned char)(q << sh);\n        if (sh > 5u && bo + 1u < cb7) c7[bo+1u] |= (unsigned char)(q >> (8u-sh));\n    }\n    const size_t enc7 = n + cb7 + ne7;\n    unsigned char *s7 = (unsigned char *)malloc(enc7);\n    if (!s7) return 2;\n    memcpy(s7, low, n); memcpy(s7+n, c7, cb7); memcpy(s7+n+cb7, e7, ne7);\n    const size_t used7 = k3_dict7_decode(o7, raw_n, s7, enc7, d7);\n    if (used7 != ne7 || memcmp(o7, r7, raw_n) != 0) {\n        fprintf(stderr, "FAIL dict7 roundtrip: escapes %zu/%zu, bytes_equal=%d\\n",\n                used7, ne7, memcmp(o7, r7, raw_n) == 0);\n        return 1;\n    }\n    if (ne7 && k3_dict7_decode(o7, raw_n, s7, enc7 - 1u, d7) != SIZE_MAX) {\n        fprintf(stderr, "FAIL dict7 accepted truncated escape payload\\n");\n        return 1;\n    }\n    printf("LOSSLESS TRUNK CODECS PASSED: dict15 %zu escapes, dict7 %zu escapes, byte-identical\\n",\n           ne, ne7);\n    free(raw); free(low); free(code); free(esc); free(out); free(src);\n    free(c7); free(e7); free(r7); free(o7); free(s7);\n    return 0;\n}'''
if needle not in s:
    raise SystemExit("unit-test footer anchor not found")
s = s.replace(needle, replacement, 1)
p.write_text(s)
print("staged adaptive dict7 lossless trunk codec")
