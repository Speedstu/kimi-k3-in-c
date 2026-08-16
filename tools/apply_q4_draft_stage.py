#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {n}')
    return s.replace(old, new, 1)

# ---- public weight format / dispatch -------------------------------------------------
p = ROOT / 'include/k3/k3.h'
s = p.read_text(encoding='utf-8')
s = one(s,
'''enum { K3_WF32 = 0, K3_WBF16 = 1, K3_WI8 = 2 };''',
'''enum { K3_WF32 = 0, K3_WBF16 = 1, K3_WI8 = 2, K3_WQ4G = 3 };''',
'weight dtype enum')
s = one(s,
'''/* Per-row int8 matmul for the draft model. W is `out` rows of [f32 scale][int8 * in].
 * No determinism contract (see K3_WI8): uses the fastest AVX2 form available. */
void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out);

/* The one call every trunk matmul goes through.''',
'''/* Per-row int8 matmul for the draft model. W is `out` rows of [f32 scale][int8 * in].
 * No determinism contract (see K3_WI8): uses the fastest AVX2 form available. */
void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out);

/* Groupwise signed int4 draft weights. Each 128-value group is stored as
 * [f32 scale][64 packed bytes], low nibble first, two's-complement signed nibbles.
 * This is proposal-only: exact K3 never carries K3_WQ4G. */
#define K3_Q4_GROUP 128
void k3_matmul_q4g(float *y, const float *x, const void *W, int in, int out);

static inline size_t k3_q4_row_bytes(int in)
{
    const int full = in / K3_Q4_GROUP, rem = in % K3_Q4_GROUP;
    return (size_t)full * (4u + K3_Q4_GROUP / 2u)
         + (rem ? 4u + (size_t)(rem + 1) / 2u : 0u);
}

/* The one call every trunk matmul goes through.''',
'q4 declarations')
s = one(s,
'''    if (wdt == K3_WBF16)     k3_matmul_bf16(y, x, (const uint16_t *)W, in, out);
    else if (wdt == K3_WI8)  k3_matmul_q8(y, x, W, in, out);
    else                     k3_matmul(y, x, (const float *)W, in, out);''',
'''    if (wdt == K3_WBF16)      k3_matmul_bf16(y, x, (const uint16_t *)W, in, out);
    else if (wdt == K3_WI8)   k3_matmul_q8(y, x, W, in, out);
    else if (wdt == K3_WQ4G)  k3_matmul_q4g(y, x, W, in, out);
    else                      k3_matmul(y, x, (const float *)W, in, out);''',
'k3_mmw q4 dispatch')
s = one(s,
'''static inline size_t k3_row_bytes(int wdt, int in)
{
    return wdt == K3_WI8 ? (size_t)4 + (size_t)in : (size_t)in * k3_wsz(wdt);
}''',
'''static inline size_t k3_row_bytes(int wdt, int in)
{
    if (wdt == K3_WI8)  return (size_t)4 + (size_t)in;
    if (wdt == K3_WQ4G) return k3_q4_row_bytes(in);
    return (size_t)in * k3_wsz(wdt);
}''',
'row bytes q4')
p.write_text(s, encoding='utf-8', newline='\n')

# ---- packed trunk dtype --------------------------------------------------------------
p = ROOT / 'src/io/k3_st.h'
s = p.read_text(encoding='utf-8')
s = one(s,
'''/* K3_DT_I8R is the packed trunk's per-row int8 draft format: each row is [f32 scale]
 * [int8 * cols]. It only ever appears in a draft trunk written by tools/int8_trunk.py. */
typedef enum { K3_DT_UNKNOWN = 0, K3_DT_U8, K3_DT_BF16, K3_DT_F16, K3_DT_F32,
               K3_DT_I8R } K3Dtype;''',
'''/* Draft-only packed trunk dtypes. I8R is [f32 row scale][int8 * cols]. Q4G is
 * groupwise signed int4 with one f32 scale per 128 values. Neither appears in the
 * released checkpoint; they are derived proposal trunks whose tokens exact K3 verifies. */
typedef enum { K3_DT_UNKNOWN = 0, K3_DT_U8, K3_DT_BF16, K3_DT_F16, K3_DT_F32,
               K3_DT_I8R, K3_DT_Q4G } K3Dtype;''',
'packed dtype enum')
p.write_text(s, encoding='utf-8', newline='\n')

p = ROOT / 'src/io/k3_trunk.c'
s = p.read_text(encoding='utf-8')
s = one(s,
'''    if (!strcmp(s, "I8R"))  return K3_DT_I8R;
    return K3_DT_UNKNOWN;''',
'''    if (!strcmp(s, "I8R"))  return K3_DT_I8R;
    if (!strcmp(s, "Q4G"))  return K3_DT_Q4G;
    return K3_DT_UNKNOWN;''',
'trunk manifest q4 dtype')
p.write_text(s, encoding='utf-8', newline='\n')

# ---- Q4 matmul kernel ----------------------------------------------------------------
p = ROOT / 'src/core/k3_ops.c'
s = p.read_text(encoding='utf-8')
needle = '''/* A whole BYTE to its two E2M1 values, so the inner loop does one 8-byte load instead
 * of masking, shifting and two separate lookups. 2 KB, built once, shared by all
 * threads after initialisation. */'''
q4 = r'''/* Groupwise signed-int4 matmul for the speculative draft trunk. Layout per row:
 * repeated groups of [f32 scale][ceil(n/2) packed bytes], group size 128. Nibbles are
 * two's-complement signed values (-8..7), low nibble = even element. The packer uses
 * symmetric absmax/7, so -8 is representable but normally unused.
 *
 * This kernel has intentionally NO exact-model determinism contract: Q4G is proposal
 * only and exact bf16 K3 verifies every emitted token. That lets the AVX2 path use float
 * FMA and a natural reduction. The goal is bandwidth: 0.53125 bytes/weight at full
 * groups versus 1.0 for I8R and 2.0 for bf16. */
void k3_matmul_q4g(float *y, const float *x, const void *W, int in, int out)
{
    const unsigned char *base = (const unsigned char *)W;
    const size_t rowb = k3_q4_row_bytes(in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const unsigned char *bp = base + (size_t)o * rowb;
        float acc = 0.0f;
        for (int g = 0, off = 0; off < in; g++, off += K3_Q4_GROUP) {
            int n = in - off;
            if (n > K3_Q4_GROUP) n = K3_Q4_GROUP;
            float scale;
            memcpy(&scale, bp, 4);
            const unsigned char *pk = bp + 4;
            const float *xg = x + off;
            float sub = 0.0f;
            int i = 0;
#if defined(__AVX2__)
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            __m256 v2 = _mm256_setzero_ps(), v3 = _mm256_setzero_ps();
            const __m128i mask = _mm_set1_epi8(0x0f);
            const __m128i sign = _mm_set1_epi8(0x08);
            for (; i + 31 < n; i += 32) {
                const __m128i b = _mm_loadu_si128((const __m128i *)(pk + (i >> 1)));
                __m128i lo = _mm_and_si128(b, mask);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
                lo = _mm_sub_epi8(_mm_xor_si128(lo, sign), sign);
                hi = _mm_sub_epi8(_mm_xor_si128(hi, sign), sign);
                const __m128i q0 = _mm_unpacklo_epi8(lo, hi);
                const __m128i q1 = _mm_unpackhi_epi8(lo, hi);
                v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),
                                     _mm256_loadu_ps(xg + i), v0);
                v1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))),
                    _mm256_loadu_ps(xg + i + 8), v1);
                v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),
                                     _mm256_loadu_ps(xg + i + 16), v2);
                v3 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))),
                    _mm256_loadu_ps(xg + i + 24), v3);
            }
            {
                const __m256 vs = _mm256_add_ps(_mm256_add_ps(v0, v1),
                                                 _mm256_add_ps(v2, v3));
                __m128 z = _mm_add_ps(_mm256_castps256_ps128(vs),
                                      _mm256_extractf128_ps(vs, 1));
                z = _mm_add_ps(z, _mm_movehl_ps(z, z));
                z = _mm_add_ss(z, _mm_shuffle_ps(z, z, 1));
                sub = _mm_cvtss_f32(z);
            }
#endif
            for (; i < n; i++) {
                const unsigned char b = pk[i >> 1];
                const int nib = (i & 1) ? (b >> 4) : (b & 0x0f);
                const int q = nib < 8 ? nib : nib - 16;
                sub += (float)q * xg[i];
            }
            acc += sub * scale;
            bp += 4u + (size_t)(n + 1) / 2u;
        }
        y[o] = acc;
    }
}

'''
if s.count(needle) != 1:
    raise RuntimeError(f'q4 insertion point: {s.count(needle)}')
s = s.replace(needle, q4 + needle, 1)
p.write_text(s, encoding='utf-8', newline='\n')

# ---- memory binder -------------------------------------------------------------------
p = ROOT / 'src/model/k3_bind.c'
s = p.read_text(encoding='utf-8')
s = one(s,
'''    int narrowed_all = 1;
    int i8_seen = 0;''',
'''    int narrowed_all = 1;
    int i8_seen = 0, q4_seen = 0;''',
'binder seen flags')
insert_before = '''        /* Per-row int8 draft weight: [f32 scale][int8 * cols] per row. A matmul weight is'''
q4bind = r'''        /* Groupwise Q4 draft tensor. The Q4 packer requires each matrix row width
         * to be a multiple of K3_Q4_GROUP, so the tensor is also just a sequence of
         * independent 128-value blocks. Narrow matmul weights point directly into the
         * run; elementwise tensors are dequantised into the existing widen area. */
        if (dt == K3_DT_Q4G) {
            if (q->narrow) {
                *q->dest = run + off;
                q4_seen = 1;
                continue;
            }
            const int64_t take = q->take;
            if (take < 0 || take % K3_Q4_GROUP != 0) {
                fprintf(stderr, "k3_bind_mem: %s bad Q4 logical size\n", q->name);
                return -1;
            }
            const int64_t blocks = take / K3_Q4_GROUP;
            const int64_t wantb = blocks * (4 + K3_Q4_GROUP / 2);
            if (nb != wantb) {
                fprintf(stderr, "k3_bind_mem: %s bad Q4 layout (%lld bytes, want %lld)\n",
                        q->name, (long long)nb, (long long)wantb);
                return -1;
            }
            w = (w + 7u) & ~(size_t)7u;
            if (w + (size_t)take * 4 > widen_cap) {
                fprintf(stderr, "k3_bind_mem: widen area too small at %s\n", q->name);
                return -1;
            }
            float *dst = (float *)(widen + w);
            const unsigned char *rp = run + off;
            for (int64_t g = 0; g < blocks; g++) {
                float scale;
                memcpy(&scale, rp, 4);
                const unsigned char *pk = rp + 4;
                float *dg = dst + g * K3_Q4_GROUP;
                for (int j = 0; j < K3_Q4_GROUP; j++) {
                    const unsigned char b = pk[j >> 1];
                    const int nib = (j & 1) ? (b >> 4) : (b & 0x0f);
                    const int qv = nib < 8 ? nib : nib - 16;
                    dg[j] = (float)qv * scale;
                }
                rp += 4 + K3_Q4_GROUP / 2;
            }
            *q->dest = dst;
            w += (size_t)take * 4;
            q4_seen = 1;
            continue;
        }
'''
if s.count(insert_before) != 1:
    raise RuntimeError('binder q4 insertion point')
s = s.replace(insert_before, q4bind + insert_before, 1)
s = one(s,
'''    if (!narrowed_all && !i8_seen) {''',
'''    if (!narrowed_all && !i8_seen && !q4_seen) {''',
'binder narrowed format guard')
s = one(s,
'''    /* An int8 draft trunk has every matmul weight as I8R (norms stay f32), so one tag
     * describes the layer. The two formats are never mixed within a packed trunk. */
    const int lw = i8_seen ? K3_WI8 : K3_WBF16;''',
'''    /* Derived draft trunks use one matrix format consistently within a layer. Refuse
     * mixed Q4/I8 because the dtype tag is per weight struct, not per tensor. */
    if (i8_seen && q4_seen) {
        fprintf(stderr, "k3_bind_mem: layer %d mixes I8R and Q4G matrices\n", L);
        return -1;
    }
    const int lw = q4_seen ? K3_WQ4G : (i8_seen ? K3_WI8 : K3_WBF16);''',
'binder q4 tag')
p.write_text(s, encoding='utf-8', newline='\n')

# ---- permanent kernel test -----------------------------------------------------------
p = ROOT / 'tests/unit/test_ops.c'
s = p.read_text(encoding='utf-8')
marker = '''int main(int argc, char **argv)\n{'''
test = r'''/* Draft-only Q4G kernel format check. This validates nibble order, sign extension,
 * group scale placement and row stride against an independent scalar decode. */
static void t_matmul_q4g(void)
{
    const int in = 256, out = 17, G = K3_Q4_GROUP;
    const size_t rowb = k3_q4_row_bytes(in);
    unsigned char *W = (unsigned char *)calloc((size_t)out, rowb);
    float *x = (float *)malloc((size_t)in * sizeof(float));
    float *got = (float *)malloc((size_t)out * sizeof(float));
    float *want = (float *)malloc((size_t)out * sizeof(float));
    if (!W || !x || !got || !want) { printf("  FAIL  matmul_q4g (alloc)\n"); g_fail++; return; }
    for (int i = 0; i < in; i++) x[i] = ((i * 37) % 101 - 50) * 0.007f;
    for (int o = 0; o < out; o++) {
        unsigned char *bp = W + (size_t)o * rowb;
        float ref = 0.0f;
        for (int g = 0; g < in / G; g++) {
            const float scale = 0.003f * (float)(1 + ((o + g) % 7));
            memcpy(bp, &scale, 4);
            unsigned char *pk = bp + 4;
            float sub = 0.0f;
            for (int j = 0; j < G; j += 2) {
                const int q0 = ((o * 11 + g * 5 + j) % 15) - 7;
                const int q1 = ((o * 13 + g * 3 + j + 1) % 15) - 7;
                pk[j >> 1] = (unsigned char)((q0 & 15) | ((q1 & 15) << 4));
                sub += (float)q0 * x[g * G + j];
                sub += (float)q1 * x[g * G + j + 1];
            }
            ref += sub * scale;
            bp += 4 + G / 2;
        }
        want[o] = ref;
    }
    k3_matmul_q4g(got, x, W, in, out);
    double worst = 0.0;
    for (int o = 0; o < out; o++) {
        const double d = fabs((double)got[o] - want[o]);
        const double tol = 2e-5 + 2e-5 * fabs((double)want[o]);
        if (d / tol > worst) worst = d / tol;
    }
    if (worst <= 1.0) { printf("  PASS  matmul_q4g    n=%d    worst=%.2fx tol\n", out, worst); g_pass++; }
    else              { printf("  FAIL  matmul_q4g    worst=%.2fx tol\n", worst); g_fail++; }
    free(W); free(x); free(got); free(want);
}

'''
if s.count(marker) != 1:
    raise RuntimeError('test main insertion point')
s = s.replace(marker, test + marker, 1)
s = one(s,
'''    t_matmul_bf16();
    t_kda_layer(dir, "kda_layer1");''',
'''    t_matmul_bf16();
    t_matmul_q4g();
    t_kda_layer(dir, "kda_layer1");''',
'test q4 call')
p.write_text(s, encoding='utf-8', newline='\n')

# ---- streaming Q4 packer -------------------------------------------------------------
p = ROOT / 'tools/q4_trunk.py'
p.write_text(r'''#!/usr/bin/env python3
"""Build a groupwise signed-int4 speculative draft trunk from a packed bf16 trunk.

Exact K3 is untouched and verifies every emitted token. Q4 affects proposal quality only.
Layout for every 2D BF16 tensor (group=128): repeated [f32 scale][64 packed int4 bytes]
for each row/group. Codes are signed two's-complement nibbles; scale is absmax/7.

Compared with I8R (~1 byte/weight), full Q4 groups use 68/128 = 0.53125 bytes/weight.
The converter streams row chunks instead of loading a multi-GB layer into RAM.

usage: python3 tools/q4_trunk.py <bf16_trunk_dir> <q4_out_dir>
"""
import json, os, sys
import numpy as np

ALIGN = 4096
GROUP = 128
TARGET_INPUT_BYTES = 16 << 20
COPY_CHUNK = 16 << 20


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def encode_rows(raw, rows, cols):
    if cols % GROUP:
        raise ValueError(f'Q4G requires cols divisible by {GROUP}, got {cols}')
    f = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(rows, cols)
    ng = cols // GROUP
    fg = f.reshape(rows, ng, GROUP)
    amax = np.abs(fg).max(axis=2)
    scale = (amax / 7.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.rint(fg / scale[:, :, None]).clip(-7, 7).astype(np.int8)
    qn = (q.astype(np.int16) & 15).astype(np.uint8)
    packed = qn[:, :, 0::2] | (qn[:, :, 1::2] << 4)
    block = np.empty((rows, ng, 4 + GROUP // 2), dtype=np.uint8)
    block[:, :, :4] = scale.view(np.uint8).reshape(rows, ng, 4)
    block[:, :, 4:] = packed
    return block.tobytes()


def copy_n(src, dst, n):
    left = n
    while left:
        b = src.read(min(left, COPY_CHUNK))
        if not b:
            raise IOError('unexpected EOF')
        dst.write(b)
        left -= len(b)


def main():
    if len(sys.argv) != 3:
        print('usage: q4_trunk.py <bf16_trunk_dir> <q4_out_dir>')
        return 2
    srcdir, outdir = sys.argv[1:]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(srcdir, 'trunk.json')) as f:
        man = json.load(f)
    src = open(os.path.join(srcdir, 'trunk.bin'), 'rb')
    dst = open(os.path.join(outdir, 'trunk.bin'), 'wb')
    outman = {k: v for k, v in man.items() if k != 'layers'}
    outman['q4_group'] = GROUP
    outman['layers'] = []
    nq = 0

    for li, lay in enumerate(man['layers']):
        pad = (-dst.tell()) % ALIGN
        if pad: dst.write(b'\0' * pad)
        file_off = dst.tell()
        run_pos = 0
        nt = {}
        items = sorted(lay['tensors'].items(), key=lambda kv: kv[1]['off'])
        for name, t in items:
            off, nb, dt = t['off'], t['nbytes'], t['dtype']
            shape = t.get('shape', [])
            src.seek(lay['file_off'] + off)
            if dt == 'BF16' and len(shape) == 2:
                rows, cols = int(shape[0]), int(shape[1])
                if cols % GROUP:
                    raise RuntimeError(f'{name}: cols={cols} not divisible by Q4 group {GROUP}')
                rchunk = max(1, TARGET_INPUT_BYTES // max(1, cols * 2))
                before = dst.tell()
                for r0 in range(0, rows, rchunk):
                    nr = min(rchunk, rows - r0)
                    raw = src.read(nr * cols * 2)
                    if len(raw) != nr * cols * 2:
                        raise IOError(f'short read at {name} row {r0}')
                    dst.write(encode_rows(raw, nr, cols))
                enb = dst.tell() - before
                expected = rows * (cols // GROUP) * (4 + GROUP // 2)
                if enb != expected:
                    raise RuntimeError(f'{name}: encoded {enb}, expected {expected}')
                nt[name] = {'off': run_pos, 'nbytes': enb, 'dtype': 'Q4G', 'shape': shape}
                run_pos += enb
                nq += 1
            else:
                copy_n(src, dst, nb)
                nt[name] = {'off': run_pos, 'nbytes': nb, 'dtype': dt, 'shape': shape}
                run_pos += nb

        pad = (-dst.tell()) % ALIGN
        if pad: dst.write(b'\0' * pad)
        run_bytes = (run_pos + ALIGN - 1) & ~(ALIGN - 1)
        nl = {k: v for k, v in lay.items() if k not in ('file_off', 'nbytes', 'tensors')}
        nl.update({'file_off': file_off, 'nbytes': run_bytes, 'tensors': nt})
        outman['layers'].append(nl)
        if (li + 1) % 10 == 0 or li + 1 == len(man['layers']):
            print(f'  q4 {li+1}/{len(man["layers"])} layers, {dst.tell()/1e9:.1f} GB out', flush=True)

    src.close(); dst.close()
    with open(os.path.join(outdir, 'trunk.json'), 'w') as f:
        json.dump(outman, f)
    sz = os.path.getsize(os.path.join(outdir, 'trunk.bin'))
    print(f'wrote {outdir}: {sz/1e9:.2f} GB, {nq} tensors Q4-quantised')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
''', encoding='utf-8', newline='\n')

print('staged Q4 draft trunk support')
