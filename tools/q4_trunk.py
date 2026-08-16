#!/usr/bin/env python3
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
