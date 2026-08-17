#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, struct, zlib, lzma
from collections import defaultdict
from pathlib import Path
MAGIC=0x31444b564b334b
HDR=struct.Struct('<QQII')
def shuffled(b: bytes)->bytes:
    return b[0::4]+b[1::4]+b[2::4]+b[3::4]
def xorb(a: bytes,b: bytes)->bytes:
    return bytes(x^y for x,y in zip(a,b))
def ratio_z(b: bytes)->float: return len(zlib.compress(b,9))/len(b)
def ratio_xz(b: bytes)->float: return len(lzma.compress(b,preset=6))/len(b)
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('--json-out'); args=ap.parse_args()
    raw=Path(args.dump).read_bytes(); off=0; layers=defaultdict(list)
    while off<len(raw):
        if off+HDR.size>len(raw): raise SystemExit('truncated record header')
        magic,key,pos,nf=HDR.unpack_from(raw,off); off+=HDR.size
        if magic!=MAGIC: raise SystemExit(f'bad magic at {off-HDR.size}: {magic:x}')
        n=nf*4
        if off+n>len(raw): raise SystemExit('truncated record data')
        layers[key].append((pos,raw[off:off+n])); off+=n
    if not layers: raise SystemExit('no records')
    all_raw=b''.join(v for rows in layers.values() for _,v in sorted(rows))
    all_shuf=b''.join(shuffled(v) for rows in layers.values() for _,v in sorted(rows))
    xor_stream=[]
    for rows in layers.values():
        rows=sorted(rows)
        prev=None
        for _,v in rows:
            xor_stream.append(v if prev is None else xorb(v,prev)); prev=v
    all_xor=b''.join(xor_stream); all_xor_shuf=b''.join(shuffled(v) for v in xor_stream)
    methods={
        'raw_zlib':ratio_z(all_raw), 'shuffle_zlib':ratio_z(all_shuf),
        'temporal_xor_zlib':ratio_z(all_xor), 'xor_shuffle_zlib':ratio_z(all_xor_shuf),
        'raw_xz':ratio_xz(all_raw), 'shuffle_xz':ratio_xz(all_shuf),
        'xor_shuffle_xz':ratio_xz(all_xor_shuf),
    }
    best=min(methods,key=methods.get)
    rows=sum(len(v) for v in layers.values())
    result={'schema':1,'layers':len(layers),'rows':rows,'raw_bytes':len(all_raw),'methods':methods,'best':best,'best_ratio':methods[best],
            'note':'tiny exact-C-cache probe; ratio is not a full-checkpoint claim'}
    print(json.dumps(result,indent=2,sort_keys=True))
    if args.json_out: Path(args.json_out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return 0
if __name__=='__main__': raise SystemExit(main())
