#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import OrderedDict, deque
import numpy as np

EXPERT_BYTES=17_547_264


def lru(trace,cap):
    q=OrderedDict(); h=0
    for k in trace:
        if k in q:
            h+=1; q.move_to_end(k)
        else:
            if len(q)>=cap: q.popitem(last=False)
            q[k]=None
    return h


def slru(trace,cap,protected_frac=0.8):
    # probation for one-touch entries, protected for entries proven reusable.
    pcap=max(1,int(cap*protected_frac)); qcap=max(1,cap-pcap)
    prob=OrderedDict(); prot=OrderedDict(); h=0
    for k in trace:
        if k in prot:
            h+=1; prot.move_to_end(k); continue
        if k in prob:
            h+=1; del prob[k]; prot[k]=None
            if len(prot)>pcap:
                x,_=prot.popitem(last=False); prob[x]=None
                if len(prob)>qcap: prob.popitem(last=False)
            continue
        prob[k]=None
        if len(prob)>qcap: prob.popitem(last=False)
    return h


def twoq(trace,cap,kin_frac=0.25,ghost_frac=1.0):
    kin=max(1,int(cap*kin_frac)); ghost_cap=max(1,int(cap*ghost_frac))
    a1=OrderedDict(); am=OrderedDict(); ghost=OrderedDict(); h=0
    def trim_resident():
        while len(a1)+len(am)>cap:
            if len(a1)>kin:
                x,_=a1.popitem(last=False); ghost[x]=None
                if len(ghost)>ghost_cap: ghost.popitem(last=False)
            elif am:
                am.popitem(last=False)
            else:
                x,_=a1.popitem(last=False); ghost[x]=None
                if len(ghost)>ghost_cap: ghost.popitem(last=False)
    for k in trace:
        if k in am:
            h+=1; am.move_to_end(k); continue
        if k in a1:
            h+=1; del a1[k]; am[k]=None; trim_resident(); continue
        if k in ghost:
            del ghost[k]
            am[k]=None; trim_resident(); continue
        a1[k]=None; trim_resident()
        while len(a1)>kin:
            x,_=a1.popitem(last=False); ghost[x]=None
            if len(ghost)>ghost_cap: ghost.popitem(last=False)
    return h


def s3fifo(trace,cap,small_frac=0.1,ghost_frac=0.9):
    # Practical S3-FIFO-like policy: new pages enter a small FIFO; repeated small pages
    # are promoted to main. Main uses a tiny frequency counter and second chances.
    small_cap=max(1,int(cap*small_frac)); main_cap=max(1,cap-small_cap)
    ghost_cap=max(1,int(cap*ghost_frac))
    small=deque(); main=deque(); ghost=deque()
    where={}       # key -> 's'/'m'
    freq={}        # resident key -> 0..3
    ghostset=set(); h=0

    def ghost_add(k):
        if k in ghostset: return
        ghost.append(k); ghostset.add(k)
        while len(ghost)>ghost_cap:
            x=ghost.popleft(); ghostset.discard(x)

    def evict_main():
        # frequency second chance; every rotation decreases one credit.
        spins=0
        while len(main)>=main_cap and main:
            k=main.popleft(); spins+=1
            f=freq.get(k,0)
            if f>0:
                freq[k]=f-1; main.append(k)
            else:
                where.pop(k,None); freq.pop(k,None); ghost_add(k); return
            # Safety only; at most a few full rotations because freq <=3.
            if spins>4*max(len(main),1):
                k=main.popleft(); where.pop(k,None); freq.pop(k,None); ghost_add(k); return

    def promote(k):
        where[k]='m'; freq[k]=min(freq.get(k,0)+1,3); main.append(k)
        evict_main()

    def trim_small():
        while len(small)>small_cap:
            k=small.popleft()
            if where.get(k)!='s': continue
            if freq.get(k,0)>0:
                promote(k)
            else:
                where.pop(k,None); freq.pop(k,None); ghost_add(k)

    for k in trace:
        w=where.get(k)
        if w is not None:
            h+=1; freq[k]=min(freq.get(k,0)+1,3)
            # FIFO deliberately does not reorder on a hit.
            continue
        if k in ghostset:
            ghostset.discard(k)   # stale deque entry is skipped when eventually popped
            evict_main(); where[k]='m'; freq[k]=1; main.append(k)
            continue
        where[k]='s'; freq[k]=0; small.append(k); trim_small()
        while len(where)>cap:
            if small: trim_small()
            else: evict_main()
    return h


def main(path):
    raw=np.fromfile(path,dtype=np.int32)
    if raw.size%2: raise SystemExit('odd trace')
    p=raw.reshape(-1,2)
    tr=((p[:,0].astype(np.int64)<<20)|p[:,1].astype(np.int64)).tolist()
    n=len(tr); ntok=max(n//1472,1)
    print(f'trace requests={n} distinct={len(set(tr))} tokens~={ntok}')
    print('GB     slots    LRU      SLRU     2Q       S3FIFO   best_vs_LRU   GBread LRU->best')
    for gb in (0.5,1,2,4,8,16,32,64,128,192):
        cap=int(gb*1e9//EXPERT_BYTES)
        if cap<17: continue
        vals={
            'LRU':lru(tr,cap),
            'SLRU':max(slru(tr,cap,f) for f in (0.6,0.75,0.85,0.9)),
            '2Q':max(twoq(tr,cap,f,g) for f in (0.1,0.2,0.25,0.33) for g in (0.5,1.0,2.0)),
            'S3':max(s3fifo(tr,cap,f,g) for f in (0.05,0.1,0.2) for g in (0.5,0.9,1.5)),
        }
        bestn=max(vals,key=vals.get); best=vals[bestn]; base=vals['LRU']
        miss0=n-base; miss1=n-best
        gb0=miss0*EXPERT_BYTES/1e9/ntok; gb1=miss1*EXPERT_BYTES/1e9/ntok
        gain=(miss0/miss1 if miss1 else float('inf'))
        print(f'{gb:5.1f} {cap:8d} '+ ' '.join(f'{100*vals[x]/n:7.2f}%' for x in ('LRU','SLRU','2Q','S3')) +
              f' {bestn:>5s} {gain:7.3f}x   {gb0:6.2f}->{gb1:6.2f}')
    return 0

if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
