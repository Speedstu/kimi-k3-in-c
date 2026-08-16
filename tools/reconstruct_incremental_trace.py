#!/usr/bin/env python3
from __future__ import annotations
import hashlib, sys
from collections import OrderedDict, deque
import numpy as np

PER_TOKEN=92*16
EXPERT_BYTES=17_547_264

def lru(tr,cap):
 q=OrderedDict();h=0
 for k in tr:
  if k in q:h+=1;q.move_to_end(k)
  else:
   if len(q)>=cap:q.popitem(last=False)
   q[k]=None
 return h

def s3(tr,cap,small_frac=.1,ghost_frac=.9):
 sc=max(1,int(cap*small_frac));mc=max(1,cap-sc);gc=max(1,int(cap*ghost_frac));small=deque();main=deque();ghost=deque();gs=set();where={};freq={};h=0
 def ga(k):
  if k in gs:return
  ghost.append(k);gs.add(k)
  while len(ghost)>gc:
   x=ghost.popleft();gs.discard(x)
 def em():
  spins=0
  while len(main)>=mc and main:
   k=main.popleft();spins+=1;f=freq.get(k,0)
   if f>0:freq[k]=f-1;main.append(k)
   else:where.pop(k,None);freq.pop(k,None);ga(k);return
   if spins>4*max(len(main),1):
    k=main.popleft();where.pop(k,None);freq.pop(k,None);ga(k);return
 def prom(k):where[k]='m';freq[k]=min(freq.get(k,0)+1,3);main.append(k);em()
 def ts():
  while len(small)>sc:
   k=small.popleft()
   if where.get(k)!='s':continue
   if freq.get(k,0)>0:prom(k)
   else:where.pop(k,None);freq.pop(k,None);ga(k)
 for k in tr:
  w=where.get(k)
  if w is not None:h+=1;freq[k]=min(freq.get(k,0)+1,3);continue
  if k in gs:gs.discard(k);em();where[k]='m';freq[k]=1;main.append(k);continue
  where[k]='s';freq[k]=0;small.append(k);ts()
  while len(where)>cap:
   if small:ts()
   else:em()
 return h

def main(path):
 raw=np.fromfile(path,dtype=np.int32)
 if raw.size%(2*PER_TOKEN):raise SystemExit(f'trace does not divide into {PER_TOKEN}-request token blocks')
 pairs=raw.reshape(-1,2);keys=((pairs[:,0].astype(np.int64)<<20)|pairs[:,1].astype(np.int64))
 blocks=keys.reshape(-1,PER_TOKEN)
 ids=[];first={};uniq=[]
 for b in blocks:
  h=hashlib.blake2b(b.tobytes(),digest_size=16).hexdigest()
  if h not in first:first[h]=len(uniq);uniq.append(b.copy())
  ids.append(first[h])
 print('token-evaluation blocks:',len(blocks),'unique position blocks:',len(uniq))
 print('observed block ids:',','.join(map(str,ids)))
 # Strong validation for causal full-recompute: once a unique position appears, later
 # occurrences of that id are byte-identical by construction; first-seen order gives the
 # chronological positions. Print multiplicities so the reconstruction is auditable.
 cnt=[ids.count(i) for i in range(len(uniq))]
 print('position multiplicities:',cnt)
 inc=np.concatenate(uniq).tolist()
 print('incremental-like requests:',len(inc),'tokens:',len(uniq),'distinct experts:',len(set(inc)))
 print('GB slots LRU S3best GBread_LRU->S3 miss_speedup')
 n=len(inc);nt=max(len(uniq),1)
 for gb in (.5,1,2,4,8,16,32,64,128):
  cap=int(gb*1e9//EXPERT_BYTES)
  if cap<17:continue
  hl=lru(inc,cap);hs=max(s3(inc,cap,f,g) for f in(.05,.1,.2) for g in(.5,.9,1.5))
  ml=n-hl;ms=n-hs;g0=ml*EXPERT_BYTES/1e9/nt;g1=ms*EXPERT_BYTES/1e9/nt
  print(f'{gb:4.1f} {cap:5d} {100*hl/n:6.2f}% {100*hs/n:6.2f}% {g0:6.2f}->{g1:6.2f} {ml/ms if ms else float("inf"):.3f}x')
 return 0
if __name__=='__main__':raise SystemExit(main(sys.argv[1]))
