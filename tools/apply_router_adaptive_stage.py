#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected one match, found {n}')
    return s.replace(old, new, 1)

# Header: separate gate storage tag from the rest of the layer matrices.
p = root / 'include/k3/k3.h'
s = p.read_text()
s = one(s,
'''    const void  *gate;                   /* router: [n_experts][hidden], tagged by wdt */
    const float *bias;                   /* [n_experts], elementwise: stays fp32       */''',
'''    const void  *gate;                   /* router: [n_experts][hidden]                */
    const float *bias;                   /* [n_experts], elementwise: stays fp32       */
    int          gate_wdt;               /* resident FP32; streamed/draft native dtype */''',
'gate_wdt field')
p.write_text(s, newline='\n')

# Core call sites use gate-specific storage tag.
p = root / 'src/core/k3_ops.c'
s = p.read_text()
s = s.replace('w->gate, w->wdt, w->bias, E, c->n_experts, K,',
              'w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,')
if s.count('w->gate, w->gate_wdt, w->bias, E, c->n_experts, K,') != 2:
    raise RuntimeError('expected two typed router call sites')
p.write_text(s, newline='\n')

# Binder: resident gates remain FP32; streaming gates stay native.
p = root / 'src/model/k3_bind.c'
s = p.read_text()
s = one(s,
'''    int             narrow;    /* 1 = keep bf16 bytes, 0 = widen to fp32             */
    const void    **dest;''',
'''    int             narrow;    /* 1 = keep bf16 bytes, 0 = widen to fp32             */
    int             router_gate;/* resident wide, streamed native storage              */
    const void    **dest;''',
'Req router_gate')
s = one(s,
'''    q->narrow = narrow && p->narrow_ok;
    q->t = NULL; q->off = 0;''',
'''    q->narrow = narrow && p->narrow_ok;
    q->router_gate = 0;
    q->t = NULL; q->off = 0;''',
'initialize router_gate')
anchor = '''/* NARROW: kept as the checkpoint's bf16. */
static void reqn(Plan *p, const void **dest, int64_t want, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, dest, 1, want, -1, fmt, ap);
    va_end(ap);
}
'''
reqg = anchor + '''
/* ROUTER GATE: resident checkpoint binding widens once to FP32 (fastest repeated
 * router), while streaming binding points directly at BF16/I8R/Q4G to avoid a full
 * gate-sized conversion every layer/token. */
static void reqg(Plan *p, const void **dest, int64_t want, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, dest, 0, want, -1, fmt, ap);
    va_end(ap);
    p->r[p->n - 1].router_gate = 1;
}
'''
s = one(s, anchor, reqg, 'reqg helper')
s = one(s,
'''        /* Gate is a real matrix too. Exact BF16 is consumed directly by k3_router;
         * draft I8R/Q4G is proposal-only and follows the layer's tagged matrix format. */
        reqn(p, &b->moe.gate, (int64_t)c->n_experts * H,
             PRE "layers.%d.block_sparse_moe.gate.weight", L);''',
'''        reqg(p, &b->moe.gate, (int64_t)c->n_experts * H,
             PRE "layers.%d.block_sparse_moe.gate.weight", L);''',
'plan router reqg')
# Resident: gate was widened by reqg, independently of other matrix storage.
s = one(s,
'''    b->kda.wdt = b->mla.wdt = b->moe.wdt = b->lay.wdt = wdt;

    /* Exactly one of kda/mla is non-NULL;''',
'''    b->kda.wdt = b->mla.wdt = b->moe.wdt = b->lay.wdt = wdt;
    if (!is_dense) b->moe.gate_wdt = K3_WF32;

    /* Exactly one of kda/mla is non-NULL;''',
'resident gate tag')
# Streaming special case must precede draft generic dequantization paths.
loop_anchor = '''        if (src->find(src->ctx, q->name, &off, &nb, &dt) != 0) {
            fprintf(stderr, "k3_bind_mem: %s not present in the packed run\\n", q->name);
            return -1;
        }
'''
special = loop_anchor + '''        if (q->router_gate) {
            if (dt == K3_DT_BF16) {
                if (nb != q->take * 2) {
                    fprintf(stderr, "k3_bind_mem: %s bad BF16 router size\\n", q->name);
                    return -1;
                }
                *q->dest = run + off;
                b->moe.gate_wdt = K3_WBF16;
                continue;
            }
            if (dt == K3_DT_I8R) {
                *q->dest = run + off;
                b->moe.gate_wdt = K3_WI8;
                i8_seen = 1;
                continue;
            }
            if (dt == K3_DT_Q4G) {
                *q->dest = run + off;
                b->moe.gate_wdt = K3_WQ4G;
                q4_seen = 1;
                continue;
            }
            if (dt == K3_DT_F32) {
                if (nb != q->take * 4) {
                    fprintf(stderr, "k3_bind_mem: %s bad FP32 router size\\n", q->name);
                    return -1;
                }
                *q->dest = run + off;
                b->moe.gate_wdt = K3_WF32;
                continue;
            }
            fprintf(stderr, "k3_bind_mem: %s unsupported router dtype %d\\n", q->name, dt);
            return -1;
        }
'''
s = one(s, loop_anchor, special, 'streaming router special case')
p.write_text(s, newline='\n')
print('staged adaptive router storage: resident FP32, streamed native')
