#!/usr/bin/env python3
from pathlib import Path

p = Path("src/core/k3_ops.c")
s = p.read_text()

late = '''#define K3_PREFILL_BATCH_MAX 64\nstatic void k3_matmul_bf16_batch(float *y, int ystride,\n                                  const float *const *xs, int batch,\n                                  const uint16_t *W, int in, int out);\n\nsize_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)\n'''
late_new = '''size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)\n'''
if late not in s:
    raise SystemExit("late MLA batch declaration anchor not found")
s = s.replace(late, late_new, 1)

early = '''void k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,\n                   int T, float *scratch,\n                   float *kvc, float *ropec, int cached, int cap)\n'''
early_new = '''#define K3_PREFILL_BATCH_MAX 64\nstatic void k3_matmul_bf16_batch(float *y, int ystride,\n                                  const float *const *xs, int batch,\n                                  const uint16_t *W, int in, int out);\n\nvoid k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,\n                   int T, float *scratch,\n                   float *kvc, float *ropec, int cached, int cap)\n'''
if early not in s:
    raise SystemExit("MLA function declaration anchor not found")
s = s.replace(early, early_new, 1)

p.write_text(s)
print("moved exact BF16 batch declaration before MLA")
