#!/usr/bin/env python3
from pathlib import Path

p = Path("tools/make_tiny_checkpoint.py")
s = p.read_text()

if "--router-bf16" in s:
    print("tiny router BF16 option already applied")
    raise SystemExit(0)

old = '''    ap.add_argument("--topk", type=int, default=2,\n                    help="routed experts selected per token (default 2)")\n    a = ap.parse_args()\n'''
new = '''    ap.add_argument("--topk", type=int, default=2,\n                    help="routed experts selected per token (default 2)")\n    ap.add_argument("--router-bf16", action="store_true",\n                    help="store MoE router gate matrices as native BF16, matching the released streamed K3 path")\n    a = ap.parse_args()\n'''
if old not in s:
    raise SystemExit("argparse anchor not found")
s = s.replace(old, new, 1)

old = '''            if is_expert(name):\n                packed, scales = mxfp4_quant(w)\n                p.copy_(torch.from_numpy(mxfp4_dequant(packed, scales)))\n            elif is_reqn(eng, name):\n                p.copy_(torch.from_numpy(bf16_roundtrip(w)))\n            # else: keep exact fp32 (reqw path)\n'''
new = '''            if is_expert(name):\n                packed, scales = mxfp4_quant(w)\n                p.copy_(torch.from_numpy(mxfp4_dequant(packed, scales)))\n            elif a.router_bf16 and eng.endswith(".block_sparse_moe.gate.weight"):\n                # The released streamed binder preserves router gates in native BF16.\n                # Round the torch model too, so ref_logits describes the exact bytes\n                # this special fixture writes rather than the default F32 router.\n                p.copy_(torch.from_numpy(bf16_roundtrip(w)))\n            elif is_reqn(eng, name):\n                p.copy_(torch.from_numpy(bf16_roundtrip(w)))\n            # else: keep exact fp32 (reqw path)\n'''
if old not in s:
    raise SystemExit("reference dtype transform anchor not found")
s = s.replace(old, new, 1)

old = '''        elif is_reqn(eng, name):\n            entry = [(eng, bf16_of(w))]\n            print("  %-78s %s %s" % (eng, entry[0][1].dtype, list(entry[0][1].shape)))\n        else:\n            entry = [(eng, ww.astype(np.float32))]\n'''
new = '''        elif a.router_bf16 and eng.endswith(".block_sparse_moe.gate.weight"):\n            # Emit BF16 directly through the order-preserving in-tree writer. Do not\n            # rewrite the safetensors file afterward: pack_trunk requires each layer's\n            # original contiguous tensor run to stay intact.\n            entry = [(eng, bf16_of(w))]\n            print("  %-78s %s %s  [native BF16 router]" %\n                  (eng, entry[0][1].dtype, list(entry[0][1].shape)))\n        elif is_reqn(eng, name):\n            entry = [(eng, bf16_of(w))]\n            print("  %-78s %s %s" % (eng, entry[0][1].dtype, list(entry[0][1].shape)))\n        else:\n            entry = [(eng, ww.astype(np.float32))]\n'''
if old not in s:
    raise SystemExit("safetensors serialization anchor not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("added native BF16 router fixture option")
