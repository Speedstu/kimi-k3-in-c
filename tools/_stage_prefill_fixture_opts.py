from pathlib import Path

p = Path('tools/make_tiny_checkpoint.py')
s = p.read_text(encoding='utf-8')

old = '''def build(seed: int, dense_intermediate: int = 96):
    torch.manual_seed(seed)
    cfg = tiny_config(moe_intermediate_size=64, intermediate_size=dense_intermediate)'''
new = '''def build(seed: int, dense_intermediate: int = 96, num_experts: int = 8,
          topk: int = 2):
    torch.manual_seed(seed)
    cfg = tiny_config(moe_intermediate_size=64, intermediate_size=dense_intermediate,
                      num_experts=num_experts, num_experts_per_token=topk)'''
if old not in s:
    raise SystemExit('build signature block not found')
s = s.replace(old, new, 1)

old = '''    ap.add_argument("--dense-intermediate", type=int, default=96,
                    help="dense layer-0 intermediate width; 512 mimics K3's oversized first layer")
    a = ap.parse_args()

    if a.dense_intermediate <= 0:
        ap.error("--dense-intermediate must be positive")
    os.makedirs(a.out_dir, exist_ok=True)
    cfg, model = build(a.seed, a.dense_intermediate)'''
new = '''    ap.add_argument("--dense-intermediate", type=int, default=96,
                    help="dense layer-0 intermediate width; 512 mimics K3's oversized first layer")
    ap.add_argument("--num-experts", type=int, default=8,
                    help="routed experts in the tiny fixture (default 8; useful for cache/prefill stress)")
    ap.add_argument("--topk", type=int, default=2,
                    help="routed experts selected per token (default 2)")
    a = ap.parse_args()

    if a.dense_intermediate <= 0:
        ap.error("--dense-intermediate must be positive")
    if a.num_experts <= 0:
        ap.error("--num-experts must be positive")
    if a.topk <= 0 or a.topk > a.num_experts or a.topk > 64:
        ap.error("--topk must be in 1..min(num-experts,64)")
    os.makedirs(a.out_dir, exist_ok=True)
    cfg, model = build(a.seed, a.dense_intermediate, a.num_experts, a.topk)'''
if old not in s:
    raise SystemExit('argparse/build call block not found')
s = s.replace(old, new, 1)

old = 'Usage: make_tiny_checkpoint.py <out_dir> [--seed N] [--prompt-ids a,b,c]'
new = ('Usage: make_tiny_checkpoint.py <out_dir> [--seed N] [--prompt-ids a,b,c]\n'
       '                                      [--num-experts N] [--topk K]')
if old not in s:
    raise SystemExit('usage line not found')
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8')
