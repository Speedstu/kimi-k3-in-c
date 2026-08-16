#!/usr/bin/env python3
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, got {n}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]
p = root / "local/k3_local.py"
s = p.read_text()

s = once(
    s,
    '''    trunk_gb: float | None = None


@dataclass
''',
    '''    trunk_gb: float | None = None
    draft_trunk: Path | None = None
    draft_trunk_gb: float = 32.0
    draft_topk: int = 4
    spec: int = 4


@dataclass
''',
    "BackendConfig draft fields",
)

# Both one-shot and streaming paths have the same optional budget tail. Insert draft
# arguments immediately before the model subprocess starts. Exactly two occurrences.
old = '''                if self.cfg.trunk_gb is not None:
                    cmd += ["--trunk-gb", str(self.cfg.trunk_gb)]
'''
new = '''                if self.cfg.trunk_gb is not None:
                    cmd += ["--trunk-gb", str(self.cfg.trunk_gb)]
                if self.cfg.draft_trunk is not None:
                    cmd += [
                        "--draft-trunk",
                        str(self.cfg.draft_trunk),
                        "--draft-trunk-gb",
                        str(self.cfg.draft_trunk_gb),
                        "--draft-topk",
                        str(self.cfg.draft_topk),
                        "--spec",
                        str(self.cfg.spec),
                    ]
'''
if s.count(old) != 2:
    raise SystemExit(f"draft command insertion: expected 2 matches, got {s.count(old)}")
s = s.replace(old, new)

s = once(
    s,
    '''        trunk_gb=args.trunk_gb,
    )
''',
    '''        trunk_gb=args.trunk_gb,
        draft_trunk=args.draft_trunk.resolve() if args.draft_trunk else None,
        draft_trunk_gb=args.draft_trunk_gb,
        draft_topk=args.draft_topk,
        spec=args.spec,
    )
''',
    "serve config draft",
)

s = once(
    s,
    '''    for path, label in [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")
''',
    '''    for path, label in [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")
    if cfg.draft_trunk is not None and not cfg.draft_trunk.exists():
        raise SystemExit(f"draft trunk path does not exist: {cfg.draft_trunk}")
    if cfg.draft_topk < 1 or cfg.spec < 1:
        raise SystemExit("--draft-topk and --spec must both be >= 1")
''',
    "serve validate draft",
)

s = once(
    s,
    '''    if state_root is not None:
        print(
            f"conversation state cache: ON ({args.state_cache_entries} entry/entries; "
            f"root {state_root})"
        )
''',
    '''    if cfg.draft_trunk is not None:
        print(
            f"sampled speculative draft: ON ({cfg.draft_trunk}, top-{cfg.draft_topk}, "
            f"spec={cfg.spec}); exact K3 p/q verification remains authoritative"
        )
    else:
        print("sampled speculative draft: OFF")
    if state_root is not None:
        print(
            f"conversation state cache: ON ({args.state_cache_entries} entry/entries; "
            f"root {state_root})"
        )
''',
    "draft startup banner",
)

s = once(
    s,
    '''    sp.add_argument("--trunk-gb", type=float)
    sp.add_argument("--host", default="127.0.0.1")
''',
    '''    sp.add_argument("--trunk-gb", type=float)
    sp.add_argument(
        "--draft-trunk",
        type=Path,
        help="optional local Q4/I8/BF16 speculative draft trunk; exact K3 still verifies",
    )
    sp.add_argument("--draft-trunk-gb", type=float, default=32.0)
    sp.add_argument("--draft-topk", type=int, default=4)
    sp.add_argument("--spec", type=int, default=4)
    sp.add_argument("--host", default="127.0.0.1")
''',
    "server CLI draft flags",
)
p.write_text(s)

# Update local docs: sampled speculation is no longer a deliberate gap.
p = root / "local/README.md"
s = p.read_text()
s = once(
    s,
    '''Sampled speculative decoding is intentionally **not** faked. The existing speculative
path is exact for greedy decode; proper sampled speculation needs rejection sampling using
both target and draft probabilities. Until that is implemented, asking for sampling plus
`--spec` / `--draft-trunk` is rejected instead of silently changing the target
distribution.
''',
    '''Sampled speculative decoding is now probability-correct. A draft proposal `y~q` is
accepted with `min(1,p(y)/q(y))`; on the first rejection the correction is drawn from the
normalised residual `(p-q)+`, and after a fully accepted block the extra token is sampled
from exact K3's next distribution. This preserves the target K3 distribution while allowing
a cheap Q4/I8 draft at `temperature=1`.

To enable it for the localhost/Kimi Code bridge, add for example:

```bash
python local/k3_local.py serve \\
  --model-dir ~/k3model --trunk ~/k3trunk-lossless --preset laptop --threads N \\
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4
```

The draft can change acceptance rate and wall-clock speed, but exact K3 remains the
verification/target distribution. Sweep draft top-k/spec length on the real machine rather
than assuming one setting is universally fastest.
''',
    "README sampled speculation",
)
p.write_text(s)
print("sampled speculative bridge/docs patch applied")
