#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected 1, got {n}")
    return s.replace(old, new, 1)

p = Path(__file__).resolve().parents[1] / "local/k3_local.py"
s = p.read_text()
s = once(s,
'''    worker_binary: Path | None = None
    worker_context: int = 1024
''',
'''    worker_binary: Path | None = None
    worker_context: int = 1024
    prefill_mb: float = 256.0
    prefill_chunk: int | None = None
''', "backend prefill fields")
s = once(s,
'''            "--context",
            str(self.context),
        ]
        if self.cfg.threads is not None:
''',
'''            "--context",
            str(self.context),
            "--prefill-mb",
            str(self.cfg.prefill_mb),
        ]
        if self.cfg.prefill_chunk is not None:
            cmd += ["--prefill-chunk", str(self.cfg.prefill_chunk)]
        if self.cfg.threads is not None:
''', "worker command prefill")
s = once(s,
'''        worker_context=args.worker_context,
    )
''',
'''        worker_context=args.worker_context,
        prefill_mb=args.prefill_mb,
        prefill_chunk=args.prefill_chunk,
    )
''', "serve backend prefill config")
s = once(s,
'''    sp.add_argument("--preset", default="laptop")
''',
'''    sp.add_argument(
        "--prefill-mb",
        type=float,
        default=256.0,
        help="transient resident prefill RAM budget in MiB (default: 256)",
    )
    sp.add_argument(
        "--prefill-chunk",
        type=int,
        help="manual prefill chunk override; normally leave unset",
    )
    sp.add_argument("--preset", default="laptop")
''', "serve prefill CLI args")
p.write_text(s)
print("adaptive prefill settings materialized")
