#!/usr/bin/env python3
from pathlib import Path


def once(s, old, new, label):
    n=s.count(old)
    if n!=1: raise SystemExit(f'{label}: expected 1, got {n}')
    return s.replace(old,new,1)

p=Path(__file__).resolve().parents[1]/'local/k3_local.py'
s=p.read_text()
s=once(s,
'''        if cfg.worker_binary is None:
            raise ValueError("resident worker requested without a worker binary")
        if cfg.draft_trunk is not None:
            raise ValueError(
                "resident worker draft support is not enabled yet; use --no-resident-worker "
                "for sampled draft acceleration until that path is merged"
            )
        self.cfg = cfg
''',
'''        if cfg.worker_binary is None:
            raise ValueError("resident worker requested without a worker binary")
        self.cfg = cfg
''','remove draft rejection')
s=once(s,
'''        if self.cfg.threads is not None:
            cmd += ["--threads", str(self.cfg.threads)]
        return cmd
''',
'''        if self.cfg.threads is not None:
            cmd += ["--threads", str(self.cfg.threads)]
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
        return cmd
''','worker command draft args')
s=once(s,
'''            generated: list[int] = []
            try:
                for line in self.proc.stdout:
''',
'''            generated: list[int] = []
            draft_stats: dict[str, Any] = {}
            try:
                for line in self.proc.stdout:
''','draft stats init')
s=once(s,
'''                    if line.startswith(f"@K3ERROR {rid} "):
                        code = int(line.split()[2])
                        raise RuntimeError(f"resident K3 worker rejected/failed request (code {code})")
                    if line.startswith(f"@K3DONE {rid} "):
''',
'''                    if line.startswith(f"@K3ERROR {rid} "):
                        code = int(line.split()[2])
                        raise RuntimeError(f"resident K3 worker rejected/failed request (code {code})")
                    if line.startswith(f"@K3DRAFT {rid} "):
                        fields = line.split()
                        if len(fields) != 8:
                            raise RuntimeError(f"malformed resident worker DRAFT line: {line!r}")
                        proposed = int(fields[3])
                        accepted = int(fields[4])
                        draft_stats = {
                            "draft_rounds": int(fields[2]),
                            "draft_proposed": proposed,
                            "draft_accepted": accepted,
                            "draft_acceptance": accepted / proposed if proposed else 0.0,
                            "draft_seconds": float(fields[5]),
                            "verify_seconds": float(fields[6]),
                        }
                        continue
                    if line.startswith(f"@K3DONE {rid} "):
''','parse draft marker')
s=once(s,
'''                        return generated, {
                            "resident_worker": True,
                            "worker_seconds": seconds,
                            "worker_cached_positions": cached,
                            "state_cache_hit_tokens": reused,
                            "state_cache_suffix_tokens": len(prompt_ids) - reused,
                        }
''',
'''                        stats = {
                            "resident_worker": True,
                            "worker_seconds": seconds,
                            "worker_cached_positions": cached,
                            "state_cache_hit_tokens": reused,
                            "state_cache_suffix_tokens": len(prompt_ids) - reused,
                        }
                        stats.update(draft_stats)
                        return generated, stats
''','return draft stats')
p.write_text(s)
print('resident sampled-draft bridge materialized')
