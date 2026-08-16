#!/usr/bin/env python3
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, got {n}")
    return text.replace(old, new, 1)

p = Path(__file__).resolve().parents[1] / "local/k3_local.py"
s = p.read_text()

s = once(s,
'''    draft_trunk_gb: float = 32.0
    draft_topk: int = 4
    spec: int = 4
''',
'''    draft_trunk_gb: float = 32.0
    draft_topk: int = 4
    spec: int = 4
    resident_worker: bool = True
    worker_binary: Path | None = None
    worker_context: int = 1024
''', "backend config worker fields")

marker = '''

class LocalK3:
'''
resident = r'''

class ResidentCBackend:
    """One warm C process: weights/index/trunk/cache and the active KV/KDA state stay live."""

    _PRESET_BUDGETS = {
        "laptop": (3.0, 1.0),
        "desktop": (16.0, 10.0),
        "workstation": (60.0, 30.0),
        "server": (110.0, 13.0),
        "max": (110.0, 109.0),
    }

    def __init__(self, cfg: BackendConfig):
        if cfg.worker_binary is None:
            raise ValueError("resident worker requested without a worker binary")
        if cfg.draft_trunk is not None:
            raise ValueError(
                "resident worker draft support is not enabled yet; use --no-resident-worker "
                "for sampled draft acceleration until that path is merged"
            )
        self.cfg = cfg
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.request_id = 0
        self.tail: deque[str] = deque(maxlen=200)
        self.context = int(cfg.worker_context)
        self.vocab = 0
        self._start_locked()
        atexit.register(self.close)

    def _budgets(self) -> tuple[float, float]:
        defaults = self._PRESET_BUDGETS.get(self.cfg.preset)
        if defaults is None:
            raise ValueError(
                f"resident worker cannot resolve preset {self.cfg.preset!r}; "
                "use a named fixed preset or explicit --trunk-gb/--cache-gb"
            )
        trunk = self.cfg.trunk_gb if self.cfg.trunk_gb is not None else defaults[0]
        cache = self.cfg.cache_gb if self.cfg.cache_gb is not None else defaults[1]
        return float(trunk), float(cache)

    def _command(self) -> list[str]:
        trunk_gb, cache_gb = self._budgets()
        cmd = [
            str(self.cfg.worker_binary),
            str(self.cfg.model_dir),
            "--trunk",
            str(self.cfg.trunk_dir),
            "--trunk-gb",
            str(trunk_gb),
            "--cache-gb",
            str(cache_gb),
            "--context",
            str(self.context),
        ]
        if self.cfg.threads is not None:
            cmd += ["--threads", str(self.cfg.threads)]
        return cmd

    def _start_locked(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        self.tail.clear()
        self.proc = subprocess.Popen(
            self._command(),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.tail.append(line)
            if line.startswith("@K3READY "):
                parts = line.split()
                if len(parts) != 3:
                    self._terminate_locked()
                    raise RuntimeError(f"malformed resident worker READY line: {line!r}")
                self.context = int(parts[1])
                self.vocab = int(parts[2])
                return
        rc = self.proc.wait()
        detail = "".join(self.tail)[-8000:]
        self.proc = None
        raise RuntimeError(f"resident K3 worker exited during startup ({rc})\n{detail}")

    def _terminate_locked(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                proc.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()

    def close(self) -> None:
        with self.lock:
            self._terminate_locked()

    def _request(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_id: int,
        on_token=None,
    ) -> tuple[list[int], dict[str, Any]]:
        if not prompt_ids:
            raise ValueError("resident worker needs at least one prompt token")
        if len(prompt_ids) + max_tokens > self.context:
            raise ValueError(
                f"resident worker context is {self.context} positions but this request needs "
                f"{len(prompt_ids) + max_tokens}; restart the server with --worker-context "
                "set high enough for the benchmark/session"
            )
        with self.lock:
            self._start_locked()
            assert self.proc is not None and self.proc.stdin is not None
            assert self.proc.stdout is not None
            self.request_id += 1
            rid = self.request_id
            header = (
                f"REQ {rid} {len(prompt_ids)} {max_tokens} {temperature:.17g} "
                f"{top_p:.17g} {int(seed)} {int(stop_id)}\n"
            )
            try:
                self.proc.stdin.write(header)
                self.proc.stdin.write(" ".join(map(str, prompt_ids)) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate_locked()
                raise RuntimeError("resident K3 worker pipe broke before request start") from exc

            generated: list[int] = []
            try:
                for line in self.proc.stdout:
                    if line.startswith(f"@K3TOKEN {rid} "):
                        token_id = int(line.split()[2])
                        generated.append(token_id)
                        if on_token is not None:
                            on_token(token_id)
                        continue
                    if line.startswith(f"@K3ERROR {rid} "):
                        code = int(line.split()[2])
                        raise RuntimeError(f"resident K3 worker rejected/failed request (code {code})")
                    if line.startswith(f"@K3DONE {rid} "):
                        fields = line.split()
                        if len(fields) != 6:
                            raise RuntimeError(f"malformed resident worker DONE line: {line!r}")
                        nout = int(fields[2])
                        cached = int(fields[3])
                        reused = int(fields[4])
                        seconds = float(fields[5])
                        if nout != len(generated):
                            raise RuntimeError(
                                "resident worker protocol drift: DONE token count differs "
                                "from streamed committed tokens"
                            )
                        return generated, {
                            "resident_worker": True,
                            "worker_seconds": seconds,
                            "worker_cached_positions": cached,
                            "state_cache_hit_tokens": reused,
                            "state_cache_suffix_tokens": len(prompt_ids) - reused,
                        }
                    self.tail.append(line)
            except BaseException:
                # A streaming HTTP client may disappear while C is in a multi-minute
                # token. There is no cancellable in-process request yet, so terminate the
                # worker rather than burn the laptop in the background. The next request
                # starts a fresh warm worker and remains numerically correct.
                self._terminate_locked()
                raise

            rc = self.proc.poll()
            detail = "".join(self.tail)[-8000:]
            self._terminate_locked()
            raise RuntimeError(f"resident K3 worker exited mid-request ({rc})\n{detail}")

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_id: int,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
        )

    def generate_stream(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_id: int,
        on_token,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
            on_token=on_token,
        )
'''
if s.count(marker) != 1:
    raise SystemExit(f"LocalK3 insertion marker: expected 1, got {s.count(marker)}")
s = s.replace(marker, resident + marker, 1)

s = once(s,
'''        self.tokenizer = LocalTokenizer(cfg.model_dir)
        self.backend = CBackend(cfg, state_root, max_state_entries)
''',
'''        self.tokenizer = LocalTokenizer(cfg.model_dir)
        if cfg.resident_worker:
            self.backend = ResidentCBackend(cfg)
        else:
            self.backend = CBackend(cfg, state_root, max_state_entries)
''', "LocalK3 backend selection")

s = once(s,
'''        spec=args.spec,
    )
''',
'''        spec=args.spec,
        resident_worker=args.resident_worker,
        worker_binary=args.worker_binary.resolve() if args.resident_worker else None,
        worker_context=args.worker_context,
    )
''', "serve backend config")

s = once(s,
'''    for path, label in [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]:
''',
'''    required_paths = [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]
    if cfg.resident_worker and cfg.worker_binary is not None:
        required_paths.append((cfg.worker_binary, "worker binary"))
    for path, label in required_paths:
''', "serve path validation")

s = once(s,
'''    state_root: Path | None = None
    if not args.no_state_cache and args.state_cache_entries > 0:
        state_root = args.state_cache_dir.expanduser().resolve()

    k3 = LocalK3(cfg, state_root, args.state_cache_entries)
''',
'''    state_root: Path | None = None
    if not cfg.resident_worker and not args.no_state_cache and args.state_cache_entries > 0:
        state_root = args.state_cache_dir.expanduser().resolve()

    k3 = LocalK3(cfg, state_root, args.state_cache_entries)
''', "serve state cache selection")

s = once(s,
'''    print("default parity profile: reasoning=max, temperature=1.0, top-p=1.0")
    if cfg.draft_trunk is not None:
''',
'''    print("default parity profile: reasoning=max, temperature=1.0, top-p=1.0")
    if cfg.resident_worker:
        print(f"resident C worker: ON (context {cfg.worker_context}; weights/cache stay hot)")
    else:
        print("resident C worker: OFF (one-shot exact backend)")
    if cfg.draft_trunk is not None:
''', "serve worker status")

s = once(s,
'''    if state_root is not None:
        print(
            f"conversation state cache: ON ({args.state_cache_entries} entry/entries; "
            f"root {state_root})"
        )
        print("note: current expanded MLA state is large; keep this cache on fast local NVMe")
    else:
        print("conversation state cache: OFF")
''',
'''    if cfg.resident_worker:
        print("conversation state: in RAM inside the resident worker")
    elif state_root is not None:
        print(
            f"conversation state cache: ON ({args.state_cache_entries} entry/entries; "
            f"root {state_root})"
        )
        print("note: current expanded MLA state is large; keep this cache on fast local NVMe")
    else:
        print("conversation state cache: OFF")
''', "serve state status")

s = once(s,
'''    sp.add_argument("--binary", type=Path, default=Path("bin/k3"))
    sp.add_argument("--preset", default="laptop")
''',
'''    sp.add_argument("--binary", type=Path, default=Path("bin/k3"))
    sp.add_argument(
        "--resident-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep model/trunk/cache/KV in one warm k3-worker process (default: on)",
    )
    sp.add_argument("--worker-binary", type=Path, default=Path("bin/k3-worker"))
    sp.add_argument(
        "--worker-context",
        type=int,
        default=1024,
        help="resident KV capacity in positions; raise for long benchmark sessions",
    )
    sp.add_argument("--preset", default="laptop")
''', "serve worker arguments")

p.write_text(s)
print('resident localhost bridge integration materialized')
