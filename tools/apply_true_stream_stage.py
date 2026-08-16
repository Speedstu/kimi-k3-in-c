#!/usr/bin/env python3
"""Guarded materialization of true C-token -> Kimi Code SSE streaming.

Deleted before merge. Every source replacement must match exactly once.
"""

from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {n}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- C CLI marker
p = root / "src/cli/k3_run.c"
s = p.read_text()
s = once(
    s,
    '"  --stop-id N           stop after emitting this token id (default disabled)\\n"\n',
    '"  --stop-id N           stop after emitting this token id (default disabled)\\n"\n'
    '"  --stream-tokens       flush one machine-readable @K3TOKEN <id> line as each\\n"\n'
    '"                        verified output token is committed; normal logs remain\\n"\n',
    "usage stream-tokens",
)
s = once(
    s,
    "    const char *preset_name = NULL;\n    int incremental = 0;\n",
    "    const char *preset_name = NULL;\n    int incremental = 0;\n    int stream_tokens = 0;\n",
    "stream flag declaration",
)
s = once(
    s,
    '        else if (!strcmp(argv[i], "--incremental")) incremental = 1;\n',
    '        else if (!strcmp(argv[i], "--incremental")) incremental = 1;\n'
    '        else if (!strcmp(argv[i], "--stream-tokens")) stream_tokens = 1;\n',
    "parse stream flag",
)
s = once(
    s,
    '''        int stop_hit = 0;
        for (int i = 0; i < emitn && nout < gen && T < Tmax; i++) {
            seq[T++] = emit[i];
            outtok[nout++] = emit[i];
            if (stop_id >= 0 && emit[i] == stop_id) { stop_hit = 1; break; }
        }
''',
    '''        int stop_hit = 0;
        for (int i = 0; i < emitn && nout < gen && T < Tmax; i++) {
            seq[T++] = emit[i];
            outtok[nout++] = emit[i];
            if (stream_tokens) {
                /* A unique prefix lets a parent process ignore the human diagnostics on
                 * stdout. Flush AFTER the token is committed to seq/outtok so a client
                 * never observes an uncommitted speculative proposal. */
                printf("@K3TOKEN %d\\n", emit[i]);
                fflush(stdout);
            }
            if (stop_id >= 0 && emit[i] == stop_id) { stop_hit = 1; break; }
        }
''',
    "emit stream marker",
)
p.write_text(s)

# -------------------------------------------------------------- Python bridge
p = root / "local/k3_local.py"
s = p.read_text()
s = once(
    s,
    "from dataclasses import dataclass\n",
    "from collections import deque\nfrom dataclasses import dataclass\n",
    "deque import",
)

# Insert a streaming backend method immediately before LocalK3.
marker = "\n\nclass LocalK3:\n"
if s.count(marker) != 1:
    raise SystemExit("LocalK3 insertion marker not unique")
backend_stream = r'''

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
        """Generate while calling ``on_token(id)`` immediately for committed tokens.

        The C child still owns inference and exact saved-state semantics.  Human CLI logs
        share stdout with a deliberately unique @K3TOKEN marker; only marker lines are
        interpreted as protocol.  If the HTTP client disconnects, the callback raises and
        the child is terminated so a laptop does not continue a multi-minute generation
        nobody is listening to.
        """
        with self.lock:
            cached = self._best_state(prompt_ids)
            cached_tokens = len(cached.tokens) if cached else 0
            suffix = prompt_ids[cached_tokens:]

            with tempfile.TemporaryDirectory(prefix="k3-local-stream-") as td:
                td_path = Path(td)
                ids_path = td_path / "prompt.ids"
                out_path = td_path / "result.json"
                ids_path.write_text(",".join(map(str, suffix)), encoding="ascii")

                save_path: Path | None = None
                if self.session_state_dir is not None:
                    save_path = self.session_state_dir / (uuid.uuid4().hex + ".k3state")

                cmd = [
                    str(self.cfg.binary),
                    str(self.cfg.model_dir),
                    "--trunk",
                    str(self.cfg.trunk_dir),
                    "--preset",
                    self.cfg.preset,
                    "--ids-file",
                    str(ids_path),
                    "--gen",
                    str(max_tokens),
                    "--incremental",
                    "--stream-tokens",
                    "--temperature",
                    str(temperature),
                    "--top-p",
                    str(top_p),
                    "--seed",
                    str(seed),
                    "--stop-id",
                    str(stop_id),
                    "--out",
                    str(out_path),
                ]
                if cached is not None:
                    cmd += ["--load-state", str(cached.path)]
                if save_path is not None:
                    cmd += ["--save-state", str(save_path)]
                if self.cfg.threads is not None:
                    cmd += ["--threads", str(self.cfg.threads)]
                if self.cfg.cache_gb is not None:
                    cmd += ["--cache-gb", str(self.cfg.cache_gb)]
                if self.cfg.trunk_gb is not None:
                    cmd += ["--trunk-gb", str(self.cfg.trunk_gb)]

                proc = subprocess.Popen(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                generated: list[int] = []
                tail: deque[str] = deque(maxlen=160)
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        if line.startswith("@K3TOKEN "):
                            token_id = int(line[9:].strip())
                            generated.append(token_id)
                            on_token(token_id)
                        else:
                            tail.append(line)
                    rc = proc.wait()
                except BaseException:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise

                if rc != 0:
                    raise RuntimeError(
                        f"local K3 backend exited with code {rc}\\n{''.join(tail)[-8000:]}"
                    )
                data = json.loads(out_path.read_text(encoding="utf-8"))
                final_generated = [int(x) for x in data["generated_ids"]]
                if final_generated != generated:
                    raise RuntimeError(
                        "stream protocol drift: emitted @K3TOKEN ids differ from result JSON"
                    )
                full_ids = [int(x) for x in data["full_ids"]]
                if save_path is not None:
                    self._remember(full_ids, save_path)
                data["state_cache_hit_tokens"] = cached_tokens
                data["state_cache_suffix_tokens"] = len(suffix)
                return generated, data
'''
s = s.replace(marker, backend_stream + marker, 1)

# Factor request validation/tokenization so stream headers are sent only after validation.
start = s.find("    def complete(self, request: dict[str, Any]) -> dict[str, Any]:\n")
end = s.find("\n\nclass Handler(BaseHTTPRequestHandler):", start)
if start < 0 or end < 0:
    raise SystemExit("LocalK3 complete block not found")
old_block = s[start:end]
new_block = r'''    def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if _contains_media(messages):
            raise ValueError(
                "this local C backend currently supports K3 text/coding input only; "
                "image/video input is rejected rather than silently discarded"
            )

        thinking = request.get("thinking")
        if isinstance(thinking, dict):
            if thinking.get("type") == "disabled":
                raise ValueError(
                    "K3 is an always-thinking model; local parity mode cannot disable thinking"
                )
            effort = thinking.get("effort", request.get("reasoning_effort", "max"))
        else:
            effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("thinking effort must be low, high, or max")

        temperature = float(request.get("temperature", 1.0))
        top_p = float(request.get("top_p", 1.0))
        max_tokens = int(
            request.get("max_tokens", request.get("max_completion_tokens", 4096))
        )
        seed = int(request.get("seed", 1))
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be in [1,4096] for the current C backend")
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")
        if int(request.get("n", 1)) != 1:
            raise ValueError("the local K3 backend currently supports n=1 only")
        if request.get("stop") not in (None, [], ""):
            raise ValueError("string stop sequences are not implemented; K3 EOS is handled exactly")
        if float(request.get("presence_penalty", 0.0)) != 0.0:
            raise ValueError("presence_penalty is unsupported in benchmark-parity mode")
        if float(request.get("frequency_penalty", 0.0)) != 0.0:
            raise ValueError("frequency_penalty is unsupported in benchmark-parity mode")

        prompt_ids = self.tokenizer.render(
            messages,
            request.get("tools"),
            effort,
            request.get("tool_choice"),
            request.get("response_format"),
        )
        return {
            "prompt_ids": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stop_id": self.tokenizer.eos_id,
            "model": request.get("model", "kimi-k3-local"),
        }

    def _result(
        self,
        prepared: dict[str, Any],
        generated: list[int],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        message = parse_xtml(self.tokenizer.decode(generated))
        stopped = bool(generated and generated[-1] == self.tokenizer.eos_id)
        finish_reason = (
            "tool_calls"
            if message.get("tool_calls")
            else ("stop" if stopped else "length")
        )
        prompt_ids = prepared["prompt_ids"]
        return {
            "id": "chatcmpl-local-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": prepared["model"],
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
            "local_stats": stats,
        }

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare(request)
        generated, stats = self.backend.generate(
            prepared["prompt_ids"],
            max_tokens=prepared["max_tokens"],
            temperature=prepared["temperature"],
            top_p=prepared["top_p"],
            seed=prepared["seed"],
            stop_id=prepared["stop_id"],
        )
        return self._result(prepared, generated, stats)

    @staticmethod
    def _safe_delta(previous: str, current: str) -> tuple[str, str]:
        """Return only text safe to commit to an irreversible SSE stream.

        Some byte-level tokenizer prefixes decode to U+FFFD until the next token arrives.
        Do not emit that unstable suffix.  If a decoder ever rewrites already-emitted
        text, hold rather than inventing a retraction the OpenAI/Kimi stream protocol does
        not have.
        """
        if not current.startswith(previous):
            return "", previous
        delta = current[len(previous) :]
        if "\ufffd" in delta:
            return "", previous
        return delta, current

    def stream_prepared(self, prepared: dict[str, Any], on_delta) -> dict[str, Any]:
        generated: list[int] = []
        sent_reasoning = ""
        sent_content = ""
        sent_tools = 0

        def on_token(token_id: int) -> None:
            nonlocal sent_reasoning, sent_content, sent_tools
            generated.append(token_id)
            message = parse_xtml(self.tokenizer.decode(generated))
            reasoning = message.get("reasoning_content") or ""
            delta, updated = self._safe_delta(sent_reasoning, reasoning)
            if delta:
                on_delta({"reasoning_content": delta})
                sent_reasoning = updated

            content = message.get("content") or ""
            delta, updated = self._safe_delta(sent_content, content)
            if delta:
                on_delta({"content": delta})
                sent_content = updated

            calls = message.get("tool_calls") or []
            if len(calls) > sent_tools:
                fresh = [
                    {"index": i, **calls[i]}
                    for i in range(sent_tools, len(calls))
                ]
                on_delta({"tool_calls": fresh})
                sent_tools = len(calls)

        streamed, stats = self.backend.generate_stream(
            prepared["prompt_ids"],
            max_tokens=prepared["max_tokens"],
            temperature=prepared["temperature"],
            top_p=prepared["top_p"],
            seed=prepared["seed"],
            stop_id=prepared["stop_id"],
            on_token=on_token,
        )
        if streamed != generated:
            raise RuntimeError("bridge token stream drift")

        # Flush any stable suffix that was held while a byte-level token decoded as U+FFFD.
        final_message = parse_xtml(self.tokenizer.decode(generated))
        reasoning = final_message.get("reasoning_content") or ""
        delta, sent_reasoning = self._safe_delta(sent_reasoning, reasoning)
        if delta:
            on_delta({"reasoning_content": delta})
        content = final_message.get("content") or ""
        delta, sent_content = self._safe_delta(sent_content, content)
        if delta:
            on_delta({"content": delta})

        return self._result(prepared, generated, stats)
'''
s = s[:start] + new_block + s[end:]

# Replace buffered Handler POST with validation-first true streaming branch.
post_start = s.find("    def do_POST(self) -> None:\n")
log_start = s.find("\n    def log_message(self, fmt: str, *args: Any) -> None:", post_start)
if post_start < 0 or log_start < 0:
    raise SystemExit("Handler do_POST block not found")
new_post = r'''    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            k3 = self.server.k3  # type: ignore[attr-defined]
            if not request.get("stream", False):
                self._json(200, k3.complete(request))
                return
            # Validate and tokenize before the HTTP 200/SSE headers. A malformed request
            # can therefore still receive a normal JSON 400 response.
            prepared = k3._prepare(request)
        except Exception as exc:
            self._json(
                400,
                {"error": {"message": str(exc), "type": type(exc).__name__}},
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        response_id = "chatcmpl-local-" + uuid.uuid4().hex
        created = int(time.time())
        model = prepared["model"]

        def event(delta: dict[str, Any], finish: str | None = None) -> None:
            packet = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}
                ],
            }
            payload = "data: " + json.dumps(packet, ensure_ascii=False) + "\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        try:
            event({"role": "assistant"})
            result = k3.stream_prepared(prepared, event)
            finish_reason = result["choices"][0]["finish_reason"]
            event({}, finish_reason)
            if request.get("stream_options", {}).get("include_usage"):
                usage_packet = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": result["usage"],
                }
                payload = "data: " + json.dumps(usage_packet, ensure_ascii=False) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            # generate_stream terminates its child when the callback raises.
            pass
        except Exception as exc:
            # Headers are already committed. Kimi/OpenAI clients tolerate an error JSON
            # event before [DONE] better than a hung connection with no terminator.
            try:
                payload = {
                    "error": {"message": str(exc), "type": type(exc).__name__}
                }
                self.wfile.write(
                    ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode(
                        "utf-8"
                    )
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass
'''
s = s[:post_start] + new_post + s[log_start:]

# Bump version/banner wording and remove obsolete buffered-stream caveat in docs later.
s = once(s, '    server_version = "K3Local/0.2"\n', '    server_version = "K3Local/0.3"\n', "server version")
p.write_text(s)

# -------------------------------------------------------------- Python tests
p = root / "tests/python/test_local_bridge.py"
s = p.read_text()
insert = r'''
    def test_safe_stream_delta_holds_unstable_unicode(self):
        delta, previous = LocalK3._safe_delta("abc", "abc\ufffd")
        self.assertEqual(delta, "")
        self.assertEqual(previous, "abc")
        delta, previous = LocalK3._safe_delta("abc", "abcé")
        self.assertEqual(delta, "é")
        self.assertEqual(previous, "abcé")

'''
needle = "    def test_loopback_guard(self):\n"
if s.count(needle) != 1:
    raise SystemExit("python test insertion point not found")
s = s.replace(needle, insert + needle, 1)
p.write_text(s)

# -------------------------------------------------------------- permanent tiny CI
p = root / ".github/workflows/local-parity.yml"
s = p.read_text()
needle = '''      - name: stop-id stops on the emitted token
        run: |
'''
if s.count(needle) != 1:
    raise SystemExit("local parity stop step not found")
stream_step = '''      - name: C token stream contains exactly the committed generated ids
        run: |
          ./bin/k3 /tmp/k3tiny --trunk /tmp/k3tiny-trunk --trunk-gb 0.05 --cache-gb 0.05 \\
            --ids 3,7,11,5,9 --gen 5 --incremental --temperature 0 --stream-tokens \\
            --out /tmp/stream.json >/tmp/stream.log
          python - <<'PY'
          import json
          expected=json.load(open('/tmp/stream.json'))['generated_ids']
          got=[]
          for line in open('/tmp/stream.log'):
              if line.startswith('@K3TOKEN '): got.append(int(line.split()[1]))
          assert got==expected, (got,expected)
          print('committed token streaming: PASS', got)
          PY
'''
s = s.replace(needle, stream_step + needle, 1)
p.write_text(s)

print("true streaming patch materialized")
