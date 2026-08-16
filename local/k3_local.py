#!/usr/bin/env python3
"""Fully local Kimi K3 chat / agent bridge.

The official K3 tokenizer code and the model checkpoint are read from ``--model-dir``;
the C engine is the only inference backend.  This process never calls Moonshot, Kimi,
Hugging Face, or another inference service.  It exposes an OpenAI-compatible localhost
endpoint so the official Kimi Code harness can drive the local model.

For coding/agent parity the defaults intentionally match the K3 release evaluation:
reasoning effort ``max``, temperature 1.0, top-p 1.0.  Single-step benchmark callers can
request top-p 0.95 explicitly.
"""

from __future__ import annotations

import argparse
import atexit
import html
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# A local model directory must be sufficient by itself.  These environment variables are
# set before transformers is imported (the import is intentionally lazy below), so a
# missing local tokenizer file fails rather than quietly downloading one from the web.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

O = "<|open|>"
C = "<|close|>"
S = "<|sep|>"
EOM = "<|end_of_msg|>"


def _attrs(text: str) -> dict[str, str]:
    return {
        key: html.unescape(value)
        for key, value in re.findall(r'([A-Za-z_][\w-]*)="([^"]*)"', text)
    }


def parse_xtml(text: str) -> dict[str, Any]:
    """Turn generated K3 XTML channels into an API-style assistant message.

    The official generation prompt starts generation *inside* the think channel, so a
    generated completion commonly begins with reasoning text and then
    ``<|close|>think<|sep|>`` rather than with a think-open token.
    """

    think_open = O + "think" + S
    think_close = C + "think" + S
    response_open = O + "response" + S
    response_close = C + "response" + S
    tools_open = O + "tools" + S
    tools_close = C + "tools" + S

    raw = text
    if raw.startswith(think_open):
        raw = raw[len(think_open) :]

    reasoning = ""
    rest = raw
    cut = raw.find(think_close)
    if cut >= 0:
        reasoning = raw[:cut]
        rest = raw[cut + len(think_close) :]
    elif response_open not in raw and tools_open not in raw:
        # The completion ended while the model was still thinking.
        reasoning = raw
        rest = ""

    content: str | None = None
    ro = rest.find(response_open)
    if ro >= 0:
        body = rest[ro + len(response_open) :]
        rc = body.find(response_close)
        if rc >= 0:
            content = body[:rc]
            rest_after_response = body[rc + len(response_close) :]
        else:
            content = body
            rest_after_response = ""
    else:
        rest_after_response = rest

    tool_calls: list[dict[str, Any]] = []
    to = rest_after_response.find(tools_open)
    if to >= 0:
        tool_body = rest_after_response[to + len(tools_open) :]
        tc = tool_body.find(tools_close)
        if tc >= 0:
            tool_body = tool_body[:tc]

        call_re = re.compile(
            re.escape(O)
            + r"call(?P<attrs>.*?)"
            + re.escape(S)
            + r"(?P<body>.*?)"
            + re.escape(C + "call" + S),
            re.DOTALL,
        )
        arg_re = re.compile(
            re.escape(O)
            + r"argument(?P<attrs>.*?)"
            + re.escape(S)
            + r"(?P<body>.*?)"
            + re.escape(C + "argument" + S),
            re.DOTALL,
        )
        json_re = re.compile(
            re.escape(O)
            + r"json(?P<attrs>.*?)"
            + re.escape(S)
            + r"(?P<body>.*?)"
            + re.escape(C + "json" + S),
            re.DOTALL,
        )

        for pos, call in enumerate(call_re.finditer(tool_body), start=1):
            ca = _attrs(call.group("attrs"))
            name = ca.get("tool", "")
            idx = ca.get("index", str(pos))
            args: dict[str, Any] = {}
            call_body = call.group("body")
            json_block = json_re.search(call_body)
            if json_block:
                try:
                    parsed = json.loads(json_block.group("body"))
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        args = {"value": parsed}
                except json.JSONDecodeError:
                    args = {"_raw": json_block.group("body")}
            else:
                for arg in arg_re.finditer(call_body):
                    aa = _attrs(arg.group("attrs"))
                    key = aa.get("key", "")
                    typ = aa.get("type", "string")
                    body = arg.group("body")
                    if typ == "string":
                        value: Any = body
                    else:
                        try:
                            value = json.loads(body)
                        except json.JSONDecodeError:
                            value = body
                    if key:
                        args[key] = value
            tool_calls.append(
                {
                    "id": f"call_{idx}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            args, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )

    if content is not None:
        content = content.replace(C + "message" + S, "").replace(EOM, "")
    reasoning = reasoning.replace(EOM, "")
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls or None,
    }


def _contains_media(messages: list[dict[str, Any]]) -> bool:
    """Return true for API message parts the current C text backend cannot encode."""

    media_types = {
        "image",
        "image_url",
        "input_image",
        "video",
        "video_url",
        "input_video",
        "audio",
        "input_audio",
    }
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in media_types:
                return True
    return False


@dataclass(frozen=True)
class BackendConfig:
    model_dir: Path
    trunk_dir: Path
    binary: Path
    preset: str = "laptop"
    threads: int | None = None
    cache_gb: float | None = None
    trunk_gb: float | None = None


@dataclass
class StateEntry:
    tokens: tuple[int, ...]
    path: Path
    touched: float


class LocalTokenizer:
    def __init__(self, model_dir: Path):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment error
            raise SystemExit(
                "The official K3 XTML tokenizer needs transformers. Install it in a "
                "venv, for example: pip install 'transformers>=4.56' tiktoken"
            ) from exc

        # trust_remote_code here means "execute the tokenizer Python source already in
        # model_dir".  local_files_only plus the offline env above forbids a web fallback.
        self.tok = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )

    @property
    def eos_id(self) -> int:
        value = self.tok.eos_token_id
        if value is None:
            raise RuntimeError("K3 tokenizer exposes no eos_token_id")
        return int(value)

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        reasoning_effort: str,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "thinking": True,
            # The local K3 encoder names this template kwarg thinking_effort.  The
            # compatible HTTP field remains reasoning_effort.
            "thinking_effort": reasoning_effort,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        ids = self.tok.apply_chat_template(messages, **kwargs)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


class CBackend:
    """Serialises access to the huge local engine and reuses exact saved prefixes."""

    def __init__(
        self,
        cfg: BackendConfig,
        state_root: Path | None,
        max_state_entries: int,
    ):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.max_state_entries = max(0, max_state_entries)
        self.entries: list[StateEntry] = []
        self.session_state_dir: Path | None = None
        if state_root is not None and self.max_state_entries > 0:
            state_root.mkdir(parents=True, exist_ok=True)
            self.session_state_dir = state_root / ("session-" + uuid.uuid4().hex)
            self.session_state_dir.mkdir(parents=True)
            atexit.register(shutil.rmtree, self.session_state_dir, True)

    def _best_state(self, prompt_ids: list[int]) -> StateEntry | None:
        best: StateEntry | None = None
        for entry in self.entries:
            n = len(entry.tokens)
            if n > len(prompt_ids):
                continue
            if tuple(prompt_ids[:n]) != entry.tokens:
                continue
            if best is None or n > len(best.tokens):
                best = entry
        if best is not None:
            best.touched = time.monotonic()
        return best

    def _remember(self, tokens: list[int], path: Path) -> None:
        if not path.is_file():
            return
        self.entries.append(StateEntry(tuple(tokens), path, time.monotonic()))
        while len(self.entries) > self.max_state_entries:
            victim = min(self.entries, key=lambda entry: entry.touched)
            self.entries.remove(victim)
            victim.path.unlink(missing_ok=True)

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
        # Running two 2.78T checkpoint processes at once on a laptop is not concurrency,
        # it is an OOM / disk-thrashing bug.  The HTTP server may have many client
        # threads, but the model backend is deliberately one-at-a-time.
        with self.lock:
            cached = self._best_state(prompt_ids)
            cached_tokens = len(cached.tokens) if cached else 0
            suffix = prompt_ids[cached_tokens:]

            with tempfile.TemporaryDirectory(prefix="k3-local-") as td:
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

                proc = subprocess.run(cmd, text=True, capture_output=True)
                if proc.returncode != 0:
                    tail = (proc.stdout + "\n" + proc.stderr)[-8000:]
                    raise RuntimeError(
                        f"local K3 backend exited with code {proc.returncode}\n{tail}"
                    )
                data = json.loads(out_path.read_text(encoding="utf-8"))
                generated = [int(x) for x in data["generated_ids"]]
                full_ids = [int(x) for x in data["full_ids"]]
                if save_path is not None:
                    self._remember(full_ids, save_path)
                data["state_cache_hit_tokens"] = cached_tokens
                data["state_cache_suffix_tokens"] = len(suffix)
                return generated, data


class LocalK3:
    def __init__(
        self,
        cfg: BackendConfig,
        state_root: Path | None,
        max_state_entries: int,
    ):
        self.tokenizer = LocalTokenizer(cfg.model_dir)
        self.backend = CBackend(cfg, state_root, max_state_entries)

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if _contains_media(messages):
            raise ValueError(
                "this local C backend currently supports K3 text/coding input only; "
                "image/video input is rejected rather than silently discarded"
            )

        effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
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

        prompt_ids = self.tokenizer.render(
            messages,
            request.get("tools"),
            effort,
            request.get("tool_choice"),
            request.get("response_format"),
        )
        generated, stats = self.backend.generate(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=self.tokenizer.eos_id,
        )
        message = parse_xtml(self.tokenizer.decode(generated))
        stopped = bool(generated and generated[-1] == self.tokenizer.eos_id)
        finish_reason = (
            "tool_calls"
            if message.get("tool_calls")
            else ("stop" if stopped else "length")
        )
        return {
            "id": "chatcmpl-local-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "kimi-k3-local"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
            "local_stats": stats,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "K3Local/0.2"
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "kimi-k3-local",
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
        elif self.path in {"/health", "/healthz"}:
            self._json(
                200,
                {
                    "status": "ok",
                    "backend": "local-c",
                    "network_inference": False,
                },
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            result = self.server.k3.complete(request)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - HTTP API boundary
            self._json(
                400,
                {"error": {"message": str(exc), "type": type(exc).__name__}},
            )
            return

        if not request.get("stream", False):
            self._json(200, result)
            return

        # This is valid SSE and therefore compatible with streaming clients, but the
        # current CLI backend completes a turn before returning.  The resident-worker
        # backend planned for this same wire protocol can later send true token deltas.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        choice = result["choices"][0]
        msg = choice["message"]
        response_id = result["id"]

        def event(delta: dict[str, Any], finish: str | None = None) -> None:
            packet = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": result["model"],
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}
                ],
            }
            payload = "data: " + json.dumps(packet, ensure_ascii=False) + "\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        try:
            event({"role": "assistant"})
            if msg.get("reasoning_content"):
                event({"reasoning_content": msg["reasoning_content"]})
            if msg.get("content") is not None:
                event({"content": msg["content"]})
            if msg.get("tool_calls"):
                event({"tool_calls": msg["tool_calls"]})
            event({}, choice["finish_reason"])
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[k3-local] " + (fmt % args))


class K3HTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(args: argparse.Namespace) -> None:
    if not _is_loopback_host(args.host) and not args.allow_remote:
        raise SystemExit(
            f"refusing to expose unauthenticated model server on {args.host}; "
            "use 127.0.0.1/localhost or pass --allow-remote explicitly"
        )

    cfg = BackendConfig(
        model_dir=args.model_dir.resolve(),
        trunk_dir=args.trunk.resolve(),
        binary=args.binary.resolve(),
        preset=args.preset,
        threads=args.threads,
        cache_gb=args.cache_gb,
        trunk_gb=args.trunk_gb,
    )
    for path, label in [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")

    state_root: Path | None = None
    if not args.no_state_cache and args.state_cache_entries > 0:
        state_root = args.state_cache_dir.expanduser().resolve()

    k3 = LocalK3(cfg, state_root, args.state_cache_entries)
    httpd = K3HTTPServer((args.host, args.port), Handler)
    httpd.k3 = k3  # type: ignore[attr-defined]
    print(f"K3 Local listening on http://{args.host}:{args.port}/v1")
    print("inference network: OFF; tokenizer + weights are local-files-only")
    print("default parity profile: reasoning=max, temperature=1.0, top-p=1.0")
    if state_root is not None:
        print(
            f"conversation state cache: ON ({args.state_cache_entries} entry/entries; "
            f"root {state_root})"
        )
        print("note: current expanded MLA state is large; keep this cache on fast local NVMe")
    else:
        print("conversation state cache: OFF")
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fully local Kimi K3 OpenAI-compatible chat / agent server"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--model-dir", type=Path, required=True)
    sp.add_argument("--trunk", type=Path, required=True)
    sp.add_argument("--binary", type=Path, default=Path("bin/k3"))
    sp.add_argument("--preset", default="laptop")
    sp.add_argument("--threads", type=int)
    sp.add_argument("--cache-gb", type=float)
    sp.add_argument("--trunk-gb", type=float)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument(
        "--state-cache-dir",
        type=Path,
        default=Path("~/.cache/k3-local/state"),
    )
    sp.add_argument(
        "--state-cache-entries",
        type=int,
        default=1,
        help="exact saved conversation prefixes retained on local NVMe (default 1)",
    )
    sp.add_argument("--no-state-cache", action="store_true")
    sp.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding to a non-loopback host; the server itself has no auth",
    )
    sp.set_defaults(func=serve)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
