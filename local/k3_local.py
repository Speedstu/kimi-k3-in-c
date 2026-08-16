#!/usr/bin/env python3
"""Fully local Kimi K3 API/agent bridge.

No request made by this program leaves the machine.  The official K3 tokenizer code is
loaded from MODEL_DIR with ``local_files_only=True`` and renders the exact XTML chat
format; ``bin/k3`` remains the inference backend.

The endpoint intentionally mirrors the useful part of OpenAI chat completions so the
*official Kimi Code CLI* can be pointed at localhost and used as the agent harness.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
    """Turn K3's generated XTML channels into an API-style assistant message.

    K3 generation normally starts *inside* the think channel because the generation
    prompt already emitted ``<|open|>think<|sep|>``.  Therefore a missing think-open is
    expected, not an error.
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
        # Truncated while still thinking.
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

        # Calls and arguments are delimited by K3's control-token separators.
        call_re = re.compile(
            re.escape(O) + r"call(?P<attrs>.*?)" + re.escape(S)
            + r"(?P<body>.*?)" + re.escape(C + "call" + S),
            re.DOTALL,
        )
        arg_re = re.compile(
            re.escape(O) + r"argument(?P<attrs>.*?)" + re.escape(S)
            + r"(?P<body>.*?)" + re.escape(C + "argument" + S),
            re.DOTALL,
        )
        json_re = re.compile(
            re.escape(O) + r"json(?P<attrs>.*?)" + re.escape(S)
            + r"(?P<body>.*?)" + re.escape(C + "json" + S),
            re.DOTALL,
        )

        for pos, call in enumerate(call_re.finditer(tool_body), start=1):
            ca = _attrs(call.group("attrs"))
            name = ca.get("tool", "")
            idx = ca.get("index", str(pos))
            args: dict[str, Any] = {}
            cb = call.group("body")
            json_block = json_re.search(cb)
            if json_block:
                try:
                    parsed = json.loads(json_block.group("body"))
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    args = {"_raw": json_block.group("body")}
            else:
                for arg in arg_re.finditer(cb):
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
                        "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
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


@dataclass
class BackendConfig:
    model_dir: Path
    trunk_dir: Path
    binary: Path
    preset: str = "laptop"
    threads: int | None = None
    cache_gb: float | None = None
    trunk_gb: float | None = None


class LocalTokenizer:
    def __init__(self, model_dir: Path):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment error
            raise SystemExit(
                "transformers is required only for the official K3 XTML tokenizer. "
                "Install it in a venv: pip install 'transformers>=4.56' tiktoken"
            ) from exc

        # LOCAL ONLY. trust_remote_code means execute the tokenizer Python files already
        # present in model_dir; local_files_only prevents a network fallback.
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
            # The official local encoder calls the template kwarg thinking_effort even
            # though the public API field is reasoning_effort.
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
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

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
        with tempfile.TemporaryDirectory(prefix="k3-local-") as td:
            td_path = Path(td)
            ids_path = td_path / "prompt.ids"
            out_path = td_path / "result.json"
            ids_path.write_text(",".join(map(str, prompt_ids)), encoding="ascii")

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
            if self.cfg.threads is not None:
                cmd += ["--threads", str(self.cfg.threads)]
            if self.cfg.cache_gb is not None:
                cmd += ["--cache-gb", str(self.cfg.cache_gb)]
            if self.cfg.trunk_gb is not None:
                cmd += ["--trunk-gb", str(self.cfg.trunk_gb)]

            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"local K3 backend exited with code {proc.returncode}")
            data = json.loads(out_path.read_text(encoding="utf-8"))
            return [int(x) for x in data["generated_ids"]], data


class LocalK3:
    def __init__(self, cfg: BackendConfig):
        self.tokenizer = LocalTokenizer(cfg.model_dir)
        self.backend = CBackend(cfg)

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
        temperature = float(request.get("temperature", 1.0))
        top_p = float(request.get("top_p", 1.0))
        max_tokens = int(request.get("max_tokens", request.get("max_completion_tokens", 4096)))
        seed = int(request.get("seed", 1))

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
        return {
            "id": "chatcmpl-local-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "kimi-k3-local"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
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
    server_version = "K3Local/0.1"

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
                    "data": [{"id": "kimi-k3-local", "object": "model", "owned_by": "local"}],
                },
            )
        elif self.path in {"/health", "/healthz"}:
            self._json(200, {"status": "ok", "backend": "local-c"})
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
            if not request.get("stream", False):
                self._json(200, result)
                return

            # The C backend currently completes the turn before returning. We still speak
            # valid SSE so Kimi Code and OpenAI-compatible clients work unchanged; a future
            # resident backend can emit these deltas token-by-token without changing the
            # wire protocol.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            choice = result["choices"][0]
            msg = choice["message"]
            rid = result["id"]

            def event(delta: dict[str, Any], finish: str | None = None) -> None:
                packet = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": result["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()

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
        except Exception as exc:  # noqa: BLE001 - API boundary
            self._json(400, {"error": {"message": str(exc), "type": type(exc).__name__}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[k3-local] " + (fmt % args))


def serve(args: argparse.Namespace) -> None:
    cfg = BackendConfig(
        model_dir=args.model_dir.resolve(),
        trunk_dir=args.trunk.resolve(),
        binary=args.binary.resolve(),
        preset=args.preset,
        threads=args.threads,
        cache_gb=args.cache_gb,
        trunk_gb=args.trunk_gb,
    )
    for path, label in [(cfg.model_dir, "model"), (cfg.trunk_dir, "trunk"), (cfg.binary, "binary")]:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")
    k3 = LocalK3(cfg)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.k3 = k3  # type: ignore[attr-defined]
    print(f"K3 Local listening on http://{args.host}:{args.port}/v1")
    print("network inference: disabled; tokenizer and weights are loaded from local paths only")
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fully local Kimi K3 OpenAI-compatible server")
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
    sp.set_defaults(func=serve)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
