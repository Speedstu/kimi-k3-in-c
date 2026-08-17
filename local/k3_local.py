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
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

# A local model directory must be sufficient by itself.  These environment variables are
# set before transformers is imported (the import is intentionally lazy below), so a
# missing local tokenizer file fails rather than quietly downloading one from the web.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

OPEN = "<|open|>"
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

    think_open = OPEN + "think" + S
    think_close = C + "think" + S
    response_open = OPEN + "response" + S
    response_close = C + "response" + S
    tools_open = OPEN + "tools" + S
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
            re.escape(OPEN)
            + r"call(?P<attrs>.*?)"
            + re.escape(S)
            + r"(?P<body>.*?)"
            + re.escape(C + "call" + S),
            re.DOTALL,
        )
        arg_re = re.compile(
            re.escape(OPEN)
            + r"argument(?P<attrs>.*?)"
            + re.escape(S)
            + r"(?P<body>.*?)"
            + re.escape(C + "argument" + S),
            re.DOTALL,
        )
        json_re = re.compile(
            re.escape(OPEN)
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
    draft_trunk: Path | None = None
    draft_trunk_gb: float = 32.0
    draft_topk: int = 4
    spec: int = 4
    resident_worker: bool = True
    worker_binary: Path | None = None
    worker_context: int = 1024
    prefill_mb: float = 256.0
    prefill_chunk: int | None = None


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


class ResidentCBackend:
    """One warm C process: weights/index/trunk/cache and the active KV/KDA state stay live."""

    _PRESET_BUDGETS: ClassVar[dict[str, tuple[float, float]]] = {
        "laptop": (3.0, 1.0),
        "desktop": (16.0, 10.0),
        "workstation": (60.0, 30.0),
        "server": (110.0, 13.0),
        "max": (110.0, 109.0),
    }

    def __init__(self, cfg: BackendConfig):
        if cfg.worker_binary is None:
            raise ValueError("resident worker requested without a worker binary")
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
            "--prefill-mb",
            str(self.cfg.prefill_mb),
        ]
        if self.cfg.prefill_chunk is not None:
            cmd += ["--prefill-chunk", str(self.cfg.prefill_chunk)]
        if self.cfg.threads is not None:
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
            draft_stats: dict[str, Any] = {}
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
                    if line.startswith(f"@K3DRAFT {rid} "):
                        fields = line.split()
                        if len(fields) != 7:
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
                        stats = {
                            "resident_worker": True,
                            "worker_seconds": seconds,
                            "worker_cached_positions": cached,
                            "state_cache_hit_tokens": reused,
                            "state_cache_suffix_tokens": len(prompt_ids) - reused,
                        }
                        stats.update(draft_stats)
                        return generated, stats
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


class LocalK3:
    def __init__(
        self,
        cfg: BackendConfig,
        state_root: Path | None,
        max_state_entries: int,
    ):
        self.tokenizer = LocalTokenizer(cfg.model_dir)
        if cfg.resident_worker:
            self.backend = ResidentCBackend(cfg)
        else:
            self.backend = CBackend(cfg, state_root, max_state_entries)

    def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
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
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
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
        backend_context = getattr(self.backend, "context", None)
        if backend_context is not None:
            backend_context = int(backend_context)
            if len(prompt_ids) + max_tokens > backend_context:
                raise ValueError(
                    f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds "
                    f"resident worker context ({backend_context}); raise --worker-context"
                )
        elif max_tokens > 4096:
            raise ValueError(
                "non-resident one-shot compatibility mode supports at most 4096 output "
                "tokens; use the default resident worker for K3 Max benchmark budgets"
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


class Handler(BaseHTTPRequestHandler):
    server_version = "K3Local/0.3"
    protocol_version = "HTTP/1.1"

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
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

    def do_POST(self) -> None:
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
        draft_trunk=args.draft_trunk.resolve() if args.draft_trunk else None,
        draft_trunk_gb=args.draft_trunk_gb,
        draft_topk=args.draft_topk,
        spec=args.spec,
        resident_worker=args.resident_worker,
        worker_binary=args.worker_binary.resolve() if args.resident_worker else None,
        worker_context=args.worker_context,
        prefill_mb=args.prefill_mb,
        prefill_chunk=args.prefill_chunk,
    )
    required_paths = [
        (cfg.model_dir, "model"),
        (cfg.trunk_dir, "trunk"),
        (cfg.binary, "binary"),
    ]
    if cfg.resident_worker and cfg.worker_binary is not None:
        required_paths.append((cfg.worker_binary, "worker binary"))
    for path, label in required_paths:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")
    if cfg.draft_trunk is not None and not cfg.draft_trunk.exists():
        raise SystemExit(f"draft trunk path does not exist: {cfg.draft_trunk}")
    if cfg.draft_topk < 1 or cfg.spec < 1:
        raise SystemExit("--draft-topk and --spec must both be >= 1")

    state_root: Path | None = None
    if not cfg.resident_worker and not args.no_state_cache and args.state_cache_entries > 0:
        state_root = args.state_cache_dir.expanduser().resolve()

    k3 = LocalK3(cfg, state_root, args.state_cache_entries)
    httpd = K3HTTPServer((args.host, args.port), Handler)
    httpd.k3 = k3  # type: ignore[attr-defined]
    print(f"K3 Local listening on http://{args.host}:{args.port}/v1")
    print("inference network: OFF; tokenizer + weights are local-files-only")
    print("default parity profile: reasoning=max, temperature=1.0, top-p=1.0")
    if cfg.resident_worker:
        print(f"resident C worker: ON (context {cfg.worker_context}; weights/cache stay hot)")
    else:
        print("resident C worker: OFF (one-shot exact backend)")
    if cfg.draft_trunk is not None:
        print(
            f"sampled speculative draft: ON ({cfg.draft_trunk}, top-{cfg.draft_topk}, "
            f"spec={cfg.spec}); exact K3 p/q verification remains authoritative"
        )
    else:
        print("sampled speculative draft: OFF")
    if cfg.resident_worker:
        print("conversation state: in RAM inside the resident worker")
    elif state_root is not None:
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
        help=(
            "resident capacity in positions (2..1048576); virtual reservation is lazy, "
            "but RAM still grows with positions actually used"
        ),
    )
    sp.add_argument(
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
    sp.add_argument("--threads", type=int)
    sp.add_argument("--cache-gb", type=float)
    sp.add_argument("--trunk-gb", type=float)
    sp.add_argument(
        "--draft-trunk",
        type=Path,
        help="optional local Q4/I8/BF16 speculative draft trunk; exact K3 still verifies",
    )
    sp.add_argument("--draft-trunk-gb", type=float, default=32.0)
    sp.add_argument("--draft-topk", type=int, default=4)
    sp.add_argument("--spec", type=int, default=4)
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
