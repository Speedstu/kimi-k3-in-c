#!/usr/bin/env python3
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, got {n}")
    return text.replace(old, new, 1)


p = Path(__file__).resolve().parents[1] / "local/k3_local.py"
s = p.read_text()

old = '''        effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
        temperature = float(request.get("temperature", 1.0))
'''
new = '''        # The official Kimi Code provider sends Moonshot's top-level `thinking`
        # object, e.g. {"type":"enabled","effort":"max","keep":"all"}.  A generic
        # OpenAI-compatible caller may instead send reasoning_effort. Support both, with
        # the official dialect taking precedence.
        thinking = request.get("thinking")
        if isinstance(thinking, dict):
            if thinking.get("type") == "disabled":
                raise ValueError("K3 is an always-thinking model; local parity mode cannot disable thinking")
            effort = thinking.get("effort", request.get("reasoning_effort", "max"))
        else:
            effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("thinking effort must be low, high, or max")
        temperature = float(request.get("temperature", 1.0))
'''
s = once(s, old, new, "Kimi thinking dialect")

old = '''        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")

        prompt_ids = self.tokenizer.render(
'''
new = '''        if temperature < 0.0:
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
'''
s = once(s, old, new, "reject unsupported sampling knobs")

old = '''            if msg.get("tool_calls"):
                event({"tool_calls": msg["tool_calls"]})
            event({}, choice["finish_reason"])
            self.wfile.write(b"data: [DONE]\\n\\n")
            self.wfile.flush()
'''
new = '''            if msg.get("tool_calls"):
                # OpenAI's streaming tool-call delta requires an `index` even though the
                # final non-streaming object does not. Kimi Code's stream parser uses it
                # to assemble parallel tool calls.
                stream_calls = [
                    {"index": i, **tool_call}
                    for i, tool_call in enumerate(msg["tool_calls"])
                ]
                event({"tool_calls": stream_calls})
            event({}, choice["finish_reason"])
            if request.get("stream_options", {}).get("include_usage"):
                usage_packet = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": result["model"],
                    "choices": [],
                    "usage": result["usage"],
                }
                payload = "data: " + json.dumps(usage_packet, ensure_ascii=False) + "\\n\\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\\n\\n")
            self.wfile.flush()
'''
s = once(s, old, new, "stream tool indices and usage")

p.write_text(s)
print("Kimi Code wire-dialect patch applied")
