#!/usr/bin/env python3
from pathlib import Path

p = Path("local/README.md")
s = p.read_text(encoding="utf-8")

old = """The Python bridge needs `transformers` only for K3's **official local chat tokenizer / XTML
template**. Put it in a venv rather than modifying the system interpreter:

```bash
python3 -m venv .venv-k3
. .venv-k3/bin/activate
pip install 'transformers>=4.56' tiktoken
```

At runtime the bridge sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and also passes
`local_files_only=True`. A missing tokenizer file therefore fails instead of silently
fetching one from the network.
"""
new = """Text/chat mode needs `transformers` only for K3's **official local tokenizer / XTML
template**. Put it in a venv rather than modifying the system interpreter. The released K3
config declares Transformers 4.56.2:

```bash
python3 -m venv .venv-k3
. .venv-k3/bin/activate
pip install 'transformers==4.56.2' tiktoken
```

Image input is optional and does **not** make PyTorch a dependency of text inference. Install
a PyTorch build appropriate for the target CPU/CUDA machine first, then:

```bash
pip install -r local/requirements-vision.txt
```

At runtime the bridge sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and passes
`local_files_only=True`. A missing tokenizer, processor, custom-code or vision-weight file
therefore fails instead of silently fetching a different component from the network. See
[`local/VISION.md`](VISION.md) for the image path.
"""
if s.count(old) != 1:
    raise SystemExit(f"dependency anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = """The response exposes both `reasoning_content` and ordinary `content`, plus OpenAI-style
`tool_calls` when K3 emits XTML tool calls.

## 4. Use the official Kimi Code agent harness against localhost
"""
new = """The response exposes both `reasoning_content` and ordinary `content`, plus OpenAI-style
`tool_calls` when K3 emits XTML tool calls.

### Image input

The resident path also accepts K3 `image` / `image_url` message parts. The bridge lazy-loads
only the released MoonViT vision tower + multimodal projector, expands the image placeholder
to the projected 7168-dimensional rows, and sends those rows to the exact C language model
through `REQMM`. Text requests never initialize that frontend. Use `--vision-device auto`
and `--vision-attention auto` for the normal laptop path; full setup and a request example
are in [`local/VISION.md`](VISION.md).

Video/audio remain fail-closed until their released processing path is independently gated.

## 4. Use the official Kimi Code agent harness against localhost
"""
if s.count(old) != 1:
    raise SystemExit(f"image section anchor count={s.count(old)}")
s = s.replace(old, new, 1)

old = """- **Text/coding first:** image/video/audio message parts are rejected instead of discarded.
  K3's vision path still needs to be integrated locally.
"""
new = """- **Vision scores are measured separately:** image input now uses the released K3
  MoonViT/projector boundary and exact C language model; hosted CI gates the boundary, while
  the actual full-checkpoint vision scores still require the self-hosted benchmark suites.
  Video/audio remain explicitly unsupported rather than being silently discarded.
"""
if s.count(old) != 1:
    raise SystemExit(f"gap anchor count={s.count(old)}")
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
print("updated local README for validated K3 image path")
