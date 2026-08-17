# Local K3 image input

Image requests use the released K3 vision stack, not a substitute vision model. The localhost bridge lazy-loads only Moonshot's official MoonViT vision tower, multimodal projector, image processor and tokenizer from the local K3 checkpoint. The 2.78T language model remains the exact C runtime.

## Install the optional frontend

Text-only use does not need PyTorch. For image input, install a PyTorch build appropriate for the target machine first, then install the parity-pinned frontend dependencies:

```bash
python -m pip install -r local/requirements-vision.txt
```

The released K3 config declares Transformers 4.56.2, so the optional requirements pin that version. The bridge sets Hugging Face/Transformers offline mode and loads K3 custom code and weights from `--model-dir`.

## Start the server

```bash
python local/k3_local.py serve \
  --model-dir ~/k3model \
  --trunk ~/k3trunk-lossless \
  --preset laptop \
  --threads N \
  --worker-context 8192 \
  --vision-device auto \
  --vision-attention auto
```

`--vision-device auto` uses CUDA when PyTorch can use it and otherwise CPU. `--vision-attention auto` prefers the released FlashAttention2 path when available on CUDA and otherwise uses K3's official eager implementation. This choice affects the vision frontend only; generated tokens still come from the exact C language model.

## OpenAI-compatible image request

```json
{
  "model": "kimi-k3-local",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image precisely."},
        {"type": "image_url", "image_url": {"url": "/absolute/path/to/image.png"}}
      ]
    }
  ],
  "thinking": {"type": "enabled", "effort": "max", "keep": "all"},
  "temperature": 1.0,
  "top_p": 0.95
}
```

The official K3 processor accepts `image` and `image_url` message parts. Local file paths, PIL images inside direct Python calls, and the media forms supported by Moonshot's processor are processed by that official frontend. The C worker receives only the projected 7168-dimensional feature rows through `REQMM`.

## Correctness boundary

Hosted CI proves the C mixed-embedding boundary by replacing an image placeholder with external rows exactly equal to known text-token embeddings and requiring identical generated tokens, including verified speculative decoding. Bridge CI proves expanded-position context accounting and fail-closed behavior.

The real-checkpoint smoke workflow `.github/workflows/k3-full-vision-smoke.yml` additionally loads the actual released MoonViT/projector weights on a self-hosted `k3-full-checkpoint` runner. Published vision benchmark parity remains unclaimed until the full benchmark workflow measures those suites and `k3_score_gate.py` accepts the results.

Video and audio remain unsupported and are rejected explicitly; they are never silently converted to text or images.
