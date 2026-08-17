#!/usr/bin/env python3
from __future__ import annotations

import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from local.k3_local import LocalK3, ResidentCBackend
from local.k3_vision import write_feature_sidecar


class FakeTokenizer:
    eos_id = 99

    def __init__(self):
        self.render_calls = 0

    def render(self, *args, **kwargs):
        self.render_calls += 1
        return [10, 11, 12]


class FakeVision:
    def __init__(self, *, prompt_positions: int = 8):
        self.prompt_positions = prompt_positions
        self.calls = 0

    def prepare(self, messages, **kwargs):
        import numpy as np

        self.calls += 1
        return SimpleNamespace(
            input_ids=[10, 163605, 12],
            features=[np.zeros((6, 4), dtype=np.float32)],
            placeholder_id=163605,
            prompt_positions=self.prompt_positions,
        )


class K3VisionTests(unittest.TestCase):
    def test_sidecar_binary_contract(self):
        import numpy as np

        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = (np.arange(8, dtype=np.float32) + 100).reshape(2, 4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "features.k3mmf"
            write_feature_sidecar(path, [a, b], 4)
            raw = path.read_bytes()

        magic, version, hidden, nimage, reserved = struct.unpack_from("<8sIIII", raw, 0)
        self.assertEqual(magic.rstrip(b"\0"), b"K3MMF1")
        self.assertEqual((version, hidden, nimage, reserved), (1, 4, 2, 0))
        off = struct.calcsize("<8sIIII")
        (n0,) = struct.unpack_from("<I", raw, off)
        off += 4
        self.assertEqual(n0, 3)
        got0 = np.frombuffer(raw, dtype="<f4", count=12, offset=off).reshape(3, 4)
        off += 12 * 4
        (n1,) = struct.unpack_from("<I", raw, off)
        off += 4
        self.assertEqual(n1, 2)
        got1 = np.frombuffer(raw, dtype="<f4", count=8, offset=off).reshape(2, 4)
        off += 8 * 4
        np.testing.assert_array_equal(got0, a)
        np.testing.assert_array_equal(got1, b)
        self.assertEqual(off, len(raw))

    def make_multimodal_k3(self, *, context: int = 64, prompt_positions: int = 8):
        k3 = LocalK3.__new__(LocalK3)
        k3.tokenizer = FakeTokenizer()
        backend = ResidentCBackend.__new__(ResidentCBackend)
        backend.context = context
        k3.backend = backend
        k3._vision = FakeVision(prompt_positions=prompt_positions)
        k3._vision_lock = threading.Lock()
        return k3

    def image_request(self, *, max_tokens: int = 4):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe exactly"},
                        {"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}},
                    ],
                }
            ],
            "thinking": {"type": "enabled", "effort": "max", "keep": "all"},
            "temperature": 1.0,
            "top_p": 0.95,
            "max_completion_tokens": max_tokens,
        }

    def test_image_uses_official_vision_preparation_not_text_render(self):
        k3 = self.make_multimodal_k3(prompt_positions=8)
        prepared = k3._prepare(self.image_request())
        self.assertEqual(k3.tokenizer.render_calls, 0)
        self.assertEqual(k3._vision.calls, 1)
        self.assertEqual(prepared["prompt_ids"], [10, 163605, 12])
        self.assertEqual(prepared["media_placeholder"], 163605)
        self.assertEqual(prepared["prompt_positions"], 8)
        self.assertEqual(len(prepared["media_features"]), 1)

    def test_image_context_gate_uses_expanded_positions(self):
        k3 = self.make_multimodal_k3(context=10, prompt_positions=8)
        with self.assertRaisesRegex(ValueError, "8 positions"):
            k3._prepare(self.image_request(max_tokens=3))

    def test_video_remains_fail_closed(self):
        k3 = self.make_multimodal_k3()
        request = self.image_request()
        request["messages"][0]["content"][1] = {
            "type": "video_url",
            "video_url": {"url": "file:///tmp/x.mp4"},
        }
        with self.assertRaisesRegex(ValueError, "unsupported media types"):
            k3._prepare(request)


if __name__ == "__main__":
    unittest.main()
