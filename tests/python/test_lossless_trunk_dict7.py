from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("lossless_trunk", ROOT / "tools/lossless_trunk.py")
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def decode7(payload: bytes, raw_n: int, dictionary: list[int]) -> bytes:
    assert raw_n % 2 == 0
    n = raw_n // 2
    cb = (3 * n + 7) // 8
    low = payload[:n]
    codes = payload[n : n + cb]
    esc = payload[n + cb :]
    ne = 0
    out = bytearray(raw_n)
    for i in range(n):
        bit = 3 * i
        bo, sh = bit >> 3, bit & 7
        w = codes[bo]
        if bo + 1 < cb:
            w |= codes[bo + 1] << 8
        if bo + 2 < cb:
            w |= codes[bo + 2] << 16
        q = (w >> sh) & 7
        out[2 * i] = low[i]
        if q < 7:
            out[2 * i + 1] = dictionary[q]
        else:
            out[2 * i + 1] = esc[ne]
            ne += 1
    if ne != len(esc):
        raise AssertionError((ne, len(esc)))
    return bytes(out)


class Dict7LosslessTests(unittest.TestCase):
    def test_roundtrip_tail_and_escapes(self) -> None:
        n = 65539  # deliberately neither /8 nor a SIMD multiple
        low = (np.arange(n, dtype=np.uint32) * 17 + 3).astype(np.uint8)
        common = np.array([0x3C, 0xBC, 0x3D, 0xBD, 0x3B, 0xBB, 0x3E], dtype=np.uint8)
        high = common[np.arange(n) % len(common)].copy()
        high[::97] = 0x71
        a = np.empty(n * 2, dtype=np.uint8)
        a[0::2], a[1::2] = low, high
        raw = a.tobytes()
        payload, dictionary, escapes = mod.encode_block_dict7(raw)
        self.assertGreater(escapes, 0)
        self.assertEqual(decode7(payload, len(raw), dictionary), raw)

    def test_dict7_beats_dict15_when_top7_cover_the_block(self) -> None:
        n = 1 << 18
        low = (np.arange(n, dtype=np.uint32) * 29 + 11).astype(np.uint8)
        common = np.array([0x3C, 0xBC, 0x3D, 0xBD, 0x3B, 0xBB, 0x3E], dtype=np.uint8)
        high = common[np.arange(n) % 7]
        a = np.empty(n * 2, dtype=np.uint8)
        a[0::2], a[1::2] = low, high
        raw = a.tobytes()
        p7, _dictionary7, _ = mod.encode_block_dict7(raw)
        p15, _, _ = mod.encode_block_dict15(raw)
        self.assertLess(len(p7), len(p15))
        codec, payload, dictionary, _ = mod.choose_block_codec(raw)
        self.assertEqual(codec, "dict7")
        self.assertEqual(decode7(payload, len(raw), dictionary), raw)
        self.assertLessEqual(mod.align_up(len(payload)), mod.align_up(len(p15)))

    def test_adaptive_selection_never_reads_more_than_raw(self) -> None:
        rng = np.random.default_rng(260817)
        raw = rng.integers(0, 256, size=2 << 20, dtype=np.uint8).tobytes()
        codec, payload, _, _ = mod.choose_block_codec(raw)
        self.assertIn(codec, {"raw", "dict15", "dict7"})
        self.assertLessEqual(mod.align_up(len(payload)), mod.align_up(len(raw)))


if __name__ == "__main__":
    unittest.main()
