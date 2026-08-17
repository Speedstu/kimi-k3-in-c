#!/usr/bin/env python3
"""Reference K3-Compact Q3 weight format.

This module defines the disk/runtime contract for the future QAT checkpoint. It is NOT
an accuracy recipe: production weights should come from quantization-aware training.
The reference quantizer exists to make the bitstream testable before expensive training.

Each row is split into groups of 128 values. A group stores:
  [2-byte BF16 scale][3-bit signed codes, little-endian bitstream]
Codes are two's-complement signed 3-bit integers. The reference exporter emits -3..3;
-4 remains decodable so the representation is complete. Full groups cost 50 bytes,
which is exactly 3.125 bits/weight including the BF16 scale.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

GROUP = 128
QMIN = -3
QMAX = 3


def f32_to_bf16_bits(x: float) -> int:
    """Round float32 to nearest-even BF16 and return the 16 stored bits."""
    u = struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]
    # RNE: add 0x7fff plus retained LSB before truncating.
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (int(bits) & 0xFFFF) << 16))[0]


def packed_code_bytes(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    return (3 * n + 7) // 8


def row_bytes(cols: int) -> int:
    if cols < 0:
        raise ValueError("cols must be non-negative")
    full, rem = divmod(cols, GROUP)
    total = full * (2 + packed_code_bytes(GROUP))
    if rem:
        total += 2 + packed_code_bytes(rem)
    return total


def pack_codes(codes: np.ndarray) -> bytes:
    q = np.asarray(codes, dtype=np.int16).reshape(-1)
    if np.any(q < -4) or np.any(q > 3):
        raise ValueError("Q3 code outside signed 3-bit range [-4,3]")
    out = bytearray(packed_code_bytes(q.size))
    for i, value in enumerate(q.tolist()):
        code = int(value) & 0x7
        bit = 3 * i
        byte = bit >> 3
        shift = bit & 7
        v = code << shift
        out[byte] |= v & 0xFF
        if shift > 5:
            out[byte + 1] |= (v >> 8) & 0xFF
    return bytes(out)


def unpack_codes(buf: bytes | bytearray | memoryview, n: int) -> np.ndarray:
    src = memoryview(buf)
    need = packed_code_bytes(n)
    if len(src) < need:
        raise ValueError(f"need {need} packed bytes, got {len(src)}")
    q = np.empty(n, dtype=np.int8)
    for i in range(n):
        bit = 3 * i
        byte = bit >> 3
        shift = bit & 7
        word = int(src[byte])
        if shift > 5 and byte + 1 < need:
            word |= int(src[byte + 1]) << 8
        code = (word >> shift) & 0x7
        q[i] = code - 8 if code & 0x4 else code
    return q


def quantize_group(x: np.ndarray) -> tuple[int, np.ndarray]:
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot quantize an empty group")
    amax = float(np.max(np.abs(values)))
    scale = amax / QMAX if amax > 0.0 else 1.0
    scale_bits = f32_to_bf16_bits(scale)
    stored_scale = bf16_bits_to_f32(scale_bits)
    if stored_scale == 0.0 or not np.isfinite(stored_scale):
        raise ValueError("invalid Q3 scale")
    # Deterministic nearest-even through NumPy rint; QAT export can supply learned codes.
    codes = np.rint(values / np.float32(stored_scale)).astype(np.int16)
    codes = np.clip(codes, QMIN, QMAX).astype(np.int8)
    return scale_bits, codes


def pack_row(row: np.ndarray) -> bytes:
    x = np.asarray(row, dtype=np.float32).reshape(-1)
    out = bytearray()
    for start in range(0, x.size, GROUP):
        chunk = x[start : start + GROUP]
        scale_bits, codes = quantize_group(chunk)
        out += struct.pack("<H", scale_bits)
        out += pack_codes(codes)
    if len(out) != row_bytes(x.size):
        raise AssertionError("Q3 row-size contract violated")
    return bytes(out)


def unpack_row(buf: bytes | bytearray | memoryview, cols: int) -> np.ndarray:
    src = memoryview(buf)
    if len(src) != row_bytes(cols):
        raise ValueError(f"row has {len(src)} bytes, expected {row_bytes(cols)}")
    y = np.empty(cols, dtype=np.float32)
    off = 0
    for start in range(0, cols, GROUP):
        n = min(GROUP, cols - start)
        scale_bits = struct.unpack_from("<H", src, off)[0]
        off += 2
        nb = packed_code_bytes(n)
        codes = unpack_codes(src[off : off + nb], n).astype(np.float32)
        off += nb
        y[start : start + n] = codes * np.float32(bf16_bits_to_f32(scale_bits))
    return y


def pack_matrix(matrix: np.ndarray) -> bytes:
    w = np.asarray(matrix, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("matrix must be [rows, cols]")
    return b"".join(pack_row(row) for row in w)


def unpack_matrix(buf: bytes, rows: int, cols: int) -> np.ndarray:
    stride = row_bytes(cols)
    if len(buf) != rows * stride:
        raise ValueError("Q3 matrix byte length does not match shape")
    return np.vstack([unpack_row(buf[r * stride : (r + 1) * stride], cols) for r in range(rows)])


def main() -> int:
    p = argparse.ArgumentParser(description="Reference K3-Compact Q3 matrix packer")
    p.add_argument("input", help=".npy float32 matrix")
    p.add_argument("output", help="output .q3 file")
    p.add_argument("--meta", help="optional JSON metadata path")
    args = p.parse_args()
    w = np.load(args.input).astype(np.float32, copy=False)
    if w.ndim != 2:
        raise SystemExit("input must be a 2D .npy matrix")
    packed = pack_matrix(w)
    Path(args.output).write_bytes(packed)
    meta = {
        "format": "k3compact-q3-bf16scale-v1",
        "rows": int(w.shape[0]),
        "cols": int(w.shape[1]),
        "group": GROUP,
        "bytes": len(packed),
        "bits_per_weight_physical": len(packed) * 8 / w.size,
    }
    if args.meta:
        Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
