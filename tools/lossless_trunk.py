#!/usr/bin/env python3
"""Convert a normal packed trunk into a byte-exact dict15-compressed trunk.

The input is the output of tools/pack_trunk.py. Every layer is split into independent
128 MiB RAW blocks. A block is encoded as:

    low-byte plane | packed 4-bit high-byte dictionary codes | escape high bytes

The 15 most common high bytes are chosen independently per block. Code 15 is an escape.
If a block does not shrink, it is stored raw. Every stored block starts and ends on a
4096-byte boundary so the runtime can keep using O_DIRECT.

This is lossless storage compression. Decompression reconstructs the original trunk run
byte-for-byte before any tensor is bound; model arithmetic and weights do not change.

usage: python3 tools/lossless_trunk.py <raw_packed_trunk> <output_dir>
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

ALIGN = 4096
RAW_BLOCK = 128 << 20


def align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


def encode_block(raw: bytes):
    if len(raw) & 1:
        raise ValueError("dict15 blocks must contain an even number of bytes")
    a = np.frombuffer(raw, dtype=np.uint8)
    low = a[0::2]
    high = a[1::2]
    hist = np.bincount(high, minlength=256)
    # Stable sort gives deterministic lowest-byte tie breaking.
    dictionary = np.argsort(-hist, kind="stable")[:15].astype(np.uint8)
    lut = np.full(256, 15, dtype=np.uint8)
    lut[dictionary] = np.arange(15, dtype=np.uint8)
    q = lut[high]
    codes = np.zeros((len(q) + 1) // 2, dtype=np.uint8)
    codes[:] = q[0::2]
    if len(q) > 1:
        codes[: len(q) // 2] |= q[1::2] << 4
    escapes = high[q == 15]
    payload = low.tobytes() + codes.tobytes() + escapes.tobytes()
    return payload, dictionary.tolist(), int(len(escapes))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: lossless_trunk.py <raw_packed_trunk> <output_dir>")
        return 2
    srcdir, outdir = sys.argv[1:]
    with open(os.path.join(srcdir, "trunk.json"), encoding="utf-8") as f:
        srcman = json.load(f)
    if any("blocks" in layer for layer in srcman.get("layers", [])):
        raise SystemExit("input trunk is already block-compressed")
    if int(srcman.get("align", 0)) != ALIGN:
        raise SystemExit(f"input trunk align must be {ALIGN}")

    os.makedirs(outdir, exist_ok=True)
    outman = copy.deepcopy(srcman)
    outman["storage_codec"] = "dict15-block-v1"
    outman["raw_block_bytes"] = RAW_BLOCK
    outman["layers"] = []

    src = open(os.path.join(srcdir, "trunk.bin"), "rb")
    dst = open(os.path.join(outdir, "trunk.bin"), "wb")
    total_raw = total_stored = total_encoded = 0
    compressed_blocks = raw_blocks = 0

    for li, layer in enumerate(srcman["layers"]):
        raw_total = int(layer["nbytes"])
        layer_out = copy.deepcopy(layer)
        layer_out["blocks"] = []
        layer_out["stored_nbytes"] = 0
        first_off = None

        for raw_off in range(0, raw_total, RAW_BLOCK):
            raw_n = min(RAW_BLOCK, raw_total - raw_off)
            if raw_n % ALIGN:
                raise RuntimeError(f"layer {li} block length {raw_n} is not {ALIGN}-aligned")
            src.seek(int(layer["file_off"]) + raw_off)
            raw = src.read(raw_n)
            if len(raw) != raw_n:
                raise OSError(f"short read on layer {li} at raw offset {raw_off}")

            payload, dictionary, nesc = encode_block(raw)
            use_codec = len(payload) < raw_n
            blob = payload if use_codec else raw
            pad = (-dst.tell()) % ALIGN
            if pad:
                dst.write(b"\0" * pad)
            file_off = dst.tell()
            if first_off is None:
                first_off = file_off
            dst.write(blob)
            stored = align_up(len(blob))
            if stored > len(blob):
                dst.write(b"\0" * (stored - len(blob)))

            block = {
                "codec": "dict15" if use_codec else "raw",
                "file_off": file_off,
                "stored_nbytes": stored,
                "encoded_nbytes": len(blob),
                "raw_off": raw_off,
                "raw_nbytes": raw_n,
            }
            if use_codec:
                block["dict"] = dictionary
                block["escapes"] = nesc
                compressed_blocks += 1
            else:
                raw_blocks += 1
            layer_out["blocks"].append(block)
            layer_out["stored_nbytes"] += stored
            total_raw += raw_n
            total_encoded += len(blob)
            total_stored += stored

        layer_out["file_off"] = int(first_off or 0)
        outman["layers"].append(layer_out)
        if (li + 1) % 10 == 0 or li + 1 == len(srcman["layers"]):
            print(
                f"  lossless {li + 1}/{len(srcman['layers'])} layers: "
                f"{total_raw / 1e9:.1f} GB raw -> {total_stored / 1e9:.1f} GB stored",
                flush=True,
            )

    src.close()
    dst.close()
    with open(os.path.join(outdir, "trunk.json"), "w", encoding="utf-8") as f:
        json.dump(outman, f)

    print(
        f"raw {total_raw / 1e9:.2f} GB -> encoded {total_encoded / 1e9:.2f} GB -> "
        f"O_DIRECT stored {total_stored / 1e9:.2f} GB"
    )
    print(
        f"ratio {total_stored / total_raw:.4f}; saved "
        f"{100.0 * (1.0 - total_stored / total_raw):.1f}%"
    )
    print(f"blocks: {compressed_blocks} dict15, {raw_blocks} raw fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
