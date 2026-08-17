#!/usr/bin/env python3
"""Smoke the released K3 vision tower/projector without instantiating the 2.78T LM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attention", default="eager")
    ap.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("k3_vision_contract.json"),
    )
    args = ap.parse_args()

    import torch
    from PIL import Image

    from local.k3_vision import K3VisionEncoder

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    enc = K3VisionEncoder(
        args.model_dir,
        device=args.device,
        attention=args.attention,
    )

    vc = enc.config.vision_config
    expected = contract["vision"]
    assert enc.placeholder_id == contract["media_placeholder_token_id"]
    assert enc.hidden == contract["language_hidden_size"]
    assert int(vc.patch_size) == expected["patch_size"]
    assert int(vc.vt_hidden_size) == expected["hidden_size"]
    assert int(vc.vt_intermediate_size) == expected["intermediate_size"]
    assert int(vc.vt_num_attention_heads) == expected["attention_heads"]
    assert int(vc.vt_num_hidden_layers) == expected["hidden_layers"]
    assert list(vc.merge_kernel_size) == expected["merge_kernel_size"]
    assert vc.merge_type == expected["merge_type"]
    assert vc.mm_projector_type == expected["projector_type"]
    assert int(vc.qkv_hidden_size) == expected["qkv_hidden_size"]

    # A deterministic local PIL object exercises Moonshot's actual media processor without
    # allowing the smoke test itself to depend on a URL or external image service.
    image = Image.new("RGB", (56, 56))
    px = image.load()
    for y in range(56):
        for x in range(56):
            px[x, y] = ((x * 7 + y * 3) & 255, (x * 5) & 255, (y * 11) & 255)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image precisely."},
            ],
        }
    ]
    prepared = enc.prepare(messages, tools=None, reasoning_effort="max")

    placeholders = sum(x == enc.placeholder_id for x in prepared.input_ids)
    assert placeholders == 1, placeholders
    assert len(prepared.features) == 1
    feature = prepared.features[0]
    assert feature.ndim == 2
    assert feature.shape[0] > 0
    assert feature.shape[1] == enc.hidden
    assert torch.isfinite(feature).all().item()
    expected_positions = len(prepared.input_ids) - placeholders + int(feature.shape[0])
    assert prepared.prompt_positions == expected_positions
    assert prepared.prompt_positions > len(prepared.input_ids)

    # The selective loader must really be only the vision/projector side: there is no
    # language_model attribute and no second copy of K3's trillions of parameters.
    assert not hasattr(enc, "language_model")
    vision_params = sum(p.numel() for p in enc.vision_tower.parameters())
    projector_params = sum(p.numel() for p in enc.mm_projector.parameters())
    assert vision_params > 0 and projector_params > 0

    print(
        "K3 FULL VISION FRONTEND SMOKE PASS",
        {
            "device": enc.device,
            "attention": enc.attention,
            "input_ids": len(prepared.input_ids),
            "feature_rows": int(feature.shape[0]),
            "feature_width": int(feature.shape[1]),
            "prompt_positions": prepared.prompt_positions,
            "vision_params": vision_params,
            "projector_params": projector_params,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
