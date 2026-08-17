#!/usr/bin/env python3
"""Selective, fully-local Kimi K3 vision front-end.

This module deliberately does *not* instantiate KimiK3ForConditionalGeneration: doing so
would allocate the 2.78T language model a second time in PyTorch.  Instead it imports the
official custom model code already present in the checkpoint directory, instantiates only
its MoonViT vision tower and multimodal projector, and loads only ``vision_tower.*`` and
``mm_projector.*`` tensors from the released safetensors index.  The projected hidden-size
rows are then consumed by the exact C language-model worker via its REQMM path.

No model/tokenizer files are downloaded.  Media URLs, when present in a request, are
handled by Moonshot's official local processor and are the only possible network input.
"""
from __future__ import annotations

import importlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_MAGIC = b"K3MMF1\0"
_HEADER = struct.Struct("<8sIIII")
_U32 = struct.Struct("<I")


@dataclass(frozen=True)
class VisionPrepared:
    input_ids: list[int]
    features: list[Any]
    placeholder_id: int
    prompt_positions: int


def _tensor_to_ids(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("the local K3 bridge currently accepts one conversation per request")
        value = value[0]
    return [int(x) for x in value]


def write_feature_sidecar(path: Path, features: list[Any], hidden: int) -> None:
    """Serialize projected media rows in the fail-closed REQMM K3MMF1 format."""
    if not features:
        raise ValueError("cannot write an empty K3 media feature sidecar")
    path = Path(path)
    with path.open("wb") as f:
        f.write(_HEADER.pack(_MAGIC, 1, int(hidden), len(features), 0))
        for feature in features:
            # Import numpy only when media is actually used; text mode has no dependency.
            import numpy as np

            if hasattr(feature, "detach"):
                feature = feature.detach().float().cpu().contiguous().numpy()
            arr = np.asarray(feature, dtype="<f4")
            if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] != hidden:
                raise ValueError(
                    f"projected K3 image feature must be [N,{hidden}], got {arr.shape}"
                )
            f.write(_U32.pack(int(arr.shape[0])))
            f.write(arr.astype("<f4", copy=False).tobytes(order="C"))


def _pick_device(torch_mod: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_mod.cuda.is_available():
        return "cuda"
    return "cpu"


def _official_model_module(model_dir: Path, config: Any) -> Any:
    """Load Moonshot's local custom modeling module without constructing the full model."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ref = getattr(config, "auto_map", {}).get("AutoModel")
    if not ref:
        raise RuntimeError("K3 config exposes no AutoModel custom-code mapping")
    cls = get_class_from_dynamic_module(
        ref,
        str(model_dir),
        local_files_only=True,
    )
    module = sys.modules.get(cls.__module__)
    if module is None:
        module = importlib.import_module(cls.__module__)
    return module


def _load_prefixed_state(
    module: Any,
    *,
    model_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
) -> None:
    """Load one released submodule by consulting the official safetensors weight map."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("K3 vision input needs the 'safetensors' Python package") from exc

    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not wanted:
        raise RuntimeError(f"full checkpoint index contains no tensors with prefix {prefix!r}")

    by_shard: dict[str, list[str]] = {}
    for key, shard in wanted.items():
        by_shard.setdefault(shard, []).append(key)

    state: dict[str, Any] = {}
    for shard, keys in by_shard.items():
        shard_path = model_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"K3 vision shard is missing: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            available = set(sf.keys())
            for key in keys:
                if key not in available:
                    raise RuntimeError(f"checkpoint index points to missing tensor {key} in {shard}")
                state[key[len(prefix) :]] = sf.get_tensor(key)

    incompatible = module.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            f"selective K3 load mismatch for {prefix}: missing={missing[:20]} "
            f"unexpected={unexpected[:20]}"
        )


class K3VisionEncoder:
    """Official MoonViT + K3 projector, loaded selectively from the local checkpoint."""

    def __init__(self, model_dir: Path, *, device: str = "auto", attention: str = "auto"):
        self.model_dir = Path(model_dir).resolve()
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "K3 vision input needs torch, transformers, safetensors, numpy and Pillow"
            ) from exc

        self.torch = torch
        self.device = _pick_device(torch, device)
        self.dtype = torch.bfloat16
        self.config = AutoConfig.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True
        )
        self.placeholder_id = int(self.config.media_placeholder_token_id)
        self.hidden = int(self.config.text_config.hidden_size)

        official = _official_model_module(self.model_dir, self.config)
        for name in (
            "VisionTowerConfig",
            "MoonViT3dPretrainedModel",
            "ProjectorConfig",
            "IdentityMap",
            "MLP",
            "PatchMergerMLP",
            "PatchMergerMLPV2",
        ):
            if not hasattr(official, name):
                raise RuntimeError(f"official K3 modeling module is missing {name}")

        # The released model code has an eager attention implementation in addition to
        # FlashAttention2.  Prefer FA2 only when explicitly requested or importable on a
        # CUDA device; otherwise remain on the official eager implementation rather than
        # substituting a third-party approximation.
        if attention == "auto":
            attn_impl = "eager"
            if self.device.startswith("cuda"):
                try:
                    import flash_attn  # noqa: F401
                except Exception:
                    pass
                else:
                    attn_impl = "flash_attention_2"
        elif attention in {"eager", "flash_attention_2"}:
            attn_impl = attention
        else:
            raise ValueError("vision attention must be auto, eager, or flash_attention_2")

        vision_cfg = self.config.vision_config
        vision_cfg._attn_implementation = attn_impl
        vt_cfg = official.VisionTowerConfig(vision_cfg)
        vt_cfg._attn_implementation = attn_impl
        self.vision_tower = official.MoonViT3dPretrainedModel(vt_cfg).to(dtype=self.dtype)

        proj_cfg = official.ProjectorConfig(vision_cfg)
        projector_cls = {
            "identity": official.IdentityMap,
            "mlp": official.MLP,
            "patchmerger": official.PatchMergerMLP,
            "patchmergerv2": official.PatchMergerMLPV2,
        }.get(proj_cfg.mm_projector_type)
        if projector_cls is None:
            raise RuntimeError(f"unsupported released K3 projector {proj_cfg.mm_projector_type!r}")
        self.mm_projector = projector_cls(proj_cfg).to(dtype=self.dtype)

        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                "selective K3 vision loading requires model.safetensors.index.json from the "
                "released full checkpoint"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError("K3 safetensors index has no weight_map")
        _load_prefixed_state(
            self.vision_tower,
            model_dir=self.model_dir,
            weight_map=weight_map,
            prefix="vision_tower.",
        )
        _load_prefixed_state(
            self.mm_projector,
            model_dir=self.model_dir,
            weight_map=weight_map,
            prefix="mm_projector.",
        )

        self.vision_tower = self.vision_tower.to(self.device).eval()
        self.mm_projector = self.mm_projector.to(self.device).eval()
        self.attention = attn_impl

    def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        reasoning_effort: str,
        tool_choice: Any = None,
        response_format: Any = None,
        thinking_enabled: bool = True,
    ) -> VisionPrepared:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "thinking": thinking_enabled,
        }
        if thinking_enabled:
            kwargs["thinking_effort"] = reasoning_effort
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format

        batch = self.processor(messages=messages, return_tensors="pt", **kwargs)
        ids = _tensor_to_ids(batch["input_ids"])
        n_placeholders = sum(x == self.placeholder_id for x in ids)
        if n_placeholders <= 0:
            raise ValueError("K3 processor produced no image placeholder for a media request")
        if "pixel_values" not in batch or "grid_thws" not in batch:
            raise ValueError("K3 processor produced no pixel_values/grid_thws for image input")

        pixel_values = batch["pixel_values"].to(self.device)
        grid_thws = batch["grid_thws"].to(self.device)
        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)
        with self.torch.inference_mode():
            image_features = self.vision_tower(pixel_values, grid_thws)
            image_features = self.mm_projector(image_features)

        features = [x.detach().float().cpu().contiguous() for x in image_features]
        if len(features) != n_placeholders:
            raise ValueError(
                f"K3 processor rendered {n_placeholders} image placeholder(s) but vision "
                f"tower returned {len(features)} feature group(s)"
            )
        for feature in features:
            if feature.ndim != 2 or feature.shape[1] != self.hidden or feature.shape[0] <= 0:
                raise ValueError(
                    f"official K3 projector returned invalid shape {tuple(feature.shape)}; "
                    f"expected [N,{self.hidden}]"
                )

        prompt_positions = len(ids) - n_placeholders + sum(int(x.shape[0]) for x in features)
        return VisionPrepared(
            input_ids=ids,
            features=features,
            placeholder_id=self.placeholder_id,
            prompt_positions=prompt_positions,
        )

    def sidecar(self, features: list[Any]):
        """Context manager yielding a temporary K3MMF1 file for one worker request."""
        encoder = self

        class _Sidecar:
            def __enter__(self):
                self.tmp = tempfile.TemporaryDirectory(prefix="k3-media-")
                self.path = Path(self.tmp.name) / "features.k3mmf"
                write_feature_sidecar(self.path, features, encoder.hidden)
                return self.path

            def __exit__(self, exc_type, exc, tb):
                self.tmp.cleanup()
                return False

        return _Sidecar()
