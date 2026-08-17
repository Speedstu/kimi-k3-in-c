#!/usr/bin/env python3
"""Fail-closed preflight for a real K3-Compact training run.

This script intentionally does NOT train a model. It verifies that a self-hosted training
runner has the teacher checkpoint, benchmark-clean verified training manifest, held-out
hash denylist and writable output space before any expensive training command is allowed
to start. Missing evidence is a hard failure, never a warning.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from compact.data_contract import load_jsonl, validate_records
from compact.plan import estimate, load_spec

DEFAULT_MIN_TEACHER_BYTES = 1_400_000_000_000
DEFAULT_MIN_SHARDS = 96
DEFAULT_MIN_FREE_OUTPUT_BYTES = 250_000_000_000


def _load_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        if len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
            raise ValueError(f"{path}:{lineno}: expected lowercase/uppercase SHA-256 hex")
        hashes.add(line)
    if not hashes:
        raise ValueError(f"{path}: held-out hash denylist is empty")
    return hashes


def _checkpoint_inventory(model_dir: Path) -> dict[str, Any]:
    if not (model_dir / "config.json").is_file():
        raise ValueError(f"teacher checkpoint missing {model_dir / 'config.json'}")
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"teacher checkpoint has no .safetensors shards in {model_dir}")
    total = sum(p.stat().st_size for p in shards)
    return {
        "model_dir": str(model_dir.resolve()),
        "safetensors_shards": len(shards),
        "safetensors_bytes": total,
    }


def _gpu_inventory() -> dict[str, Any]:
    """Best-effort metadata only; GPU policy is caller-configurable and checked separately."""
    try:
        import torch
    except ModuleNotFoundError:
        return {"torch_available": False, "cuda_available": False, "cuda_devices": 0}
    cuda = bool(torch.cuda.is_available())
    return {
        "torch_available": True,
        "cuda_available": cuda,
        "cuda_devices": int(torch.cuda.device_count()) if cuda else 0,
    }


def run_preflight(
    *,
    model_dir: Path,
    train_manifest: Path,
    heldout_hashes: Path,
    output_dir: Path,
    spec_path: Path,
    denied_source_prefixes: tuple[str, ...] = ("eval/", "benchmark/test/", "heldout/"),
    min_teacher_bytes: int = DEFAULT_MIN_TEACHER_BYTES,
    min_shards: int = DEFAULT_MIN_SHARDS,
    min_free_output_bytes: int = DEFAULT_MIN_FREE_OUTPUT_BYTES,
    min_cuda_devices: int = 0,
) -> dict[str, Any]:
    if min_teacher_bytes <= 0 or min_shards <= 0 or min_free_output_bytes <= 0:
        raise ValueError("minimum resource requirements must be positive")
    if min_cuda_devices < 0:
        raise ValueError("min_cuda_devices may not be negative")

    spec = load_spec(spec_path)
    architecture = estimate(spec)
    if not architecture["fits_checkpoint_budget"]:
        raise ValueError("K3-Compact architecture no longer fits its checkpoint budget")

    teacher = _checkpoint_inventory(model_dir)
    if int(teacher["safetensors_shards"]) < min_shards:
        raise ValueError(
            f"teacher checkpoint incomplete: {teacher['safetensors_shards']} shards < {min_shards}"
        )
    if int(teacher["safetensors_bytes"]) < min_teacher_bytes:
        raise ValueError(
            f"teacher checkpoint too small: {teacher['safetensors_bytes']} bytes < {min_teacher_bytes}"
        )

    if not train_manifest.is_file():
        raise ValueError(f"training manifest missing: {train_manifest}")
    if not heldout_hashes.is_file():
        raise ValueError(f"held-out hash denylist missing: {heldout_hashes}")
    denied = _load_hashes(heldout_hashes)
    records = load_jsonl(train_manifest)
    data_counts = validate_records(
        records,
        denied_hashes=denied,
        denied_source_prefixes=denied_source_prefixes,
    )
    # A specialist run must actually contain every promised capability domain.
    for domain in ("code", "agentic", "cyber", "general"):
        if data_counts.get(domain, 0) <= 0:
            raise ValueError(f"training manifest has zero {domain!r} records")

    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".k3compact-write-probe"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    usage = shutil.disk_usage(output_dir)
    if usage.free < min_free_output_bytes:
        raise ValueError(
            f"output filesystem free space {usage.free} < required {min_free_output_bytes} bytes"
        )

    gpu = _gpu_inventory()
    if min_cuda_devices:
        if not gpu["torch_available"]:
            raise ValueError("PyTorch is required when min_cuda_devices > 0")
        if int(gpu["cuda_devices"]) < min_cuda_devices:
            raise ValueError(
                f"training runner has {gpu['cuda_devices']} CUDA devices < required {min_cuda_devices}"
            )

    report = {
        "pass": True,
        "teacher": teacher,
        "data": data_counts,
        "heldout_hashes": len(denied),
        "output_dir": str(output_dir.resolve()),
        "output_free_bytes": usage.free,
        "gpu": gpu,
        "architecture": {
            "total_params_b": architecture["total_params_b"],
            "active_params_b": architecture["active_params_b"],
            "estimated_checkpoint_gb": architecture["estimated_checkpoint_gb"],
        },
        "claim": "preflight only; no K3-Compact quality or superiority claim",
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("K3_MODEL_DIR"))
    p.add_argument("--train-manifest", default=os.environ.get("K3_COMPACT_TRAIN_MANIFEST"))
    p.add_argument("--heldout-hashes", default=os.environ.get("K3_COMPACT_HELDOUT_HASHES"))
    p.add_argument("--output-dir", default=os.environ.get("K3_COMPACT_OUTPUT_DIR"))
    p.add_argument("--spec", default="compact/k3_compact.json")
    p.add_argument("--min-teacher-bytes", type=int, default=DEFAULT_MIN_TEACHER_BYTES)
    p.add_argument("--min-shards", type=int, default=DEFAULT_MIN_SHARDS)
    p.add_argument("--min-free-output-bytes", type=int, default=DEFAULT_MIN_FREE_OUTPUT_BYTES)
    p.add_argument("--min-cuda-devices", type=int, default=int(os.environ.get("K3_COMPACT_MIN_CUDA_DEVICES", "0")))
    p.add_argument("--json-out")
    args = p.parse_args()

    required = {
        "--model-dir/K3_MODEL_DIR": args.model_dir,
        "--train-manifest/K3_COMPACT_TRAIN_MANIFEST": args.train_manifest,
        "--heldout-hashes/K3_COMPACT_HELDOUT_HASHES": args.heldout_hashes,
        "--output-dir/K3_COMPACT_OUTPUT_DIR": args.output_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("missing required full-training inputs: " + ", ".join(missing))

    report = run_preflight(
        model_dir=Path(args.model_dir),
        train_manifest=Path(args.train_manifest),
        heldout_hashes=Path(args.heldout_hashes),
        output_dir=Path(args.output_dir),
        spec_path=Path(args.spec),
        min_teacher_bytes=args.min_teacher_bytes,
        min_shards=args.min_shards,
        min_free_output_bytes=args.min_free_output_bytes,
        min_cuda_devices=args.min_cuda_devices,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
