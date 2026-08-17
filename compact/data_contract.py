#!/usr/bin/env python3
"""Provenance and contamination gates for K3-Compact specialist training data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

DOMAINS = frozenset({"code", "agentic", "cyber", "general"})
ALLOWED_SPLITS = frozenset({"train", "dev"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text_hash(text: str) -> str:
    # Normalize line endings only. Do NOT lowercase/strip content; exact provenance should
    # distinguish semantically different prompts and retain meaningful whitespace/code.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalized.encode("utf-8"))


@dataclass(frozen=True)
class TrainingRecord:
    sample_id: str
    domain: str
    split: str
    source_id: str
    prompt_sha256: str
    chosen_sha256: str
    rejected_sha256: str
    verifier_id: str
    verifier_passed: bool

    @classmethod
    def from_mapping(cls, x: Mapping[str, object]) -> "TrainingRecord":
        return cls(
            sample_id=str(x["sample_id"]),
            domain=str(x["domain"]),
            split=str(x["split"]),
            source_id=str(x["source_id"]),
            prompt_sha256=str(x["prompt_sha256"]),
            chosen_sha256=str(x["chosen_sha256"]),
            rejected_sha256=str(x["rejected_sha256"]),
            verifier_id=str(x["verifier_id"]),
            verifier_passed=bool(x["verifier_passed"]),
        )


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_records(
    records: Iterable[TrainingRecord],
    *,
    denied_hashes: set[str] | frozenset[str] = frozenset(),
    denied_source_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    """Fail on duplicates, held-out/eval sources, invalid provenance or unverified pairs."""
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    counts = {domain: 0 for domain in DOMAINS}
    total = 0

    for r in records:
        total += 1
        if not r.sample_id or r.sample_id in seen_ids:
            raise ValueError(f"duplicate/empty sample_id: {r.sample_id!r}")
        seen_ids.add(r.sample_id)
        if r.domain not in DOMAINS:
            raise ValueError(f"unsupported domain {r.domain!r}")
        if r.split not in ALLOWED_SPLITS:
            raise ValueError(f"training manifest may not contain held-out split {r.split!r}")
        if not r.source_id:
            raise ValueError(f"{r.sample_id}: empty source_id")
        if any(r.source_id.startswith(prefix) for prefix in denied_source_prefixes):
            raise ValueError(f"{r.sample_id}: source is on held-out denylist: {r.source_id}")
        hashes = (r.prompt_sha256, r.chosen_sha256, r.rejected_sha256)
        if any(not _valid_sha256(h) for h in hashes):
            raise ValueError(f"{r.sample_id}: malformed SHA-256 provenance")
        blocked = denied_hashes.intersection(hashes)
        if blocked:
            raise ValueError(f"{r.sample_id}: content hash collides with held-out evaluation data")
        if r.prompt_sha256 in seen_prompts:
            raise ValueError(f"{r.sample_id}: duplicate prompt hash {r.prompt_sha256}")
        seen_prompts.add(r.prompt_sha256)
        if r.chosen_sha256 == r.rejected_sha256:
            raise ValueError(f"{r.sample_id}: chosen/rejected are identical")
        if not r.verifier_id or not r.verifier_passed:
            raise ValueError(f"{r.sample_id}: verified-improvement pair lacks a passing verifier")
        counts[r.domain] += 1

    if total == 0:
        raise ValueError("training manifest is empty")
    return {"total": total, **counts}


def load_jsonl(path: str | Path) -> list[TrainingRecord]:
    out: list[TrainingRecord] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
            out.append(TrainingRecord.from_mapping(x))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid training record: {exc}") from exc
    return out
