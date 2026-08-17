# Moonshot K3 Vendor Verifier parity

This repository pins Moonshot's public `Kimi-Vendor-Verifier` by exact commit instead of copying its prompts or scoring code. The pinned contract is `benchmarks/k3_vendor_verifier_contract.json`; the orchestrator is `benchmarks/k3_vendor_verifier.py`; the self-hosted entrypoint is `.github/workflows/k3-official-vendor-verifier.yml`.

## Pinned upstream

Repository: `MoonshotAI/Kimi-Vendor-Verifier`

Commit: `3dad65a760a8867cda72f6dd8848d876a4e851b4`

`prepare` clones that repository, checks out the exact SHA, verifies `HEAD`, and performs the Git LFS pull required by the official fixtures. A different upstream commit is a hard failure, not an implicit benchmark upgrade.

## Moonshot K3 targets pinned here

| Official verifier suite | Moonshot submitted K3 score | Local gate |
|---|---:|---|
| OCRBench | 0.89 | direct official Inspect eval against localhost |
| MMMU Pro Vision | 0.82 | direct official Inspect eval against localhost |
| BEAM (1M) | 0.31 | official 700-question generation + explicit LLM judge |
| DeepSWE | 0.675 | official Pier + Kimi Code measurement required |

OCRBench and MMMU are fully automatable on a self-hosted runner that has the released K3 checkpoint. The workflow first runs Moonshot's API/tool/prompt-token preflight suites, then invokes upstream `eval.py` with the published K3 Max parameters (`thinking=max`, preserved thinking, temperature 1.0, top-p 0.95 and the published output budgets). It reads Inspect's own `accuracy` metric and fails below Moonshot's submitted score.

BEAM is intentionally split exactly as upstream documents it: `beam_generate` produces answers with the local K3 endpoint and the exact local K3 tokenizer; `beam_judge` requires an explicitly configured judge endpoint/model and gates the mean official judge score. A partial `--limit` generation is bring-up only and is never reported as parity.

DeepSWE is not reimplemented in this repository. Moonshot's public protocol uses Pier and Kimi Code on 113 tasks. `deepswe_gate` accepts only the score produced by that run and refuses values below 0.675.

## Laptop execution

The self-hosted workflow uses one benchmark request at a time by default. It can consume the runner's already autotuned exact settings (`K3_THREADS`, `K3_ASYNC_IO_THREADS`, exact/lossless `K3_TRUNK_DIR`, optional verified `K3_DRAFT_TRUNK_DIR`, draft top-k/spec width). Draft proposals never become benchmark outputs without exact K3 verification.

The local model endpoint remains `127.0.0.1`; inference does not go to Moonshot or another model provider. Dataset downloads and, for BEAM scoring, the explicitly configured judge are separate from K3 inference.

## What a green hosted CI means

Hosted CI can prove the API contract, exact C model oracles, mixed-image prefill parity, and the verifier orchestration/commit pin. It cannot fabricate the 1.56 TB full-checkpoint scores. OCR/MMMU score parity turns green only after the self-hosted official verifier actually measures them; BEAM additionally needs the official judge phase; DeepSWE needs Pier.
