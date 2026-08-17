# K3 Max parity contract

The target is not "looks like K3". The target is the released `moonshotai/Kimi-K3`
checkpoint with the published K3 Max inference/harness contract, while runtime
optimisations are accepted only behind exact/token or benchmark gates.

## Public K3 Max contract pinned by this repository

Source snapshots are the official Moonshot K3 README and Kimi Vendor Verifier, checked on
2026-08-17.

- context window: 1,048,576 positions;
- thinking: always enabled; benchmark effort `max`;
- temperature: 1.0;
- top-p: 0.95 on published single-step tasks, 1.0 on agentic tasks;
- preserved thinking: complete assistant reasoning/tool history is passed back on later
  turns;
- the vendor verifier uses output budgets up to 98,304 tokens for K3 MMMU;
- coding benchmarks that Moonshot identifies as Kimi-Code runs must use the Kimi Code
  harness against the localhost K3 endpoint rather than a home-grown agent loop.

The machine-readable score snapshot is `benchmarks/k3_max_reference.json` and
`benchmarks/k3_score_gate.py` fails closed on a missing or lower score by default.

## Exactness boundary

The released bf16/MXFP4 K3 remains authoritative. Q4/reduced-top-k trunks are proposal
models only. No draft token may be committed until the exact model accepts/corrects it.
Greedy optimisations must preserve the exact token stream. Sampled speculation must
preserve the target distribution and, for the seeded regression harness, preserve the
historical effective width-8 schedule.

The resident worker is allowed to accept generation budgets larger than the legacy
one-shot CLI ceiling because its sequence buffers are sized from `--context`. It must
still reject `prompt + max_tokens > context`; the Python bridge performs the same check
before starting expensive inference.

## Capability gates

### Text / coding / agentic

The local bridge loads the official tokenizer/template from the checkpoint in offline mode,
uses K3 XTML, exposes `reasoning_content` and tool calls, preserves thinking, and can be
driven by Kimi Code. `local/kimi-code-benchmark.toml.example` is the benchmark profile.

A result is not called benchmark-parity until the actual full-checkpoint harness result is
fed to `benchmarks/k3_score_gate.py` and passes against the pinned references.

### Vision

The current C runtime does not yet implement the released MoonViT-V2/image path. Media is
rejected instead of silently converted to text. Therefore the vision section in
`k3_max_reference.json` is deliberately a red/unverified gate, not a claimed score.
Full-suite K3 Max parity requires native vision inference and the same multimodal benchmark
protocol before those rows can turn green.

## Performance rule

Speed is subordinate to parity. A speed change may be promoted only when the relevant
exact token/distribution gate passes; benchmark-facing changes additionally require the
score gate on the real checkpoint/harness. The laptop may have very different wall-clock
latency from Moonshot's serving hardware without that implying lower model intelligence.

Official references:

- https://github.com/MoonshotAI/Kimi-K3
- https://github.com/MoonshotAI/Kimi-Vendor-Verifier
