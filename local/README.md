# K3 Local — official-model chat / coding harness, no inference API

This directory turns the C engine into a **fully local** Kimi K3 chat and coding-agent
backend. The model weights stay on this machine. The HTTP endpoint is localhost and exists
only so normal clients — especially the official **Kimi Code** CLI used by K3's coding
evaluations — can talk to the local engine without a custom fork of the agent harness.

## What “parity” means

There are three different things that are easy to mix together:

1. **Model parity** — the released `moonshotai/Kimi-K3` checkpoint is authoritative. The
   exact C path does not replace it with a distilled model.
2. **Behavior / harness parity** — use K3's own local XTML chat template, always-on
   thinking with effort `max`, preserved reasoning history, tool calls, temperature 1.0,
   and the Kimi Code agent loop. This is what this directory adds.
3. **Datacenter latency parity** — a laptop cannot match an H20/H100-class cluster merely
   by changing software. The C backend instead attacks storage/RAM bottlenecks and scales
   down to consumer hardware. Never interpret “same model/harness” as a promise of the
   hosted service's wall-clock latency.

For the published **agentic coding** evaluation profile, run `reasoning=max`,
`temperature=1.0`, `top_p=1.0`. For the published single-step profile use top-p 0.95. The
localhost server defaults to the agentic values when a client omits sampling fields.

## 1. Build and prepare the exact local model

Follow the repository root README first: download the official checkpoint, build `bin/k3`,
pack the trunk, optionally convert it to the byte-exact lossless representation, and run
`make test`.

Example paths used below:

```text
~/k3model/             official K3 checkpoint + tokenizer files
~/k3trunk-lossless/    exact lossless packed trunk
./bin/k3               C inference engine
```

The Python bridge needs `transformers` only for K3's **official local chat tokenizer / XTML
template**. Put it in a venv rather than modifying the system interpreter:

```bash
python3 -m venv .venv-k3
. .venv-k3/bin/activate
pip install 'transformers>=4.56' tiktoken
```

At runtime the bridge sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and also passes
`local_files_only=True`. A missing tokenizer file therefore fails instead of silently
fetching one from the network.

## 2. Find the fastest thread count on this machine

Do not assume every logical CPU is faster. Use the real-workload sweep already shipped by
this fork:

```bash
K3_SWEEP_REPEATS=3 benchmarks/thread-sweep.sh ~/k3model \
  --trunk ~/k3trunk-lossless --preset laptop --incremental \
  --ids 1008,10484,318,15383,387 --gen 4
```

Keep the recommended `N` for the server command below.

## 3. Start the completely local API-compatible bridge

```bash
. .venv-k3/bin/activate
python local/k3_local.py serve \
  --model-dir ~/k3model \
  --trunk ~/k3trunk-lossless \
  --preset laptop \
  --threads N
```

It listens on `http://127.0.0.1:8000/v1` by default. The server refuses a non-loopback bind
unless `--allow-remote` is explicitly supplied; the endpoint itself has no authentication.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

A direct chat-completions request, still fully local:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"kimi-k3-local",
    "messages":[{"role":"user","content":"Fix the race in this C queue."}],
    "thinking":{"type":"enabled","effort":"max","keep":"all"},
    "temperature":1.0,
    "top_p":1.0,
    "max_completion_tokens":4096
  }'
```

The response exposes both `reasoning_content` and ordinary `content`, plus OpenAI-style
`tool_calls` when K3 emits XTML tool calls.

## 4. Use the official Kimi Code agent harness against localhost

For ordinary interactive use, copy the safe profile:

```bash
mkdir -p ~/.kimi-code
cp local/kimi-code-local.toml.example ~/.kimi-code/config.toml
kimi
```

That config uses Kimi Code's own `kimi` protocol adapter but sets its `base_url` to
`http://127.0.0.1:8000/v1`. The dummy `local-only` credential satisfies the client's
non-empty credential check; the local bridge neither validates nor forwards it.

Important parity settings in the profile:

```toml
capabilities = ["thinking", "always_thinking", "tool_use"]
support_efforts = ["max"]
default_effort = "max"

[thinking]
enabled = true
effort = "max"
keep = "all"
```

`keep = "all"` matters: Kimi Code sends the assistant's previous thinking back in later
turns, which is part of the intended K3 multi-turn/tool behavior.

### Benchmark profile

`local/kimi-code-benchmark.toml.example` declares the official 1,048,576-token model
window and uses autonomous permission mode. Use it only inside the disposable environment
provided by the benchmark:

```bash
export KIMI_CODE_HOME="$PWD/.kimi-code-bench"
mkdir -p "$KIMI_CODE_HOME"
cp local/kimi-code-benchmark.toml.example "$KIMI_CODE_HOME/config.toml"
kimi
```

The **declared** model window and the **physically affordable** local context are separate.
The current exact C incremental cache stores expanded MLA K/V in fp32 (~2.37 MB per
position across the 24 MLA layers), so a small machine will correctly refuse a huge
context before allocation. The next performance target is a latent MLA cache; do not hide
this constraint by letting the OS OOM-kill the run.

## Conversation state reuse

By default the bridge keeps one exact saved prefix under:

```text
~/.cache/k3-local/state/
```

When Kimi Code sends the next full conversation, the bridge tokenizes it with the official
XTML template and compares it against saved token prefixes. If an exact prefix matches, it
passes only the new suffix to `bin/k3 --load-state`: prior KDA/MLA state is restored instead
of recomputing the whole conversation. If even one token differs, the cache is ignored and
the full prompt is evaluated — correctness wins over a guessed reuse.

Current state files contain expanded MLA KV and can therefore be large. Controls:

```bash
--state-cache-entries 1   # default; exact-prefix LRU
--state-cache-dir PATH
--no-state-cache
```

Keep the cache directory on fast local NVMe.

## Sampling correctness

The C engine now supports:

```text
--temperature X
--top-p X
--seed N
--stop-id N
--ids-file PATH
--stream-tokens
```

Temperature 0 remains the legacy exact greedy path. Temperature 1 / top-p 1 is the coding
agent profile. A fixed seed is reproducible inside this engine.

Sampled speculative decoding is intentionally **not** faked. The existing speculative
path is exact for greedy decode; proper sampled speculation needs rejection sampling using
both target and draft probabilities. Until that is implemented, asking for sampling plus
`--spec` / `--draft-trunk` is rejected instead of silently changing the target
distribution.

## True committed-token streaming

With `stream: true`, the bridge starts the C engine with `--stream-tokens`. After a token
has been verified and committed to the C sequence/output buffers, the engine immediately
flushes one machine-readable line:

```text
@K3TOKEN <token-id>
```

The marker is emitted **after commit**, not when a speculative draft merely proposes a
token. Python ignores all other human CLI diagnostics, accumulates the committed ids,
re-decodes them through the official local K3 tokenizer, and sends new
`reasoning_content`, response text, and completed tool calls to Kimi Code as SSE deltas.

Byte-level tokenization has one awkward edge: an incomplete UTF-8 byte sequence can decode
temporarily as U+FFFD until the next token arrives. The bridge therefore holds only that
unstable suffix and releases it as soon as the next token makes the text stable; it never
sends a character it would later need to retract.

If the SSE client disconnects, the callback aborts and the C child is terminated instead
of continuing a multi-minute generation nobody is consuming. At normal completion the
bridge also compares every streamed marker id with the final `generated_ids` JSON; any
protocol drift is an error.

The permanent tiny-checkpoint CI independently performs the same comparison, so true
streaming is part of the correctness gate rather than a UI-only feature.

## Current gaps, stated explicitly

- **Text/coding first:** image/video/audio message parts are rejected instead of discarded.
  K3's vision path still needs to be integrated locally.
- **Process startup per turn:** true token streaming is live now, and saved state avoids
  recomputing exact conversation prefixes, but the one-shot C process still reopens the
  checkpoint/index/trunk/cache for each HTTP request. A resident C worker is the next
  latency step and will also preserve the warm expert cache between tool turns.
- **Long context:** the exact expanded fp32 MLA cache is the current context-memory wall.
  A latent-cache kernel can reduce it dramatically, but it must be numerically gated
  before becoming a default.
- **Benchmark numbers are measured, not inherited:** using the same checkpoint and harness
  is a prerequisite for reproducing a score; it is not evidence that a local run already
  achieved that score. Run the actual benchmark suite before quoting a number.

Those gaps are deliberately visible because the goal is a credible local K3 runtime, not
an API-shaped demo that silently changes model behavior.
