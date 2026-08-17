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
./bin/k3               one-shot C inference engine
./bin/k3-worker        resident C inference worker (default localhost backend)
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
  --threads N \
  --worker-context 1024
```

`k3_local.py` now starts **one resident `bin/k3-worker` process by default**. The
safetensors index, exact packed trunk mappings, model head, expert cache, recurrent KDA
state and MLA KV stay alive between HTTP/tool turns. If the next XTML prompt extends the
previous exact token sequence, only the pending last token plus the new suffix is fed; a
bifurcation resets conversation state but keeps weights and the warm expert cache open.
Use `--no-resident-worker` only for A/B testing or a feature not yet supported by the
worker.

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
window, a 98,304-token maximum output budget for the longest published K3 verifier profile,
and autonomous permission mode. Use it only inside the disposable environment provided by
the benchmark. Start the resident server with a `--worker-context` large enough for the
rendered prompt plus that output ceiling; the bridge refuses undersized contexts rather
than silently truncating reasoning:

```bash
export KIMI_CODE_HOME="$PWD/.kimi-code-bench"
mkdir -p "$KIMI_CODE_HOME"
cp local/kimi-code-benchmark.toml.example "$KIMI_CODE_HOME/config.toml"
kimi
```

The **declared** model window and the **physically affordable** local context are separate.
The exact C incremental cache still stores expanded MLA K/V in fp32 (~2.37 MB per used
position across the 24 MLA layers). The resident worker now reserves its configured KV
capacity with lazy anonymous virtual memory on supported POSIX systems, so merely choosing
a larger capacity no longer faults every KV page into RAM. Physical use still grows as
positions are actually written. This removes an artificial startup/reset cost; it does
**not** make a million used tokens fit in laptop RAM.

## Conversation state reuse

The default path is now **in-RAM resident reuse**, not a multi-GB state file per tool
turn. `k3-worker` retains the active exact sequence, KDA recurrence and MLA KV. Every new
request still sends the full canonical XTML token sequence; the worker reuses state only
when the entire previous sequence is an exact prefix. It reports the exact number of
prefix tokens reused. Any token mismatch causes a conversation-state reset, while the
loaded checkpoint/trunk and expert cache remain warm.

The resident KV capacity is explicit and can now be configured up to the declared K3
window:

```bash
--worker-context 1024      # conservative default
--worker-context 1048576   # maximum VIRTUAL capacity; not a laptop-RAM promise
```

On a 64-bit POSIX build, large exact/draft KV regions use anonymous lazy mappings and do
not fall back to a giant `calloc` if that virtual reservation fails. The worker prints the
per-used-position KV cost and total virtual reservation at startup. CI boots a tiny K3
worker with `--context 1048576`, performs an exact request, and rejects `1048577` before
model allocation. This validates the capacity mechanism, not the feasibility of filling a
million positions with the released model.

The worker reserves KV address space lazily. Prompt prefill is also RAM-bounded, but no
longer hard-coded to 64 tokens: by default `--prefill-mb 256` chooses the largest batch
(up to 8192 tokens) whose hidden/residual/scratch estimate fits that transient budget. If
an allocation still fails because of fragmentation/rlimits, the worker halves the chunk
until it fits. Larger chunks matter because every chunk is another whole-model/trunk sweep;
the right value is therefore a direct RAM-vs-I/O speed tradeoff. A conversation reset
zeros only true KDA recurrent/ShortConv state: setting
`cached=0` makes old MLA KV rows unreachable, and every row used by the next conversation
is fully overwritten before it can be read. On anonymous mappings the worker also gives
the actually-touched dead KV pages back to the OS with best-effort `MADV_DONTNEED`, rather
than writing zeros through gigabytes. This reclamation is a memory optimisation only;
correctness still comes from the cache-position invariant. Permanent CI gates a 130-token
prompt across two chunk boundaries and a long-to-short reset against one-shot K3.

RAM is still proportional to **used** context. Expanded fp32 MLA KV costs roughly 2.37 MB
per used position across the 24 MLA layers (and a resident draft has its own state), so
`--worker-context` remains a real capacity decision rather than a promise that the full
advertised 1M-token model window is affordable locally.

The older disk-backed prefix cache remains available with `--no-resident-worker`:

```bash
--no-resident-worker
--state-cache-entries 1
--state-cache-dir PATH
--no-state-cache
```

That fallback is useful for A/B tests and can retain multiple prefixes, but the resident
worker is the low-latency path for the normal linear Kimi Code tool loop.

### Prefill speed tuning

The localhost server exposes the same controls:

```bash
--prefill-mb 256        # automatic transient-RAM budget, normal path
--prefill-chunk 512     # manual override for measurement/debugging
```

Do not assume that the largest chunk is fastest: resident trunk pages, expert-cache size,
NVMe bandwidth and available RAM all matter. Measure the actual machine with an already
tokenised representative prompt:

```bash
python benchmarks/prefill-sweep.py ~/k3model ~/k3trunk-lossless prompt.ids \
  --trunk-gb 3 --cache-gb 1 --budgets 64,128,256,512,1024 --repeats 2
```

The sweep starts the same local `k3-worker` for each budget, refuses to rank a candidate if
its greedy token differs from the baseline, and prints the median request time plus the
selected chunk. Use the recommended `--prefill-mb` for that PC/workload. Permanent CI also
checks manual 1/16/64/128 and automatic chunks against one-shot exact K3, including the
`temperature=1` sampled-draft path.

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

Sampled speculative decoding is now probability-correct. A draft proposal `y~q` is
accepted with `min(1,p(y)/q(y))`; on the first rejection the correction is drawn from the
normalised residual `(p-q)+`, and after a fully accepted block the extra token is sampled
from exact K3's next distribution. This preserves the target K3 distribution while allowing
a cheap Q4/I8 draft at `temperature=1`.

To enable it for the localhost/Kimi Code bridge, add for example:

```bash
python local/k3_local.py serve \
  --model-dir ~/k3model --trunk ~/k3trunk-lossless --preset laptop --threads N \
  --draft-trunk ~/k3draft-q4 --draft-trunk-gb 32 --draft-topk 4 --spec 4
```

With these flags the **exact trunk and the proposal trunk are both resident worker
resources**: their packed mappings and independent KDA/MLA states stay alive across Kimi
Code tool turns. The shared expert cache stays warm too. The worker only reuses a prior
conversation state when the entire previous XTML token sequence is an exact prefix; a
branch resets both exact and draft conversation states together without reopening either
trunk.

The worker deliberately mirrors the one-shot decoder's full-block scheduling as well as
its p/q mathematics. A fixed seed therefore gives the same output in one-shot and
resident modes, including near `max_tokens` where the decoder falls back to serial exact
decode instead of consuming extra speculative RNG draws. Permanent tiny-checkpoint CI
gates first-turn and warm second-turn parity against the one-shot engine.

The draft can change acceptance rate and wall-clock speed, but exact K3 remains the
verification/target distribution. The worker reports proposal rounds, proposed/accepted
tokens, draft time and exact verification time, so sweep draft top-k/spec length on the
real machine rather than assuming one setting is universally fastest.

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
- **Draft quality is hardware/workload dependent:** Q4/I8 proposal inference is now
  resident and probability-correct, but a low acceptance rate can still make speculation
  slower than serial exact decode. Use the emitted draft metrics / sweep instead of
  treating one top-k/spec pair as a universal default.
- **Long context:** the exact expanded fp32 MLA cache is the current context-memory wall.
  A latent-cache kernel can reduce it dramatically, but it must be numerically gated
  before becoming a default.
- **Benchmark numbers are measured, not inherited:** using the same checkpoint and harness
  is a prerequisite for reproducing a score; it is not evidence that a local run already
  achieved that score. Run the actual benchmark suite before quoting a number.

Those gaps are deliberately visible because the goal is a credible local K3 runtime, not
an API-shaped demo that silently changes model behavior.
