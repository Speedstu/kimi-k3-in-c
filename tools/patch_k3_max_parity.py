#!/usr/bin/env python3
"""Stage the long-reasoning K3 Max parity fixes.

This transform intentionally changes no model arithmetic. It only removes an HTTP/worker
output ceiling that was lower than K3's published benchmark budgets, keeps sampled spec
width compatible with the one-shot decoder, and updates the benchmark-facing profile/tests.
"""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: anchor count {n}, expected 1\nANCHOR:\n{old}")
    p.write_text(s.replace(old, new, 1))


# The resident worker allocates request/sequence storage from its configured context. The
# legacy K3_MAX_GEN cap belongs to the one-shot CLI's compatibility ceiling, not to this
# context-sized worker.
replace_once(
    "src/cli/k3_worker.c",
    "        if (np <= 0 || gen <= 0 || np + gen > context || gen > K3_MAX_GEN) bad = 1;\n",
    "        if (np <= 0 || gen <= 0 || np + gen > context) bad = 1;\n",
)

# After greedy K3_SPEC_MAX is widened, sampled worker requests must retain the one-shot
# decoder's historical effective width-8 cap or a fixed seed would consume a different
# proposal/accept RNG sequence.
replace_once(
    "src/cli/k3_worker.c",
    """            const int request_tmax = np + gen + 1;
            const int can_full_spec = draft_dir &&
                T + spec_n + 1 < request_tmax &&
                base + spec_n + 1 <= w.kv_cap;
            const int want_drafts = can_full_spec ? spec_n : 0;
""",
    """            const int request_tmax = np + gen + 1;
            int request_spec_n = spec_n;
            if (temperature > 0.0 && request_spec_n > K3_SPEC_SAMPLE_MAX)
                request_spec_n = K3_SPEC_SAMPLE_MAX;
            const int can_full_spec = draft_dir &&
                T + request_spec_n + 1 < request_tmax &&
                base + request_spec_n + 1 <= w.kv_cap;
            const int want_drafts = can_full_spec ? request_spec_n : 0;
""",
)

# The bridge used to reject >4096 before it even knew the resident worker's actual
# configured capacity. Validate positive output here, render the real XTML prompt, then
# enforce prompt+output against the worker context. Keep the one-shot fallback at 4096.
replace_once(
    "local/k3_local.py",
    """        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be in [1,4096] for the current C backend")
""",
    """        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
""",
)
replace_once(
    "local/k3_local.py",
    """        prompt_ids = self.tokenizer.render(
            messages,
            request.get("tools"),
            effort,
            request.get("tool_choice"),
            request.get("response_format"),
        )
        return {
""",
    """        prompt_ids = self.tokenizer.render(
            messages,
            request.get("tools"),
            effort,
            request.get("tool_choice"),
            request.get("response_format"),
        )
        backend_context = getattr(self.backend, "context", None)
        if backend_context is not None:
            backend_context = int(backend_context)
            if len(prompt_ids) + max_tokens > backend_context:
                raise ValueError(
                    f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds "
                    f"resident worker context ({backend_context}); raise --worker-context"
                )
        elif max_tokens > 4096:
            raise ValueError(
                "non-resident one-shot compatibility mode supports at most 4096 output "
                "tokens; use the default resident worker for K3 Max benchmark budgets"
            )
        return {
""",
)

# The official K3 vendor verifier uses up to 98,304 output tokens for MMMU at thinking
# effort max. Kimi Code may finish much sooner; this is a ceiling, not a forced length.
replace_once(
    "local/kimi-code-benchmark.toml.example",
    """# Keep the model request aligned with the released preserved-thinking example and the
# current exact C generation buffer instead of deriving a huge output budget from 1M ctx.
max_output_size = 4096
""",
    """# K3's published verifier uses output budgets up to 98,304 tokens (MMMU, thinking
# max). The resident worker accepts this when --worker-context covers prompt + output.
max_output_size = 98304
""",
)

# Keep permanent CI aware that the interactive and benchmark profiles intentionally use
# different output ceilings.
replace_once(
    ".github/workflows/local-parity.yml",
    """          for path in ['local/kimi-code-local.toml.example', 'local/kimi-code-benchmark.toml.example']:
              with open(path, 'rb') as f:
                  cfg = tomllib.load(f)
              provider = cfg['providers']['local-k3']
              assert provider['base_url'] == 'http://127.0.0.1:8000/v1'
              assert cfg['thinking']['enabled'] is True
              assert cfg['thinking']['effort'] == 'max'
              assert cfg['thinking']['keep'] == 'all'
              assert cfg['models'][cfg['default_model']]['max_output_size'] == 4096
              print(path, 'PASS')
""",
    """          expected_outputs = {
              'local/kimi-code-local.toml.example': 4096,
              'local/kimi-code-benchmark.toml.example': 98304,
          }
          for path, expected_output in expected_outputs.items():
              with open(path, 'rb') as f:
                  cfg = tomllib.load(f)
              provider = cfg['providers']['local-k3']
              assert provider['base_url'] == 'http://127.0.0.1:8000/v1'
              assert cfg['thinking']['enabled'] is True
              assert cfg['thinking']['effort'] == 'max'
              assert cfg['thinking']['keep'] == 'all'
              assert cfg['models'][cfg['default_model']]['max_output_size'] == expected_output
              print(path, 'PASS')
""",
)

# Add bridge-level regression tests without importing the heavy tokenizer.
replace_once(
    "tests/python/test_local_bridge.py",
    """    def test_unsupported_penalty_is_rejected(self):
""",
    """    def test_resident_worker_accepts_official_98304_output_budget(self):
        k3 = self.make_k3()
        k3.backend.context = 131072
        prepared = k3._prepare(
            {
                "messages": [{"role": "user", "content": "benchmark"}],
                "thinking": {"type": "enabled", "effort": "max", "keep": "all"},
                "temperature": 1.0,
                "top_p": 0.95,
                "max_completion_tokens": 98304,
            }
        )
        self.assertEqual(prepared["max_tokens"], 98304)

    def test_resident_worker_rejects_budget_beyond_configured_context(self):
        k3 = self.make_k3()
        k3.backend.context = 65536
        with self.assertRaisesRegex(ValueError, "raise --worker-context"):
            k3._prepare(
                {
                    "messages": [{"role": "user", "content": "benchmark"}],
                    "max_completion_tokens": 98304,
                }
            )

    def test_nonresident_fallback_keeps_legacy_output_ceiling(self):
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "one-shot compatibility"):
            k3._prepare(
                {
                    "messages": [{"role": "user", "content": "benchmark"}],
                    "max_completion_tokens": 4097,
                }
            )

    def test_unsupported_penalty_is_rejected(self):
""",
)

# Documentation: distinguish ordinary interactive settings from the benchmark ceiling.
replace_once(
    "local/README.md",
    """`local/kimi-code-benchmark.toml.example` declares the official 1,048,576-token model
window and uses autonomous permission mode. Use it only inside the disposable environment
provided by the benchmark:
""",
    """`local/kimi-code-benchmark.toml.example` declares the official 1,048,576-token model
window, a 98,304-token maximum output budget for the longest published K3 verifier profile,
and autonomous permission mode. Use it only inside the disposable environment provided by
the benchmark. Start the resident server with a `--worker-context` large enough for the
rendered prompt plus that output ceiling; the bridge refuses undersized contexts rather
than silently truncating reasoning:
""",
)

print("applied K3 Max long-reasoning parity transform")
