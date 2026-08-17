#!/usr/bin/env python3
from __future__ import annotations

import unittest

from local.k3_local import (
    BackendConfig, LocalK3, ResidentCBackend, _auto_worker_budgets,
    _is_loopback_host, parse_xtml,
)


class FakeTokenizer:
    eos_id = 99

    def __init__(self):
        self.render_call = None

    def render(
        self, messages, tools, effort, tool_choice=None, response_format=None,
        thinking_enabled=True,
    ):
        self.render_call = {
            "messages": messages,
            "tools": tools,
            "effort": effort,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "thinking_enabled": thinking_enabled,
        }
        return [10, 11, 12]

    def decode(self, ids):
        assert ids == [41, 99]
        return (
            "reasoning trace"
            "<|close|>think<|sep|>"
            "<|open|>response<|sep|>done"
            "<|close|>response<|sep|>"
        )


class FakeBackend:
    def __init__(self):
        self.call = None

    def generate(self, prompt_ids, **kwargs):
        self.call = (prompt_ids, kwargs)
        return [41, 99], {
            "generated_ids": [41, 99],
            "full_ids": [10, 11, 12, 41, 99],
            "seconds_per_token": 0.01,
        }


class LocalBridgeTests(unittest.TestCase):
    def make_k3(self):
        k3 = LocalK3.__new__(LocalK3)
        k3.tokenizer = FakeTokenizer()
        k3.backend = FakeBackend()
        return k3

    def test_kimi_thinking_object_takes_precedence(self):
        k3 = self.make_k3()
        result = k3.complete(
            {
                "model": "kimi-k3-local",
                "messages": [{"role": "user", "content": "fix it"}],
                "thinking": {"type": "enabled", "effort": "max", "keep": "all"},
                "reasoning_effort": "low",
                "temperature": 1.0,
                "top_p": 1.0,
                "seed": 17,
                "max_completion_tokens": 32,
            }
        )
        self.assertEqual(k3.tokenizer.render_call["effort"], "max")
        _, kwargs = k3.backend.call
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(result["choices"][0]["message"]["reasoning_content"], "reasoning trace")
        self.assertEqual(result["choices"][0]["message"]["content"], "done")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_non_thinking_uses_official_template_switch_and_temperature(self):
        k3 = self.make_k3()
        result = k3.complete(
            {
                "messages": [{"role": "user", "content": "x"}],
                "thinking": {"type": "disabled"},
                "temperature": 0.6,
                "top_p": 0.95,
            }
        )
        self.assertFalse(k3.tokenizer.render_call["thinking_enabled"])
        self.assertIsNotNone(result["choices"][0]["message"])

    def test_vendor_sampling_constraints_fail_closed(self):
        k3 = self.make_k3()
        base = {"messages": [{"role": "user", "content": "x"}]}
        for bad in (-0.1, 1.1, 2.0):
            with self.assertRaisesRegex(ValueError, "temperature"):
                k3._prepare({**base, "temperature": bad})
        with self.assertRaisesRegex(ValueError, "top_p"):
            k3._prepare({**base, "top_p": 0.8})
        with self.assertRaisesRegex(ValueError, "temperature"):
            k3._prepare({**base, "thinking": {"type": "disabled"}, "temperature": 1.0})

    def test_media_requires_resident_worker_not_silent_drop(self):
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "requires the default resident C worker"):
            k3.complete(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image_url", "image_url": {"url": "file:///x.png"}},
                            ],
                        }
                    ]
                }
            )

    def test_resident_worker_accepts_official_98304_output_budget(self):
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
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "presence_penalty"):
            k3.complete(
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "presence_penalty": 0.5,
                }
            )

    def test_xtml_tool_call_parsing(self):
        text = (
            "inspect first"
            "<|close|>think<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="Bash" index="1"<|sep|>'
            '<|open|>argument key="command" type="string"<|sep|>git status'
            "<|close|>argument<|sep|>"
            '<|open|>argument key="timeout" type="number"<|sep|>15'
            "<|close|>argument<|sep|>"
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
        )
        message = parse_xtml(text)
        self.assertEqual(message["reasoning_content"], "inspect first")
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "Bash")
        self.assertEqual(
            call["function"]["arguments"], '{"command":"git status","timeout":15}'
        )

    def test_safe_stream_delta_holds_unstable_unicode(self):
        delta, previous = LocalK3._safe_delta("abc", "abc\ufffd")
        self.assertEqual(delta, "")
        self.assertEqual(previous, "abc")
        delta, previous = LocalK3._safe_delta("abc", "abcé")
        self.assertEqual(delta, "é")
        self.assertEqual(previous, "abcé")

    def test_auto_worker_budget_uses_32gb_for_exact_trunk_not_expert_lru(self):
        trunk, cache = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1024)
        self.assertGreater(trunk, 18.0)
        self.assertLess(trunk, 24.0)
        self.assertEqual(cache, 0.5)

    def test_auto_worker_budget_reserves_hot_kv_for_large_virtual_context(self):
        small_ctx = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1024)
        huge_ctx = _auto_worker_budgets(32.0, prefill_mb=256.0, worker_context=1048576)
        self.assertLess(huge_ctx[0], small_ctx[0])
        self.assertEqual(huge_ctx[1], 0.5)
        # Virtual 1M context must not reserve 1M physical KV rows at startup.
        self.assertGreater(huge_ctx[0], 14.0)

    def test_auto_worker_budget_fills_trunk_before_expert_cache(self):
        trunk, cache = _auto_worker_budgets(192.0, prefill_mb=256.0, worker_context=1024)
        self.assertEqual(trunk, 111.0)
        self.assertGreater(cache, 0.5)

    def test_auto_worker_budget_fails_closed_when_host_is_too_busy(self):
        with self.assertRaisesRegex(RuntimeError, "Close memory-heavy apps"):
            _auto_worker_budgets(8.0, prefill_mb=256.0, worker_context=1024)

    def test_default_backend_config_is_machine_aware(self):
        cfg = BackendConfig(
            model_dir=__import__('pathlib').Path('/model'),
            trunk_dir=__import__('pathlib').Path('/trunk'),
            binary=__import__('pathlib').Path('/bin/k3'),
        )
        self.assertEqual(cfg.preset, "auto")

    def test_loopback_guard(self):
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("::1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertFalse(_is_loopback_host("0.0.0.0"))
        self.assertFalse(_is_loopback_host("192.168.1.50"))


if __name__ == "__main__":
    unittest.main()
