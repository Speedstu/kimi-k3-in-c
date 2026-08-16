#!/usr/bin/env python3
from __future__ import annotations

import unittest

from local.k3_local import LocalK3, _is_loopback_host, parse_xtml


class FakeTokenizer:
    eos_id = 99

    def __init__(self):
        self.render_call = None

    def render(self, messages, tools, effort, tool_choice=None, response_format=None):
        self.render_call = {
            "messages": messages,
            "tools": tools,
            "effort": effort,
            "tool_choice": tool_choice,
            "response_format": response_format,
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

    def test_always_thinking_cannot_be_disabled(self):
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "always-thinking"):
            k3.complete(
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": {"type": "disabled"},
                }
            )

    def test_media_is_rejected_not_silently_dropped(self):
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "text/coding input only"):
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

    def test_loopback_guard(self):
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("::1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertFalse(_is_loopback_host("0.0.0.0"))
        self.assertFalse(_is_loopback_host("192.168.1.50"))


if __name__ == "__main__":
    unittest.main()
