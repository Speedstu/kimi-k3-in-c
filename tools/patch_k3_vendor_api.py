#!/usr/bin/env python3
"""Stage API-compatibility fixes required by Moonshot's pinned Vendor Verifier."""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: anchor count {n}, expected 1\n--- anchor ---\n{old}")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")


replace_once(
    "local/k3_local.py",
    """        reasoning_effort: str,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "thinking": True,
            # The local K3 encoder names this template kwarg thinking_effort.  The
            # compatible HTTP field remains reasoning_effort.
            "thinking_effort": reasoning_effort,
        }
""",
    """        reasoning_effort: str,
        tool_choice: Any = None,
        response_format: Any = None,
        thinking_enabled: bool = True,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "thinking": thinking_enabled,
        }
        if thinking_enabled:
            # The local K3 encoder names this template kwarg thinking_effort.  The
            # compatible HTTP field remains reasoning_effort.
            kwargs["thinking_effort"] = reasoning_effort
""",
)

replace_once(
    "local/k3_local.py",
    """        thinking = request.get("thinking")
        if isinstance(thinking, dict):
            if thinking.get("type") == "disabled":
                raise ValueError(
                    "K3 is an always-thinking model; local parity mode cannot disable thinking"
                )
            effort = thinking.get("effort", request.get("reasoning_effort", "max"))
        else:
            effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("thinking effort must be low, high, or max")

        temperature = float(request.get("temperature", 1.0))
        top_p = float(request.get("top_p", 1.0))
""",
    """        thinking = request.get("thinking")
        thinking_enabled = True
        if isinstance(thinking, dict):
            thinking_type = thinking.get("type", "enabled")
            if thinking_type not in {"enabled", "disabled"}:
                raise ValueError("thinking.type must be enabled or disabled")
            thinking_enabled = thinking_type != "disabled"
            # Moonshot's current K3 contract gives thinking.effort precedence when both
            # extension fields are present; reasoning_effort applies when effort is absent.
            effort = thinking.get("effort", request.get("reasoning_effort", "max"))
        else:
            effort = request.get("reasoning_effort", "max")
        if effort not in {"low", "high", "max"}:
            raise ValueError("thinking effort must be low, high, or max")

        default_temperature = 1.0 if thinking_enabled else 0.6
        temperature = float(request.get("temperature", default_temperature))
        top_p = float(request.get("top_p", 1.0))
""",
)

replace_once(
    "local/k3_local.py",
    """        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0,1]")
""",
    """        if thinking_enabled:
            if not 0.0 <= temperature <= 1.0:
                raise ValueError("thinking temperature must be in [0,1]")
        elif temperature != 0.6:
            raise ValueError("non-thinking K3 temperature is fixed at 0.6")
        # The official verifier requires 0.95 and the K3 agentic recipe uses 1.0.
        # Reject unrelated nucleus values rather than silently accepting a non-reference
        # sampling policy.
        if top_p not in {0.95, 1.0}:
            raise ValueError("K3 top_p must be 0.95 or 1.0")
""",
)

replace_once(
    "local/k3_local.py",
    """                response_format=request.get("response_format"),
            )
""",
    """                response_format=request.get("response_format"),
                thinking_enabled=thinking_enabled,
            )
""",
)

replace_once(
    "local/k3_local.py",
    """                request.get("response_format"),
            )
""",
    """                request.get("response_format"),
                thinking_enabled=thinking_enabled,
            )
""",
)

replace_once(
    "local/k3_local.py",
    """            "media_placeholder": media_placeholder,
        }
""",
    """            "media_placeholder": media_placeholder,
            "thinking_enabled": thinking_enabled,
        }
""",
)

# Vision processor uses the same official chat-template switch as text.
replace_once(
    "local/k3_vision.py",
    """        reasoning_effort: str,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> VisionPrepared:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "thinking": True,
            "thinking_effort": reasoning_effort,
        }
""",
    """        reasoning_effort: str,
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
""",
)

# Existing unit-test fake tokenizer must accept the new template switch.
replace_once(
    "tests/python/test_local_bridge.py",
    """    def render(self, messages, tools, effort, tool_choice=None, response_format=None):
""",
    """    def render(
        self, messages, tools, effort, tool_choice=None, response_format=None,
        thinking_enabled=True,
    ):
""",
)
replace_once(
    "tests/python/test_local_bridge.py",
    """            "response_format": response_format,
        }
""",
    """            "response_format": response_format,
            "thinking_enabled": thinking_enabled,
        }
""",
)

p = Path("tests/python/test_local_bridge.py")
s = p.read_text(encoding="utf-8")
anchor = """    def test_always_thinking_cannot_be_disabled(self):
        k3 = self.make_k3()
        with self.assertRaisesRegex(ValueError, "always-thinking"):
            k3.complete(
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "thinking": {"type": "disabled"},
                }
            )

"""
replacement = """    def test_non_thinking_uses_official_template_switch_and_temperature(self):
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

"""
if s.count(anchor) != 1:
    raise SystemExit(f"tests/python/test_local_bridge.py: disabled-test anchor={s.count(anchor)}")
p.write_text(s.replace(anchor, replacement, 1), encoding="utf-8")

print("applied Moonshot K3 Vendor Verifier API compatibility transform")
