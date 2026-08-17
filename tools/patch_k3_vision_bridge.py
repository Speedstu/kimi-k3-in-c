#!/usr/bin/env python3
"""Stage the official K3 image processor/vision path into the localhost bridge."""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: anchor count {n}, expected 1\n--- anchor ---\n{old}")
    p.write_text(s.replace(old, new, 1))


# ---------------------------------------------------------------- media classification
replace_once(
    "local/k3_local.py",
    """def _contains_media(messages: list[dict[str, Any]]) -> bool:
    \"\"\"Return true for API message parts the current C text backend cannot encode.\"\"\"

    media_types = {
        \"image\",
        \"image_url\",
        \"input_image\",
        \"video\",
        \"video_url\",
        \"input_video\",
        \"audio\",
        \"input_audio\",
    }
    for message in messages:
        content = message.get(\"content\")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get(\"type\") in media_types:
                return True
    return False
""",
    """def _media_types(messages: list[dict[str, Any]]) -> set[str]:
    \"\"\"Return media part types without ever silently discarding unsupported input.\"\"\"

    media_types = {
        \"image\",
        \"image_url\",
        \"input_image\",
        \"video\",
        \"video_url\",
        \"input_video\",
        \"audio\",
        \"input_audio\",
    }
    found: set[str] = set()
    for message in messages:
        content = message.get(\"content\")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get(\"type\") in media_types:
                found.add(str(part.get(\"type\")))
    return found


def _contains_media(messages: list[dict[str, Any]]) -> bool:
    return bool(_media_types(messages))
""",
)

# ---------------------------------------------------------------- config
replace_once(
    "local/k3_local.py",
    """    prefill_mb: float = 256.0
    prefill_chunk: int | None = None
""",
    """    prefill_mb: float = 256.0
    prefill_chunk: int | None = None
    vision_device: str = \"auto\"
    vision_attention: str = \"auto\"
""",
)

# ---------------------------------------------------------------- resident worker protocol
replace_once(
    "local/k3_local.py",
    """        seed: int,
        stop_id: int,
        on_token=None,
    ) -> tuple[list[int], dict[str, Any]]:
        if not prompt_ids:
            raise ValueError(\"resident worker needs at least one prompt token\")
        if len(prompt_ids) + max_tokens > self.context:
            raise ValueError(
                f\"resident worker context is {self.context} positions but this request needs \"
                f\"{len(prompt_ids) + max_tokens}; restart the server with --worker-context \"
                \"set high enough for the benchmark/session\"
            )
""",
    """        seed: int,
        stop_id: int,
        on_token=None,
        media_path: Path | None = None,
        media_placeholder: int | None = None,
        prompt_positions: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        if not prompt_ids:
            raise ValueError(\"resident worker needs at least one prompt token\")
        positions = len(prompt_ids) if prompt_positions is None else int(prompt_positions)
        if positions < 1 or positions + max_tokens > self.context:
            raise ValueError(
                f\"resident worker context is {self.context} positions but this request needs \"
                f\"{positions + max_tokens}; restart the server with --worker-context \"
                \"set high enough for the benchmark/session\"
            )
        if media_path is not None:
            if media_placeholder is None:
                raise ValueError(\"media request is missing K3 placeholder id\")
            media_path = Path(media_path).resolve()
            if not media_path.is_file():
                raise ValueError(f\"media feature sidecar does not exist: {media_path}\")
            if any(ch.isspace() for ch in str(media_path)):
                raise ValueError(\"media feature sidecar path cannot contain whitespace\")
""",
)

replace_once(
    "local/k3_local.py",
    """            header = (
                f\"REQ {rid} {len(prompt_ids)} {max_tokens} {temperature:.17g} \"
                f\"{top_p:.17g} {int(seed)} {int(stop_id)}\\n\"
            )
""",
    """            if media_path is None:
                header = (
                    f\"REQ {rid} {len(prompt_ids)} {max_tokens} {temperature:.17g} \"
                    f\"{top_p:.17g} {int(seed)} {int(stop_id)}\\n\"
                )
            else:
                header = (
                    f\"REQMM {rid} {len(prompt_ids)} {max_tokens} {temperature:.17g} \"
                    f\"{top_p:.17g} {int(seed)} {int(stop_id)} {int(media_placeholder)} \"
                    f\"{media_path}\\n\"
                )
""",
)

replace_once(
    "local/k3_local.py",
    """                            \"state_cache_suffix_tokens\": len(prompt_ids) - reused,
""",
    """                            \"state_cache_suffix_tokens\": positions - reused,
                            \"prompt_positions\": positions,
                            \"multimodal\": media_path is not None,
""",
)

# Public resident generate methods accept the optional mixed-input metadata.
replace_once(
    "local/k3_local.py",
    """        seed: int,
        stop_id: int,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
        )

    def generate_stream(
""",
    """        seed: int,
        stop_id: int,
        media_path: Path | None = None,
        media_placeholder: int | None = None,
        prompt_positions: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
            media_path=media_path,
            media_placeholder=media_placeholder,
            prompt_positions=prompt_positions,
        )

    def generate_stream(
""",
)

replace_once(
    "local/k3_local.py",
    """        seed: int,
        stop_id: int,
        on_token,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
            on_token=on_token,
        )


class LocalK3:
""",
    """        seed: int,
        stop_id: int,
        on_token,
        media_path: Path | None = None,
        media_placeholder: int | None = None,
        prompt_positions: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        return self._request(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_id=stop_id,
            on_token=on_token,
            media_path=media_path,
            media_placeholder=media_placeholder,
            prompt_positions=prompt_positions,
        )


class LocalK3:
""",
)

# ---------------------------------------------------------------- LocalK3 lazy official vision front-end
replace_once(
    "local/k3_local.py",
    """    ):
        self.tokenizer = LocalTokenizer(cfg.model_dir)
        if cfg.resident_worker:
            self.backend = ResidentCBackend(cfg)
        else:
            self.backend = CBackend(cfg, state_root, max_state_entries)

    def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
""",
    """    ):
        self.cfg = cfg
        self.tokenizer = LocalTokenizer(cfg.model_dir)
        self._vision = None
        self._vision_lock = threading.Lock()
        if cfg.resident_worker:
            self.backend = ResidentCBackend(cfg)
        else:
            self.backend = CBackend(cfg, state_root, max_state_entries)

    def _vision_frontend(self):
        if self._vision is not None:
            return self._vision
        with self._vision_lock:
            if self._vision is None:
                try:
                    from .k3_vision import K3VisionEncoder
                except ImportError:  # direct `python local/k3_local.py`
                    from k3_vision import K3VisionEncoder
                self._vision = K3VisionEncoder(
                    self.cfg.model_dir,
                    device=self.cfg.vision_device,
                    attention=self.cfg.vision_attention,
                )
        return self._vision

    def _media_sidecar(self, prepared: dict[str, Any]):
        features = prepared.get(\"media_features\")
        if not features:
            return None
        return self._vision_frontend().sidecar(features)

    def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
""",
)

# Replace old blanket rejection with a capability/fail-closed split.
replace_once(
    "local/k3_local.py",
    """        if _contains_media(messages):
            raise ValueError(
                \"this local C backend currently supports K3 text/coding input only; \"
                \"image/video input is rejected rather than silently discarded\"
            )

        thinking = request.get(\"thinking\")
""",
    """        media_types = _media_types(messages)
        unsupported_media = media_types - {\"image\", \"image_url\"}
        if unsupported_media:
            raise ValueError(
                \"the released K3 processor path currently supports image/image_url only; \"
                f\"unsupported media types: {sorted(unsupported_media)}\"
            )
        if media_types and not isinstance(self.backend, ResidentCBackend):
            raise ValueError(
                \"K3 image input requires the default resident C worker; one-shot mode \"
                \"refuses media rather than using a different model path\"
            )

        thinking = request.get(\"thinking\")
""",
)

# Render text exactly as before, or let the official K3 processor render multimodal XTML
# and produce the projected image features.
replace_once(
    "local/k3_local.py",
    """        prompt_ids = self.tokenizer.render(
            messages,
            request.get(\"tools\"),
            effort,
            request.get(\"tool_choice\"),
            request.get(\"response_format\"),
        )
        backend_context = getattr(self.backend, \"context\", None)
        if backend_context is not None:
            backend_context = int(backend_context)
            if len(prompt_ids) + max_tokens > backend_context:
                raise ValueError(
                    f\"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds \"
                    f\"resident worker context ({backend_context}); raise --worker-context\"
                )
""",
    """        media_features = None
        media_placeholder = None
        if media_types:
            vision = self._vision_frontend().prepare(
                messages,
                tools=request.get(\"tools\"),
                reasoning_effort=effort,
                tool_choice=request.get(\"tool_choice\"),
                response_format=request.get(\"response_format\"),
            )
            prompt_ids = vision.input_ids
            prompt_positions = vision.prompt_positions
            media_features = vision.features
            media_placeholder = vision.placeholder_id
        else:
            prompt_ids = self.tokenizer.render(
                messages,
                request.get(\"tools\"),
                effort,
                request.get(\"tool_choice\"),
                request.get(\"response_format\"),
            )
            prompt_positions = len(prompt_ids)
        backend_context = getattr(self.backend, \"context\", None)
        if backend_context is not None:
            backend_context = int(backend_context)
            if prompt_positions + max_tokens > backend_context:
                raise ValueError(
                    f\"prompt ({prompt_positions} positions) + max_tokens ({max_tokens}) exceeds \"
                    f\"resident worker context ({backend_context}); raise --worker-context\"
                )
""",
)

replace_once(
    "local/k3_local.py",
    """            \"stop_id\": self.tokenizer.eos_id,
            \"model\": request.get(\"model\", \"kimi-k3-local\"),
        }
""",
    """            \"stop_id\": self.tokenizer.eos_id,
            \"model\": request.get(\"model\", \"kimi-k3-local\"),
            \"prompt_positions\": prompt_positions,
            \"media_features\": media_features,
            \"media_placeholder\": media_placeholder,
        }
""",
)

# Usage reports language-model positions, which is what K3's context limit measures.
replace_once(
    "local/k3_local.py",
    """        prompt_ids = prepared[\"prompt_ids\"]
        return {
""",
    """        prompt_ids = prepared[\"prompt_ids\"]
        prompt_positions = int(prepared.get(\"prompt_positions\", len(prompt_ids)))
        return {
""",
)
replace_once(
    "local/k3_local.py",
    """                \"prompt_tokens\": len(prompt_ids),
                \"completion_tokens\": len(generated),
                \"total_tokens\": len(prompt_ids) + len(generated),
""",
    """                \"prompt_tokens\": prompt_positions,
                \"completion_tokens\": len(generated),
                \"total_tokens\": prompt_positions + len(generated),
""",
)

# Complete and stream keep a sidecar alive only for the duration of the resident request.
replace_once(
    "local/k3_local.py",
    """    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare(request)
        generated, stats = self.backend.generate(
            prepared[\"prompt_ids\"],
            max_tokens=prepared[\"max_tokens\"],
            temperature=prepared[\"temperature\"],
            top_p=prepared[\"top_p\"],
            seed=prepared[\"seed\"],
            stop_id=prepared[\"stop_id\"],
        )
        return self._result(prepared, generated, stats)
""",
    """    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare(request)
        sidecar = self._media_sidecar(prepared)
        if sidecar is None:
            generated, stats = self.backend.generate(
                prepared[\"prompt_ids\"],
                max_tokens=prepared[\"max_tokens\"],
                temperature=prepared[\"temperature\"],
                top_p=prepared[\"top_p\"],
                seed=prepared[\"seed\"],
                stop_id=prepared[\"stop_id\"],
            )
        else:
            with sidecar as media_path:
                generated, stats = self.backend.generate(
                    prepared[\"prompt_ids\"],
                    max_tokens=prepared[\"max_tokens\"],
                    temperature=prepared[\"temperature\"],
                    top_p=prepared[\"top_p\"],
                    seed=prepared[\"seed\"],
                    stop_id=prepared[\"stop_id\"],
                    media_path=media_path,
                    media_placeholder=prepared[\"media_placeholder\"],
                    prompt_positions=prepared[\"prompt_positions\"],
                )
        return self._result(prepared, generated, stats)
""",
)

replace_once(
    "local/k3_local.py",
    """        streamed, stats = self.backend.generate_stream(
            prepared[\"prompt_ids\"],
            max_tokens=prepared[\"max_tokens\"],
            temperature=prepared[\"temperature\"],
            top_p=prepared[\"top_p\"],
            seed=prepared[\"seed\"],
            stop_id=prepared[\"stop_id\"],
            on_token=on_token,
        )
""",
    """        sidecar = self._media_sidecar(prepared)
        if sidecar is None:
            streamed, stats = self.backend.generate_stream(
                prepared[\"prompt_ids\"],
                max_tokens=prepared[\"max_tokens\"],
                temperature=prepared[\"temperature\"],
                top_p=prepared[\"top_p\"],
                seed=prepared[\"seed\"],
                stop_id=prepared[\"stop_id\"],
                on_token=on_token,
            )
        else:
            with sidecar as media_path:
                streamed, stats = self.backend.generate_stream(
                    prepared[\"prompt_ids\"],
                    max_tokens=prepared[\"max_tokens\"],
                    temperature=prepared[\"temperature\"],
                    top_p=prepared[\"top_p\"],
                    seed=prepared[\"seed\"],
                    stop_id=prepared[\"stop_id\"],
                    on_token=on_token,
                    media_path=media_path,
                    media_placeholder=prepared[\"media_placeholder\"],
                    prompt_positions=prepared[\"prompt_positions\"],
                )
""",
)

# ---------------------------------------------------------------- CLI/config
replace_once(
    "local/k3_local.py",
    """        prefill_mb=args.prefill_mb,
        prefill_chunk=args.prefill_chunk,
    )
""",
    """        prefill_mb=args.prefill_mb,
        prefill_chunk=args.prefill_chunk,
        vision_device=args.vision_device,
        vision_attention=args.vision_attention,
    )
""",
)

replace_once(
    "local/k3_local.py",
    """    sp.add_argument(\"--preset\", default=\"laptop\")
""",
    """    sp.add_argument(
        \"--vision-device\",
        default=\"auto\",
        help=\"official MoonViT/projector device: auto, cpu, cuda, cuda:0, ...\",
    )
    sp.add_argument(
        \"--vision-attention\",
        choices=[\"auto\", \"eager\", \"flash_attention_2\"],
        default=\"auto\",
        help=\"official K3 vision attention implementation (default auto)\",
    )
    sp.add_argument(\"--preset\", default=\"laptop\")
""",
)

replace_once(
    "local/k3_local.py",
    """    print(\"default parity profile: reasoning=max, temperature=1.0, top-p=1.0\")
""",
    """    print(\"default parity profile: reasoning=max, temperature=1.0, top-p=1.0\")
    print(
        f\"image input: official K3 MoonViT/projector lazy-loaded on {cfg.vision_device} \"
        f\"(attention={cfg.vision_attention}); video/audio remain fail-closed\"
    )
""",
)

print("applied official K3 vision bridge transform")
