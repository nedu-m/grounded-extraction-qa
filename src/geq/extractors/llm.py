"""The real LLM extractor: the thing the harness exists to police.

Two design choices matter here, and both come from the same observation. Models
are unreliable at counting characters and reliable at copying them, so:

1. The model is asked for a **verbatim quote**, never for offsets. The quote is
   mapped back to a span with :func:`locate_quote`. Offsets produced by a model
   are a fabricated citation waiting to happen.
2. Its ``value`` may only be what appears inside its own quote. A value that is
   not in the quote fails the grounding check, which is the point.

The default provider is DeepSeek, which is OpenAI-compatible. Anything else
speaking the same protocol works by setting ``GEQ_LLM_BASE_URL`` and
``GEQ_LLM_MODEL``, so this file is not a DeepSeek integration, it is a
chat-completions integration that defaults to DeepSeek.

Nothing here runs in the offline test suite: the parts worth testing are
transported out into :func:`quote_must_contain_value` and
:func:`geq.extractors.base.locate_quote`, which have no network in them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..models import Claim, Extraction, FieldKind, FieldSpec, GoldenCase
from .base import locate_quote

__all__ = ["DEFAULT_MODEL", "LLMConfig", "LLMExtractor", "quote_must_contain_value"]

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV_VARS = ("DEEPSEEK_API_KEY", "GEQ_LLM_API_KEY", "OPENAI_API_KEY")
_TIMEOUT_SECONDS = 60

SYSTEM_PROMPT = """You extract single fields from financial documents.

Rules, all of them mandatory:
- Return the value exactly as it appears in the document, character for character.
- Also return "quote": the shortest verbatim run of text from the document that
  contains that value. The quote must contain the value as a literal substring.
- If the document does not state the field, return null for that field.
- Never infer, compute, round, or reformat a value, and never carry a scale
  across. If the document says "13,504.5", the value is "13,504.5"."""


def quote_must_contain_value(quote: str, value: str) -> bool:
    """Contract check: a quote that does not contain its value is not evidence.

    Runs in-process on the model's own output before it reaches the scorer, so a
    non-compliant response is reported as uncited rather than quietly accepted.
    """
    if not quote or not value:
        return False
    return value in quote


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Where the chat-completions endpoint lives and how to reach it."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    api_key_env: str | None = None
    """Which environment variable the key was read from, for diagnostics."""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LLMConfig:
        env = os.environ if environ is None else environ
        source = next((name for name in API_KEY_ENV_VARS if env.get(name)), None)
        return cls(
            model=env.get("GEQ_LLM_MODEL") or DEFAULT_MODEL,
            base_url=(env.get("GEQ_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            api_key=env[source] if source else None,
            api_key_env=source,
        )

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def public_view(self) -> dict[str, Any]:
        """Everything except the key, for health endpoints and reports."""
        return {
            "configured": self.configured,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
        }


class LLMExtractor:
    """Extractor backed by an OpenAI-compatible chat completion (DeepSeek by default)."""

    name = "llm"

    def __init__(
        self,
        specs: Mapping[str, FieldSpec] | None = None,
        *,
        config: LLMConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        base = config or LLMConfig.from_env()
        self._specs: Mapping[str, FieldSpec] = specs or {}
        self._config = LLMConfig(
            model=model or base.model,
            base_url=(base_url or base.base_url).rstrip("/"),
            api_key=api_key if api_key is not None else base.api_key,
            api_key_env=base.api_key_env,
        )

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def config(self) -> LLMConfig:
        return self._config

    # -- transport ------------------------------------------------------

    def _complete(self, document: str, spec: FieldSpec) -> dict[str, Any]:
        if not self._config.configured:
            raise RuntimeError(
                "no LLM API key is set. Export DEEPSEEK_API_KEY, or run the "
                "deterministic revisions with --extractor rules:v3."
            )
        user = (
            f"Field to extract: {spec.name}\n"
            f"Expected kind: {spec.kind.value}\n"
            f"Canonical label in the document: {spec.label!r}\n\n"
            f"DOCUMENT\n--------\n{document}\n--------\n\n"
            'Reply with JSON only: {"value": <string or null>, "quote": <string or null>}'
        )
        payload = json.dumps(
            {
                "model": self._config.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self._config.completions_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - needs network
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):  # pragma: no cover - defensive
            raise RuntimeError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed

    # -- entry point ----------------------------------------------------

    def extract(self, case: GoldenCase) -> Extraction:  # pragma: no cover - needs network
        claims: dict[str, Claim] = {}
        notes: list[str] = []

        for name, spec in self._specs.items():
            if spec.kind is FieldKind.TEXT and name == "entity_name":
                # Cheap and deterministic; a model adds cost and no accuracy here.
                continue
            try:
                answer = self._complete(case.text, spec)
            except RuntimeError as exc:
                notes.append(f"{name}: {exc}")
                break

            value = answer.get("value")
            quote = answer.get("quote")
            if not value or not quote:
                continue
            if not quote_must_contain_value(quote, value):
                notes.append(
                    f"{name}: model returned a value not present in its own quote; "
                    f"treated as uncited"
                )
                claims[name] = Claim(field=name, raw_value=str(value), span=None)
                continue

            located = locate_quote(case.text, quote)
            if located is None:
                notes.append(f"{name}: quoted text not found in the document; treated as uncited")
                claims[name] = Claim(field=name, raw_value=str(value), span=None)
                continue

            span, occurrences = located
            if occurrences > 1:
                notes.append(f"{name}: quote occurs {occurrences} times; first occurrence used")
            claims[name] = Claim(field=name, raw_value=str(value), span=span)

        return Extraction(
            doc_id=case.doc_id, extractor=self.name, claims=claims, notes=tuple(notes)
        )
