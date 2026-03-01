"""Core routing types for multi-provider dispatch.

Defines the data structures used by the provider abstraction layer and
routing system. These types are imported by config, models, provider
implementations, the router, and the orchestrator.

Separated from ``models.py`` to avoid circular imports once
``ModelResponse`` gains a ``routing`` field that references
``RoutingDecision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Vendor(StrEnum):
    """Supported inference providers.

    Values are lowercase strings matching the provider keys used in
    ``config.py``'s ``providers`` dict and ``_PROVIDER_ENV_MAP``.
    Inherits from ``str`` so values serialize naturally to JSON.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    XAI = "xai"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class RoutedRequest:
    """A model request annotated with routing info.

    Created by the ProviderRouter when dispatching requests.
    Groups related request metadata so providers receive everything
    they need in a single object.

    Attributes:
        vendor: Target provider for this request.
        model_id: Provider-specific model identifier.
        model_alias: Human-readable short name (e.g. "claude").
        round_number: Debate round (0=initial, 1+=reflection, -1=synthesis).
        messages: Chat messages in OpenAI-compatible format.
        context: Optional per-panelist pre-prompt content injected before
            the user query. Used for RAG augmentation and experiment payloads.
    """

    vendor: Vendor
    model_id: str
    model_alias: str
    round_number: int
    messages: list[dict[str, Any]]
    context: str | None = None


@dataclass
class RoutingDecision:
    """Record of how a request was routed.

    Attached to every ModelResponse for transcript provenance.
    Serializable to dict for JSON transcript storage.

    Attributes:
        vendor: Which provider handled the request.
        mode: Routing mode that was in effect ("auto", "direct", "openrouter").
        via_openrouter: Whether the request went through OpenRouter
            (True even in "auto" mode if no direct key was available).
    """

    vendor: Vendor
    mode: str
    via_openrouter: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with vendor as string, mode, and via_openrouter flag.
        """
        return {
            "vendor": self.vendor.value,
            "mode": self.mode,
            "via_openrouter": self.via_openrouter,
        }
