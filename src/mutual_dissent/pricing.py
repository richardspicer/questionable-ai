"""Model pricing — fetch and cache per-token pricing from OpenRouter.

Fetches pricing data from the OpenRouter ``/api/v1/models`` endpoint
(public, no auth required) and caches it for the lifetime of the
``PricingCache`` instance. Computes per-response USD cost from
input/output token counts and pricing data.

Typical usage::

    from mutual_dissent.pricing import PricingCache, compute_response_cost

    cache = PricingCache(alias_map=config._model_aliases_v2)
    await cache.prefetch()
    pricing = await cache.get_pricing("anthropic/claude-sonnet-4.5")
    cost = compute_response_cost(response, pricing)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import httpx

from mutual_dissent.models import ModelResponse

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
FETCH_TIMEOUT = 15.0  # seconds — generous but bounded


@dataclass
class ModelPricing:
    """Per-token pricing and context metadata for a model.

    Attributes:
        prompt_price: USD per token for input/prompt tokens.
        completion_price: USD per token for output/completion tokens.
        context_length: Maximum context window in tokens, or None if unknown.
    """

    prompt_price: float
    completion_price: float
    context_length: int | None = None


class PricingCache:
    """Session-scoped cache for model pricing from OpenRouter.

    Fetches pricing from ``/api/v1/models`` on first access (or via
    ``prefetch()``), caches for the lifetime of the object. No API key
    required — the models endpoint is public.

    Supports both OpenRouter model IDs (e.g. ``anthropic/claude-sonnet-4.5``)
    and vendor-native model IDs (e.g. ``claude-sonnet-4-5-20250929``) by
    mapping through the provided alias configuration.

    Args:
        alias_map: Model alias configuration mapping alias names to dicts
            with ``"openrouter"`` and optionally ``"direct"`` keys.
            Used to translate vendor-native model IDs to OpenRouter IDs
            for pricing lookup.
    """

    def __init__(self, alias_map: dict[str, dict[str, str]] | None = None) -> None:
        self._cache: dict[str, ModelPricing] | None = None
        self._alias_map = alias_map or {}
        # Build reverse map: direct model ID → OpenRouter model ID.
        self._direct_to_openrouter: dict[str, str] = {}
        for ids in self._alias_map.values():
            direct_id = ids.get("direct")
            openrouter_id = ids.get("openrouter")
            if direct_id and openrouter_id:
                self._direct_to_openrouter[direct_id] = openrouter_id

    async def prefetch(self) -> None:
        """Eagerly fetch and cache all model pricing.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._cache is None:
            await self._fetch_all()

    async def get_pricing(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a model by model ID.

        Tries exact match against OpenRouter model IDs first, then
        attempts to resolve vendor-native IDs via the alias map.

        Args:
            model_id: Model ID — either OpenRouter format
                (e.g. ``"anthropic/claude-sonnet-4.5"``) or vendor-native
                (e.g. ``"claude-sonnet-4-5-20250929"``).

        Returns:
            ModelPricing if found, None if model not in catalog or
            pricing fetch failed.
        """
        if self._cache is None:
            await self._fetch_all()
        if not self._cache:
            return None

        # Try exact match (OpenRouter ID).
        pricing = self._cache.get(model_id)
        if pricing is not None:
            return pricing

        # Try mapping vendor-native ID → OpenRouter ID.
        openrouter_id = self._direct_to_openrouter.get(model_id)
        if openrouter_id:
            return self._cache.get(openrouter_id)

        return None

    async def get_context_length(self, model_id: str) -> int | None:
        """Get maximum context length for a model.

        Args:
            model_id: Model ID — OpenRouter or vendor-native format.

        Returns:
            Context length in tokens, or None if unknown.
        """
        pricing = await self.get_pricing(model_id)
        if pricing is None:
            return None
        return pricing.context_length

    async def _fetch_all(self) -> None:
        """Fetch all model pricing from OpenRouter.

        On failure (network error, timeout, malformed response), sets
        cache to empty dict and logs to stderr. Cost will show as
        unavailable rather than crashing.
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(FETCH_TIMEOUT),
            ) as client:
                resp = await client.get(OPENROUTER_MODELS_URL)

            if resp.status_code != 200:
                print(
                    f"Warning: OpenRouter models endpoint returned {resp.status_code}, "
                    "cost tracking unavailable.",
                    file=sys.stderr,
                )
                self._cache = {}
                return

            data = resp.json()
            self._cache = _parse_pricing_response(data)

        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            print(
                f"Warning: Failed to fetch model pricing: {exc}. Cost tracking unavailable.",
                file=sys.stderr,
            )
            self._cache = {}


def _parse_pricing_response(data: dict[str, Any]) -> dict[str, ModelPricing]:
    """Parse the OpenRouter models API response into a pricing map.

    Args:
        data: Parsed JSON response from ``/api/v1/models``.

    Returns:
        Mapping of model ID to ModelPricing. Models with missing or
        unparseable pricing are silently skipped.
    """
    result: dict[str, ModelPricing] = {}
    for model in data.get("data", []):
        model_id = model.get("id")
        pricing = model.get("pricing")
        if not model_id or not pricing:
            continue
        try:
            prompt_str = pricing.get("prompt", "0")
            completion_str = pricing.get("completion", "0")
            ctx_len = model.get("context_length")
            result[model_id] = ModelPricing(
                prompt_price=float(prompt_str),
                completion_price=float(completion_str),
                context_length=int(ctx_len) if ctx_len is not None else None,
            )
        except (ValueError, TypeError):  # fmt: skip
            continue
    return result


def compute_response_cost(
    resp: ModelResponse,
    pricing: ModelPricing | None,
) -> float | None:
    """Compute USD cost for a single model response.

    Requires both pricing data and input/output token split to compute
    cost. Returns None if either is unavailable.

    Args:
        resp: Model response with optional token split data.
        pricing: Per-token pricing for the model, or None.

    Returns:
        Cost in USD, or None if pricing or token data is missing.
    """
    if pricing is None:
        return None
    if resp.input_tokens is None or resp.output_tokens is None:
        return None
    return resp.input_tokens * pricing.prompt_price + resp.output_tokens * pricing.completion_price
