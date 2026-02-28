"""Tests for cost tracking and pricing module.

Covers: ModelResponse token split fields, PricingCache fetch (success,
failure, cache hit), compute_response_cost(), _compute_stats() with
cost data, cost display in terminal and markdown, and backward
compatibility with old transcripts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mutual_dissent.display import _format_cost_summary, format_markdown, render_debate
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.pricing import (
    ModelPricing,
    PricingCache,
    _parse_pricing_response,
    compute_response_cost,
)

# ---------------------------------------------------------------------------
# ModelResponse token split fields
# ---------------------------------------------------------------------------


class TestModelResponseTokenSplit:
    """ModelResponse input_tokens and output_tokens fields."""

    def test_defaults_to_none(self) -> None:
        """input_tokens and output_tokens default to None."""
        r = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
        )
        assert r.input_tokens is None
        assert r.output_tokens is None

    def test_explicit_values(self) -> None:
        """input_tokens and output_tokens can be set explicitly."""
        r = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            token_count=150,
            input_tokens=100,
            output_tokens=50,
        )
        assert r.input_tokens == 100
        assert r.output_tokens == 50
        assert r.token_count == 150

    def test_to_dict_includes_token_split(self) -> None:
        """to_dict() includes input_tokens and output_tokens."""
        r = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            token_count=150,
            input_tokens=100,
            output_tokens=50,
        )
        d = r.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["token_count"] == 150

    def test_to_dict_none_values(self) -> None:
        """to_dict() includes None for missing token split."""
        r = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
        )
        d = r.to_dict()
        assert d["input_tokens"] is None
        assert d["output_tokens"] is None


# ---------------------------------------------------------------------------
# compute_response_cost
# ---------------------------------------------------------------------------


class TestComputeResponseCost:
    """compute_response_cost() returns correct USD cost."""

    def test_with_pricing_and_tokens(self) -> None:
        """Correct cost computed from pricing and token split."""
        resp = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            input_tokens=1000,
            output_tokens=500,
        )
        pricing = ModelPricing(prompt_price=0.000003, completion_price=0.000015)
        cost = compute_response_cost(resp, pricing)
        assert cost is not None
        assert abs(cost - (1000 * 0.000003 + 500 * 0.000015)) < 1e-10

    def test_without_pricing(self) -> None:
        """Returns None when pricing is None."""
        resp = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            input_tokens=1000,
            output_tokens=500,
        )
        assert compute_response_cost(resp, None) is None

    def test_without_input_tokens(self) -> None:
        """Returns None when input_tokens is missing."""
        resp = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            output_tokens=500,
        )
        pricing = ModelPricing(prompt_price=0.000003, completion_price=0.000015)
        assert compute_response_cost(resp, pricing) is None

    def test_without_output_tokens(self) -> None:
        """Returns None when output_tokens is missing."""
        resp = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            input_tokens=1000,
        )
        pricing = ModelPricing(prompt_price=0.000003, completion_price=0.000015)
        assert compute_response_cost(resp, pricing) is None

    def test_zero_tokens(self) -> None:
        """Zero tokens returns zero cost."""
        resp = ModelResponse(
            model_id="test/model",
            model_alias="test",
            round_number=0,
            content="hello",
            input_tokens=0,
            output_tokens=0,
        )
        pricing = ModelPricing(prompt_price=0.000003, completion_price=0.000015)
        cost = compute_response_cost(resp, pricing)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# _parse_pricing_response
# ---------------------------------------------------------------------------


class TestParsePricingResponse:
    """_parse_pricing_response() parses OpenRouter models API response."""

    def test_valid_response(self) -> None:
        """Parses a well-formed models response."""
        data = {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4.5",
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                },
                {
                    "id": "openai/gpt-5.2",
                    "pricing": {"prompt": "0.000005", "completion": "0.000010"},
                },
            ]
        }
        result = _parse_pricing_response(data)
        assert len(result) == 2
        assert result["anthropic/claude-sonnet-4.5"].prompt_price == 0.000003
        assert result["anthropic/claude-sonnet-4.5"].completion_price == 0.000015
        assert result["openai/gpt-5.2"].prompt_price == 0.000005

    def test_missing_pricing_skipped(self) -> None:
        """Models without pricing are silently skipped."""
        data = {
            "data": [
                {"id": "model-a", "pricing": {"prompt": "0.001", "completion": "0.002"}},
                {"id": "model-b"},
            ]
        }
        result = _parse_pricing_response(data)
        assert len(result) == 1
        assert "model-a" in result

    def test_empty_data(self) -> None:
        """Empty data array returns empty dict."""
        result = _parse_pricing_response({"data": []})
        assert result == {}

    def test_missing_data_key(self) -> None:
        """Missing 'data' key returns empty dict."""
        result = _parse_pricing_response({})
        assert result == {}

    def test_invalid_pricing_value(self) -> None:
        """Non-numeric pricing value skips that model."""
        data = {
            "data": [
                {"id": "model-a", "pricing": {"prompt": "invalid", "completion": "0.002"}},
            ]
        }
        result = _parse_pricing_response(data)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# PricingCache
# ---------------------------------------------------------------------------


def _mock_models_response() -> httpx.Response:
    """Build a mock httpx.Response for the models endpoint."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-4.5",
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                "context_length": 200000,
            },
            {
                "id": "openai/gpt-5.2",
                "pricing": {"prompt": "0.000005", "completion": "0.000010"},
                "context_length": 128000,
            },
        ]
    }
    return resp


class TestPricingCache:
    """PricingCache fetches and caches model pricing."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self) -> None:
        """Fetches pricing from OpenRouter on first access."""
        cache = PricingCache()
        mock_resp = _mock_models_response()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            pricing = await cache.get_pricing("anthropic/claude-sonnet-4.5")
        finally:
            httpx.AsyncClient = original

        assert pricing is not None
        assert pricing.prompt_price == 0.000003
        assert pricing.completion_price == 0.000015

    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        """Second access uses cached data (no second fetch)."""
        cache = PricingCache()
        mock_resp = _mock_models_response()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            await cache.get_pricing("anthropic/claude-sonnet-4.5")
            await cache.get_pricing("openai/gpt-5.2")
        finally:
            httpx.AsyncClient = original

        # Only one GET call despite two lookups.
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self) -> None:
        """Network failure results in None pricing (no crash)."""
        cache = PricingCache()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            pricing = await cache.get_pricing("anthropic/claude-sonnet-4.5")
        finally:
            httpx.AsyncClient = original

        assert pricing is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self) -> None:
        """Non-200 status results in None pricing (no crash)."""
        cache = PricingCache()

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=error_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            pricing = await cache.get_pricing("anthropic/claude-sonnet-4.5")
        finally:
            httpx.AsyncClient = original

        assert pricing is None

    @pytest.mark.asyncio
    async def test_direct_model_id_lookup(self) -> None:
        """Vendor-native model IDs are resolved via alias map."""
        alias_map = {
            "claude": {
                "openrouter": "anthropic/claude-sonnet-4.5",
                "direct": "claude-sonnet-4-5-20250929",
            },
        }
        cache = PricingCache(alias_map=alias_map)
        mock_resp = _mock_models_response()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            pricing = await cache.get_pricing("claude-sonnet-4-5-20250929")
        finally:
            httpx.AsyncClient = original

        assert pricing is not None
        assert pricing.prompt_price == 0.000003

    @pytest.mark.asyncio
    async def test_unknown_model_returns_none(self) -> None:
        """Unknown model ID returns None."""
        cache = PricingCache()
        mock_resp = _mock_models_response()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            pricing = await cache.get_pricing("unknown/model")
        finally:
            httpx.AsyncClient = original

        assert pricing is None

    @pytest.mark.asyncio
    async def test_prefetch_idempotent(self) -> None:
        """Calling prefetch() twice only fetches once."""
        cache = PricingCache()
        mock_resp = _mock_models_response()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        original = httpx.AsyncClient
        try:
            httpx.AsyncClient = lambda **kwargs: mock_client  # type: ignore[assignment,misc]
            await cache.prefetch()
            await cache.prefetch()
        finally:
            httpx.AsyncClient = original

        assert mock_client.get.call_count == 1


# ---------------------------------------------------------------------------
# _compute_stats with cost data
# ---------------------------------------------------------------------------


class TestComputeStatsWithCost:
    """_compute_stats() populates cost when pricing is available."""

    @pytest.mark.asyncio
    async def test_with_pricing(self) -> None:
        """Stats include total_cost_usd and per-model cost_usd."""
        from mutual_dissent.orchestrator import _compute_stats

        transcript = _make_costed_transcript()

        # Build a pricing cache with pre-populated data.
        cache = PricingCache()
        cache._cache = {
            "anthropic/claude-sonnet-4.5": ModelPricing(
                prompt_price=0.000003, completion_price=0.000015
            ),
            "openai/gpt-5.2": ModelPricing(prompt_price=0.000005, completion_price=0.000010),
        }

        stats = await _compute_stats(transcript, cache)

        assert stats["total_cost_usd"] is not None
        assert stats["total_cost_usd"] > 0
        assert stats["per_model"]["claude"]["cost_usd"] > 0
        assert stats["per_model"]["gpt"]["cost_usd"] > 0
        # claude has 2 responses: initial (500 in, 200 out) + synthesis (200 in, 100 out)
        assert stats["per_model"]["claude"]["input_tokens"] == 700
        assert stats["per_model"]["claude"]["output_tokens"] == 300

    @pytest.mark.asyncio
    async def test_without_pricing(self) -> None:
        """Stats have total_cost_usd=None when no pricing cache."""
        from mutual_dissent.orchestrator import _compute_stats

        transcript = _make_costed_transcript()
        stats = await _compute_stats(transcript)

        assert stats["total_cost_usd"] is None
        assert stats["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_empty_cache(self) -> None:
        """Stats have total_cost_usd=None with empty pricing cache."""
        from mutual_dissent.orchestrator import _compute_stats

        transcript = _make_costed_transcript()

        cache = PricingCache()
        cache._cache = {}

        stats = await _compute_stats(transcript, cache)

        assert stats["total_cost_usd"] is None

    @pytest.mark.asyncio
    async def test_per_model_token_split(self) -> None:
        """Per-model stats include input_tokens and output_tokens."""
        from mutual_dissent.orchestrator import _compute_stats

        transcript = _make_costed_transcript()
        stats = await _compute_stats(transcript)

        assert "input_tokens" in stats["per_model"]["claude"]
        assert "output_tokens" in stats["per_model"]["claude"]


# ---------------------------------------------------------------------------
# Cost display
# ---------------------------------------------------------------------------


class TestCostDisplay:
    """Cost display in terminal and markdown output."""

    def test_format_cost_summary_with_data(self) -> None:
        """_format_cost_summary returns formatted cost string."""
        transcript = _make_costed_transcript()
        transcript.metadata["stats"] = {
            "total_cost_usd": 0.0234,
            "per_model": {
                "claude": {"cost_usd": 0.0150},
                "gpt": {"cost_usd": 0.0084},
            },
        }
        result = _format_cost_summary(transcript)
        assert "$0.0234" in result
        assert "claude: $0.0150" in result
        assert "gpt: $0.0084" in result

    def test_format_cost_summary_no_stats(self) -> None:
        """_format_cost_summary returns empty string with no stats."""
        transcript = _make_costed_transcript()
        result = _format_cost_summary(transcript)
        assert result == ""

    def test_format_cost_summary_null_cost(self) -> None:
        """_format_cost_summary returns empty string when cost is None."""
        transcript = _make_costed_transcript()
        transcript.metadata["stats"] = {"total_cost_usd": None}
        result = _format_cost_summary(transcript)
        assert result == ""

    def test_markdown_includes_cost(self) -> None:
        """Markdown output includes cost line when data is available."""
        transcript = _make_costed_transcript()
        transcript.metadata["stats"] = {
            "total_cost_usd": 0.0234,
            "per_model": {
                "claude": {"cost_usd": 0.015},
                "gpt": {"cost_usd": 0.0084},
            },
        }
        result = format_markdown(transcript)
        assert "**Cost:**" in result
        assert "$0.0234" in result

    def test_markdown_omits_cost_when_unavailable(self) -> None:
        """Markdown output omits cost when data is unavailable."""
        transcript = _make_costed_transcript()
        result = format_markdown(transcript)
        assert "**Cost:**" not in result

    def test_terminal_render_with_cost(self) -> None:
        """render_debate() doesn't crash with cost data (smoke test)."""
        transcript = _make_costed_transcript()
        transcript.metadata["stats"] = {
            "total_cost_usd": 0.0234,
            "per_model": {
                "claude": {"cost_usd": 0.015},
                "gpt": {"cost_usd": 0.0084},
            },
        }
        render_debate(transcript)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Old transcripts without token split data load correctly."""

    def test_old_transcript_loads(self) -> None:
        """Transcript with only token_count (no input/output split) loads."""
        from mutual_dissent.transcript import _parse_response

        data = {
            "model_id": "test/model",
            "model_alias": "test",
            "round_number": 0,
            "content": "hello",
            "token_count": 42,
        }
        resp = _parse_response(data)
        assert resp.token_count == 42
        assert resp.input_tokens is None
        assert resp.output_tokens is None

    def test_new_transcript_loads(self) -> None:
        """Transcript with all token fields loads correctly."""
        from mutual_dissent.transcript import _parse_response

        data = {
            "model_id": "test/model",
            "model_alias": "test",
            "round_number": 0,
            "content": "hello",
            "token_count": 150,
            "input_tokens": 100,
            "output_tokens": 50,
        }
        resp = _parse_response(data)
        assert resp.token_count == 150
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50


# ---------------------------------------------------------------------------
# Provider token split extraction
# ---------------------------------------------------------------------------


class TestOpenRouterTokenSplit:
    """OpenRouter provider extracts input/output token split."""

    @pytest.mark.asyncio
    async def test_token_split_from_usage(self) -> None:
        """complete() sets input_tokens and output_tokens from usage."""
        from mutual_dissent.providers.openrouter import OpenRouterProvider

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Reply"}}],
            "usage": {
                "total_tokens": 150,
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }

        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=resp)
            result = await provider.complete("model/id", prompt="Hello")

        assert result.token_count == 150
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_missing_split_returns_none(self) -> None:
        """Missing prompt_tokens/completion_tokens returns None."""
        from mutual_dissent.providers.openrouter import OpenRouterProvider

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "Reply"}}],
            "usage": {"total_tokens": 150},
        }

        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=resp)
            result = await provider.complete("model/id", prompt="Hello")

        assert result.token_count == 150
        assert result.input_tokens is None
        assert result.output_tokens is None


class TestAnthropicTokenSplit:
    """Anthropic provider extracts input/output token split."""

    @pytest.mark.asyncio
    async def test_token_split_from_usage(self) -> None:
        """complete() sets input_tokens and output_tokens from usage."""
        from mutual_dissent.providers.anthropic import AnthropicProvider

        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "content": [{"type": "text", "text": "Reply"}],
            "usage": {"input_tokens": 80, "output_tokens": 40},
        }

        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=resp)
            result = await provider.complete("claude-model", prompt="Hello")

        assert result.token_count == 120  # 80 + 40
        assert result.input_tokens == 80
        assert result.output_tokens == 40


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------


class TestContextLength:
    """ModelPricing.context_length field and PricingCache.get_context_length()."""

    def test_model_pricing_context_length_default(self) -> None:
        """context_length defaults to None."""
        pricing = ModelPricing(prompt_price=0.001, completion_price=0.002)
        assert pricing.context_length is None

    def test_model_pricing_context_length_set(self) -> None:
        """context_length can be set explicitly."""
        pricing = ModelPricing(prompt_price=0.001, completion_price=0.002, context_length=200000)
        assert pricing.context_length == 200000

    def test_parse_pricing_response_captures_context_length(self) -> None:
        """_parse_pricing_response captures context_length from API data."""
        data = {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4.5",
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    "context_length": 200000,
                },
            ]
        }
        result = _parse_pricing_response(data)
        assert result["anthropic/claude-sonnet-4.5"].context_length == 200000

    def test_parse_pricing_response_missing_context_length(self) -> None:
        """context_length is None when missing from API response."""
        data = {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4.5",
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                },
            ]
        }
        result = _parse_pricing_response(data)
        assert result["anthropic/claude-sonnet-4.5"].context_length is None

    @pytest.mark.asyncio
    async def test_get_context_length(self) -> None:
        """get_context_length returns context_length for a known model."""
        cache = PricingCache()
        cache._cache = {
            "anthropic/claude-sonnet-4.5": ModelPricing(
                prompt_price=0.000003,
                completion_price=0.000015,
                context_length=200000,
            ),
        }
        ctx_len = await cache.get_context_length("anthropic/claude-sonnet-4.5")
        assert ctx_len == 200000

    @pytest.mark.asyncio
    async def test_get_context_length_unknown_model(self) -> None:
        """get_context_length returns None for unknown model."""
        cache = PricingCache()
        cache._cache = {}
        ctx_len = await cache.get_context_length("unknown/model")
        assert ctx_len is None

    @pytest.mark.asyncio
    async def test_get_context_length_direct_id(self) -> None:
        """get_context_length resolves vendor-native model IDs via alias map."""
        alias_map = {
            "claude": {
                "openrouter": "anthropic/claude-sonnet-4.5",
                "direct": "claude-sonnet-4-5-20250929",
            },
        }
        cache = PricingCache(alias_map=alias_map)
        cache._cache = {
            "anthropic/claude-sonnet-4.5": ModelPricing(
                prompt_price=0.000003,
                completion_price=0.000015,
                context_length=200000,
            ),
        }
        ctx_len = await cache.get_context_length("claude-sonnet-4-5-20250929")
        assert ctx_len == 200000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_costed_transcript() -> DebateTranscript:
    """Build a transcript with token split data for cost tests."""
    return DebateTranscript(
        query="What is X?",
        panel=["claude", "gpt"],
        synthesizer_id="claude",
        max_rounds=1,
        rounds=[
            DebateRound(
                round_number=0,
                round_type="initial",
                responses=[
                    ModelResponse(
                        model_id="anthropic/claude-sonnet-4.5",
                        model_alias="claude",
                        round_number=0,
                        content="Claude response.",
                        role="initial",
                        token_count=700,
                        input_tokens=500,
                        output_tokens=200,
                    ),
                    ModelResponse(
                        model_id="openai/gpt-5.2",
                        model_alias="gpt",
                        round_number=0,
                        content="GPT response.",
                        role="initial",
                        token_count=600,
                        input_tokens=400,
                        output_tokens=200,
                    ),
                ],
            ),
        ],
        synthesis=ModelResponse(
            model_id="anthropic/claude-sonnet-4.5",
            model_alias="claude",
            round_number=-1,
            content="Synthesis.",
            role="synthesis",
            token_count=300,
            input_tokens=200,
            output_tokens=100,
        ),
    )
