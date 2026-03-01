"""Tests for the provider abstraction layer.

Covers: ABC validation logic (_resolve_messages), OpenRouterProvider
construction, mocked complete() with prompt and messages, error handling
(timeout, HTTP error), complete_parallel ordering, backward compat shim,
and async context manager lifecycle.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mutual_dissent.models import ModelResponse
from mutual_dissent.providers.base import Provider
from mutual_dissent.providers.openrouter import (
    OpenRouterError,
    OpenRouterProvider,
)

# ---------------------------------------------------------------------------
# ABC _resolve_messages validation
# ---------------------------------------------------------------------------


class TestResolveMessages:
    """Provider._resolve_messages() enforces exactly-one-of messages/prompt."""

    def test_prompt_only(self) -> None:
        result = Provider._resolve_messages(None, "Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_messages_only(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        result = Provider._resolve_messages(msgs, None)
        assert result is msgs

    def test_both_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Provider._resolve_messages([{"role": "user", "content": "Hi"}], "Hello")

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            Provider._resolve_messages(None, None)

    def test_empty_messages_list_accepted(self) -> None:
        """An empty list is valid — the provider may add system messages."""
        result = Provider._resolve_messages([], None)
        assert result == []


# ---------------------------------------------------------------------------
# OpenRouterProvider construction
# ---------------------------------------------------------------------------


class TestOpenRouterProviderConstruction:
    """OpenRouterProvider.__init__() validation."""

    def test_valid_api_key(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        assert provider._api_key == "sk-or-test"

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            OpenRouterProvider(api_key="")

    def test_custom_timeout(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test", timeout=30.0)
        assert provider._timeout == 30.0

    def test_default_timeout(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        assert provider._timeout == 120.0


# ---------------------------------------------------------------------------
# OpenRouterError exception
# ---------------------------------------------------------------------------


class TestOpenRouterError:
    """OpenRouterError stores status_code and detail."""

    def test_attributes(self) -> None:
        err = OpenRouterError(429, "Rate limited")
        assert err.status_code == 429
        assert err.detail == "Rate limited"
        assert "429" in str(err)
        assert "Rate limited" in str(err)


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestAsyncContextManager:
    """OpenRouterProvider async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_enter_creates_client(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        assert provider._client is None
        async with provider:
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_exit_closes_client(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            pass
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_complete_outside_context_raises(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="context manager"):
            await provider.complete("model/id", prompt="Hello")


# ---------------------------------------------------------------------------
# Mocked complete() — prompt path
# ---------------------------------------------------------------------------


def _mock_success_response() -> httpx.Response:
    """Build a mock httpx.Response for a successful completion."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": "Hello back!"}}],
        "usage": {"total_tokens": 42},
    }
    return resp


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestCompleteWithPrompt:
    """OpenRouterProvider.complete() using prompt parameter."""

    @pytest.mark.asyncio
    async def test_success_returns_model_response(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=_mock_success_response())

            result = await provider.complete(
                "anthropic/claude-sonnet-4.5",
                prompt="Hello",
                model_alias="claude",
                round_number=0,
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello back!"
        assert result.model_id == "anthropic/claude-sonnet-4.5"
        assert result.model_alias == "claude"
        assert result.round_number == 0
        assert result.token_count == 42
        assert result.error is None
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_prompt_sent_as_user_message(self) -> None:
        """Verify the prompt is wrapped in a user message."""
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_success_response())
            provider._client.post = mock_post

            await provider.complete("model/id", prompt="Test prompt")

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["messages"] == [{"role": "user", "content": "Test prompt"}]

    @pytest.mark.asyncio
    async def test_default_alias_from_model_id(self) -> None:
        """model_alias defaults to last segment of model_id."""
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=_mock_success_response())

            result = await provider.complete("vendor/model-name", prompt="Hi")

        assert result.model_alias == "model-name"


# ---------------------------------------------------------------------------
# Mocked complete() — messages path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestCompleteWithMessages:
    """OpenRouterProvider.complete() using messages parameter."""

    @pytest.mark.asyncio
    async def test_messages_sent_directly(self) -> None:
        """Messages list is sent to the API as-is."""
        provider = OpenRouterProvider(api_key="sk-or-test")
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_success_response())
            provider._client.post = mock_post

            result = await provider.complete(
                "openai/gpt-5.2",
                messages=messages,
                model_alias="gpt",
            )

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["messages"] == messages

        assert result.content == "Hello back!"
        assert result.model_alias == "gpt"

    @pytest.mark.asyncio
    async def test_both_messages_and_prompt_raises(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            with pytest.raises(ValueError, match="not both"):
                await provider.complete(
                    "model/id",
                    messages=[{"role": "user", "content": "Hi"}],
                    prompt="Hello",
                )

    @pytest.mark.asyncio
    async def test_neither_messages_nor_prompt_raises(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            with pytest.raises(ValueError, match="Provide either"):
                await provider.complete("model/id")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestErrorHandling:
    """OpenRouterProvider.complete() error paths."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_response(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test", timeout=5.0)
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

            result = await provider.complete(
                "model/slow",
                prompt="Hello",
                model_alias="slow",
            )

        assert result.error is not None
        assert "timed out" in result.error
        assert result.content == ""
        assert result.model_alias == "slow"

    @pytest.mark.asyncio
    async def test_http_error_returns_error_response(self) -> None:
        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 429
        error_resp.json.return_value = {"error": {"message": "Rate limited"}}

        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=error_resp)

            result = await provider.complete(
                "model/busy",
                prompt="Hello",
                model_alias="busy",
            )

        assert result.error is not None
        assert "429" in result.error
        assert "Rate limited" in result.error
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_malformed_response_body(self) -> None:
        """Response with unexpected structure still returns a ModelResponse."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"unexpected": "data"}

        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=resp)

            result = await provider.complete("model/id", prompt="Hello")

        assert "Failed to parse" in result.content
        assert result.error is None
        assert result.token_count is None


# ---------------------------------------------------------------------------
# complete_parallel ordering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestCompleteParallel:
    """complete_parallel() preserves request order."""

    @pytest.mark.asyncio
    async def test_order_preserved(self) -> None:
        """Results come back in the same order as requests."""

        def _make_response(model_id: str) -> httpx.Response:
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": f"Reply from {model_id}"}}],
            }
            return resp

        provider = OpenRouterProvider(api_key="sk-or-test")
        async with provider:
            assert provider._client is not None

            async def _side_effect(url: str, json: dict[str, Any]) -> httpx.Response:
                return _make_response(json["model"])

            provider._client.post = AsyncMock(side_effect=_side_effect)

            requests = [
                {"model_id": "vendor/model-a", "prompt": "Hi"},
                {"model_id": "vendor/model-b", "prompt": "Hi"},
                {"model_id": "vendor/model-c", "prompt": "Hi"},
            ]
            results = await provider.complete_parallel(requests)

        assert len(results) == 3
        assert results[0].model_id == "vendor/model-a"
        assert results[1].model_id == "vendor/model-b"
        assert results[2].model_id == "vendor/model-c"
        assert "model-a" in results[0].content
        assert "model-b" in results[1].content
        assert "model-c" in results[2].content


# ---------------------------------------------------------------------------
# Backward compatibility shim
# ---------------------------------------------------------------------------


class TestBackwardCompatShim:
    """client.py shim re-exports work correctly."""

    def test_openrouter_client_is_openrouter_provider(self) -> None:
        from mutual_dissent.client import OpenRouterClient

        assert OpenRouterClient is OpenRouterProvider

    def test_openrouter_error_importable(self) -> None:
        from mutual_dissent.client import OpenRouterError as ShimError

        assert ShimError is OpenRouterError

    def test_shim_import_still_works(self) -> None:
        """The client.py shim still resolves for external callers."""
        from mutual_dissent.client import OpenRouterClient

        client = OpenRouterClient(api_key="sk-or-compat")
        assert isinstance(client, OpenRouterProvider)
