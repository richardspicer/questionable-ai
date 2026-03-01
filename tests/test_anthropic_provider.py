"""Tests for the Anthropic provider.

Covers: construction validation, system message extraction (none, single,
multiple), mocked complete() with prompt and messages, timeout handling,
HTTP error handling, malformed response handling, content block extraction
(single text, multiple text blocks, no text blocks), token count computation,
and async context manager lifecycle.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mutual_dissent.models import ModelResponse
from mutual_dissent.providers.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    AnthropicProvider,
    _extract_content,
    _extract_system,
    _extract_token_count,
)

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestAnthropicProviderConstruction:
    """AnthropicProvider.__init__() validation."""

    def test_valid_api_key(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._api_key == "sk-ant-test"

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            AnthropicProvider(api_key="")

    def test_custom_timeout(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test", timeout=30.0)
        assert provider._timeout == 30.0

    def test_default_timeout(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._timeout == 120.0

    def test_custom_max_tokens(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test", max_tokens=8192)
        assert provider._max_tokens == 8192

    def test_default_max_tokens(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._max_tokens == 4096


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestAsyncContextManager:
    """AnthropicProvider async context manager lifecycle."""

    @pytest.mark.asyncio
    async def test_enter_creates_client(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._client is None
        async with provider:
            assert provider._client is not None

    @pytest.mark.asyncio
    async def test_exit_closes_client(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            pass
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_complete_outside_context_raises(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        with pytest.raises(RuntimeError, match="context manager"):
            await provider.complete("claude-sonnet-4-5-20250929", prompt="Hello")

    @pytest.mark.asyncio
    async def test_headers_set_correctly(self) -> None:
        """Verify x-api-key and anthropic-version headers are set."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            headers = provider._client.headers
            assert headers["x-api-key"] == "sk-ant-test"
            assert headers["anthropic-version"] == ANTHROPIC_VERSION
            assert headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# System message extraction
# ---------------------------------------------------------------------------


class TestExtractSystem:
    """_extract_system() separates system messages from chat messages."""

    def test_no_system_messages(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        system_text, remaining = _extract_system(messages)
        assert system_text is None
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_single_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system_text, remaining = _extract_system(messages)
        assert system_text == "Be helpful."
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_multiple_system_messages(self) -> None:
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        system_text, remaining = _extract_system(messages)
        assert system_text == "Be helpful.\n\nBe concise."
        assert remaining == [{"role": "user", "content": "Hello"}]

    def test_system_message_preserves_chat_order(self) -> None:
        """Non-system messages maintain their original order."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        system_text, remaining = _extract_system(messages)
        assert system_text == "System prompt."
        assert len(remaining) == 3
        assert remaining[0]["content"] == "First"
        assert remaining[1]["content"] == "Reply"
        assert remaining[2]["content"] == "Second"

    def test_empty_messages(self) -> None:
        system_text, remaining = _extract_system([])
        assert system_text is None
        assert remaining == []


# ---------------------------------------------------------------------------
# Content block extraction
# ---------------------------------------------------------------------------


class TestExtractContent:
    """_extract_content() extracts text from Anthropic content block arrays."""

    def test_single_text_block(self) -> None:
        data = {"content": [{"type": "text", "text": "Hello!"}]}
        assert _extract_content(data) == "Hello!"

    def test_multiple_text_blocks(self) -> None:
        data = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ],
        }
        assert _extract_content(data) == "Hello world!"

    def test_non_text_blocks_skipped(self) -> None:
        data = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
                {"type": "tool_use", "id": "tool_1", "name": "calc"},
            ],
        }
        assert _extract_content(data) == "The answer is 42."

    def test_no_text_blocks(self) -> None:
        data = {"content": [{"type": "tool_use", "id": "tool_1", "name": "calc"}]}
        assert _extract_content(data) == "[No text content in response]"

    def test_missing_content_key(self) -> None:
        data = {"unexpected": "data"}
        assert "Failed to parse" in _extract_content(data)

    def test_empty_content_array(self) -> None:
        data = {"content": []}
        assert _extract_content(data) == "[No text content in response]"


# ---------------------------------------------------------------------------
# Token count extraction
# ---------------------------------------------------------------------------


class TestExtractTokenCount:
    """_extract_token_count() sums input + output tokens."""

    def test_both_fields_present(self) -> None:
        data = {"usage": {"input_tokens": 12, "output_tokens": 6}}
        assert _extract_token_count(data) == 18

    def test_missing_usage(self) -> None:
        assert _extract_token_count({}) is None

    def test_missing_output_tokens(self) -> None:
        data = {"usage": {"input_tokens": 12}}
        assert _extract_token_count(data) is None

    def test_missing_input_tokens(self) -> None:
        data = {"usage": {"output_tokens": 6}}
        assert _extract_token_count(data) is None


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_anthropic_success() -> httpx.Response:
    """Build a mock httpx.Response for a successful Anthropic completion."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello back!"}],
        "model": "claude-sonnet-4-5-20250929",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }
    return resp


# ---------------------------------------------------------------------------
# Mocked complete() — prompt path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestCompleteWithPrompt:
    """AnthropicProvider.complete() using prompt parameter."""

    @pytest.mark.asyncio
    async def test_success_returns_model_response(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=_mock_anthropic_success())

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hello",
                model_alias="claude",
                round_number=0,
            )

        assert isinstance(result, ModelResponse)
        assert result.content == "Hello back!"
        assert result.model_id == "claude-sonnet-4-5-20250929"
        assert result.model_alias == "claude"
        assert result.round_number == 0
        assert result.token_count == 18
        assert result.error is None
        assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_prompt_sent_as_user_message(self) -> None:
        """Verify the prompt is wrapped in a user message."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_anthropic_success())
            provider._client.post = mock_post

            await provider.complete("claude-sonnet-4-5-20250929", prompt="Test prompt")

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["messages"] == [{"role": "user", "content": "Test prompt"}]

    @pytest.mark.asyncio
    async def test_max_tokens_always_sent(self) -> None:
        """max_tokens is required by Anthropic and must appear in payload."""
        provider = AnthropicProvider(api_key="sk-ant-test", max_tokens=2048)
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_anthropic_success())
            provider._client.post = mock_post

            await provider.complete("claude-sonnet-4-5-20250929", prompt="Hello")

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_default_alias_is_model_id(self) -> None:
        """model_alias defaults to the full model_id (no slash splitting)."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=_mock_anthropic_success())

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hi",
            )

        assert result.model_alias == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_posts_to_anthropic_url(self) -> None:
        """Requests go to the Anthropic Messages API endpoint."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_anthropic_success())
            provider._client.post = mock_post

            await provider.complete("claude-sonnet-4-5-20250929", prompt="Hello")

            call_args = mock_post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert url == ANTHROPIC_API_URL


# ---------------------------------------------------------------------------
# Mocked complete() — messages path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestCompleteWithMessages:
    """AnthropicProvider.complete() using messages parameter."""

    @pytest.mark.asyncio
    async def test_messages_with_system_hoisted(self) -> None:
        """System messages are extracted and hoisted to top-level system field."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_anthropic_success())
            provider._client.post = mock_post

            await provider.complete(
                "claude-sonnet-4-5-20250929",
                messages=messages,
                model_alias="claude",
            )

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["system"] == "Be helpful."
            assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_messages_without_system(self) -> None:
        """No system field in payload when no system messages exist."""
        provider = AnthropicProvider(api_key="sk-ant-test")
        messages = [{"role": "user", "content": "Hello"}]
        async with provider:
            assert provider._client is not None
            mock_post = AsyncMock(return_value=_mock_anthropic_success())
            provider._client.post = mock_post

            await provider.complete(
                "claude-sonnet-4-5-20250929",
                messages=messages,
            )

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "system" not in payload
            assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_both_messages_and_prompt_raises(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            with pytest.raises(ValueError, match="not both"):
                await provider.complete(
                    "claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hi"}],
                    prompt="Hello",
                )

    @pytest.mark.asyncio
    async def test_neither_messages_nor_prompt_raises(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            with pytest.raises(ValueError, match="Provide either"):
                await provider.complete("claude-sonnet-4-5-20250929")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestErrorHandling:
    """AnthropicProvider.complete() error paths."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_response(self) -> None:
        provider = AnthropicProvider(api_key="sk-ant-test", timeout=5.0)
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(
                side_effect=httpx.TimeoutException("timed out"),
            )

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hello",
                model_alias="claude",
            )

        assert result.error is not None
        assert "timed out" in result.error
        assert result.content == ""
        assert result.model_alias == "claude"

    @pytest.mark.asyncio
    async def test_http_error_returns_error_response(self) -> None:
        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 400
        error_resp.json.return_value = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "max_tokens: 100001 > 64000",
            },
        }

        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=error_resp)

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hello",
                model_alias="claude",
            )

        assert result.error is not None
        assert "400" in result.error
        assert "max_tokens" in result.error
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_malformed_response_body(self) -> None:
        """Response with unexpected structure still returns a ModelResponse."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"unexpected": "data"}

        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=resp)

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hello",
            )

        assert "Failed to parse" in result.content
        assert result.error is None
        assert result.token_count is None

    @pytest.mark.asyncio
    async def test_error_response_non_json(self) -> None:
        """Non-JSON error response falls back to text body."""
        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 502
        error_resp.json.side_effect = ValueError("not json")
        error_resp.text = "Bad Gateway"

        provider = AnthropicProvider(api_key="sk-ant-test")
        async with provider:
            assert provider._client is not None
            provider._client.post = AsyncMock(return_value=error_resp)

            result = await provider.complete(
                "claude-sonnet-4-5-20250929",
                prompt="Hello",
            )

        assert result.error is not None
        assert "502" in result.error
        assert "Bad Gateway" in result.error


# ---------------------------------------------------------------------------
# Import from providers package
# ---------------------------------------------------------------------------


class TestProviderImport:
    """AnthropicProvider is importable from the providers package."""

    def test_importable_from_providers(self) -> None:
        from mutual_dissent.providers import AnthropicProvider as Imported

        assert Imported is AnthropicProvider
