"""OpenRouter API client for model interactions.

Async HTTP client that sends chat completion requests to OpenRouter's
unified API endpoint. Supports parallel fan-out to multiple models and
tracks response latency and token usage.

Typical usage::

    import asyncio
    from mutual_dissent.client import OpenRouterClient

    async def main():
        async with OpenRouterClient(api_key="sk-or-...") as client:
            response = await client.complete("anthropic/claude-sonnet-4.5", "Hello")

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from mutual_dissent.models import ModelResponse

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 120.0  # seconds — generous for slow models
APP_SITE_URL = "https://richardspicer.io"
APP_NAME = "Mutual Dissent"


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns an error.

    Attributes:
        status_code: HTTP status code from the API response.
        detail: Error detail string from the API response body.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"OpenRouter API error {status_code}: {detail}")


class OpenRouterClient:
    """Async client for the OpenRouter chat completions API.

    Uses httpx.AsyncClient for connection pooling and async I/O.
    Designed to be used as an async context manager.

    Args:
        api_key: OpenRouter API key.
        timeout: Request timeout in seconds. Defaults to 120s.

    Example::

        async with OpenRouterClient(api_key="sk-or-...") as client:
            resp = await client.complete("openai/gpt-5.2", "What is 2+2?")
            print(resp.content)
    """

    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY env var "
                "or add api_key to ~/.mutual-dissent/config.toml"
            )
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenRouterClient:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": APP_SITE_URL,
                "X-Title": APP_NAME,
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        model_id: str,
        prompt: str,
        *,
        model_alias: str = "",
        round_number: int = 0,
    ) -> ModelResponse:
        """Send a chat completion request to a single model.

        Args:
            model_id: OpenRouter model identifier.
            prompt: The user message content.
            model_alias: Human-readable name for logging. Defaults to
                the model_id if not provided.
            round_number: Which debate round this belongs to.

        Returns:
            ModelResponse with the model's reply, timing, and token stats.

        Raises:
            OpenRouterError: If the API returns a non-2xx status code.
            RuntimeError: If the client is used outside a context manager.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        alias = model_alias or model_id.split("/")[-1]
        start = time.monotonic()

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            resp = await self._client.post(OPENROUTER_API_URL, json=payload)
        except httpx.TimeoutException:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ModelResponse(
                model_id=model_id,
                model_alias=alias,
                round_number=round_number,
                content="",
                latency_ms=elapsed_ms,
                error=f"Request timed out after {self._timeout}s",
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if resp.status_code != 200:
            error_detail = _extract_error(resp)
            return ModelResponse(
                model_id=model_id,
                model_alias=alias,
                round_number=round_number,
                content="",
                latency_ms=elapsed_ms,
                error=f"HTTP {resp.status_code}: {error_detail}",
            )

        data = resp.json()
        content = _extract_content(data)
        token_count = _extract_token_count(data)

        return ModelResponse(
            model_id=model_id,
            model_alias=alias,
            round_number=round_number,
            content=content,
            latency_ms=elapsed_ms,
            token_count=token_count,
        )

    async def complete_parallel(
        self,
        requests: list[dict[str, Any]],
    ) -> list[ModelResponse]:
        """Send multiple completion requests in parallel.

        Args:
            requests: List of keyword argument dicts for complete().
                Each dict should contain at minimum 'model_id' and 'prompt'.

        Returns:
            List of ModelResponse objects, one per request, in the same order.

        Example::

            results = await client.complete_parallel([
                {"model_id": "anthropic/claude-sonnet-4.5", "prompt": "Hi"},
                {"model_id": "openai/gpt-5.2", "prompt": "Hi"},
            ])
        """
        tasks = [self.complete(**req) for req in requests]
        return list(await asyncio.gather(*tasks))


def _extract_content(data: dict[str, Any]) -> str:
    """Extract the assistant message content from an API response.

    Args:
        data: Parsed JSON response body.

    Returns:
        The text content of the first choice, or an error description.
    """
    try:
        content: str = data["choices"][0]["message"]["content"]
        return content
    except KeyError, IndexError, TypeError:
        return f"[Failed to parse response: {data}]"


def _extract_token_count(data: dict[str, Any]) -> int | None:
    """Extract total token usage from an API response.

    Args:
        data: Parsed JSON response body.

    Returns:
        Total token count if available, None otherwise.
    """
    usage = data.get("usage")
    if usage and "total_tokens" in usage:
        return int(usage["total_tokens"])
    return None


def _extract_error(resp: httpx.Response) -> str:
    """Extract error detail from a non-2xx API response.

    Args:
        resp: The httpx response object.

    Returns:
        Human-readable error description.
    """
    try:
        body = resp.json()
        error = body.get("error", {})
        if isinstance(error, dict):
            return str(error.get("message", str(body)))
        return str(error)
    except Exception:
        return str(resp.text[:500])
