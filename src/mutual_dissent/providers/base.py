"""Abstract base class for model API providers.

Defines the ``Provider`` interface that all vendor-specific implementations
must follow.  Providers handle auth, endpoints, request formatting, and
response normalization.  All providers return ``ModelResponse`` objects.

Subclasses must implement ``complete()``, ``__aenter__()``, and
``__aexit__()``.  The default ``complete_parallel()`` fans out via
``asyncio.gather`` and can be overridden for provider-specific batching.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from mutual_dissent.models import ModelResponse


class Provider(ABC):
    """Base class for all model API providers.

    Providers handle vendor-specific auth, endpoints, request formatting,
    and response normalization. All providers return ``ModelResponse`` objects.

    Designed as an async context manager for connection lifecycle.
    """

    @abstractmethod
    async def complete(
        self,
        model_id: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        model_alias: str = "",
        round_number: int = 0,
    ) -> ModelResponse:
        """Send a completion request.

        Accepts either ``messages`` (list of chat messages) or ``prompt``
        (single user message string).  Exactly one must be provided.

        Args:
            model_id: Provider-specific model identifier.
            messages: Chat messages in OpenAI-compatible format.
            prompt: Single user message string (convenience shorthand).
            model_alias: Human-readable short name for logging.
            round_number: Debate round (0=initial, 1+=reflection, -1=synthesis).

        Returns:
            ModelResponse with the model's reply, timing, and token stats.

        Raises:
            ValueError: If both ``messages`` and ``prompt`` are provided,
                or if neither is provided.
        """
        ...

    async def complete_parallel(
        self,
        requests: list[dict[str, Any]],
    ) -> list[ModelResponse]:
        """Fan out multiple completion requests in parallel.

        The default implementation uses ``asyncio.gather``.  Subclasses may
        override for provider-specific batching APIs.

        Args:
            requests: List of keyword argument dicts for ``complete()``.
                Each dict should contain at minimum ``model_id`` and either
                ``prompt`` or ``messages``.

        Returns:
            List of ``ModelResponse`` objects in the same order as *requests*.
        """
        tasks = [self.complete(**req) for req in requests]
        return list(await asyncio.gather(*tasks))

    @staticmethod
    def _resolve_messages(
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
    ) -> list[dict[str, Any]]:
        """Normalize ``messages``/``prompt`` into a message list.

        Exactly one of the two parameters must be provided.  If ``prompt``
        is given it is wrapped in ``[{"role": "user", "content": prompt}]``.

        Args:
            messages: Chat messages in OpenAI-compatible format, or ``None``.
            prompt: Single user message string, or ``None``.

        Returns:
            A list of message dicts ready to send to the API.

        Raises:
            ValueError: If both or neither argument is provided.
        """
        if messages is not None and prompt is not None:
            raise ValueError("Provide either 'messages' or 'prompt', not both.")
        if messages is None and prompt is None:
            raise ValueError("Provide either 'messages' or 'prompt'.")
        if messages is not None:
            return messages
        return [{"role": "user", "content": prompt}]

    @abstractmethod
    async def __aenter__(self) -> Provider:
        """Enter the async context manager (open connections)."""
        ...

    @abstractmethod
    async def __aexit__(self, *exc: Any) -> None:
        """Exit the async context manager (close connections)."""
        ...
