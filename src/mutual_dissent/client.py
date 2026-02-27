"""Backward compatibility — imports moved to providers package."""

from mutual_dissent.providers.openrouter import (
    OpenRouterError,
)
from mutual_dissent.providers.openrouter import (
    OpenRouterProvider as OpenRouterClient,
)

__all__ = ["OpenRouterClient", "OpenRouterError"]
