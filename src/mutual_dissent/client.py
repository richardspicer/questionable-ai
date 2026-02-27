"""Backward compatibility shim — imports moved to providers package.

Deprecated: The orchestrator now uses ProviderRouter directly.
This module exists only for external callers that import OpenRouterClient.
"""

from mutual_dissent.providers.openrouter import (
    OpenRouterError,
)
from mutual_dissent.providers.openrouter import (
    OpenRouterProvider as OpenRouterClient,
)

__all__ = ["OpenRouterClient", "OpenRouterError"]
