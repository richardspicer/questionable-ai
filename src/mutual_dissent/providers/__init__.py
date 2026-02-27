"""Provider abstraction layer for multi-vendor API access.

Re-exports the public interface so callers can write::

    from mutual_dissent.providers import Provider, OpenRouterProvider
"""

from mutual_dissent.providers.base import Provider
from mutual_dissent.providers.openrouter import OpenRouterProvider

__all__ = ["Provider", "OpenRouterProvider"]
