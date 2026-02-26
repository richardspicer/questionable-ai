"""Configuration management for Mutual Dissent.

Handles API key resolution, model alias mapping, and user-configurable defaults.
Configuration is loaded from TOML file (~/.mutual-dissent/config.toml) with
environment variable overrides.

Typical usage::

    from mutual_dissent.config import load_config

    config = load_config()
    api_key = config.api_key
    model_id = config.resolve_model("claude")
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path.home() / ".mutual-dissent"
CONFIG_PATH = APP_DIR / "config.toml"
TRANSCRIPT_DIR = APP_DIR / "transcripts"

# Default model aliases → OpenRouter model IDs.
# Verified against OpenRouter offerings as of 2026-02-21.
DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "claude": "anthropic/claude-sonnet-4.5",
    "gpt": "openai/gpt-5.2",
    "gemini": "google/gemini-2.5-pro",
    "grok": "x-ai/grok-4",
}

DEFAULT_PANEL = ["claude", "gpt", "gemini", "grok"]
DEFAULT_SYNTHESIZER = "claude"
DEFAULT_ROUNDS = 1
MAX_ROUNDS = 3


@dataclass
class Config:
    """Application configuration.

    Attributes:
        api_key: OpenRouter API key. Resolved from OPENROUTER_API_KEY env var
            or config file, in that priority order.
        model_aliases: Mapping of short names to OpenRouter model IDs.
        default_panel: Default list of model aliases for the debate panel.
        default_synthesizer: Default model alias for synthesis.
        default_rounds: Default number of reflection rounds.
    """

    api_key: str = ""
    model_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))
    default_panel: list[str] = field(default_factory=lambda: list(DEFAULT_PANEL))
    default_synthesizer: str = DEFAULT_SYNTHESIZER
    default_rounds: int = DEFAULT_ROUNDS

    def resolve_model(self, alias_or_id: str) -> str:
        """Resolve a model alias to an OpenRouter model ID.

        Args:
            alias_or_id: Either a short alias (e.g. "claude") or a full
                OpenRouter model ID (e.g. "anthropic/claude-sonnet-4.5").

        Returns:
            The OpenRouter model ID string.

        Raises:
            ValueError: If the alias is not found and doesn't look like a
                full model ID (i.e. doesn't contain a slash).
        """
        if alias_or_id in self.model_aliases:
            return self.model_aliases[alias_or_id]
        if "/" in alias_or_id:
            return alias_or_id
        raise ValueError(
            f"Unknown model alias '{alias_or_id}'. "
            f"Known aliases: {', '.join(sorted(self.model_aliases.keys()))}. "
            f"Or pass a full OpenRouter model ID (e.g. 'anthropic/claude-sonnet-4.5')."
        )

    def resolve_panel(self, panel: list[str]) -> list[str]:
        """Resolve a list of aliases/IDs to OpenRouter model IDs.

        Args:
            panel: List of model aliases or full OpenRouter model IDs.

        Returns:
            List of resolved OpenRouter model IDs.

        Raises:
            ValueError: If any alias cannot be resolved.
        """
        return [self.resolve_model(m) for m in panel]


def load_config() -> Config:
    """Load configuration from file and environment.

    Resolution order for API key:
        1. OPENROUTER_API_KEY environment variable
        2. api_key in config.toml
        3. Empty string (will fail at request time)

    Returns:
        Populated Config instance.
    """
    # One-time migration from old config path.
    old_app_dir = Path.home() / ".questionable-ai"
    if old_app_dir.exists() and not APP_DIR.exists():
        import shutil

        shutil.copytree(old_app_dir, APP_DIR)

    config = Config()

    # Load TOML config if it exists.
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)

        config.api_key = data.get("api_key", "")

        if "model_aliases" in data:
            config.model_aliases.update(data["model_aliases"])

        if "defaults" in data:
            defaults = data["defaults"]
            if "panel" in defaults:
                config.default_panel = defaults["panel"]
            if "synthesizer" in defaults:
                config.default_synthesizer = defaults["synthesizer"]
            if "rounds" in defaults:
                config.default_rounds = min(int(defaults["rounds"]), MAX_ROUNDS)

    # Environment variable overrides config file for API key.
    env_key = os.environ.get("OPENROUTER_API_KEY", "")
    if env_key:
        config.api_key = env_key

    return config


def ensure_dirs() -> None:
    """Create application directories if they don't exist."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
