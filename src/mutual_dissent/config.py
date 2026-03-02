"""Configuration management for Mutual Dissent.

Handles API key resolution, model alias mapping, provider routing config,
and user-configurable defaults. Configuration is loaded from TOML file
(~/.mutual-dissent/config.toml) with environment variable overrides.

Supports both Phase 1 (flat aliases, single API key) and Phase 1.5
(multi-provider keys, per-model routing, dual model IDs) config formats.

Typical usage::

    from mutual_dissent.config import load_config

    config = load_config()
    api_key = config.api_key
    model_id = config.resolve_model("claude")
    direct_id = config.resolve_model("claude", direct=True)
    key = config.get_provider_key("anthropic")
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

# Phase 1.5 dual-ID aliases: alias → {"openrouter": ..., "direct": ...}.
# The "direct" key is the vendor-native model ID for direct API calls.
DEFAULT_MODEL_ALIASES_V2: dict[str, dict[str, str]] = {
    "claude": {
        "openrouter": "anthropic/claude-sonnet-4.5",
        "direct": "claude-sonnet-4-5-20250929",
    },
    "gpt": {
        "openrouter": "openai/gpt-5.2",
        "direct": "gpt-5.2",
    },
    "gemini": {
        "openrouter": "google/gemini-2.5-pro",
    },
    "grok": {
        "openrouter": "x-ai/grok-4",
    },
}

DEFAULT_PANEL = ["claude", "gpt", "gemini", "grok"]
DEFAULT_SYNTHESIZER = "claude"
DEFAULT_ROUNDS = 1
MAX_ROUNDS = 3

# Env var name → provider key in the providers dict.
_ENV_VAR_MAP: dict[str, str] = {
    "OPENROUTER_API_KEY": "openrouter",
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
    "GOOGLE_API_KEY": "google",
    "XAI_API_KEY": "xai",
    "GROQ_API_KEY": "groq",
}

# Reverse: provider key → env var name.
_PROVIDER_ENV_MAP: dict[str, str] = {v: k for k, v in _ENV_VAR_MAP.items()}


@dataclass
class Config:
    """Application configuration.

    Attributes:
        api_key: OpenRouter API key. Resolved from providers["openrouter"],
            legacy top-level api_key, or OPENROUTER_API_KEY env var.
        providers: Mapping of provider name to API key
            (e.g. {"openrouter": "sk-or-...", "anthropic": "sk-ant-..."}).
        routing: Per-alias routing mode. Keys are model aliases or
            "default_mode". Values are "auto", "direct", or "openrouter".
        model_aliases: Mapping of short names to OpenRouter model IDs.
            Maintained for backward compatibility with Phase 1 callers.
        _model_aliases_v2: Internal dual-ID alias map. alias → {"openrouter": ..., "direct": ...}.
        default_panel: Default list of model aliases for the debate panel.
        default_synthesizer: Default model alias for synthesis.
        default_rounds: Default number of reflection rounds.
    """

    api_key: str = ""
    providers: dict[str, str] = field(default_factory=dict)
    routing: dict[str, str] = field(default_factory=lambda: {"default_mode": "auto"})
    model_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))
    _model_aliases_v2: dict[str, dict[str, str]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_MODEL_ALIASES_V2.items()},
        repr=False,
    )
    default_panel: list[str] = field(default_factory=lambda: list(DEFAULT_PANEL))
    default_synthesizer: str = DEFAULT_SYNTHESIZER
    default_rounds: int = DEFAULT_ROUNDS

    def resolve_model(self, alias_or_id: str, *, direct: bool = False) -> str:
        """Resolve a model alias to a model ID.

        By default returns the OpenRouter model ID. With ``direct=True``,
        returns the vendor-native model ID for direct API calls. Falls back
        to the OpenRouter ID if no direct ID is configured.

        Args:
            alias_or_id: Either a short alias (e.g. "claude") or a full
                OpenRouter model ID (e.g. "anthropic/claude-sonnet-4.5").
            direct: If True, return the vendor-native model ID instead of
                the OpenRouter ID.

        Returns:
            The resolved model ID string.

        Raises:
            ValueError: If the alias is not found and doesn't look like a
                full model ID (i.e. doesn't contain a slash).
        """
        if alias_or_id in self._model_aliases_v2:
            ids = self._model_aliases_v2[alias_or_id]
            if direct:
                return ids.get("direct", ids.get("openrouter", ""))
            return ids.get("openrouter", "")
        if "/" in alias_or_id:
            return alias_or_id
        raise ValueError(
            f"Unknown model alias '{alias_or_id}'. "
            f"Known aliases: {', '.join(sorted(self._model_aliases_v2.keys()))}. "
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

    def get_provider_key(self, vendor: str) -> str | None:
        """Get the API key for a given provider.

        Checks the ``providers`` dict first, then falls back to the
        corresponding environment variable. Returns None if no key is
        configured.

        Note:
            The env var fallback is defensive — ``load_config()`` already
            applies env overrides to ``providers`` via ``_apply_env_overrides()``.
            The fallback here covers manually constructed ``Config()`` instances
            (e.g. in tests) that bypass ``load_config()``.

        Args:
            vendor: Provider name (e.g. "openrouter", "anthropic", "openai").

        Returns:
            The API key string, or None if not configured.
        """
        key = self.providers.get(vendor, "")
        if key:
            return key
        env_var = _PROVIDER_ENV_MAP.get(vendor)
        if env_var:
            env_val = os.environ.get(env_var, "")
            if env_val:
                return env_val
        return None


def _normalize_aliases(
    raw: dict[str, str | dict[str, str]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Normalize model aliases from TOML into both flat and v2 formats.

    Handles mixed configs where some aliases are flat strings (Phase 1)
    and others are nested dicts (Phase 1.5).

    Args:
        raw: Raw model_aliases from TOML. Values are either strings
            (Phase 1: "anthropic/claude-sonnet-4.5") or dicts
            (Phase 1.5: {"openrouter": "...", "direct": "..."}).

    Returns:
        Tuple of (flat_aliases, v2_aliases) where flat_aliases maps
        alias → OpenRouter ID and v2_aliases maps alias → {"openrouter": ..., "direct": ...}.
    """
    flat: dict[str, str] = {}
    v2: dict[str, dict[str, str]] = {}
    for alias, value in raw.items():
        if isinstance(value, str):
            flat[alias] = value
            v2[alias] = {"openrouter": value}
        elif isinstance(value, dict):
            v2[alias] = dict(value)
            flat[alias] = value.get("openrouter", "")
    return flat, v2


def _apply_toml(config: Config, data: dict[str, Any]) -> None:
    """Apply parsed TOML data to a Config instance.

    Handles both Phase 1 (flat aliases, single api_key) and Phase 1.5
    (multi-provider keys, routing, dual model IDs) config formats.

    Args:
        config: Config instance to populate.
        data: Parsed TOML dictionary.
    """
    # Legacy top-level api_key.
    legacy_api_key: str = data.get("api_key", "")

    # --- Providers ---
    if "providers" in data:
        for toml_key, value in data["providers"].items():
            # Keys are like "openrouter_api_key" → strip "_api_key" suffix.
            if toml_key.endswith("_api_key") and value:
                config.providers[toml_key[: -len("_api_key")]] = value

    # Legacy api_key → providers["openrouter"] (only if not already set).
    if legacy_api_key and "openrouter" not in config.providers:
        config.providers["openrouter"] = legacy_api_key

    # Set api_key from providers["openrouter"] or legacy.
    config.api_key = config.providers.get("openrouter", legacy_api_key)

    # --- Routing ---
    if "routing" in data:
        config.routing = {"default_mode": "auto"}
        config.routing.update(data["routing"])

    # --- Model aliases ---
    if "model_aliases" in data:
        flat, v2 = _normalize_aliases(data["model_aliases"])
        config.model_aliases.update(flat)
        config._model_aliases_v2.update(v2)

    # --- Defaults ---
    if "defaults" in data:
        defaults: dict[str, Any] = data["defaults"]
        if "panel" in defaults:
            config.default_panel = defaults["panel"]
        if "synthesizer" in defaults:
            config.default_synthesizer = defaults["synthesizer"]
        if "rounds" in defaults:
            config.default_rounds = min(int(defaults["rounds"]), MAX_ROUNDS)


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides to provider keys.

    Args:
        config: Config instance to update.
    """
    for env_var, provider_name in _ENV_VAR_MAP.items():
        env_val = os.environ.get(env_var, "")
        if env_val:
            config.providers[provider_name] = env_val

    # OPENROUTER_API_KEY env var also sets api_key for backward compat.
    openrouter_key = config.providers.get("openrouter", "")
    if openrouter_key:
        config.api_key = openrouter_key


def load_config() -> Config:
    """Load configuration from file and environment.

    Supports both Phase 1 (flat aliases, single api_key) and Phase 1.5
    (multi-provider keys, routing, dual model IDs) config formats.

    Resolution order for OpenRouter API key:
        1. OPENROUTER_API_KEY environment variable
        2. [providers].openrouter_api_key in config.toml
        3. Top-level api_key in config.toml (legacy)
        4. Empty string (will fail at request time)

    Returns:
        Populated Config instance.
    """
    # One-time migration from old config path.
    old_app_dir = Path.home() / ".questionable-ai"
    if old_app_dir.exists() and not APP_DIR.exists():
        import shutil

        shutil.copytree(old_app_dir, APP_DIR)

    config = Config()

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            _apply_toml(config, tomllib.load(f))

    _apply_env_overrides(config)

    return config


def write_config(
    config: Config,
    path: Path | None = None,
    *,
    env_providers: set[str] | None = None,
) -> None:
    """Serialize a Config to TOML and write to disk.

    Builds a Phase 1.5 format TOML document from the Config dataclass.
    Provider keys sourced from environment variables are excluded -- only
    values the user explicitly set through the UI should be persisted.

    If the file already exists, its permissions are preserved after write.

    Args:
        config: Config instance to serialize.
        path: File path to write. Defaults to CONFIG_PATH.
        env_providers: Set of provider names whose keys came from env vars
            and should be excluded from the written file.
    """
    import tomlkit

    target = path or CONFIG_PATH
    env_provs = env_providers or set()

    # Capture existing permissions before overwriting.
    existing_mode: int | None = None
    if target.exists():
        existing_mode = target.stat().st_mode & 0o777

    doc = tomlkit.document()

    # --- Providers ---
    providers_table = tomlkit.table()
    for provider_name, key_value in sorted(config.providers.items()):
        if provider_name in env_provs:
            continue
        if key_value:
            providers_table.add(f"{provider_name}_api_key", key_value)
    doc.add("providers", providers_table)

    # --- Routing ---
    routing_table = tomlkit.table()
    for rkey, rval in sorted(config.routing.items()):
        routing_table.add(rkey, rval)
    doc.add("routing", routing_table)

    # --- Model aliases (Phase 1.5 nested format) ---
    # Only _model_aliases_v2 is serialized; the flat model_aliases dict
    # is derived from it during load and is not a source of truth.
    aliases_table = tomlkit.table()
    for alias, ids in sorted(config._model_aliases_v2.items()):
        alias_sub = tomlkit.table()
        for id_key, id_val in sorted(ids.items()):
            alias_sub.add(id_key, id_val)
        aliases_table.add(alias, alias_sub)
    doc.add("model_aliases", aliases_table)

    # --- Defaults ---
    defaults_table = tomlkit.table()
    defaults_table.add("panel", config.default_panel)
    defaults_table.add("synthesizer", config.default_synthesizer)
    defaults_table.add("rounds", config.default_rounds)
    doc.add("defaults", defaults_table)

    # Write: parent dirs, then file.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tomlkit.dumps(doc), encoding="utf-8")

    # Restore permissions if file existed before.
    if existing_mode is not None:
        target.chmod(existing_mode)


def ensure_dirs() -> None:
    """Create application directories if they don't exist."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
