"""Configuration page for the Mutual Dissent web interface.

Renders an interactive form for editing application settings: debate defaults,
provider API keys, routing modes, and model alias mappings. Form state is
populated from the current Config and bound to NiceGUI inputs for two-way
reactivity.
"""

from __future__ import annotations

import os
from typing import Any

from nicegui import ui

from mutual_dissent.config import (
    _PROVIDER_ENV_MAP,
    DEFAULT_MODEL_ALIASES_V2,
    Config,
    load_config,
)

# Canonical list of providers shown in the UI.
_PROVIDERS = ["openrouter", "anthropic", "openai", "google", "xai", "groq"]

# Routing mode options for dropdowns.
_ROUTING_MODES = ["auto", "direct", "openrouter"]


def _build_form_state(config: Config) -> dict[str, Any]:
    """Extract all editable fields from Config into a dict for form binding.

    Creates a flat dictionary whose keys map directly to NiceGUI bind targets.
    Provider key sources are classified as ``"env"`` (from environment variable),
    ``"file"`` (from config file), or ``"none"`` (not configured).

    Args:
        config: The loaded Config instance to extract values from.

    Returns:
        Dictionary with keys: panel, synthesizer, rounds, providers,
        provider_sources, routing, aliases.
    """
    # --- Provider sources ---
    provider_sources: dict[str, str] = {}
    for provider in _PROVIDERS:
        env_var = _PROVIDER_ENV_MAP.get(provider, "")
        key_value = config.providers.get(provider, "")
        env_value = os.environ.get(env_var, "") if env_var else ""

        if key_value and env_value:
            provider_sources[provider] = "env"
        elif key_value:
            provider_sources[provider] = "file"
        else:
            provider_sources[provider] = "none"

    # --- Aliases: deep-copy v2 aliases ---
    aliases: dict[str, dict[str, str]] = {}
    for alias, ids in config._model_aliases_v2.items():
        aliases[alias] = dict(ids)

    return {
        "panel": list(config.default_panel),
        "synthesizer": config.default_synthesizer,
        "rounds": config.default_rounds,
        "providers": dict(config.providers),
        "provider_sources": provider_sources,
        "routing": dict(config.routing),
        "aliases": aliases,
    }


def _render_defaults_section(state: dict[str, Any]) -> None:
    """Render the Debate Defaults expansion panel.

    Contains controls for panel model selection, synthesizer choice,
    and number of reflection rounds.

    Args:
        state: Mutable form state dict with keys ``panel``, ``synthesizer``,
            and ``rounds``.
    """
    alias_options = list(DEFAULT_MODEL_ALIASES_V2.keys())

    with ui.expansion("Debate Defaults", icon="tune").classes("w-full"):
        with ui.column().classes("gap-4 p-4 w-full"):
            ui.select(
                label="Panel models",
                options=alias_options,
                multiple=True,
            ).bind_value(state, "panel").props("outlined dense").classes("w-full")

            ui.select(
                label="Synthesizer",
                options=alias_options,
            ).bind_value(state, "synthesizer").props("outlined dense").classes("w-full")

            ui.number(
                label="Reflection rounds",
                min=1,
                max=3,
                step=1,
            ).bind_value(state, "rounds").props("outlined dense").classes("w-full")


def _render_providers_section(state: dict[str, Any]) -> None:
    """Render the Provider API Keys expansion panel.

    Shows one row per provider with a password input and status indicator.
    Keys sourced from environment variables are displayed read-only.

    Args:
        state: Mutable form state dict with keys ``providers`` and
            ``provider_sources``.
    """
    with ui.expansion("Provider API Keys", icon="key").classes("w-full"):
        with ui.column().classes("gap-3 p-4 w-full"):
            for provider in _PROVIDERS:
                env_var = _PROVIDER_ENV_MAP.get(provider, "")
                source = state["provider_sources"].get(provider, "none")
                has_key = bool(state["providers"].get(provider, ""))

                with ui.row().classes("items-center gap-2 w-full"):
                    # Status icon
                    if has_key:
                        ui.icon("check_circle").classes("text-green-500")
                    else:
                        ui.icon("cancel").classes("text-red-500")

                    if source == "env":
                        # Read-only: key from environment variable
                        ui.input(
                            label=(f"{provider} (from environment: {env_var})"),
                            password=True,
                            password_toggle_button=True,
                            value=state["providers"].get(provider, ""),
                        ).props("readonly outlined dense").classes("flex-grow")
                    else:
                        # Editable: key from file or not set
                        hint = f"or set {env_var}" if env_var else ""
                        ui.input(
                            label=f"{provider}",
                            password=True,
                            password_toggle_button=True,
                        ).bind_value(state["providers"], provider).props(
                            "outlined dense" + (f' hint="{hint}"' if hint else "")
                        ).classes("flex-grow")


def _update_routing_override(state: dict[str, Any], alias: str, value: str) -> None:
    """Update or remove a per-alias routing override in form state.

    When the user selects ``"(use default)"``, the alias entry is removed
    from the routing dict so it inherits the default mode. Otherwise the
    selected routing mode is stored.

    Args:
        state: Mutable form state dict containing ``routing`` key.
        alias: Model alias being configured.
        value: Selected routing mode, or ``"(use default)"`` to clear.
    """
    if value == "(use default)":
        state["routing"].pop(alias, None)
    else:
        state["routing"][alias] = value


def _render_routing_section(state: dict[str, Any]) -> None:
    """Render the Routing Mode expansion panel.

    Contains a default routing mode selector and per-alias override rows.

    Args:
        state: Mutable form state dict with key ``routing``.
    """
    alias_keys = list(DEFAULT_MODEL_ALIASES_V2.keys())
    override_options = ["(use default)"] + _ROUTING_MODES

    with ui.expansion("Routing Mode", icon="alt_route").classes("w-full"):
        with ui.column().classes("gap-4 p-4 w-full"):
            ui.select(
                label="Default routing mode",
                options=_ROUTING_MODES,
            ).bind_value(state["routing"], "default_mode").props("outlined dense").classes("w-full")

            ui.separator()
            ui.label("Per-model overrides").classes("text-sm text-gray-400")

            for alias in alias_keys:
                current = state["routing"].get(alias, "(use default)")

                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(alias).classes("w-24 font-mono")
                    ui.select(
                        label=f"{alias} routing",
                        options=override_options,
                        value=current,
                        on_change=lambda e, a=alias: _update_routing_override(state, a, e.value),
                    ).props("outlined dense").classes("flex-grow")


def _render_aliases_section(state: dict[str, Any]) -> None:
    """Render the Model Aliases expansion panel.

    Displays a table of model aliases with editable OpenRouter and Direct
    model ID fields.

    Args:
        state: Mutable form state dict with key ``aliases``.
    """
    with ui.expansion("Model Aliases", icon="label").classes("w-full"):
        with ui.column().classes("gap-2 p-4 w-full"):
            # Header row
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label("Alias").classes("w-24 font-bold text-sm")
                ui.label("OpenRouter ID").classes("flex-grow font-bold text-sm")
                ui.label("Direct ID").classes("flex-grow font-bold text-sm")

            # Data rows
            for alias in sorted(state["aliases"].keys()):
                ids = state["aliases"][alias]
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(alias).classes("w-24 font-mono")
                    ui.input(
                        label="OpenRouter",
                    ).bind_value(ids, "openrouter").props("outlined dense").classes("flex-grow")
                    ui.input(
                        label="Direct",
                    ).bind_value(ids, "direct").props("outlined dense").classes("flex-grow")


def render() -> None:
    """Render the full configuration page.

    Loads the current config, builds reactive form state, and renders
    all four configuration sections: defaults, providers, routing, and
    model aliases.
    """
    config = load_config()
    state = _build_form_state(config)

    ui.label("Configuration").classes("text-2xl font-mono")

    with ui.column().classes("w-full max-w-4xl gap-2"):
        _render_defaults_section(state)
        _render_providers_section(state)
        _render_routing_section(state)
        _render_aliases_section(state)
