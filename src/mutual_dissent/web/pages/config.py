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
    write_config,
)
from mutual_dissent.models import ModelResponse
from mutual_dissent.types import RoutingDecision

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


# OpenRouter model-ID prefix → provider key in the config providers dict.
# Used by validation to check whether a panel model has a routable key.
_OR_PREFIX_TO_PROVIDER: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "x-ai": "xai",
    "groq": "groq",
}


def _validate_form_state(
    state: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Check form state for errors and warnings before saving.

    Errors block the save operation; warnings are shown but do not
    prevent saving.

    Args:
        state: The current form state dict with keys ``providers``,
            ``provider_sources``, ``panel``, and ``aliases``.

    Returns:
        Tuple of (errors, warnings) where each is a list of
        human-readable message strings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Error: no API key configured for any provider ---
    has_any_key = False
    for provider in _PROVIDERS:
        source = state["provider_sources"].get(provider, "none")
        key_val = state["providers"].get(provider, "")
        if source == "env" or key_val:
            has_any_key = True
            break
    if not has_any_key:
        errors.append(
            "No API key configured for any provider. Set at least one provider key before saving."
        )

    # --- Warning: panel model with no route ---
    providers_dict: dict[str, str] = state.get("providers", {})
    sources_dict: dict[str, str] = state.get("provider_sources", {})
    aliases_dict: dict[str, dict[str, str]] = state.get("aliases", {})
    has_openrouter = bool(
        providers_dict.get("openrouter", "") or sources_dict.get("openrouter") == "env"
    )

    for alias in state.get("panel", []):
        ids = aliases_dict.get(alias, {})
        or_id = ids.get("openrouter", "")
        prefix = or_id.split("/")[0] if "/" in or_id else ""
        vendor_provider = _OR_PREFIX_TO_PROVIDER.get(prefix, "")

        has_direct_key = (
            bool(
                providers_dict.get(vendor_provider, "")
                or sources_dict.get(vendor_provider) == "env"
            )
            if vendor_provider
            else False
        )

        if not has_openrouter and not has_direct_key:
            warnings.append(
                f"Panel model '{alias}' may have no route: "
                f"no OpenRouter key and no direct key for its vendor."
            )

    return errors, warnings


def _apply_form_to_config(state: dict[str, Any]) -> Config:
    """Build a Config instance from the current form state.

    Translates the flat form state dictionary back into a fully
    populated Config dataclass. Only non-empty provider keys are
    included. The flat ``model_aliases`` dict is derived from the
    v2 aliases for backward compatibility.

    Args:
        state: The current form state dict with keys ``panel``,
            ``synthesizer``, ``rounds``, ``providers``,
            ``routing``, and ``aliases``.

    Returns:
        A new Config instance reflecting the form state.
    """
    # Filter out empty provider keys.
    providers = {k: v for k, v in state.get("providers", {}).items() if v}

    # Build v2 aliases from form state.
    v2_aliases: dict[str, dict[str, str]] = {}
    for alias, ids in state.get("aliases", {}).items():
        v2_aliases[alias] = dict(ids)

    # Derive flat aliases from v2 (openrouter IDs only).
    flat_aliases: dict[str, str] = {
        alias: ids.get("openrouter", "") for alias, ids in v2_aliases.items()
    }

    cfg = Config(
        api_key=providers.get("openrouter", ""),
        providers=providers,
        routing=dict(state.get("routing", {"default_mode": "auto"})),
        model_aliases=flat_aliases,
        _model_aliases_v2=v2_aliases,
        default_panel=list(state.get("panel", [])),
        default_synthesizer=state.get("synthesizer", "claude"),
        default_rounds=int(state.get("rounds", 1)),
    )
    return cfg


async def _handle_save(state: dict[str, Any]) -> None:
    """Validate form state and write configuration to disk.

    Runs validation first. If errors are found, shows a negative
    notification and aborts. Warnings are shown but do not block
    the save. On success, calls ``write_config()`` and shows a
    success notification.

    Args:
        state: The current form state dict.
    """
    errors, warnings = _validate_form_state(state)

    if errors:
        for err in errors:
            ui.notify(err, type="negative")
        return

    for warn in warnings:
        ui.notify(warn, type="warning")

    # Determine which providers came from env vars (exclude from file).
    env_providers: set[str] = {
        provider
        for provider, source in state.get("provider_sources", {}).items()
        if source == "env"
    }

    cfg = _apply_form_to_config(state)
    write_config(cfg, env_providers=env_providers)
    ui.notify("Configuration saved.", type="positive")


async def _handle_test_providers(
    state: dict[str, Any],
    results_container: ui.column,
) -> None:
    """Test provider connectivity using the current form state.

    Builds a temporary Config from form state, collects unique model
    aliases from the panel and synthesizer, then calls
    ``_run_config_test()`` to send a test prompt to each. Results are
    rendered in the provided container with alias, vendor, route,
    model ID, latency, and status.

    Args:
        state: The current form state dict.
        results_container: NiceGUI column to render test results into.
    """
    from mutual_dissent.cli import _run_config_test

    results_container.clear()

    cfg = _apply_form_to_config(state)

    # Collect unique aliases: panel + synthesizer.
    panel = list(state.get("panel", []))
    synthesizer = state.get("synthesizer", "")
    aliases = list(dict.fromkeys(panel + ([synthesizer] if synthesizer else [])))

    if not aliases:
        with results_container:
            ui.label("No models to test.").classes("text-orange-400")
        return

    # Show spinner while testing.
    with results_container:
        ui.spinner("dots", size="lg")

    try:
        results = await _run_config_test(cfg, aliases)
    except Exception as exc:
        results_container.clear()
        with results_container:
            ui.label(f"Error: {exc}").classes("text-red-500 font-bold")
        return

    # Clear spinner, render results.
    results_container.clear()
    with results_container:
        # Header row
        with ui.row().classes("items-center gap-4 w-full"):
            ui.label("Alias").classes("w-20 font-bold text-sm")
            ui.label("Vendor").classes("w-24 font-bold text-sm")
            ui.label("Route").classes("w-24 font-bold text-sm")
            ui.label("Model ID").classes("flex-grow font-bold text-sm")
            ui.label("Latency").classes("w-16 font-bold text-sm text-right")
            ui.label("Status").classes("w-8 font-bold text-sm")

        for result in results:
            alias = str(result["alias"])
            decision = result["decision"]
            response = result["response"]

            assert isinstance(decision, RoutingDecision)
            assert isinstance(response, ModelResponse)

            vendor_str = decision.vendor.value
            route_str = "openrouter" if decision.via_openrouter else "direct"
            model_id = response.model_id
            latency_ms = response.latency_ms
            error = response.error

            if error:
                latency_str = "\u2014"
                status_icon = "cancel"
                status_class = "text-red-500"
            else:
                latency_str = f"{latency_ms / 1000:.1f}s" if latency_ms is not None else "\u2014"
                status_icon = "check_circle"
                status_class = "text-green-500"

            with ui.row().classes("items-center gap-4 w-full"):
                ui.label(alias).classes("w-20 font-mono")
                ui.label(vendor_str).classes("w-24")
                ui.label(route_str).classes("w-24")
                ui.label(model_id).classes("flex-grow text-gray-400 text-sm")
                ui.label(latency_str).classes("w-16 text-right")
                ui.icon(status_icon).classes(status_class)

            if error:
                ui.label(f"  {error}").classes("text-red-400 text-sm ml-8")


def render() -> None:
    """Render the full configuration page.

    Loads the current config, builds reactive form state, and renders
    all four configuration sections (defaults, providers, routing,
    model aliases) plus Save and Test Providers action buttons.
    """
    config = load_config()
    state = _build_form_state(config)

    ui.label("Configuration").classes("text-2xl font-mono")

    with ui.column().classes("w-full max-w-4xl gap-2"):
        _render_defaults_section(state)
        _render_providers_section(state)
        _render_routing_section(state)
        _render_aliases_section(state)

        ui.separator()

        with ui.row().classes("w-full justify-between items-center"):
            ui.button(
                "Save",
                icon="save",
                on_click=lambda: _handle_save(state),
            ).props("color=primary")
            ui.button(
                "Test Providers",
                icon="science",
                on_click=lambda: _handle_test_providers(state, test_results),
            ).props("color=secondary outlined")

        # Results container placed after buttons — the lambda captures the
        # variable name, and by the time the user clicks, it's assigned.
        test_results: ui.column = ui.column().classes("w-full mt-4")
