"""Tests for Phase 1.5 config schema.

Covers: new schema loading, backward compatibility, env var override,
resolve_model with dual IDs, get_provider_key, routing mode access,
and write_config() serialization.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from mutual_dissent.config import Config, load_config, write_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PHASE_1_TOML = """\
api_key = "sk-or-phase1"

[model_aliases]
claude = "anthropic/claude-sonnet-4.5"
gpt = "openai/gpt-5.2"

[defaults]
panel = ["claude", "gpt"]
synthesizer = "claude"
rounds = 1
"""

PHASE_1_5_TOML = """\
api_key = "sk-or-legacy"

[providers]
openrouter_api_key = "sk-or-new"
anthropic_api_key = "sk-ant-test"

[routing]
default_mode = "auto"
claude = "direct"
gpt = "openrouter"
gemini = "auto"

[model_aliases]
claude.openrouter = "anthropic/claude-sonnet-4.5"
claude.direct = "claude-sonnet-4-5-20250929"
gpt.openrouter = "openai/gpt-5.2"
gpt.direct = "gpt-5.2"
gemini.openrouter = "google/gemini-2.5-pro"

[defaults]
panel = ["claude", "gpt", "gemini"]
synthesizer = "claude"
rounds = 2
"""

MIXED_TOML = """\
api_key = "sk-or-mixed"

[model_aliases]
claude.openrouter = "anthropic/claude-sonnet-4.5"
claude.direct = "claude-sonnet-4-5-20250929"
gpt = "openai/gpt-5.2"
"""

PROVIDERS_ONLY_TOML = """\
[providers]
openrouter_api_key = "sk-or-providers"
anthropic_api_key = "sk-ant-providers"
"""

LEGACY_API_KEY_ONLY_TOML = """\
api_key = "sk-or-legacy-only"
"""

BOTH_API_KEYS_TOML = """\
api_key = "sk-or-legacy"

[providers]
openrouter_api_key = "sk-or-providers-wins"
"""


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory."""
    return tmp_path / ".mutual-dissent"


def _write_config(config_dir: Path, content: str) -> Path:
    """Write TOML content to a config file in the given directory."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(content)
    return config_path


def _load_with_config(config_dir: Path, content: str, env: dict[str, str] | None = None) -> Config:
    """Write config and load it, patching CONFIG_PATH and env vars."""
    config_path = _write_config(config_dir, content)
    clean_env = {
        "OPENROUTER_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "XAI_API_KEY": "",
        "GROQ_API_KEY": "",
    }
    if env:
        clean_env.update(env)
    with (
        patch("mutual_dissent.config.CONFIG_PATH", config_path),
        patch.dict(os.environ, clean_env, clear=False),
    ):
        return load_config()


# ---------------------------------------------------------------------------
# Phase 1.5 full schema loading
# ---------------------------------------------------------------------------


class TestPhase15SchemaLoading:
    """load_config() correctly parses a full Phase 1.5 config.toml."""

    def test_providers_loaded(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.providers["openrouter"] == "sk-or-new"
        assert config.providers["anthropic"] == "sk-ant-test"

    def test_routing_loaded(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.routing["default_mode"] == "auto"
        assert config.routing["claude"] == "direct"
        assert config.routing["gpt"] == "openrouter"
        assert config.routing["gemini"] == "auto"

    def test_nested_model_aliases_loaded(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.resolve_model("claude") == "anthropic/claude-sonnet-4.5"
        assert config.resolve_model("claude", direct=True) == "claude-sonnet-4-5-20250929"
        assert config.resolve_model("gpt") == "openai/gpt-5.2"
        assert config.resolve_model("gpt", direct=True) == "gpt-5.2"

    def test_defaults_loaded(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.default_panel == ["claude", "gpt", "gemini"]
        assert config.default_synthesizer == "claude"
        assert config.default_rounds == 2

    def test_api_key_backward_compat(self, config_dir: Path) -> None:
        """config.api_key still works and returns OpenRouter key."""
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.api_key == "sk-or-new"


# ---------------------------------------------------------------------------
# Phase 1 backward compatibility
# ---------------------------------------------------------------------------


class TestPhase1BackwardCompat:
    """load_config() correctly parses a Phase 1 config.toml."""

    def test_flat_aliases_still_work(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.resolve_model("claude") == "anthropic/claude-sonnet-4.5"
        assert config.resolve_model("gpt") == "openai/gpt-5.2"

    def test_api_key_from_top_level(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.api_key == "sk-or-phase1"

    def test_providers_populated_from_legacy_key(self, config_dir: Path) -> None:
        """Legacy api_key should map to providers['openrouter']."""
        config = _load_with_config(config_dir, LEGACY_API_KEY_ONLY_TOML)
        assert config.providers["openrouter"] == "sk-or-legacy-only"

    def test_resolve_panel_still_works(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_TOML)
        ids = config.resolve_panel(["claude", "gpt"])
        assert ids == ["anthropic/claude-sonnet-4.5", "openai/gpt-5.2"]

    def test_flat_aliases_resolve_model_direct_falls_back(self, config_dir: Path) -> None:
        """Phase 1 flat alias: direct=True returns OpenRouter ID as fallback."""
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.resolve_model("claude", direct=True) == "anthropic/claude-sonnet-4.5"

    def test_defaults_from_phase1(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.default_panel == ["claude", "gpt"]
        assert config.default_synthesizer == "claude"
        assert config.default_rounds == 1


# ---------------------------------------------------------------------------
# Mixed aliases (some flat, some nested)
# ---------------------------------------------------------------------------


class TestMixedAliases:
    """Config with both flat (Phase 1) and nested (Phase 1.5) aliases."""

    def test_nested_alias_has_dual_ids(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, MIXED_TOML)
        assert config.resolve_model("claude") == "anthropic/claude-sonnet-4.5"
        assert config.resolve_model("claude", direct=True) == "claude-sonnet-4-5-20250929"

    def test_flat_alias_still_resolves(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, MIXED_TOML)
        assert config.resolve_model("gpt") == "openai/gpt-5.2"

    def test_flat_alias_direct_falls_back(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, MIXED_TOML)
        assert config.resolve_model("gpt", direct=True) == "openai/gpt-5.2"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    """Env vars override config file values for all provider keys."""

    def test_openrouter_env_overrides_config(self, config_dir: Path) -> None:
        config = _load_with_config(
            config_dir, PHASE_1_5_TOML, env={"OPENROUTER_API_KEY": "sk-or-env"}
        )
        assert config.providers["openrouter"] == "sk-or-env"
        assert config.api_key == "sk-or-env"

    def test_anthropic_env_overrides_config(self, config_dir: Path) -> None:
        config = _load_with_config(
            config_dir, PHASE_1_5_TOML, env={"ANTHROPIC_API_KEY": "sk-ant-env"}
        )
        assert config.providers["anthropic"] == "sk-ant-env"

    def test_openai_env_sets_provider(self, config_dir: Path) -> None:
        config = _load_with_config(
            config_dir, PHASE_1_5_TOML, env={"OPENAI_API_KEY": "sk-openai-env"}
        )
        assert config.providers["openai"] == "sk-openai-env"

    def test_google_env_sets_provider(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML, env={"GOOGLE_API_KEY": "goog-env"})
        assert config.providers["google"] == "goog-env"

    def test_xai_env_sets_provider(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML, env={"XAI_API_KEY": "xai-env"})
        assert config.providers["xai"] == "xai-env"

    def test_groq_env_sets_provider(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML, env={"GROQ_API_KEY": "groq-env"})
        assert config.providers["groq"] == "groq-env"

    def test_env_var_overrides_legacy_api_key(self, config_dir: Path) -> None:
        """Env var takes priority over both legacy api_key and providers section."""
        config = _load_with_config(
            config_dir, BOTH_API_KEYS_TOML, env={"OPENROUTER_API_KEY": "sk-or-env-wins"}
        )
        assert config.api_key == "sk-or-env-wins"
        assert config.providers["openrouter"] == "sk-or-env-wins"


# ---------------------------------------------------------------------------
# resolve_model()
# ---------------------------------------------------------------------------


class TestResolveModel:
    """resolve_model() with dual IDs."""

    def test_default_returns_openrouter_id(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.resolve_model("claude") == "anthropic/claude-sonnet-4.5"

    def test_direct_returns_vendor_native_id(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.resolve_model("claude", direct=True) == "claude-sonnet-4-5-20250929"

    def test_no_direct_id_falls_back_to_openrouter(self, config_dir: Path) -> None:
        """gemini has no .direct in PHASE_1_5_TOML, should fall back."""
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.resolve_model("gemini", direct=True) == "google/gemini-2.5-pro"

    def test_full_model_id_passthrough(self, config_dir: Path) -> None:
        """Full model IDs with slash pass through unchanged."""
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.resolve_model("anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet-4.5"

    def test_unknown_alias_raises(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        with pytest.raises(ValueError, match="Unknown model alias"):
            config.resolve_model("nonexistent")

    def test_resolve_model_no_config_file(self) -> None:
        """Default config should resolve built-in aliases."""
        fake_path = Path("/nonexistent/config.toml")
        clean_env = {
            "OPENROUTER_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "XAI_API_KEY": "",
            "GROQ_API_KEY": "",
        }
        with (
            patch("mutual_dissent.config.CONFIG_PATH", fake_path),
            patch.dict(os.environ, clean_env, clear=False),
        ):
            config = load_config()
        assert config.resolve_model("claude") == "anthropic/claude-sonnet-4.5"


# ---------------------------------------------------------------------------
# get_provider_key()
# ---------------------------------------------------------------------------


class TestGetProviderKey:
    """get_provider_key() returns key from config or env, None if missing."""

    def test_returns_key_from_config(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.get_provider_key("openrouter") == "sk-or-new"

    def test_returns_key_from_config_anthropic(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.get_provider_key("anthropic") == "sk-ant-test"

    def test_returns_none_when_not_configured(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.get_provider_key("anthropic") is None

    def test_env_var_fallback(self, config_dir: Path) -> None:
        """get_provider_key checks env var if not in providers dict."""
        config = _load_with_config(
            config_dir, PHASE_1_TOML, env={"ANTHROPIC_API_KEY": "sk-ant-env-fallback"}
        )
        assert config.get_provider_key("anthropic") == "sk-ant-env-fallback"

    def test_returns_none_for_unknown_vendor(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.get_provider_key("unknown_vendor") is None

    def test_empty_string_treated_as_missing(self, config_dir: Path) -> None:
        """Empty string in config is not a valid key — returns None."""
        toml_content = """\
[providers]
openrouter_api_key = ""
"""
        config = _load_with_config(config_dir, toml_content)
        assert config.get_provider_key("openrouter") is None


# ---------------------------------------------------------------------------
# Routing mode access
# ---------------------------------------------------------------------------


class TestRoutingMode:
    """Routing mode per alias is accessible from Config."""

    def test_routing_mode_for_alias(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.routing["claude"] == "direct"
        assert config.routing["gpt"] == "openrouter"

    def test_default_routing_mode(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, PHASE_1_5_TOML)
        assert config.routing["default_mode"] == "auto"

    def test_no_routing_section_defaults_to_auto(self, config_dir: Path) -> None:
        """Phase 1 config without [routing] should default to auto."""
        config = _load_with_config(config_dir, PHASE_1_TOML)
        assert config.routing["default_mode"] == "auto"


# ---------------------------------------------------------------------------
# Provider key precedence: [providers] wins over top-level api_key
# ---------------------------------------------------------------------------


class TestApiKeyPrecedence:
    """When both legacy api_key and [providers] exist, providers wins."""

    def test_providers_openrouter_wins_over_legacy(self, config_dir: Path) -> None:
        config = _load_with_config(config_dir, BOTH_API_KEYS_TOML)
        assert config.api_key == "sk-or-providers-wins"
        assert config.providers["openrouter"] == "sk-or-providers-wins"


# ---------------------------------------------------------------------------
# write_config() serialization
# ---------------------------------------------------------------------------


class TestWriteConfig:
    """write_config() serializes Config to TOML and roundtrips correctly."""

    def _load_roundtrip(self, config_path: Path) -> Config:
        """Load config from a written file, suppressing env var contamination."""
        clean_env = {
            "OPENROUTER_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "XAI_API_KEY": "",
            "GROQ_API_KEY": "",
        }
        with (
            patch("mutual_dissent.config.CONFIG_PATH", config_path),
            patch.dict(os.environ, clean_env, clear=False),
        ):
            return load_config()

    def test_roundtrip_defaults(self, tmp_path: Path) -> None:
        """write_config then load_config reproduces defaults (panel, synthesizer, rounds)."""
        config = Config(
            default_panel=["claude", "gpt"],
            default_synthesizer="gpt",
            default_rounds=2,
        )
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path)

        loaded = self._load_roundtrip(config_path)

        assert loaded.default_panel == ["claude", "gpt"]
        assert loaded.default_synthesizer == "gpt"
        assert loaded.default_rounds == 2

    def test_roundtrip_providers(self, tmp_path: Path) -> None:
        """write_config preserves provider keys (not env-sourced ones)."""
        config = Config(
            providers={"openrouter": "sk-or-test", "anthropic": "sk-ant-test"},
        )
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path)

        loaded = self._load_roundtrip(config_path)

        assert loaded.providers["openrouter"] == "sk-or-test"
        assert loaded.providers["anthropic"] == "sk-ant-test"

    def test_roundtrip_routing(self, tmp_path: Path) -> None:
        """write_config preserves routing config."""
        config = Config(
            routing={"default_mode": "auto", "claude": "direct", "gpt": "openrouter"},
        )
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path)

        loaded = self._load_roundtrip(config_path)

        assert loaded.routing["default_mode"] == "auto"
        assert loaded.routing["claude"] == "direct"
        assert loaded.routing["gpt"] == "openrouter"

    def test_roundtrip_model_aliases_v2(self, tmp_path: Path) -> None:
        """write_config preserves dual-ID model aliases."""
        aliases_v2 = {
            "claude": {
                "openrouter": "anthropic/claude-sonnet-4.5",
                "direct": "claude-sonnet-4-5-20250929",
            },
            "gpt": {
                "openrouter": "openai/gpt-5.2",
                "direct": "gpt-5.2",
            },
        }
        config = Config(_model_aliases_v2=aliases_v2)
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path)

        loaded = self._load_roundtrip(config_path)

        assert loaded.resolve_model("claude") == "anthropic/claude-sonnet-4.5"
        assert loaded.resolve_model("claude", direct=True) == "claude-sonnet-4-5-20250929"
        assert loaded.resolve_model("gpt") == "openai/gpt-5.2"
        assert loaded.resolve_model("gpt", direct=True) == "gpt-5.2"

    def test_skips_env_sourced_keys(self, tmp_path: Path) -> None:
        """write_config excludes keys that came from environment variables."""
        config = Config(
            providers={
                "openrouter": "sk-or-from-env",
                "anthropic": "sk-ant-from-file",
            },
        )
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path, env_providers={"openrouter"})

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # openrouter should be excluded, anthropic should be present.
        providers = data.get("providers", {})
        assert "openrouter_api_key" not in providers
        assert providers["anthropic_api_key"] == "sk-ant-from-file"

    def test_produces_valid_toml(self, tmp_path: Path) -> None:
        """write_config output is parseable by tomllib and has expected sections."""
        config = Config(
            providers={"openrouter": "sk-or-test"},
            routing={"default_mode": "auto"},
            default_panel=["claude", "gpt"],
            default_synthesizer="claude",
            default_rounds=1,
        )
        config_path = tmp_path / "config.toml"
        write_config(config, path=config_path)

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        assert "providers" in data
        assert "routing" in data
        assert "model_aliases" in data
        assert "defaults" in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_config creates parent directories if needed."""
        nested_path = tmp_path / "a" / "b" / "c" / "config.toml"
        config = Config()
        write_config(config, path=nested_path)

        assert nested_path.exists()
        # Verify it's valid TOML.
        with open(nested_path, "rb") as f:
            data = tomllib.load(f)
        assert "defaults" in data

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not supported on Windows")
    def test_preserves_file_permissions(self, tmp_path: Path) -> None:
        """write_config preserves existing file permissions on overwrite."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        config_path.chmod(0o600)

        config = Config()
        write_config(config, path=config_path)

        mode = config_path.stat().st_mode & 0o777
        assert mode == 0o600
