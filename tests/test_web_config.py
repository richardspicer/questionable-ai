"""Tests for the config page form logic.

Covers: form state population from Config, defaults section behavior.
Does NOT start NiceGUI (tests pure logic helpers).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from mutual_dissent.config import DEFAULT_MODEL_ALIASES_V2, Config


class TestBuildFormState:
    """_build_form_state() populates form data from Config."""

    def test_defaults_populated(self) -> None:
        """Form state includes default panel, synthesizer, and rounds."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        cfg.default_panel = ["claude", "gpt"]
        cfg.default_synthesizer = "gpt"
        cfg.default_rounds = 2

        state = _build_form_state(cfg)

        assert state["panel"] == ["claude", "gpt"]
        assert state["synthesizer"] == "gpt"
        assert state["rounds"] == 2

    def test_aliases_populated(self) -> None:
        """Form state includes all model aliases with both IDs."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        state = _build_form_state(cfg)

        assert "aliases" in state
        for alias in DEFAULT_MODEL_ALIASES_V2:
            assert alias in state["aliases"]
            assert "openrouter" in state["aliases"][alias]


class TestProviderSources:
    """_build_form_state() correctly identifies provider key sources."""

    def test_env_source_detected(self) -> None:
        """Provider with env var set is marked as 'env'."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        cfg.providers = {"anthropic": "sk-ant-from-env"}

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-from-env"}):
            state = _build_form_state(cfg)

        assert state["provider_sources"]["anthropic"] == "env"

    def test_file_source_detected(self) -> None:
        """Provider with key in config but not env is marked as 'file'."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        cfg.providers = {"anthropic": "sk-ant-from-file"}

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            state = _build_form_state(cfg)

        assert state["provider_sources"]["anthropic"] == "file"

    def test_none_source_for_missing(self) -> None:
        """Provider with no key is marked as 'none'."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        cfg.providers = {}

        # Clear all provider env vars so none are detected
        env_patch = {
            "OPENROUTER_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "XAI_API_KEY": "",
            "GROQ_API_KEY": "",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            state = _build_form_state(cfg)

        assert state["provider_sources"]["openrouter"] == "none"


class TestRoutingState:
    """Form state routing section populated correctly."""

    def test_routing_includes_default_mode(self) -> None:
        """Routing state includes default_mode."""
        from mutual_dissent.web.pages.config import _build_form_state

        cfg = Config()
        cfg.routing = {"default_mode": "direct", "claude": "direct"}
        state = _build_form_state(cfg)

        assert state["routing"]["default_mode"] == "direct"
        assert state["routing"]["claude"] == "direct"
