"""Tests for core routing types.

Covers: Vendor enum values and serialization, RoutedRequest construction,
RoutingDecision construction and to_dict(), and vendor value alignment
with config provider keys.
"""

from __future__ import annotations

import json

from mutual_dissent.config import _PROVIDER_ENV_MAP
from mutual_dissent.types import RoutedRequest, RoutingDecision, Vendor

# ---------------------------------------------------------------------------
# Vendor enum
# ---------------------------------------------------------------------------


class TestVendor:
    """Vendor enum has correct values and serialization behavior."""

    def test_all_seven_members(self) -> None:
        assert len(Vendor) == 7

    def test_expected_values(self) -> None:
        assert Vendor.ANTHROPIC.value == "anthropic"
        assert Vendor.OPENAI.value == "openai"
        assert Vendor.GOOGLE.value == "google"
        assert Vendor.XAI.value == "xai"
        assert Vendor.GROQ.value == "groq"
        assert Vendor.OPENROUTER.value == "openrouter"
        assert Vendor.OLLAMA.value == "ollama"

    def test_str_serialization(self) -> None:
        """Vendor inherits str, so str() and f-strings work naturally."""
        assert str(Vendor.ANTHROPIC) == "Vendor.ANTHROPIC" or Vendor.ANTHROPIC == "anthropic"
        assert f"{Vendor.ANTHROPIC.value}" == "anthropic"

    def test_equality_with_string(self) -> None:
        """str enum members compare equal to their string values."""
        assert Vendor.ANTHROPIC == "anthropic"
        assert Vendor.OPENROUTER == "openrouter"

    def test_json_serializable_via_value(self) -> None:
        """Vendor.value is a plain string, JSON-serializable."""
        data = {"vendor": Vendor.ANTHROPIC.value}
        result = json.dumps(data)
        assert '"anthropic"' in result

    def test_lookup_by_value(self) -> None:
        assert Vendor("anthropic") is Vendor.ANTHROPIC
        assert Vendor("ollama") is Vendor.OLLAMA


# ---------------------------------------------------------------------------
# Vendor alignment with config._PROVIDER_ENV_MAP
# ---------------------------------------------------------------------------


class TestVendorConfigAlignment:
    """Vendor enum values align with config.py provider keys."""

    def test_all_provider_env_map_keys_are_valid_vendors(self) -> None:
        """Every provider key in _PROVIDER_ENV_MAP must be a valid Vendor."""
        vendor_values = {v.value for v in Vendor}
        for provider_key in _PROVIDER_ENV_MAP:
            assert provider_key in vendor_values, (
                f"Provider key '{provider_key}' from _PROVIDER_ENV_MAP is not a valid Vendor value"
            )

    def test_vendor_lookup_from_provider_keys(self) -> None:
        """Vendor(key) should succeed for every _PROVIDER_ENV_MAP key."""
        for provider_key in _PROVIDER_ENV_MAP:
            vendor = Vendor(provider_key)
            assert vendor.value == provider_key


# ---------------------------------------------------------------------------
# RoutedRequest
# ---------------------------------------------------------------------------


class TestRoutedRequest:
    """RoutedRequest dataclass construction."""

    def test_construction(self) -> None:
        req = RoutedRequest(
            vendor=Vendor.ANTHROPIC,
            model_id="claude-sonnet-4-5-20250929",
            model_alias="claude",
            round_number=0,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.vendor is Vendor.ANTHROPIC
        assert req.model_id == "claude-sonnet-4-5-20250929"
        assert req.model_alias == "claude"
        assert req.round_number == 0
        assert len(req.messages) == 1

    def test_synthesis_round_number(self) -> None:
        """round_number -1 indicates synthesis."""
        req = RoutedRequest(
            vendor=Vendor.OPENAI,
            model_id="gpt-5.2",
            model_alias="gpt",
            round_number=-1,
            messages=[],
        )
        assert req.round_number == -1

    def test_messages_format(self) -> None:
        """Messages follow OpenAI chat format."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        req = RoutedRequest(
            vendor=Vendor.GOOGLE,
            model_id="gemini-2.5-pro",
            model_alias="gemini",
            round_number=1,
            messages=messages,
        )
        assert req.messages[0]["role"] == "system"
        assert req.messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    """RoutingDecision dataclass construction and serialization."""

    def test_construction(self) -> None:
        decision = RoutingDecision(
            vendor=Vendor.ANTHROPIC,
            mode="direct",
            via_openrouter=False,
        )
        assert decision.vendor is Vendor.ANTHROPIC
        assert decision.mode == "direct"
        assert decision.via_openrouter is False

    def test_to_dict(self) -> None:
        decision = RoutingDecision(
            vendor=Vendor.OPENAI,
            mode="auto",
            via_openrouter=True,
        )
        d = decision.to_dict()
        assert d == {
            "vendor": "openai",
            "mode": "auto",
            "via_openrouter": True,
        }

    def test_to_dict_vendor_is_string(self) -> None:
        """to_dict() serializes vendor as a plain string, not an Enum."""
        decision = RoutingDecision(
            vendor=Vendor.GROQ,
            mode="openrouter",
            via_openrouter=True,
        )
        d = decision.to_dict()
        assert isinstance(d["vendor"], str)
        assert d["vendor"] == "groq"

    def test_to_dict_json_serializable(self) -> None:
        """to_dict() output is fully JSON-serializable."""
        decision = RoutingDecision(
            vendor=Vendor.OLLAMA,
            mode="direct",
            via_openrouter=False,
        )
        result = json.dumps(decision.to_dict())
        parsed = json.loads(result)
        assert parsed["vendor"] == "ollama"
        assert parsed["via_openrouter"] is False

    def test_openrouter_mode(self) -> None:
        decision = RoutingDecision(
            vendor=Vendor.OPENROUTER,
            mode="openrouter",
            via_openrouter=True,
        )
        assert decision.mode == "openrouter"
        assert decision.via_openrouter is True
