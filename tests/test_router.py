"""Tests for the ProviderRouter.

Covers: vendor resolution (alias, full model ID, unknown), routing decisions
for all mode x key x provider combinations, provider lifecycle management,
dispatch correctness, mixed-provider parallel fan-out, no-provider error
handling, warning logging for direct-mode fallbacks, and package import.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import AsyncMock

import pytest

from mutual_dissent.config import Config
from mutual_dissent.models import ModelResponse
from mutual_dissent.providers.router import ProviderRouter, _resolve_vendor
from mutual_dissent.types import RoutingDecision, Vendor

ROUTER_LOGGER = "mutual_dissent.providers.router"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    openrouter_key: str = "",
    anthropic_key: str = "",
    routing: dict[str, str] | None = None,
) -> Config:
    """Create a Config with specific provider keys and routing."""
    providers: dict[str, str] = {}
    if openrouter_key:
        providers["openrouter"] = openrouter_key
    if anthropic_key:
        providers["anthropic"] = anthropic_key
    return Config(
        api_key=openrouter_key,
        providers=providers,
        routing=routing or {"default_mode": "auto"},
    )


def _mock_response(
    model_id: str = "test-model",
    alias: str = "test",
) -> ModelResponse:
    """Create a minimal ModelResponse for mocking."""
    return ModelResponse(
        model_id=model_id,
        model_alias=alias,
        round_number=0,
        content="mock response",
    )


# ---------------------------------------------------------------------------
# Vendor resolution
# ---------------------------------------------------------------------------


class TestResolveVendor:
    """_resolve_vendor() determines the vendor for a given alias or ID."""

    def test_alias_claude(self) -> None:
        config = Config()
        assert _resolve_vendor("claude", config) == Vendor.ANTHROPIC

    def test_alias_gpt(self) -> None:
        config = Config()
        assert _resolve_vendor("gpt", config) == Vendor.OPENAI

    def test_alias_gemini(self) -> None:
        config = Config()
        assert _resolve_vendor("gemini", config) == Vendor.GOOGLE

    def test_alias_grok(self) -> None:
        config = Config()
        assert _resolve_vendor("grok", config) == Vendor.XAI

    def test_full_model_id_anthropic(self) -> None:
        config = Config()
        assert _resolve_vendor("anthropic/claude-sonnet-4.5", config) == Vendor.ANTHROPIC

    def test_full_model_id_openai(self) -> None:
        config = Config()
        assert _resolve_vendor("openai/gpt-5.2", config) == Vendor.OPENAI

    def test_full_model_id_google(self) -> None:
        config = Config()
        assert _resolve_vendor("google/gemini-2.5-pro", config) == Vendor.GOOGLE

    def test_full_model_id_xai(self) -> None:
        config = Config()
        assert _resolve_vendor("x-ai/grok-4", config) == Vendor.XAI

    def test_unknown_prefix_with_slash(self) -> None:
        config = Config()
        assert _resolve_vendor("unknown-vendor/some-model", config) == Vendor.OPENROUTER

    def test_unknown_string_no_slash(self) -> None:
        config = Config()
        assert _resolve_vendor("totally-unknown", config) == Vendor.OPENROUTER

    def test_alias_takes_priority_over_slash_parse(self) -> None:
        """An alias that happens to contain a slash still resolves via v2."""
        config = Config()
        # "claude" is a known alias → resolves via v2 to Vendor.ANTHROPIC,
        # not by parsing the string itself.
        assert _resolve_vendor("claude", config) == Vendor.ANTHROPIC


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------


class TestRoutingDecisions:
    """ProviderRouter.route() returns correct decisions for all mode combos."""

    def test_openrouter_mode_always_via_openrouter(self) -> None:
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
            routing={"default_mode": "openrouter"},
        )
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert decision.via_openrouter is True
        assert decision.mode == "openrouter"
        assert decision.vendor == Vendor.ANTHROPIC

    def test_direct_mode_with_key_and_provider(self) -> None:
        config = _make_config(
            anthropic_key="sk-ant-test",
            routing={"default_mode": "direct"},
        )
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert decision.via_openrouter is False
        assert decision.mode == "direct"
        assert decision.vendor == Vendor.ANTHROPIC

    def test_direct_mode_without_key_falls_back(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _make_config(routing={"default_mode": "direct"})
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            decision = router.route("claude")
        assert decision.via_openrouter is True
        assert "no API key" in caplog.text

    def test_direct_mode_without_provider_class_falls_back(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """gpt has a key but no direct provider implementation."""
        config = _make_config(routing={"default_mode": "direct"})
        config.providers["openai"] = "sk-openai-test"
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            decision = router.route("gpt")
        assert decision.via_openrouter is True
        assert "no provider implementation" in caplog.text

    def test_auto_mode_with_key_and_provider(self) -> None:
        config = _make_config(
            anthropic_key="sk-ant-test",
            routing={"default_mode": "auto"},
        )
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert decision.via_openrouter is False
        assert decision.mode == "auto"

    def test_auto_mode_without_key(self) -> None:
        config = _make_config(routing={"default_mode": "auto"})
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert decision.via_openrouter is True

    def test_auto_mode_without_provider_class(self) -> None:
        """auto mode for a vendor with key but no direct provider (e.g. gpt)."""
        config = _make_config(routing={"default_mode": "auto"})
        config.providers["openai"] = "sk-openai-test"
        router = ProviderRouter(config)
        decision = router.route("gpt")
        assert decision.via_openrouter is True

    def test_per_alias_override(self) -> None:
        """Per-alias routing overrides the default mode."""
        config = _make_config(
            anthropic_key="sk-ant-test",
            routing={"default_mode": "openrouter", "claude": "direct"},
        )
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert decision.via_openrouter is False
        assert decision.mode == "direct"

    def test_per_alias_does_not_affect_other_aliases(self) -> None:
        """A per-alias override for claude doesn't change gpt's routing."""
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
            routing={"default_mode": "openrouter", "claude": "direct"},
        )
        router = ProviderRouter(config)
        decision = router.route("gpt")
        assert decision.via_openrouter is True
        assert decision.mode == "openrouter"

    def test_routing_decision_is_dataclass(self) -> None:
        config = _make_config(routing={"default_mode": "auto"})
        router = ProviderRouter(config)
        decision = router.route("claude")
        assert isinstance(decision, RoutingDecision)
        assert hasattr(decision, "vendor")
        assert hasattr(decision, "mode")
        assert hasattr(decision, "via_openrouter")


# ---------------------------------------------------------------------------
# Provider lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestProviderLifecycle:
    """__aenter__ opens and __aexit__ closes providers based on config."""

    @pytest.mark.asyncio
    async def test_opens_openrouter_when_key_exists(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            assert router._openrouter is not None

    @pytest.mark.asyncio
    async def test_no_openrouter_without_key(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            assert router._openrouter is None

    @pytest.mark.asyncio
    async def test_opens_direct_provider_when_key_exists(self) -> None:
        config = _make_config(anthropic_key="sk-ant-test")
        async with ProviderRouter(config) as router:
            assert "anthropic" in router._providers

    @pytest.mark.asyncio
    async def test_no_direct_provider_without_key(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            assert "anthropic" not in router._providers

    @pytest.mark.asyncio
    async def test_opens_both_when_both_keys_exist(self) -> None:
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
        )
        async with ProviderRouter(config) as router:
            assert router._openrouter is not None
            assert "anthropic" in router._providers

    @pytest.mark.asyncio
    async def test_exit_clears_providers(self) -> None:
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
        )
        router = ProviderRouter(config)
        await router.__aenter__()
        assert router._openrouter is not None
        assert len(router._providers) == 1
        await router.__aexit__(None, None, None)
        assert router._openrouter is None
        assert len(router._providers) == 0

    @pytest.mark.asyncio
    async def test_no_keys_opens_nothing(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            assert router._openrouter is None
            assert len(router._providers) == 0


# ---------------------------------------------------------------------------
# Dispatch correctness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestDispatch:
    """complete() calls the right provider with the right model_id."""

    @pytest.mark.asyncio
    async def test_routes_to_openrouter(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response("anthropic/claude-sonnet-4.5", "claude")
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            result = await router.complete("claude", prompt="Hello")

            router._openrouter.complete.assert_called_once()  # type: ignore[union-attr]
            call_args = router._openrouter.complete.call_args  # type: ignore[union-attr]
            assert call_args[0][0] == "anthropic/claude-sonnet-4.5"
            assert result.content == "mock response"

    @pytest.mark.asyncio
    async def test_routes_to_direct_provider(self) -> None:
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
        )
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response("claude-sonnet-4-5-20250929", "claude")
            router._providers["anthropic"].complete = AsyncMock(  # type: ignore[assignment]
                return_value=mock_resp,
            )

            result = await router.complete("claude", prompt="Hello")

            router._providers["anthropic"].complete.assert_called_once()  # type: ignore[attr-defined]
            call_args = router._providers["anthropic"].complete.call_args  # type: ignore[attr-defined]
            assert call_args[0][0] == "claude-sonnet-4-5-20250929"
            assert result.content == "mock response"

    @pytest.mark.asyncio
    async def test_model_alias_defaults_to_alias_or_id(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response()
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            await router.complete("claude", prompt="Hello")

            call_kwargs = router._openrouter.complete.call_args  # type: ignore[union-attr]
            assert call_kwargs.kwargs["model_alias"] == "claude"

    @pytest.mark.asyncio
    async def test_explicit_model_alias_preserved(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response()
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            await router.complete("claude", prompt="Hello", model_alias="my-claude")

            call_kwargs = router._openrouter.complete.call_args  # type: ignore[union-attr]
            assert call_kwargs.kwargs["model_alias"] == "my-claude"

    @pytest.mark.asyncio
    async def test_round_number_passed_through(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response()
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            await router.complete("claude", prompt="Hello", round_number=2)

            call_kwargs = router._openrouter.complete.call_args  # type: ignore[union-attr]
            assert call_kwargs.kwargs["round_number"] == 2

    @pytest.mark.asyncio
    async def test_messages_passed_through(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        messages = [{"role": "user", "content": "Hello"}]
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response()
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            await router.complete("claude", messages=messages)

            call_kwargs = router._openrouter.complete.call_args  # type: ignore[union-attr]
            assert call_kwargs.kwargs["messages"] == messages


# ---------------------------------------------------------------------------
# No provider available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestNoProviderAvailable:
    """complete() returns error ModelResponse when no provider is available."""

    @pytest.mark.asyncio
    async def test_no_providers_at_all(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            result = await router.complete("claude", prompt="Hello")
            assert result.error is not None
            assert "No provider available" in result.error
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_only_openrouter_for_vendor_without_direct(self) -> None:
        """gpt needs OpenRouter but no OpenRouter key configured."""
        config = _make_config()
        async with ProviderRouter(config) as router:
            result = await router.complete("gpt", prompt="Hello")
            assert result.error is not None
            assert result.content == ""

    @pytest.mark.asyncio
    async def test_error_response_has_correct_metadata(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            result = await router.complete(
                "claude",
                prompt="Hello",
                model_alias="my-claude",
                round_number=1,
            )
            assert result.model_alias == "my-claude"
            assert result.round_number == 1
            assert result.model_id == "claude"


# ---------------------------------------------------------------------------
# Mixed parallel fan-out
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestMixedParallel:
    """complete_parallel() routes different requests to different providers."""

    @pytest.mark.asyncio
    async def test_mixed_providers_in_parallel(self) -> None:
        config = _make_config(
            openrouter_key="sk-or-test",
            anthropic_key="sk-ant-test",
        )
        async with ProviderRouter(config) as router:
            anthropic_resp = _mock_response("claude-sonnet-4-5-20250929", "claude")
            openrouter_resp = _mock_response("openai/gpt-5.2", "gpt")

            router._providers["anthropic"].complete = AsyncMock(  # type: ignore[assignment]
                return_value=anthropic_resp,
            )
            router._openrouter.complete = AsyncMock(  # type: ignore[union-attr]
                return_value=openrouter_resp,
            )

            results = await router.complete_parallel(
                [
                    {"alias_or_id": "claude", "prompt": "Hello from Claude"},
                    {"alias_or_id": "gpt", "prompt": "Hello from GPT"},
                ]
            )

            assert len(results) == 2
            # claude goes to direct Anthropic provider.
            router._providers["anthropic"].complete.assert_called_once()  # type: ignore[attr-defined]
            # gpt goes to OpenRouter.
            router._openrouter.complete.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            resp_claude = _mock_response("anthropic/claude-sonnet-4.5", "claude")
            resp_gpt = _mock_response("openai/gpt-5.2", "gpt")

            call_count = 0

            async def mock_complete(model_id: str, **kwargs: object) -> ModelResponse:
                nonlocal call_count
                call_count += 1
                if "claude" in model_id:
                    return resp_claude
                return resp_gpt

            router._openrouter.complete = mock_complete  # type: ignore[union-attr, assignment]

            results = await router.complete_parallel(
                [
                    {"alias_or_id": "claude", "prompt": "First"},
                    {"alias_or_id": "gpt", "prompt": "Second"},
                ]
            )

            assert results[0].model_alias == "claude"
            assert results[1].model_alias == "gpt"
            assert call_count == 2


# ---------------------------------------------------------------------------
# Warning logging
# ---------------------------------------------------------------------------


class TestWarningLogging:
    """Direct mode fallbacks produce appropriate log warnings."""

    def test_direct_no_key_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _make_config(routing={"default_mode": "direct"})
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            router.route("claude")
        assert "no API key" in caplog.text
        assert "anthropic" in caplog.text

    def test_direct_no_provider_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = _make_config(routing={"default_mode": "direct"})
        config.providers["openai"] = "sk-openai-test"
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            router.route("gpt")
        assert "no provider implementation" in caplog.text
        assert "openai" in caplog.text

    def test_auto_mode_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """auto mode falling back to OpenRouter should NOT log warnings."""
        config = _make_config(routing={"default_mode": "auto"})
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            router.route("claude")
        router_warnings = [r for r in caplog.records if r.name == ROUTER_LOGGER]
        assert len(router_warnings) == 0

    def test_openrouter_mode_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """openrouter mode should NOT log warnings even with direct key."""
        config = _make_config(
            anthropic_key="sk-ant-test",
            routing={"default_mode": "openrouter"},
        )
        router = ProviderRouter(config)
        with caplog.at_level(logging.WARNING, logger=ROUTER_LOGGER):
            router.route("claude")
        router_warnings = [r for r in caplog.records if r.name == ROUTER_LOGGER]
        assert len(router_warnings) == 0


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class TestImport:
    """ProviderRouter is importable from the providers package."""

    def test_importable_from_providers(self) -> None:
        from mutual_dissent.providers import ProviderRouter as Imported

        assert Imported is ProviderRouter

    def test_routing_decision_importable(self) -> None:
        from mutual_dissent.providers import RoutingDecision as Imported

        assert Imported is RoutingDecision

    def test_vendor_importable(self) -> None:
        from mutual_dissent.providers import Vendor as Imported

        assert Imported is Vendor

    def test_vendor_is_types_vendor(self) -> None:
        """providers re-export is the canonical types.Vendor."""
        from mutual_dissent.providers import Vendor as ProvidersVendor
        from mutual_dissent.types import Vendor as TypesVendor

        assert ProvidersVendor is TypesVendor

    def test_routing_decision_is_types_routing_decision(self) -> None:
        """providers re-export is the canonical types.RoutingDecision."""
        from mutual_dissent.providers import RoutingDecision as ProvidersRD
        from mutual_dissent.types import RoutingDecision as TypesRD

        assert ProvidersRD is TypesRD


# ---------------------------------------------------------------------------
# Routing provenance on responses
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestRoutingProvenance:
    """complete() attaches routing info to every response."""

    @pytest.mark.asyncio
    async def test_routing_populated_on_success(self) -> None:
        config = _make_config(openrouter_key="sk-or-test")
        async with ProviderRouter(config) as router:
            mock_resp = _mock_response("anthropic/claude-sonnet-4.5", "claude")
            router._openrouter.complete = AsyncMock(return_value=mock_resp)  # type: ignore[union-attr]

            result = await router.complete("claude", prompt="Hello")

            assert result.routing is not None
            assert result.routing["vendor"] == "anthropic"
            assert "mode" in result.routing
            assert "via_openrouter" in result.routing

    @pytest.mark.asyncio
    async def test_routing_populated_on_error(self) -> None:
        config = _make_config()
        async with ProviderRouter(config) as router:
            result = await router.complete("claude", prompt="Hello")

            assert result.error is not None
            assert result.routing is not None
            assert result.routing["vendor"] == "anthropic"
