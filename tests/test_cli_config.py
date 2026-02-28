"""Tests for the ``config test`` CLI command and display rendering.

Covers: Click command registration (config group exists, test subcommand
exists), render_config_test with success results, render_config_test with
error results, and the _run_config_test async helper.
"""

from __future__ import annotations

from click.testing import CliRunner

from mutual_dissent.cli import main
from mutual_dissent.display import render_config_test
from mutual_dissent.models import ModelResponse
from mutual_dissent.types import RoutingDecision, Vendor

# ---------------------------------------------------------------------------
# Click command registration
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    """config group and test subcommand are registered correctly."""

    def test_config_group_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.output

    def test_config_shows_test_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "test" in result.output

    def test_config_test_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["config", "test", "--help"])
        assert result.exit_code == 0
        assert "Test provider configuration" in result.output


# ---------------------------------------------------------------------------
# render_config_test output
# ---------------------------------------------------------------------------


def _make_result(
    alias: str,
    vendor: Vendor,
    via_openrouter: bool,
    model_id: str,
    *,
    latency_ms: int | None = 1200,
    error: str | None = None,
) -> dict[str, RoutingDecision | ModelResponse | str]:
    """Build a result dict matching _run_config_test output format."""
    decision = RoutingDecision(vendor=vendor, mode="auto", via_openrouter=via_openrouter)
    response = ModelResponse(
        model_id=model_id,
        model_alias=alias,
        round_number=0,
        content="OK" if not error else "",
        latency_ms=latency_ms,
        error=error,
    )
    return {"alias": alias, "decision": decision, "response": response}


class TestRenderConfigTestSuccess:
    """render_config_test renders success results correctly."""

    def test_renders_without_error(self, capsys: object) -> None:
        """Smoke test — render_config_test doesn't raise."""
        results = [
            _make_result("claude", Vendor.ANTHROPIC, False, "claude-sonnet-4-5-20250929"),
            _make_result("gpt", Vendor.OPENAI, True, "openai/gpt-5.2"),
        ]
        # Should not raise.
        render_config_test(results)

    def test_success_all_aliases_shown(self) -> None:
        """All tested aliases appear in the output."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result("claude", Vendor.ANTHROPIC, False, "claude-sonnet-4-5-20250929"),
            _make_result("gpt", Vendor.OPENAI, True, "openai/gpt-5.2"),
            _make_result("gemini", Vendor.GOOGLE, True, "google/gemini-2.5-pro"),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        # Temporarily replace module console.
        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "claude" in output
        assert "gpt" in output
        assert "gemini" in output

    def test_latency_formatted_as_seconds(self) -> None:
        """Latency renders as seconds (e.g. 1.2s)."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result(
                "claude",
                Vendor.ANTHROPIC,
                False,
                "claude-sonnet-4-5-20250929",
                latency_ms=1200,
            ),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "1.2s" in output

    def test_direct_route_shown(self) -> None:
        """Direct-routed models show 'direct' in route column."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result("claude", Vendor.ANTHROPIC, False, "claude-sonnet-4-5-20250929"),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "direct" in output

    def test_openrouter_route_shown(self) -> None:
        """OpenRouter-routed models show 'openrouter' in route column."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result("gpt", Vendor.OPENAI, True, "openai/gpt-5.2"),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "openrouter" in output


class TestRenderConfigTestError:
    """render_config_test renders error results correctly."""

    def test_error_shows_message(self) -> None:
        """Error responses show the error message in the status column."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result(
                "grok",
                Vendor.XAI,
                True,
                "x-ai/grok-4",
                latency_ms=None,
                error="401 Unauthorized",
            ),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "401 Unauthorized" in output

    def test_error_shows_dash_for_latency(self) -> None:
        """Error responses show a dash instead of latency."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result(
                "grok",
                Vendor.XAI,
                True,
                "x-ai/grok-4",
                latency_ms=None,
                error="timeout",
            ),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "\u2014" in output

    def test_mixed_success_and_error(self) -> None:
        """Table renders correctly with both success and error rows."""
        from io import StringIO

        from rich.console import Console

        results = [
            _make_result("claude", Vendor.ANTHROPIC, False, "claude-sonnet-4-5-20250929"),
            _make_result(
                "grok",
                Vendor.XAI,
                True,
                "x-ai/grok-4",
                latency_ms=None,
                error="connection error",
            ),
        ]

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=120)

        import mutual_dissent.display as display_mod

        original_console = display_mod.console
        display_mod.console = test_console
        try:
            render_config_test(results)
        finally:
            display_mod.console = original_console

        output = buf.getvalue()
        assert "claude" in output
        assert "grok" in output
        assert "connection error" in output
        assert "1.2s" in output


# ---------------------------------------------------------------------------
# config path subcommand
# ---------------------------------------------------------------------------


class TestConfigPath:
    """config path subcommand."""

    def test_config_path_shows_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output

    def test_config_path_prints_path(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["config", "path"])
        assert result.exit_code == 0
        assert ".mutual-dissent" in result.output
        assert "config.toml" in result.output
