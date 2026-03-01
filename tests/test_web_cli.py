"""Tests for the ``serve`` CLI command registration.

Covers: Click command registration, help text, and option defaults.
Does NOT start the NiceGUI server (create_app is mocked).
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from mutual_dissent.cli import main


class TestServeCommandRegistration:
    """serve command is registered and has correct options."""

    def test_serve_appears_in_help(self) -> None:
        """serve command is listed in main --help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "serve" in result.output

    def test_serve_help_shows_description(self) -> None:
        """serve --help shows the command description."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the web UI server" in result.output

    def test_serve_help_shows_port_option(self) -> None:
        """serve --help lists --port option."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_serve_help_shows_host_option(self) -> None:
        """serve --help lists --host option."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output

    def test_serve_help_shows_no_open_option(self) -> None:
        """serve --help lists --no-open option."""
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--no-open" in result.output


class TestServeCommandExecution:
    """serve command calls create_app with correct arguments."""

    def test_serve_default_args(self) -> None:
        """serve with no options passes defaults to create_app."""
        runner = CliRunner()
        with patch("mutual_dissent.web.app.create_app") as mock_create:
            result = runner.invoke(main, ["serve"])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(host="127.0.0.1", port=8080, show=True)

    def test_serve_custom_port(self) -> None:
        """serve --port passes custom port to create_app."""
        runner = CliRunner()
        with patch("mutual_dissent.web.app.create_app") as mock_create:
            result = runner.invoke(main, ["serve", "--port", "9000"])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(host="127.0.0.1", port=9000, show=True)

    def test_serve_custom_host(self) -> None:
        """serve --host passes custom host to create_app."""
        runner = CliRunner()
        with patch("mutual_dissent.web.app.create_app") as mock_create:
            result = runner.invoke(main, ["serve", "--host", "0.0.0.0"])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(host="0.0.0.0", port=8080, show=True)

    def test_serve_no_open(self) -> None:
        """serve --no-open passes show=False to create_app."""
        runner = CliRunner()
        with patch("mutual_dissent.web.app.create_app") as mock_create:
            result = runner.invoke(main, ["serve", "--no-open"])
        assert result.exit_code == 0
        mock_create.assert_called_once_with(host="127.0.0.1", port=8080, show=False)

    def test_serve_all_options(self) -> None:
        """serve with all options passes everything to create_app."""
        runner = CliRunner()
        with patch("mutual_dissent.web.app.create_app") as mock_create:
            result = runner.invoke(
                main, ["serve", "--port", "3000", "--host", "0.0.0.0", "--no-open"]
            )
        assert result.exit_code == 0
        mock_create.assert_called_once_with(host="0.0.0.0", port=3000, show=False)
