"""Tests for render_transcript_list display, ``list`` and ``show`` CLI commands.

Covers: Rich table rendering of transcript metadata (with data, empty list,
multiple rows, zero tokens), Click command registration and invocation,
transcript display by ID with partial matching and error handling.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from rich.console import Console

import mutual_dissent.display as display_mod
from mutual_dissent.cli import main
from mutual_dissent.display import render_transcript_list

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_render(transcripts: list[dict[str, Any]]) -> str:
    """Render transcript list and capture the Rich output as a string.

    Args:
        transcripts: List of transcript summary dicts.

    Returns:
        Captured terminal output string.
    """
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, width=120)

    original_console = display_mod.console
    display_mod.console = test_console
    try:
        render_transcript_list(transcripts)
    finally:
        display_mod.console = original_console

    return buf.getvalue()


def _make_transcript_summary(
    *,
    short_id: str = "abcd1234",
    date: str = "2026-02-28",
    panel: str = "claude, gpt",
    synthesizer: str = "claude",
    tokens: int = 500,
    query: str = "What is the meaning of life?",
) -> dict[str, Any]:
    """Build a transcript summary dict matching list_transcripts() output.

    Args:
        short_id: 8-char transcript ID prefix.
        date: ISO date string.
        panel: Comma-separated panel aliases.
        synthesizer: Synthesizer alias.
        tokens: Total token count.
        query: Truncated query text.

    Returns:
        Dict matching list_transcripts() output format.
    """
    return {
        "id": f"{short_id}-full-uuid-placeholder",
        "short_id": short_id,
        "date": date,
        "panel": panel,
        "synthesizer": synthesizer,
        "tokens": tokens,
        "query": query,
        "file": f"{date}_{short_id}.json",
    }


# ---------------------------------------------------------------------------
# TestRenderTranscriptList
# ---------------------------------------------------------------------------


class TestRenderTranscriptList:
    """render_transcript_list() renders transcript metadata as a Rich table."""

    def test_renders_table_with_data(self) -> None:
        """Single transcript row shows all expected fields."""
        transcripts = [
            _make_transcript_summary(
                short_id="abcd1234",
                date="2026-02-28",
                panel="claude, gpt",
                synthesizer="gpt",
                tokens=1500,
                query="What is quantum computing?",
            )
        ]

        output = _capture_render(transcripts)

        assert "abcd1234" in output
        assert "2026-02-28" in output
        assert "claude" in output
        assert "gpt" in output
        assert "1,500" in output
        assert "quantum computing" in output

    def test_renders_empty_message(self) -> None:
        """Empty transcript list shows 'No transcripts found' message."""
        output = _capture_render([])

        assert "No transcripts found" in output

    def test_multiple_rows(self) -> None:
        """Multiple transcripts each appear in the output."""
        transcripts = [
            _make_transcript_summary(short_id="aaaa1111"),
            _make_transcript_summary(short_id="bbbb2222"),
        ]

        output = _capture_render(transcripts)

        assert "aaaa1111" in output
        assert "bbbb2222" in output

    def test_zero_tokens_shown_as_dash(self) -> None:
        """Zero tokens render as an em dash instead of '0'."""
        transcripts = [_make_transcript_summary(tokens=0)]

        output = _capture_render(transcripts)

        assert "\u2014" in output


# ---------------------------------------------------------------------------
# TestListCommand
# ---------------------------------------------------------------------------


class TestListCommand:
    """``list`` CLI command is registered and functional."""

    def test_list_registered(self) -> None:
        """Main help output includes the list command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_list_shows_help(self) -> None:
        """list --help shows the --limit option."""
        runner = CliRunner()
        result = runner.invoke(main, ["list", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output

    def test_list_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """list command succeeds with an empty transcript directory."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0

    def test_list_with_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """list command shows transcript short_id when data exists."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)

        data: dict[str, Any] = {
            "transcript_id": "deadbeef-1234-5678-9abc-def012345678",
            "query": "Test question for list",
            "panel": ["claude", "gpt"],
            "synthesizer_id": "claude",
            "max_rounds": 1,
            "rounds": [],
            "synthesis": None,
            "created_at": "2026-02-28T12:00:00+00:00",
            "metadata": {},
        }
        filepath = tmp_path / "2026-02-28_deadbeef.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "deadbeef" in result.output


# ---------------------------------------------------------------------------
# Helpers — transcript JSON fixtures for show tests
# ---------------------------------------------------------------------------


def _make_transcript_json(
    *,
    transcript_id: str = "abcd1234-5678-9abc-def0-123456789abc",
    query: str = "What is the meaning of life?",
    panel: list[str] | None = None,
    synthesizer_id: str = "claude",
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Build a complete transcript JSON dict suitable for load_transcript.

    Includes one initial round with a single response and a synthesis
    response so that ``render_debate`` can render it fully.

    Args:
        transcript_id: Full UUID for the transcript.
        query: The debate query.
        panel: List of panel model aliases.
        synthesizer_id: Synthesizer model alias.
        max_rounds: Number of configured reflection rounds.

    Returns:
        Dict matching the DebateTranscript.to_dict() format.
    """
    if panel is None:
        panel = ["claude", "gpt"]

    return {
        "transcript_id": transcript_id,
        "query": query,
        "panel": panel,
        "synthesizer_id": synthesizer_id,
        "max_rounds": max_rounds,
        "rounds": [
            {
                "round_number": 0,
                "round_type": "initial",
                "responses": [
                    {
                        "model_id": "anthropic/claude-sonnet-4",
                        "model_alias": "claude",
                        "round_number": 0,
                        "content": "Initial response from claude.",
                        "timestamp": "2026-02-28T12:00:00+00:00",
                        "token_count": 100,
                        "latency_ms": 500,
                        "error": None,
                        "role": "initial",
                        "routing": None,
                        "analysis": {},
                    }
                ],
            }
        ],
        "synthesis": {
            "model_id": "anthropic/claude-sonnet-4",
            "model_alias": "claude",
            "round_number": -1,
            "content": "Synthesized answer.",
            "timestamp": "2026-02-28T12:01:00+00:00",
            "token_count": 200,
            "latency_ms": 800,
            "error": None,
            "role": "synthesis",
            "routing": None,
            "analysis": {},
        },
        "created_at": "2026-02-28T12:00:00+00:00",
        "metadata": {},
    }


def _write_transcript(
    tmp_path: Path,
    transcript_id: str,
    **kwargs: Any,
) -> Path:
    """Write a transcript JSON file into the given directory.

    File naming follows the ``{date}_{short_id}.json`` convention expected
    by ``load_transcript``.

    Args:
        tmp_path: Directory to write the file into.
        transcript_id: Full UUID for the transcript.
        **kwargs: Additional overrides passed to ``_make_transcript_json``.

    Returns:
        Path to the written JSON file.
    """
    data = _make_transcript_json(transcript_id=transcript_id, **kwargs)
    short_id = transcript_id[:8]
    filepath = tmp_path / f"2026-02-28_{short_id}.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# TestShowCommand
# ---------------------------------------------------------------------------


class TestShowCommand:
    """``show`` CLI command displays a saved debate transcript."""

    def test_show_registered(self) -> None:
        """Main help output includes the show command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "show" in result.output

    def test_show_shows_help(self) -> None:
        """show --help shows --verbose and --output options."""
        runner = CliRunner()
        result = runner.invoke(main, ["show", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
        assert "--output" in result.output

    def test_show_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show exits 1 when no transcript matches the given ID."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["show", "nonexist"])
        assert result.exit_code == 1
        assert "no transcript" in result.output.lower() or "not found" in result.output.lower()

    def test_show_id_too_short(self) -> None:
        """show exits 1 when ID is fewer than 4 characters."""
        runner = CliRunner()
        result = runner.invoke(main, ["show", "abc"])
        assert result.exit_code == 1
        assert "at least 4" in result.output.lower()

    def test_show_loads_transcript(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show exits 0 and renders when given a valid transcript ID."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        _write_transcript(tmp_path, "abcd1234-5678-9abc-def0-123456789abc")

        runner = CliRunner()
        result = runner.invoke(main, ["show", "abcd1234"])
        assert result.exit_code == 0

    def test_show_json_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show --output json emits valid JSON with the transcript_id."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "jsontest-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)

        runner = CliRunner()
        result = runner.invoke(main, ["show", "jsontest", "--output", "json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data["transcript_id"] == tid

    def test_show_ambiguous_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show exits 1 when multiple transcripts match the ID prefix."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        _write_transcript(tmp_path, "abcdaaaa-1111-2222-3333-444455556666")
        # Second transcript — different full ID but same 4-char prefix.
        data2 = _make_transcript_json(transcript_id="abcdbbbb-1111-2222-3333-444455556666")
        filepath2 = tmp_path / "2026-02-28_abcdbbbb.json"
        filepath2.write_text(json.dumps(data2), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["show", "abcd"])
        assert result.exit_code == 1
        assert "ambiguous" in result.output.lower() or "matches" in result.output.lower()
