"""Tests for format_markdown() display and --file / --output markdown CLI options.

Covers: markdown rendering (default, verbose, errors, no synthesis),
file writing for all output formats, terminal+file fallback behavior,
and Click option registration on ask, show, and replay commands.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from mutual_dissent.cli import main
from mutual_dissent.display import format_markdown
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TIME = datetime(2026, 2, 28, 14, 30, 0, tzinfo=UTC)


def _make_response(
    alias: str,
    round_number: int,
    *,
    role: str = "",
    content: str = "",
    model_id: str | None = None,
    token_count: int | None = 100,
    latency_ms: int | None = 2100,
    error: str | None = None,
) -> ModelResponse:
    """Build a ModelResponse with predictable defaults.

    Args:
        alias: Model alias (e.g. "claude").
        round_number: Round number for the response.
        role: Debate role.
        content: Response text.
        model_id: Full model ID. Defaults to "vendor/{alias}-model".
        token_count: Token count.
        latency_ms: Latency in milliseconds.
        error: Error message, if any.

    Returns:
        A ModelResponse with controlled fields.
    """
    return ModelResponse(
        model_id=model_id or f"vendor/{alias}-model",
        model_alias=alias,
        round_number=round_number,
        content=content or f"{alias} response for round {round_number}",
        timestamp=FIXED_TIME,
        token_count=token_count,
        latency_ms=latency_ms,
        error=error,
        role=role,
    )


def _make_transcript(
    *,
    query: str = "What is the meaning of life?",
    panel: list[str] | None = None,
    synthesizer_id: str = "claude",
    include_reflection: bool = False,
    synthesis_error: str | None = None,
    no_synthesis: bool = False,
    no_timing: bool = False,
) -> DebateTranscript:
    """Build a DebateTranscript for testing format_markdown().

    Args:
        query: The debate query.
        panel: Panel aliases. Defaults to ["claude", "gpt"].
        synthesizer_id: Synthesizer alias.
        include_reflection: If True, add a reflection round.
        synthesis_error: If set, synthesis has this error.
        no_synthesis: If True, synthesis is None.
        no_timing: If True, responses have no token/latency data.

    Returns:
        A DebateTranscript with controlled fields.
    """
    panel = panel or ["claude", "gpt"]
    token_count = None if no_timing else 100
    latency_ms = None if no_timing else 2100

    # Initial round.
    initial_responses = [
        _make_response(alias, 0, role="initial", token_count=token_count, latency_ms=latency_ms)
        for alias in panel
    ]
    rounds = [DebateRound(round_number=0, round_type="initial", responses=initial_responses)]

    # Optional reflection round.
    if include_reflection:
        reflection_responses = [
            _make_response(
                alias, 1, role="reflection", token_count=token_count, latency_ms=latency_ms
            )
            for alias in panel
        ]
        rounds.append(
            DebateRound(round_number=1, round_type="reflection", responses=reflection_responses)
        )

    # Synthesis.
    synthesis = None
    if not no_synthesis:
        synthesis = _make_response(
            synthesizer_id,
            -1,
            role="synthesis",
            content="The synthesized answer to life.",
            token_count=token_count,
            latency_ms=latency_ms,
            error=synthesis_error,
        )

    max_rounds = 2 if include_reflection else 1

    return DebateTranscript(
        transcript_id="abcd1234-5678-9abc-def0-123456789abc",
        query=query,
        panel=panel,
        synthesizer_id=synthesizer_id,
        max_rounds=max_rounds,
        rounds=rounds,
        synthesis=synthesis,
        created_at=FIXED_TIME,
    )


def _make_transcript_json(
    *,
    transcript_id: str = "abcd1234-5678-9abc-def0-123456789abc",
    query: str = "What is the meaning of life?",
) -> dict[str, object]:
    """Build a transcript JSON dict for CLI tests.

    Args:
        transcript_id: Full UUID.
        query: The debate query.

    Returns:
        Dict matching DebateTranscript.to_dict() format.
    """
    return {
        "transcript_id": transcript_id,
        "query": query,
        "panel": ["claude", "gpt"],
        "synthesizer_id": "claude",
        "max_rounds": 1,
        "rounds": [
            {
                "round_number": 0,
                "round_type": "initial",
                "responses": [
                    {
                        "model_id": "vendor/claude-model",
                        "model_alias": "claude",
                        "round_number": 0,
                        "content": "Initial response from claude.",
                        "timestamp": "2026-02-28T14:30:00+00:00",
                        "token_count": 100,
                        "latency_ms": 2100,
                        "error": None,
                        "role": "initial",
                        "routing": None,
                        "analysis": {},
                    },
                    {
                        "model_id": "vendor/gpt-model",
                        "model_alias": "gpt",
                        "round_number": 0,
                        "content": "Initial response from gpt.",
                        "timestamp": "2026-02-28T14:30:00+00:00",
                        "token_count": 100,
                        "latency_ms": 2100,
                        "error": None,
                        "role": "initial",
                        "routing": None,
                        "analysis": {},
                    },
                ],
            }
        ],
        "synthesis": {
            "model_id": "vendor/claude-model",
            "model_alias": "claude",
            "round_number": -1,
            "content": "Synthesized answer.",
            "timestamp": "2026-02-28T14:30:00+00:00",
            "token_count": 200,
            "latency_ms": 800,
            "error": None,
            "role": "synthesis",
            "routing": None,
            "analysis": {},
        },
        "created_at": "2026-02-28T14:30:00+00:00",
        "metadata": {},
    }


def _write_transcript(tmp_path: Path, transcript_id: str) -> Path:
    """Write a transcript JSON file to tmp_path.

    Args:
        tmp_path: Directory for the file.
        transcript_id: Full UUID for the transcript.

    Returns:
        Path to the written file.
    """
    data = _make_transcript_json(transcript_id=transcript_id)
    short_id = transcript_id[:8]
    filepath = tmp_path / f"2026-02-28_{short_id}.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# TestFormatMarkdownDefault
# ---------------------------------------------------------------------------


class TestFormatMarkdownDefault:
    """format_markdown() default mode — synthesis + metadata only."""

    def test_contains_heading_with_query(self) -> None:
        """Output starts with a heading containing the query."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "# Debate: What is the meaning of life?" in result

    def test_long_query_truncated(self) -> None:
        """Queries longer than 80 chars are truncated with ellipsis."""
        long_query = "A" * 100
        transcript = _make_transcript(query=long_query)
        result = format_markdown(transcript)
        assert f"# Debate: {'A' * 80}..." in result

    def test_contains_synthesis_section(self) -> None:
        """Output includes synthesis heading and content."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "## Synthesis" in result
        assert "The synthesized answer to life." in result

    def test_contains_synthesizer_info(self) -> None:
        """Output includes synthesizer alias and model ID."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "**Synthesized by:** claude" in result
        assert "**Model:** vendor/claude-model" in result

    def test_contains_metadata(self) -> None:
        """Output includes metadata footer with all fields."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "**Transcript:** abcd1234" in result
        assert "**Panel:** claude, gpt" in result
        assert "**Synthesizer:** claude" in result
        assert "**Rounds:** 1" in result
        assert "**Date:** 2026-02-28 14:30 UTC" in result

    def test_contains_token_count(self) -> None:
        """Output includes formatted token count when available."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "**Tokens:**" in result

    def test_omits_tokens_when_no_data(self) -> None:
        """Token count is omitted when no timing data is available."""
        transcript = _make_transcript(no_timing=True)
        result = format_markdown(transcript)
        assert "**Tokens:**" not in result

    def test_default_omits_round_content(self) -> None:
        """Default mode does not include initial or reflection rounds."""
        transcript = _make_transcript(include_reflection=True)
        result = format_markdown(transcript)
        assert "## Initial Round" not in result
        assert "## Reflection Round" not in result

    def test_no_rich_markup(self) -> None:
        """Output contains no Rich markup tags."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "[bold]" not in result
        assert "[red]" not in result
        assert "[dim]" not in result

    def test_synthesis_timing_metadata(self) -> None:
        """Synthesis section includes timing metadata."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert "*2.1s · 100 tokens*" in result


# ---------------------------------------------------------------------------
# TestFormatMarkdownVerbose
# ---------------------------------------------------------------------------


class TestFormatMarkdownVerbose:
    """format_markdown() verbose mode — all rounds + synthesis."""

    def test_includes_initial_round(self) -> None:
        """Verbose output includes the initial round heading."""
        transcript = _make_transcript()
        result = format_markdown(transcript, verbose=True)
        assert "## Initial Round" in result

    def test_includes_reflection_round(self) -> None:
        """Verbose output includes reflection round headings."""
        transcript = _make_transcript(include_reflection=True)
        result = format_markdown(transcript, verbose=True)
        assert "## Reflection Round 1" in result

    def test_includes_model_responses(self) -> None:
        """Verbose output includes per-model response headings."""
        transcript = _make_transcript()
        result = format_markdown(transcript, verbose=True)
        assert "### claude" in result
        assert "### gpt" in result

    def test_includes_model_metadata(self) -> None:
        """Verbose output includes model ID and timing per response."""
        transcript = _make_transcript()
        result = format_markdown(transcript, verbose=True)
        assert "*vendor/claude-model · 2.1s · 100 tokens*" in result

    def test_includes_response_content(self) -> None:
        """Verbose output includes the actual response text."""
        transcript = _make_transcript()
        result = format_markdown(transcript, verbose=True)
        assert "claude response for round 0" in result
        assert "gpt response for round 0" in result

    def test_still_includes_synthesis(self) -> None:
        """Verbose output also includes synthesis section."""
        transcript = _make_transcript()
        result = format_markdown(transcript, verbose=True)
        assert "## Synthesis" in result
        assert "The synthesized answer to life." in result


# ---------------------------------------------------------------------------
# TestFormatMarkdownEdgeCases
# ---------------------------------------------------------------------------


class TestFormatMarkdownEdgeCases:
    """format_markdown() edge cases — errors, no synthesis."""

    def test_error_response_in_round(self) -> None:
        """Error responses render as **Error:** in verbose mode."""
        transcript = _make_transcript()
        # Replace first response with an error.
        transcript.rounds[0].responses[0] = _make_response(
            "claude", 0, role="initial", error="API timeout"
        )
        result = format_markdown(transcript, verbose=True)
        assert "**Error:** API timeout" in result

    def test_synthesis_error(self) -> None:
        """Synthesis error renders as **Error:** in the synthesis section."""
        transcript = _make_transcript(synthesis_error="Synthesis failed")
        result = format_markdown(transcript)
        assert "**Error:** Synthesis failed" in result

    def test_no_synthesis(self) -> None:
        """When synthesis is None, renders 'No synthesis available.'."""
        transcript = _make_transcript(no_synthesis=True)
        result = format_markdown(transcript)
        assert "No synthesis available." in result

    def test_is_pure_function(self) -> None:
        """format_markdown() returns a string and has no side effects."""
        transcript = _make_transcript()
        result = format_markdown(transcript)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestCliOptionRegistration
# ---------------------------------------------------------------------------


class TestCliOptionRegistration:
    """--output markdown and --file options are registered on ask, show, replay."""

    def test_ask_has_new_options(self) -> None:
        """ask --help shows markdown output choice and --file option."""
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert "markdown" in result.output
        assert "--file" in result.output

    def test_show_has_new_options(self) -> None:
        """show --help shows markdown output choice and --file option."""
        runner = CliRunner()
        result = runner.invoke(main, ["show", "--help"])
        assert "markdown" in result.output
        assert "--file" in result.output

    def test_replay_has_new_options(self) -> None:
        """replay --help shows markdown output choice and --file option."""
        runner = CliRunner()
        result = runner.invoke(main, ["replay", "--help"])
        assert "markdown" in result.output
        assert "--file" in result.output


# ---------------------------------------------------------------------------
# TestShowMarkdownOutput
# ---------------------------------------------------------------------------


class TestShowMarkdownOutput:
    """show --output markdown emits markdown to stdout."""

    def test_show_markdown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show --output markdown prints markdown with synthesis and metadata."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "mdtest01-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)

        runner = CliRunner()
        result = runner.invoke(main, ["show", "mdtest01", "--output", "markdown"])
        assert result.exit_code == 0
        assert "# Debate:" in result.output
        assert "## Synthesis" in result.output
        assert "**Transcript:** mdtest01" in result.output


# ---------------------------------------------------------------------------
# TestFileOutput
# ---------------------------------------------------------------------------


class TestFileOutput:
    """--file flag writes output to disk for all formats."""

    def test_file_markdown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show --output markdown --file writes markdown to disk."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "filemd01-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)
        out_file = tmp_path / "output.md"

        runner = CliRunner()
        result = runner.invoke(
            main, ["show", "filemd01", "--output", "markdown", "--file", str(out_file)]
        )
        assert result.exit_code == 0
        content = out_file.read_text(encoding="utf-8")
        assert "# Debate:" in content
        assert "## Synthesis" in content

    def test_file_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show --output json --file writes valid JSON to disk."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "filejsn1-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)
        out_file = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            main, ["show", "filejsn1", "--output", "json", "--file", str(out_file)]
        )
        assert result.exit_code == 0
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["transcript_id"] == tid

    def test_file_terminal_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """show --file with terminal output degrades to markdown."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "filetrm1-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)
        out_file = tmp_path / "output.md"

        runner = CliRunner()
        result = runner.invoke(main, ["show", "filetrm1", "--file", str(out_file)])
        assert result.exit_code == 0
        content = out_file.read_text(encoding="utf-8")
        # Terminal + file should degrade to markdown content.
        assert "# Debate:" in content

    def test_file_no_stdout_leak(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--file writes to disk without leaking content to stdout."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        tid = "filecnf1-5678-9abc-def0-123456789abc"
        _write_transcript(tmp_path, tid)
        out_file = tmp_path / "output.md"

        runner = CliRunner()
        result = runner.invoke(
            main, ["show", "filecnf1", "--output", "markdown", "--file", str(out_file)]
        )
        assert result.exit_code == 0
        assert out_file.exists()
        # Stdout should not contain the markdown content (it went to the file).
        assert "# Debate:" not in result.output
