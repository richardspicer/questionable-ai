"""Tests for run_replay() orchestrator function and replay CLI command.

Covers: re-synthesize-only mode, additional-rounds mode, synthesizer
override, metadata linking, round numbering continuity, and CLI behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mutual_dissent.cli import main
from mutual_dissent.config import Config
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.orchestrator import run_replay

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PANEL = ["claude", "gpt"]
SYNTHESIZER = "claude"


def _make_response(
    alias: str,
    round_number: int,
    *,
    role: str = "",
    content: str | None = None,
    token_count: int = 100,
) -> ModelResponse:
    """Build a ModelResponse with predictable values.

    Args:
        alias: Model alias (e.g. "claude").
        round_number: Round number for the response.
        role: Debate role — "initial", "reflection", or "synthesis".
        content: Response text. Defaults to a generated string.
        token_count: Token count for the response.

    Returns:
        A ModelResponse with predictable fields.
    """
    return ModelResponse(
        model_id=f"vendor/{alias}-model",
        model_alias=alias,
        round_number=round_number,
        content=content or f"{alias} round {round_number} response",
        role=role,
        token_count=token_count,
    )


def _make_source_transcript(
    panel: list[str] | None = None,
    synthesizer_id: str = SYNTHESIZER,
    num_rounds: int = 1,
) -> DebateTranscript:
    """Build a DebateTranscript with initial round + N reflection rounds + synthesis.

    Args:
        panel: Panel aliases. Defaults to PANEL.
        synthesizer_id: Synthesizer alias.
        num_rounds: Number of reflection rounds (in addition to the initial round).

    Returns:
        A complete DebateTranscript suitable as a replay source.
    """
    panel = panel or list(PANEL)

    rounds: list[DebateRound] = []

    # Initial round (round 0).
    initial_responses = [_make_response(alias, 0, role="initial") for alias in panel]
    rounds.append(DebateRound(round_number=0, round_type="initial", responses=initial_responses))

    # Reflection rounds (1..num_rounds).
    for rn in range(1, num_rounds + 1):
        reflection_responses = [_make_response(alias, rn, role="reflection") for alias in panel]
        rounds.append(
            DebateRound(round_number=rn, round_type="reflection", responses=reflection_responses)
        )

    synthesis = _make_response(synthesizer_id, -1, role="synthesis", content="Original synthesis")

    return DebateTranscript(
        query="What is the meaning of life?",
        panel=panel,
        synthesizer_id=synthesizer_id,
        max_rounds=num_rounds,
        rounds=rounds,
        synthesis=synthesis,
        metadata={"version": "0.1.0"},
    )


def _make_mock_router() -> MagicMock:
    """Create an AsyncMock ProviderRouter that works as an async context manager.

    The mock's ``complete_parallel`` returns a list of ModelResponse objects
    with predictable content (one per request). The ``complete`` method
    returns a single ModelResponse for synthesis calls.

    Returns:
        MagicMock configured as an async context manager with complete
        and complete_parallel methods.
    """
    router = MagicMock()

    async def _enter(*_args: object) -> MagicMock:
        return router

    async def _exit(*_args: object) -> None:
        pass

    router.__aenter__ = _enter
    router.__aexit__ = _exit

    # complete_parallel: return one response per request dict.
    async def _complete_parallel(requests: list[dict[str, object]]) -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id=f"vendor/{req['model_alias']}-model",
                model_alias=str(req["model_alias"]),
                round_number=int(req.get("round_number", 0)),
                content=f"mock reflection {req['model_alias']} round {req.get('round_number', 0)}",
                token_count=50,
            )
            for req in requests
        ]

    # complete: return a single synthesis response.
    async def _complete(
        alias_or_id: str,
        *,
        messages: list[dict[str, object]] | None = None,
        prompt: str | None = None,
        model_alias: str = "",
        round_number: int = 0,
    ) -> ModelResponse:
        return ModelResponse(
            model_id=f"vendor/{model_alias or alias_or_id}-model",
            model_alias=model_alias or alias_or_id,
            round_number=round_number,
            content=f"mock synthesis by {model_alias or alias_or_id}",
            token_count=75,
        )

    router.complete_parallel = _complete_parallel
    router.complete = _complete

    return router


def _config() -> Config:
    """Build a minimal Config suitable for replay tests.

    Returns:
        Config with OpenRouter key set.
    """
    return Config(
        api_key="test-key",
        providers={"openrouter": "test-key"},
    )


# ---------------------------------------------------------------------------
# TestRunReplayResynthesizeOnly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestRunReplayResynthesizeOnly:
    """run_replay() with additional_rounds=0 re-synthesizes without adding rounds."""

    @pytest.fixture
    def source(self) -> DebateTranscript:
        """Source transcript with 1 reflection round."""
        return _make_source_transcript(num_rounds=1)

    @pytest.mark.asyncio
    async def test_new_transcript_id(self, source: DebateTranscript) -> None:
        """Result transcript has a different ID from source."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        assert result.transcript_id != source.transcript_id

    @pytest.mark.asyncio
    async def test_rounds_preserved(self, source: DebateTranscript) -> None:
        """Same number and types of rounds as source (no additional rounds)."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        assert len(result.rounds) == len(source.rounds)
        for src_round, res_round in zip(source.rounds, result.rounds, strict=True):
            assert res_round.round_type == src_round.round_type
            assert res_round.round_number == src_round.round_number

    @pytest.mark.asyncio
    async def test_new_synthesis(self, source: DebateTranscript) -> None:
        """Synthesis content comes from the mock, not the source."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        assert result.synthesis is not None
        assert result.synthesis.content != source.synthesis.content
        assert "mock synthesis" in result.synthesis.content

    @pytest.mark.asyncio
    async def test_source_metadata_linked(self, source: DebateTranscript) -> None:
        """metadata['source_transcript_id'] links to the source transcript."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        assert result.metadata["source_transcript_id"] == source.transcript_id

    @pytest.mark.asyncio
    async def test_replay_config_metadata(self, source: DebateTranscript) -> None:
        """metadata['replay_config'] records defaults for re-synthesize mode."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        replay_config = result.metadata["replay_config"]
        assert replay_config["synthesizer_override"] is None
        assert replay_config["additional_rounds"] == 0


# ---------------------------------------------------------------------------
# TestRunReplayWithAdditionalRounds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestRunReplayWithAdditionalRounds:
    """run_replay() with additional_rounds>0 appends reflection rounds."""

    @pytest.fixture
    def source(self) -> DebateTranscript:
        """Source transcript with 1 reflection round (2 rounds total: initial + 1 reflection)."""
        return _make_source_transcript(num_rounds=1)

    @pytest.mark.asyncio
    async def test_rounds_appended(self, source: DebateTranscript) -> None:
        """Total rounds = source rounds + additional rounds."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), additional_rounds=2)

        assert len(result.rounds) == len(source.rounds) + 2

    @pytest.mark.asyncio
    async def test_round_numbers_continue(self, source: DebateTranscript) -> None:
        """New rounds start numbering from len(source.rounds)."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), additional_rounds=2)

        source_count = len(source.rounds)
        new_rounds = result.rounds[source_count:]
        assert new_rounds[0].round_number == source_count
        assert new_rounds[1].round_number == source_count + 1

    @pytest.mark.asyncio
    async def test_new_rounds_are_reflection_type(self, source: DebateTranscript) -> None:
        """All appended rounds have round_type == 'reflection'."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), additional_rounds=2)

        source_count = len(source.rounds)
        for rnd in result.rounds[source_count:]:
            assert rnd.round_type == "reflection"

    @pytest.mark.asyncio
    async def test_max_rounds_updated(self, source: DebateTranscript) -> None:
        """max_rounds = source.max_rounds + additional_rounds."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), additional_rounds=3)

        assert result.max_rounds == source.max_rounds + 3

    @pytest.mark.asyncio
    async def test_replay_config_records_additional_rounds(self, source: DebateTranscript) -> None:
        """metadata['replay_config']['additional_rounds'] tracks the count."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), additional_rounds=2)

        assert result.metadata["replay_config"]["additional_rounds"] == 2


# ---------------------------------------------------------------------------
# TestRunReplaySynthesizerOverride
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestRunReplaySynthesizerOverride:
    """run_replay() synthesizer parameter overrides the source's synthesizer."""

    @pytest.fixture
    def source(self) -> DebateTranscript:
        """Source transcript synthesized by 'claude'."""
        return _make_source_transcript(synthesizer_id="claude", num_rounds=1)

    @pytest.mark.asyncio
    async def test_override_synthesizer(self, source: DebateTranscript) -> None:
        """Result uses the specified override synthesizer."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), synthesizer="gpt")

        assert result.synthesizer_id == "gpt"

    @pytest.mark.asyncio
    async def test_override_recorded_in_metadata(self, source: DebateTranscript) -> None:
        """replay_config['synthesizer_override'] records the override value."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config(), synthesizer="gpt")

        assert result.metadata["replay_config"]["synthesizer_override"] == "gpt"

    @pytest.mark.asyncio
    async def test_no_override_uses_source_synthesizer(self, source: DebateTranscript) -> None:
        """Without override, result uses the source transcript's synthesizer."""
        mock_router = _make_mock_router()
        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            result = await run_replay(source, _config())

        assert result.synthesizer_id == source.synthesizer_id
        assert result.metadata["replay_config"]["synthesizer_override"] is None


# ---------------------------------------------------------------------------
# TestRunReplayValidation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slow asyncio overhead on Windows — validated on Ubuntu CI",
)
class TestRunReplayValidation:
    """run_replay() rejects invalid inputs."""

    @pytest.mark.asyncio
    async def test_negative_additional_rounds_raises(self) -> None:
        """Negative additional_rounds raises ValueError."""
        source = _make_source_transcript()
        with pytest.raises(ValueError, match="additional_rounds must be >= 0"):
            await run_replay(source, _config(), additional_rounds=-1)


# ---------------------------------------------------------------------------
# Helpers -- transcript JSON fixtures for CLI tests
# ---------------------------------------------------------------------------


def _make_transcript_json(
    *,
    transcript_id: str = "abcd1234-5678-9abc-def0-123456789abc",
    query: str = "What is the meaning of life?",
    panel: list[str] | None = None,
    synthesizer_id: str = "claude",
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Build a complete transcript JSON dict for CLI test fixtures.

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
                        "model_id": f"provider/{alias}",
                        "model_alias": alias,
                        "round_number": 0,
                        "content": f"Initial response from {alias}.",
                        "timestamp": "2026-02-28T12:00:00+00:00",
                        "token_count": 100,
                        "latency_ms": 500,
                        "error": None,
                        "role": "initial",
                        "routing": None,
                        "analysis": {},
                    }
                    for alias in panel
                ],
            }
        ],
        "synthesis": {
            "model_id": f"provider/{synthesizer_id}",
            "model_alias": synthesizer_id,
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


# ---------------------------------------------------------------------------
# TestReplayCommand -- CLI
# ---------------------------------------------------------------------------


class TestReplayCommand:
    """``replay`` CLI command registration and behavior."""

    def test_replay_registered(self) -> None:
        """Main help output includes the replay command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "replay" in result.output

    def test_replay_shows_help(self) -> None:
        """replay --help shows expected options."""
        runner = CliRunner()
        result = runner.invoke(main, ["replay", "--help"])
        assert result.exit_code == 0
        assert "--synthesizer" in result.output
        assert "--rounds" in result.output
        assert "--verbose" in result.output
        assert "--no-save" in result.output
        assert "--output" in result.output
        assert "TRANSCRIPT_ID" in result.output

    def test_replay_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """replay exits 1 when no transcript matches the given ID."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        runner = CliRunner()
        result = runner.invoke(main, ["replay", "nonexist"])
        assert result.exit_code == 1
        assert "no transcript" in result.output.lower() or "not found" in result.output.lower()

    def test_replay_id_too_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """replay exits 1 when ID is fewer than 4 characters."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        runner = CliRunner()
        result = runner.invoke(main, ["replay", "abc"])
        assert result.exit_code == 1
        assert "at least 4" in result.output.lower()

    def test_replay_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """replay exits 1 when no API key is configured."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr("mutual_dissent.config.CONFIG_PATH", Path("/nonexistent"))

        runner = CliRunner()
        result = runner.invoke(main, ["replay", "abcd1234"])
        assert result.exit_code == 1
        assert "api key" in result.output.lower()

    def test_replay_json_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """replay --output json --no-save emits valid JSON with fresh transcript ID."""
        monkeypatch.setattr("mutual_dissent.transcript.TRANSCRIPT_DIR", tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Write a source transcript.
        tid = "replayjs-5678-9abc-def0-123456789abc"
        data = _make_transcript_json(transcript_id=tid)
        filepath = tmp_path / f"2026-02-28_{tid[:8]}.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        # Patch run_replay to avoid real API calls.
        async def fake_replay(
            source: DebateTranscript,
            config: Any,
            *,
            synthesizer: str | None = None,
            additional_rounds: int = 0,
            ground_truth: str | None = None,
        ) -> DebateTranscript:
            return DebateTranscript(
                query=source.query,
                panel=list(source.panel),
                synthesizer_id=synthesizer or source.synthesizer_id,
                max_rounds=source.max_rounds + additional_rounds,
                rounds=list(source.rounds),
                synthesis=ModelResponse(
                    model_id="provider/claude",
                    model_alias="claude",
                    round_number=-1,
                    content="Replayed synthesis.",
                    role="synthesis",
                ),
                metadata={"source_transcript_id": source.transcript_id},
            )

        monkeypatch.setattr("mutual_dissent.cli.run_replay", fake_replay)

        runner = CliRunner()
        result = runner.invoke(main, ["replay", "replayjs", "--output", "json", "--no-save"])
        assert result.exit_code == 0

        output_data = json.loads(result.output)
        assert output_data["transcript_id"] != tid
        assert output_data["metadata"]["source_transcript_id"] == tid
