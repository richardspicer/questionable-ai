"""Tests for run_replay() orchestrator function.

Covers: re-synthesize-only mode, additional-rounds mode, synthesizer
override, metadata linking, and round numbering continuity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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


class TestRunReplayValidation:
    """run_replay() rejects invalid inputs."""

    @pytest.mark.asyncio
    async def test_negative_additional_rounds_raises(self) -> None:
        """Negative additional_rounds raises ValueError."""
        source = _make_source_transcript()
        with pytest.raises(ValueError, match="additional_rounds must be >= 0"):
            await run_replay(source, _config(), additional_rounds=-1)
