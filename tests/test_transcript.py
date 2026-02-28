"""Tests for transcript deserialization.

Covers: _parse_transcript_file() full deserialization of rounds, responses,
synthesis, timestamps, and backward compatibility with old transcripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.transcript import _parse_transcript_file


def _write_transcript(tmp_path: Path, data: dict[str, Any]) -> Path:
    """Write a transcript dict to a JSON file and return the path.

    Args:
        tmp_path: Pytest temporary directory.
        data: Transcript data to serialize.

    Returns:
        Path to the written JSON file.
    """
    filepath = tmp_path / "2026-02-28_abcd1234.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


def _make_response_dict(
    *,
    model_id: str = "anthropic/claude-sonnet-4.5",
    model_alias: str = "claude",
    round_number: int = 0,
    content: str = "Test response content",
    timestamp: str = "2026-02-28T12:00:00+00:00",
    token_count: int | None = 150,
    latency_ms: int | None = 1200,
    error: str | None = None,
    role: str = "initial",
    routing: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a response dict matching ModelResponse.to_dict() format.

    Args:
        model_id: OpenRouter model identifier.
        model_alias: Short display name.
        round_number: Round number (0=initial, 1+=reflection, -1=synthesis).
        content: Response text.
        timestamp: ISO 8601 timestamp string.
        token_count: Token usage count.
        latency_ms: Response time in ms.
        error: Error message or None.
        role: Debate role string.
        routing: Routing decision dict.
        analysis: Analysis metadata dict.

    Returns:
        Dictionary matching the serialized ModelResponse format.
    """
    d: dict[str, Any] = {
        "model_id": model_id,
        "model_alias": model_alias,
        "round_number": round_number,
        "content": content,
        "timestamp": timestamp,
        "token_count": token_count,
        "latency_ms": latency_ms,
        "error": error,
        "role": role,
        "routing": routing,
        "analysis": analysis if analysis is not None else {},
    }
    return d


def _make_full_transcript_dict() -> dict[str, Any]:
    """Build a complete transcript dict with rounds and synthesis.

    Returns:
        Dictionary matching the serialized DebateTranscript format.
    """
    return {
        "transcript_id": "abcd1234-5678-9abc-def0-123456789abc",
        "query": "What is the meaning of life?",
        "panel": ["claude", "gpt"],
        "synthesizer_id": "claude",
        "max_rounds": 1,
        "rounds": [
            {
                "round_number": 0,
                "round_type": "initial",
                "responses": [
                    _make_response_dict(
                        model_alias="claude",
                        content="Claude's initial response",
                        role="initial",
                    ),
                    _make_response_dict(
                        model_id="openai/gpt-4o",
                        model_alias="gpt",
                        content="GPT's initial response",
                        role="initial",
                    ),
                ],
            },
            {
                "round_number": 1,
                "round_type": "reflection",
                "responses": [
                    _make_response_dict(
                        model_alias="claude",
                        round_number=1,
                        content="Claude's reflection",
                        role="reflection",
                    ),
                    _make_response_dict(
                        model_id="openai/gpt-4o",
                        model_alias="gpt",
                        round_number=1,
                        content="GPT's reflection",
                        role="reflection",
                    ),
                ],
            },
        ],
        "synthesis": _make_response_dict(
            model_alias="claude",
            round_number=-1,
            content="Synthesis of the debate",
            role="synthesis",
            token_count=300,
            latency_ms=2500,
            routing={"vendor": "anthropic", "mode": "auto", "via_openrouter": True},
            analysis={"confidence": 0.95},
        ),
        "created_at": "2026-02-28T10:30:00+00:00",
        "metadata": {"version": "0.1.0"},
    }


class TestParseTranscriptFile:
    """Full deserialization of JSON transcript files into dataclass instances."""

    def test_rounds_deserialized(self, tmp_path: Path) -> None:
        """Rounds list is populated with correct count, number, and type."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        assert len(transcript.rounds) == 2
        assert isinstance(transcript.rounds[0], DebateRound)
        assert transcript.rounds[0].round_number == 0
        assert transcript.rounds[0].round_type == "initial"
        assert transcript.rounds[1].round_number == 1
        assert transcript.rounds[1].round_type == "reflection"

    def test_responses_in_round(self, tmp_path: Path) -> None:
        """Responses in a round have correct aliases."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        round_0 = transcript.rounds[0]
        assert len(round_0.responses) == 2
        aliases = [r.model_alias for r in round_0.responses]
        assert aliases == ["claude", "gpt"]

    def test_response_fields(self, tmp_path: Path) -> None:
        """All fields on a response match the source data."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        resp = transcript.rounds[0].responses[0]
        assert isinstance(resp, ModelResponse)
        assert resp.model_id == "anthropic/claude-sonnet-4.5"
        assert resp.content == "Claude's initial response"
        assert resp.token_count == 150
        assert resp.latency_ms == 1200
        assert resp.error is None
        assert resp.role == "initial"
        assert resp.routing is None
        assert resp.analysis == {}

    def test_synthesis_deserialized(self, tmp_path: Path) -> None:
        """Synthesis is a ModelResponse with correct fields."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        assert transcript.synthesis is not None
        assert isinstance(transcript.synthesis, ModelResponse)
        assert transcript.synthesis.model_alias == "claude"
        assert transcript.synthesis.round_number == -1
        assert transcript.synthesis.content == "Synthesis of the debate"
        assert transcript.synthesis.role == "synthesis"
        assert transcript.synthesis.token_count == 300
        assert transcript.synthesis.latency_ms == 2500
        assert transcript.synthesis.routing == {
            "vendor": "anthropic",
            "mode": "auto",
            "via_openrouter": True,
        }
        assert transcript.synthesis.analysis == {"confidence": 0.95}

    def test_timestamp_parsed_to_datetime(self, tmp_path: Path) -> None:
        """created_at, response timestamps, and synthesis timestamp are datetime."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        assert isinstance(transcript.created_at, datetime)
        assert isinstance(transcript.rounds[0].responses[0].timestamp, datetime)
        assert transcript.synthesis is not None
        assert isinstance(transcript.synthesis.timestamp, datetime)

    def test_created_at_parsed(self, tmp_path: Path) -> None:
        """Year, month, day match expected values from the source data."""
        data = _make_full_transcript_dict()
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        assert transcript.created_at.year == 2026
        assert transcript.created_at.month == 2
        assert transcript.created_at.day == 28

    def test_old_transcript_missing_role_routing_analysis(self, tmp_path: Path) -> None:
        """Transcripts missing role/routing/analysis still parse with defaults."""
        data = _make_full_transcript_dict()
        # Remove new fields from all responses to simulate old format
        for rnd in data["rounds"]:
            for resp in rnd["responses"]:
                resp.pop("role", None)
                resp.pop("routing", None)
                resp.pop("analysis", None)
        if data["synthesis"]:
            data["synthesis"].pop("role", None)
            data["synthesis"].pop("routing", None)
            data["synthesis"].pop("analysis", None)

        filepath = _write_transcript(tmp_path, data)
        transcript = _parse_transcript_file(filepath)

        resp = transcript.rounds[0].responses[0]
        assert resp.role == ""
        assert resp.routing is None
        assert resp.analysis == {}

        assert transcript.synthesis is not None
        assert transcript.synthesis.role == ""
        assert transcript.synthesis.routing is None
        assert transcript.synthesis.analysis == {}

    def test_null_synthesis(self, tmp_path: Path) -> None:
        """Transcript with null synthesis has synthesis=None."""
        data = _make_full_transcript_dict()
        data["synthesis"] = None
        filepath = _write_transcript(tmp_path, data)

        transcript = _parse_transcript_file(filepath)

        assert transcript.synthesis is None

    def test_round_trip_serialization(self, tmp_path: Path) -> None:
        """Create a transcript, serialize, parse back, verify key fields match."""
        original = DebateTranscript(
            transcript_id="roundtrip-1234-5678-9abc-def012345678",
            query="Round trip test query",
            panel=["claude", "gpt"],
            synthesizer_id="gpt",
            max_rounds=2,
            metadata={"version": "test"},
        )
        original.rounds.append(
            DebateRound(
                round_number=0,
                round_type="initial",
                responses=[
                    ModelResponse(
                        model_id="anthropic/claude-sonnet-4.5",
                        model_alias="claude",
                        round_number=0,
                        content="Round trip response",
                        token_count=100,
                        latency_ms=500,
                        role="initial",
                    ),
                ],
            )
        )
        original.synthesis = ModelResponse(
            model_id="openai/gpt-4o",
            model_alias="gpt",
            round_number=-1,
            content="Round trip synthesis",
            token_count=200,
            latency_ms=800,
            role="synthesis",
        )

        # Serialize
        filepath = tmp_path / "2026-02-28_roundtri.json"
        filepath.write_text(json.dumps(original.to_dict(), indent=2), encoding="utf-8")

        # Parse back
        restored = _parse_transcript_file(filepath)

        assert restored.transcript_id == original.transcript_id
        assert restored.query == original.query
        assert restored.panel == original.panel
        assert restored.synthesizer_id == original.synthesizer_id
        assert restored.max_rounds == original.max_rounds
        assert len(restored.rounds) == 1
        assert restored.rounds[0].round_number == 0
        assert restored.rounds[0].responses[0].content == "Round trip response"
        assert restored.synthesis is not None
        assert restored.synthesis.content == "Round trip synthesis"
        assert restored.synthesis.model_alias == "gpt"
        assert isinstance(restored.created_at, datetime)
