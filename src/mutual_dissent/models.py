"""Data models for debate transcripts and API responses.

Defines the core data structures used throughout Mutual Dissent for representing
model responses, debate rounds, and complete debate transcripts. All models
are serializable to JSON for transcript logging.

Typical usage::

    from mutual_dissent.models import DebateTranscript, ModelResponse

    response = ModelResponse(
        model_id="anthropic/claude-sonnet-4.5",
        model_alias="claude",
        round_number=0,
        content="...",
    )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ModelResponse:
    """Single response from one model in one round.

    Attributes:
        model_id: OpenRouter model identifier (e.g. "anthropic/claude-sonnet-4.5").
        model_alias: Human-readable short name (e.g. "claude").
        round_number: 0 for initial, 1+ for reflection, -1 for synthesis.
        content: Full response text.
        timestamp: When the response was received (UTC).
        token_count: Total tokens used, if reported by API.
        latency_ms: Response time in milliseconds.
        error: Error message if the call failed, None on success.
    """

    model_id: str
    model_alias: str
    round_number: int
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    token_count: int | None = None
    latency_ms: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with all fields, datetime as ISO string.
        """
        return {
            "model_id": self.model_id,
            "model_alias": self.model_alias,
            "round_number": self.round_number,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class DebateRound:
    """All responses from one round of the debate.

    Attributes:
        round_number: 0 for initial, 1+ for reflection.
        round_type: One of "initial", "reflection", "synthesis".
        responses: List of ModelResponse objects from this round.
    """

    round_number: int
    round_type: str
    responses: list[ModelResponse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with round metadata and serialized responses.
        """
        return {
            "round_number": self.round_number,
            "round_type": self.round_type,
            "responses": [r.to_dict() for r in self.responses],
        }


@dataclass
class DebateTranscript:
    """Complete record of a debate session.

    Attributes:
        transcript_id: Unique identifier (UUID4).
        query: Original user query.
        panel: List of OpenRouter model IDs that participated.
        synthesizer_id: Model ID selected for synthesis.
        max_rounds: Configured number of reflection rounds.
        rounds: List of completed debate rounds.
        synthesis: Final synthesized response, if completed.
        created_at: When the debate started (UTC).
        metadata: Additional context (version, config, etc.).
    """

    transcript_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    panel: list[str] = field(default_factory=list)
    synthesizer_id: str = ""
    max_rounds: int = 1
    rounds: list[DebateRound] = field(default_factory=list)
    synthesis: ModelResponse | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Full transcript as a nested dictionary suitable for JSON output.
        """
        return {
            "transcript_id": self.transcript_id,
            "query": self.query,
            "panel": self.panel,
            "synthesizer_id": self.synthesizer_id,
            "max_rounds": self.max_rounds,
            "rounds": [r.to_dict() for r in self.rounds],
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @property
    def short_id(self) -> str:
        """First 8 characters of the transcript ID for display.

        Returns:
            Truncated UUID string.
        """
        return self.transcript_id[:8]
