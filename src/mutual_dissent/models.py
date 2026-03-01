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
class ExperimentMetadata:
    """Metadata linking a debate to a research experiment.

    Serializes into ``DebateTranscript.metadata["experiment"]`` to track
    cross-tool research runs (CounterSignal, CounterAgent, manual).

    Attributes:
        experiment_id: Groups related debate runs under one experiment.
        source_tool: Originating tool — "countersignal", "counteragent",
            or "manual".
        campaign_id: Optional link to an external campaign or scan ID.
        condition: Description of the experimental variable being tested.
        variables: Key-value pairs of parameter values for this run.
        finding_ref: Reference code for a research finding (e.g. "MD-003").
    """

    experiment_id: str
    source_tool: str = "manual"
    campaign_id: str | None = None
    condition: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    finding_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with all fields suitable for JSON storage.
        """
        return {
            "experiment_id": self.experiment_id,
            "source_tool": self.source_tool,
            "campaign_id": self.campaign_id,
            "condition": self.condition,
            "variables": self.variables,
            "finding_ref": self.finding_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetadata:
        """Deserialize from a JSON-compatible dictionary.

        Args:
            data: Dictionary with ExperimentMetadata fields.

        Returns:
            ExperimentMetadata instance.
        """
        return cls(
            experiment_id=data["experiment_id"],
            source_tool=data.get("source_tool", "manual"),
            campaign_id=data.get("campaign_id"),
            condition=data.get("condition", ""),
            variables=data.get("variables", {}),
            finding_ref=data.get("finding_ref"),
        )


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
        input_tokens: Prompt/input tokens, if reported by API.
        output_tokens: Completion/output tokens, if reported by API.
        latency_ms: Response time in milliseconds.
        error: Error message if the call failed, None on success.
        role: Debate role — "initial", "reflection", or "synthesis".
            Empty string when not set.
        routing: Serialized RoutingDecision (via to_dict()). None when
            not set. Stored as dict to keep models.py free of routing
            type imports and simplify JSON deserialization.
        analysis: Reserved for future scoring metadata. Empty dict for now.
    """

    model_id: str
    model_alias: str
    round_number: int
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    token_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    error: str | None = None
    role: str = ""
    routing: dict[str, Any] | None = None
    analysis: dict[str, Any] = field(default_factory=dict)

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
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "role": self.role,
            "routing": self.routing,
            "analysis": self.analysis,
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
        panel: List of model aliases that participated (e.g. ["claude", "gpt"]).
        synthesizer_id: Model alias selected for synthesis (e.g. "claude").
        max_rounds: Configured number of reflection rounds.
        rounds: List of completed debate rounds.
        synthesis: Final synthesized response, if completed.
        created_at: When the debate started (UTC).
        metadata: Additional context (version, resolved_config, stats, etc.).
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
        # Serialize metadata values that have their own to_dict().
        metadata = dict(self.metadata)
        experiment = metadata.get("experiment")
        if isinstance(experiment, ExperimentMetadata):
            metadata["experiment"] = experiment.to_dict()

        return {
            "transcript_id": self.transcript_id,
            "query": self.query,
            "panel": self.panel,
            "synthesizer_id": self.synthesizer_id,
            "max_rounds": self.max_rounds,
            "rounds": [r.to_dict() for r in self.rounds],
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "created_at": self.created_at.isoformat(),
            "metadata": metadata,
        }

    @property
    def short_id(self) -> str:
        """First 8 characters of the transcript ID for display.

        Returns:
            Truncated UUID string.
        """
        return self.transcript_id[:8]
