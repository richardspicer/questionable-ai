"""Tests for cross-tool research scaffolding (Phase 3 prerequisites).

Covers: RoutedRequest.context field, ExperimentMetadata dataclass,
per-panelist context injection, round-level event hook, experiment
metadata serialization roundtrip, and display rendering with/without
experiment metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mutual_dissent.display import format_markdown
from mutual_dissent.models import (
    DebateRound,
    DebateTranscript,
    ExperimentMetadata,
    ModelResponse,
)
from mutual_dissent.orchestrator import (
    _fire_round_hook,
    _inject_context,
    run_debate,
)
from mutual_dissent.transcript import _parse_transcript_file
from mutual_dissent.types import RoutedRequest, Vendor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TIME = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)


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
        role: Debate role.
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
        timestamp=FIXED_TIME,
        role=role,
        token_count=token_count,
    )


def _make_transcript(
    *,
    experiment: ExperimentMetadata | None = None,
) -> DebateTranscript:
    """Build a DebateTranscript for display tests.

    Args:
        experiment: Optional experiment metadata to attach.

    Returns:
        A minimal DebateTranscript.
    """
    metadata: dict[str, Any] = {"version": "0.1.0"}
    if experiment is not None:
        metadata["experiment"] = experiment

    return DebateTranscript(
        transcript_id="scaff123-5678-9abc-def0-123456789abc",
        query="What is the meaning of life?",
        panel=["claude", "gpt"],
        synthesizer_id="claude",
        max_rounds=1,
        rounds=[
            DebateRound(
                round_number=0,
                round_type="initial",
                responses=[
                    _make_response("claude", 0, role="initial"),
                    _make_response("gpt", 0, role="initial"),
                ],
            ),
        ],
        synthesis=_make_response("claude", -1, role="synthesis", content="Synthesized answer."),
        created_at=FIXED_TIME,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# RoutedRequest.context
# ---------------------------------------------------------------------------


class TestRoutedRequestContext:
    """RoutedRequest has an optional context field."""

    def test_context_defaults_to_none(self) -> None:
        """Context field defaults to None when not provided."""
        req = RoutedRequest(
            vendor=Vendor.ANTHROPIC,
            model_id="claude-sonnet-4-5-20250929",
            model_alias="claude",
            round_number=0,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.context is None

    def test_context_can_be_set(self) -> None:
        """Context field accepts a string value."""
        req = RoutedRequest(
            vendor=Vendor.ANTHROPIC,
            model_id="claude-sonnet-4-5-20250929",
            model_alias="claude",
            round_number=0,
            messages=[],
            context="RAG context: relevant document excerpts",
        )
        assert req.context == "RAG context: relevant document excerpts"


# ---------------------------------------------------------------------------
# ExperimentMetadata
# ---------------------------------------------------------------------------


class TestExperimentMetadata:
    """ExperimentMetadata dataclass construction, defaults, and serialization."""

    def test_required_field(self) -> None:
        """experiment_id is required."""
        em = ExperimentMetadata(experiment_id="exp-001")
        assert em.experiment_id == "exp-001"

    def test_defaults(self) -> None:
        """Optional fields have correct defaults."""
        em = ExperimentMetadata(experiment_id="exp-001")
        assert em.source_tool == "manual"
        assert em.campaign_id is None
        assert em.condition == ""
        assert em.variables == {}
        assert em.finding_ref is None

    def test_variables_default_is_independent(self) -> None:
        """Each instance gets its own variables dict."""
        em1 = ExperimentMetadata(experiment_id="a")
        em2 = ExperimentMetadata(experiment_id="b")
        em1.variables["key"] = "value"
        assert "key" not in em2.variables

    def test_full_construction(self) -> None:
        """All fields can be set explicitly."""
        em = ExperimentMetadata(
            experiment_id="exp-042",
            source_tool="countersignal",
            campaign_id="campaign-99",
            condition="rag-augmented",
            variables={"model": "claude", "context_size": 2048},
            finding_ref="MD-003",
        )
        assert em.experiment_id == "exp-042"
        assert em.source_tool == "countersignal"
        assert em.campaign_id == "campaign-99"
        assert em.condition == "rag-augmented"
        assert em.variables == {"model": "claude", "context_size": 2048}
        assert em.finding_ref == "MD-003"

    def test_to_dict(self) -> None:
        """to_dict() includes all fields."""
        em = ExperimentMetadata(
            experiment_id="exp-001",
            source_tool="counteragent",
            campaign_id="c-1",
            condition="baseline",
            variables={"rounds": 2},
            finding_ref="MCP-001",
        )
        d = em.to_dict()
        assert d == {
            "experiment_id": "exp-001",
            "source_tool": "counteragent",
            "campaign_id": "c-1",
            "condition": "baseline",
            "variables": {"rounds": 2},
            "finding_ref": "MCP-001",
        }

    def test_to_dict_json_serializable(self) -> None:
        """to_dict() output is fully JSON-serializable."""
        em = ExperimentMetadata(experiment_id="exp-001")
        result = json.dumps(em.to_dict())
        parsed = json.loads(result)
        assert parsed["experiment_id"] == "exp-001"

    def test_from_dict_full(self) -> None:
        """from_dict() reconstructs all fields."""
        data = {
            "experiment_id": "exp-001",
            "source_tool": "countersignal",
            "campaign_id": "c-1",
            "condition": "rag",
            "variables": {"k": "v"},
            "finding_ref": "MD-001",
        }
        em = ExperimentMetadata.from_dict(data)
        assert em.experiment_id == "exp-001"
        assert em.source_tool == "countersignal"
        assert em.campaign_id == "c-1"
        assert em.condition == "rag"
        assert em.variables == {"k": "v"}
        assert em.finding_ref == "MD-001"

    def test_from_dict_defaults(self) -> None:
        """from_dict() applies defaults for missing optional fields."""
        data = {"experiment_id": "exp-002"}
        em = ExperimentMetadata.from_dict(data)
        assert em.source_tool == "manual"
        assert em.campaign_id is None
        assert em.condition == ""
        assert em.variables == {}
        assert em.finding_ref is None

    def test_roundtrip(self) -> None:
        """to_dict() → from_dict() produces an equivalent object."""
        original = ExperimentMetadata(
            experiment_id="rt-001",
            source_tool="counteragent",
            campaign_id="camp-5",
            condition="poisoned",
            variables={"injection": True, "delay_ms": 500},
            finding_ref="MD-007",
        )
        restored = ExperimentMetadata.from_dict(original.to_dict())
        assert restored.experiment_id == original.experiment_id
        assert restored.source_tool == original.source_tool
        assert restored.campaign_id == original.campaign_id
        assert restored.condition == original.condition
        assert restored.variables == original.variables
        assert restored.finding_ref == original.finding_ref


# ---------------------------------------------------------------------------
# _inject_context
# ---------------------------------------------------------------------------


class TestInjectContext:
    """_inject_context() prepends per-panelist context to prompts."""

    def test_no_context_map(self) -> None:
        """Returns prompt unchanged when panelist_context is None."""
        result = _inject_context("original prompt", "claude", None)
        assert result == "original prompt"

    def test_empty_context_map(self) -> None:
        """Returns prompt unchanged when panelist_context is empty dict."""
        result = _inject_context("original prompt", "claude", {})
        assert result == "original prompt"

    def test_alias_not_in_map(self) -> None:
        """Returns prompt unchanged when alias is not in context map."""
        result = _inject_context("original prompt", "gpt", {"claude": "ctx"})
        assert result == "original prompt"

    def test_context_prepended(self) -> None:
        """Context is prepended with double newline separator."""
        result = _inject_context("original prompt", "claude", {"claude": "RAG context here"})
        assert result == "RAG context here\n\noriginal prompt"

    def test_only_matching_alias_gets_context(self) -> None:
        """Only the matching alias's context is injected."""
        ctx = {"claude": "Claude context", "gpt": "GPT context"}
        result_claude = _inject_context("prompt", "claude", ctx)
        result_gpt = _inject_context("prompt", "gpt", ctx)
        assert result_claude.startswith("Claude context")
        assert result_gpt.startswith("GPT context")


# ---------------------------------------------------------------------------
# _fire_round_hook
# ---------------------------------------------------------------------------


class TestFireRoundHook:
    """_fire_round_hook() invokes callback and handles errors."""

    @pytest.mark.asyncio
    async def test_none_callback_is_noop(self) -> None:
        """No error when callback is None."""
        rnd = DebateRound(round_number=0, round_type="initial")
        await _fire_round_hook(None, rnd)  # Should not raise.

    @pytest.mark.asyncio
    async def test_callback_receives_round(self) -> None:
        """Callback receives the completed DebateRound."""
        received: list[DebateRound] = []

        async def hook(rnd: DebateRound) -> None:
            received.append(rnd)

        debate_round = DebateRound(round_number=1, round_type="reflection")
        await _fire_round_hook(hook, debate_round)

        assert len(received) == 1
        assert received[0] is debate_round

    @pytest.mark.asyncio
    async def test_callback_exception_logged_not_propagated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Callback exceptions are logged but do not propagate."""

        async def bad_hook(rnd: DebateRound) -> None:
            raise RuntimeError("callback boom")

        debate_round = DebateRound(round_number=0, round_type="initial")

        with caplog.at_level(logging.ERROR, logger="mutual_dissent.orchestrator"):
            await _fire_round_hook(bad_hook, debate_round)

        assert "callback boom" in caplog.text
        assert "on_round_complete callback failed" in caplog.text


# ---------------------------------------------------------------------------
# Orchestrator integration — context and callback with run_debate
# ---------------------------------------------------------------------------


def _make_mock_router() -> MagicMock:
    """Create a mock ProviderRouter for orchestrator tests.

    Returns:
        MagicMock configured as an async context manager.
    """
    router = MagicMock()

    async def _enter(*_args: object) -> MagicMock:
        return router

    async def _exit(*_args: object) -> None:
        pass

    router.__aenter__ = _enter
    router.__aexit__ = _exit

    async def _complete_parallel(
        requests: list[dict[str, object]],
    ) -> list[ModelResponse]:
        return [
            ModelResponse(
                model_id=f"vendor/{req['model_alias']}-model",
                model_alias=str(req["model_alias"]),
                round_number=int(req.get("round_number", 0)),
                content=str(req.get("prompt", "")),
                token_count=50,
            )
            for req in requests
        ]

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
            content=f"synthesis by {model_alias or alias_or_id}",
            token_count=75,
        )

    router.complete_parallel = _complete_parallel
    router.complete = _complete

    return router


def _test_config() -> Any:
    """Build a minimal Config for tests.

    Returns:
        Config with test key set.
    """
    from mutual_dissent.config import Config

    return Config(
        api_key="test-key",
        providers={"openrouter": "test-key"},
    )


class TestOrchestratorContextIntegration:
    """run_debate() passes panelist_context through to prompts and metadata."""

    @pytest.mark.asyncio
    async def test_context_in_prompt(self) -> None:
        """Panelist context appears in the prompt passed to complete_parallel."""
        captured_requests: list[list[dict[str, object]]] = []
        mock_router = _make_mock_router()

        original_cp = mock_router.complete_parallel

        async def _capture_cp(
            requests: list[dict[str, object]],
        ) -> list[ModelResponse]:
            captured_requests.append(requests)
            return await original_cp(requests)

        mock_router.complete_parallel = _capture_cp

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            await run_debate(
                "test query",
                _test_config(),
                panel=["claude", "gpt"],
                rounds=1,
                panelist_context={"claude": "Claude RAG context"},
            )

        # Initial round requests (first call to complete_parallel).
        initial_requests = captured_requests[0]
        claude_prompt = str(
            next(r["prompt"] for r in initial_requests if r["model_alias"] == "claude")
        )
        gpt_prompt = str(next(r["prompt"] for r in initial_requests if r["model_alias"] == "gpt"))

        assert "Claude RAG context" in claude_prompt
        assert "Claude RAG context" not in gpt_prompt

    @pytest.mark.asyncio
    async def test_context_persists_across_rounds(self) -> None:
        """Context is injected in both initial and reflection rounds."""
        captured_requests: list[list[dict[str, object]]] = []
        mock_router = _make_mock_router()

        original_cp = mock_router.complete_parallel

        async def _capture_cp(
            requests: list[dict[str, object]],
        ) -> list[ModelResponse]:
            captured_requests.append(requests)
            return await original_cp(requests)

        mock_router.complete_parallel = _capture_cp

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            await run_debate(
                "test query",
                _test_config(),
                panel=["claude"],
                rounds=1,
                panelist_context={"claude": "Persistent context"},
            )

        # Should have 2 calls: initial + 1 reflection.
        assert len(captured_requests) >= 2
        for reqs in captured_requests:
            claude_prompt = str(next(r["prompt"] for r in reqs if r["model_alias"] == "claude"))
            assert "Persistent context" in claude_prompt

    @pytest.mark.asyncio
    async def test_context_stored_in_metadata(self) -> None:
        """panelist_context is stored in transcript.metadata."""
        mock_router = _make_mock_router()
        ctx = {"claude": "ctx-claude", "gpt": "ctx-gpt"}

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            transcript = await run_debate(
                "test query",
                _test_config(),
                panel=["claude", "gpt"],
                rounds=1,
                panelist_context=ctx,
            )

        assert transcript.metadata["panelist_context"] == ctx

    @pytest.mark.asyncio
    async def test_no_context_metadata_when_none(self) -> None:
        """panelist_context key absent from metadata when not provided."""
        mock_router = _make_mock_router()

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            transcript = await run_debate(
                "test query",
                _test_config(),
                panel=["claude"],
                rounds=1,
            )

        assert "panelist_context" not in transcript.metadata


class TestOrchestratorCallbackIntegration:
    """run_debate() fires on_round_complete for each round."""

    @pytest.mark.asyncio
    async def test_callback_fires_for_each_round(self) -> None:
        """Callback fires for initial, reflection, and synthesis rounds."""
        received: list[DebateRound] = []

        async def hook(rnd: DebateRound) -> None:
            received.append(rnd)

        mock_router = _make_mock_router()

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            await run_debate(
                "test query",
                _test_config(),
                panel=["claude"],
                rounds=1,
                on_round_complete=hook,
            )

        # initial + 1 reflection + synthesis = 3 callbacks.
        assert len(received) == 3
        assert received[0].round_type == "initial"
        assert received[1].round_type == "reflection"
        assert received[2].round_type == "synthesis"

    @pytest.mark.asyncio
    async def test_callback_error_does_not_abort_debate(self) -> None:
        """A failing callback does not prevent the debate from completing."""
        call_count = 0

        async def bad_hook(rnd: DebateRound) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("hook error")

        mock_router = _make_mock_router()

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            transcript = await run_debate(
                "test query",
                _test_config(),
                panel=["claude"],
                rounds=1,
                on_round_complete=bad_hook,
            )

        # Debate should complete despite hook errors.
        assert transcript.synthesis is not None
        assert call_count == 3  # All hooks were attempted.

    @pytest.mark.asyncio
    async def test_no_callback_works(self) -> None:
        """Debate completes normally with no callback (default behavior)."""
        mock_router = _make_mock_router()

        with patch("mutual_dissent.orchestrator.ProviderRouter", return_value=mock_router):
            transcript = await run_debate(
                "test query",
                _test_config(),
                panel=["claude"],
                rounds=1,
            )

        assert transcript.synthesis is not None
        assert len(transcript.rounds) == 2  # initial + 1 reflection


# ---------------------------------------------------------------------------
# ExperimentMetadata serialization roundtrip via transcript
# ---------------------------------------------------------------------------


class TestExperimentMetadataTranscriptRoundtrip:
    """ExperimentMetadata serializes to/from JSON transcripts correctly."""

    def test_roundtrip_via_transcript_file(self, tmp_path: Path) -> None:
        """ExperimentMetadata survives save → load via transcript file."""
        experiment = ExperimentMetadata(
            experiment_id="exp-rt-001",
            source_tool="countersignal",
            campaign_id="camp-42",
            condition="rag-augmented",
            variables={"context_size": 4096},
            finding_ref="MD-003",
        )
        transcript = _make_transcript(experiment=experiment)

        # Serialize to file.
        filepath = tmp_path / "2026-03-01_scaff123.json"
        filepath.write_text(json.dumps(transcript.to_dict(), indent=2), encoding="utf-8")

        # Deserialize.
        restored = _parse_transcript_file(filepath)

        em = restored.metadata.get("experiment")
        assert isinstance(em, ExperimentMetadata)
        assert em.experiment_id == "exp-rt-001"
        assert em.source_tool == "countersignal"
        assert em.campaign_id == "camp-42"
        assert em.condition == "rag-augmented"
        assert em.variables == {"context_size": 4096}
        assert em.finding_ref == "MD-003"

    def test_no_experiment_backward_compat(self, tmp_path: Path) -> None:
        """Transcripts without experiment metadata load without error."""
        transcript = _make_transcript()  # No experiment.

        filepath = tmp_path / "2026-03-01_scaff123.json"
        filepath.write_text(json.dumps(transcript.to_dict(), indent=2), encoding="utf-8")

        restored = _parse_transcript_file(filepath)
        assert "experiment" not in restored.metadata

    def test_to_dict_serializes_experiment(self) -> None:
        """DebateTranscript.to_dict() converts ExperimentMetadata to dict."""
        experiment = ExperimentMetadata(experiment_id="exp-ser-001")
        transcript = _make_transcript(experiment=experiment)

        data = transcript.to_dict()
        em_data = data["metadata"]["experiment"]

        assert isinstance(em_data, dict)
        assert em_data["experiment_id"] == "exp-ser-001"

    def test_to_dict_without_experiment(self) -> None:
        """DebateTranscript.to_dict() works without experiment metadata."""
        transcript = _make_transcript()
        data = transcript.to_dict()

        # Should be JSON-serializable without error.
        json.dumps(data)
        assert "experiment" not in data["metadata"]


# ---------------------------------------------------------------------------
# Display rendering with experiment metadata
# ---------------------------------------------------------------------------


class TestDisplayExperimentMetadata:
    """Experiment metadata appears in terminal and markdown output."""

    def test_markdown_includes_experiment(self) -> None:
        """format_markdown() includes experiment line when present."""
        experiment = ExperimentMetadata(
            experiment_id="exp-disp-001",
            source_tool="countersignal",
        )
        transcript = _make_transcript(experiment=experiment)
        result = format_markdown(transcript)

        assert "**Experiment:** exp-disp-001 (countersignal)" in result

    def test_markdown_omits_experiment_when_absent(self) -> None:
        """format_markdown() has no experiment line without metadata."""
        transcript = _make_transcript()
        result = format_markdown(transcript)

        assert "**Experiment:**" not in result

    def test_markdown_experiment_manual_source(self) -> None:
        """Manual source_tool renders correctly."""
        experiment = ExperimentMetadata(experiment_id="exp-m-001")
        transcript = _make_transcript(experiment=experiment)
        result = format_markdown(transcript)

        assert "**Experiment:** exp-m-001 (manual)" in result

    def test_render_debate_with_experiment(self, capsys: pytest.CaptureFixture[str]) -> None:
        """render_debate() shows experiment info in terminal output."""
        from mutual_dissent.display import render_debate

        experiment = ExperimentMetadata(
            experiment_id="exp-term-001",
            source_tool="counteragent",
        )
        transcript = _make_transcript(experiment=experiment)
        render_debate(transcript)

        captured = capsys.readouterr()
        assert "exp-term-001" in captured.out
        assert "counteragent" in captured.out

    def test_render_debate_without_experiment(self, capsys: pytest.CaptureFixture[str]) -> None:
        """render_debate() omits experiment line when absent."""
        from mutual_dissent.display import render_debate

        transcript = _make_transcript()
        render_debate(transcript)

        captured = capsys.readouterr()
        assert "Experiment" not in captured.out
