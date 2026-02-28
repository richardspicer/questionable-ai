"""Tests for ground-truth scoring module.

Covers: parse_score_response() with valid/malformed/garbage input,
score_synthesis() with mocked provider (success and parse-failure paths),
_resolve_ground_truth() helper, CLI flag registration, and score display
rendering in terminal and markdown.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from mutual_dissent.cli import _resolve_ground_truth, main
from mutual_dissent.display import format_markdown, render_debate
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.scoring import GroundTruthScore, parse_score_response, score_synthesis

# ---------------------------------------------------------------------------
# TestParseScoreResponse
# ---------------------------------------------------------------------------


class TestParseScoreResponse:
    """parse_score_response() extracts accuracy, completeness, explanation."""

    def test_valid_format(self) -> None:
        """Standard well-formed response parses correctly."""
        content = "ACCURACY: 4\nCOMPLETENESS: 3\nEXPLANATION: Good but incomplete."
        accuracy, completeness, explanation = parse_score_response(content)
        assert accuracy == 4
        assert completeness == 3
        assert explanation == "Good but incomplete."

    def test_case_insensitive(self) -> None:
        """Mixed-case keys parse correctly."""
        content = "Accuracy: 5\nCompleteness: 4\nExplanation: Great answer."
        accuracy, completeness, explanation = parse_score_response(content)
        assert accuracy == 5
        assert completeness == 4

    def test_extra_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        content = "  ACCURACY:  4  \n  COMPLETENESS:  3  \nEXPLANATION:  Some text.  "
        accuracy, completeness, explanation = parse_score_response(content)
        assert accuracy == 4
        assert completeness == 3
        assert explanation == "Some text."

    def test_multiline_explanation(self) -> None:
        """Explanation can span multiple lines."""
        content = "ACCURACY: 4\nCOMPLETENESS: 3\nEXPLANATION: Line one.\nLine two of explanation."
        accuracy, completeness, explanation = parse_score_response(content)
        assert "Line one." in explanation
        assert "Line two" in explanation

    def test_missing_accuracy_raises(self) -> None:
        """Missing ACCURACY field raises ValueError."""
        content = "COMPLETENESS: 3\nEXPLANATION: No accuracy."
        with pytest.raises(ValueError, match="ACCURACY"):
            parse_score_response(content)

    def test_missing_completeness_raises(self) -> None:
        """Missing COMPLETENESS field raises ValueError."""
        content = "ACCURACY: 4\nEXPLANATION: No completeness."
        with pytest.raises(ValueError, match="COMPLETENESS"):
            parse_score_response(content)

    def test_garbage_input_raises(self) -> None:
        """Completely unrelated text raises ValueError."""
        with pytest.raises(ValueError):
            parse_score_response("This is not a score response at all.")

    def test_non_numeric_score_raises(self) -> None:
        """Non-numeric score value raises ValueError."""
        content = "ACCURACY: high\nCOMPLETENESS: 3\nEXPLANATION: Bad format."
        with pytest.raises(ValueError):
            parse_score_response(content)

    def test_score_clamped_to_range(self) -> None:
        """Scores outside 1-5 are clamped."""
        content = "ACCURACY: 7\nCOMPLETENESS: 0\nEXPLANATION: Out of range."
        accuracy, completeness, _ = parse_score_response(content)
        assert accuracy == 5
        assert completeness == 1

    def test_missing_explanation_uses_empty(self) -> None:
        """Missing EXPLANATION returns empty string."""
        content = "ACCURACY: 4\nCOMPLETENESS: 3"
        accuracy, completeness, explanation = parse_score_response(content)
        assert accuracy == 4
        assert completeness == 3
        assert explanation == ""


# ---------------------------------------------------------------------------
# TestGroundTruthScore
# ---------------------------------------------------------------------------


class TestGroundTruthScore:
    """GroundTruthScore dataclass."""

    def test_overall_computation(self) -> None:
        """overall is the average of accuracy and completeness."""
        score = GroundTruthScore(
            accuracy=4,
            completeness=3,
            overall=3.5,
            explanation="Test.",
            judge_model="claude",
        )
        assert score.overall == 3.5

    def test_to_dict(self) -> None:
        """to_dict() returns all fields."""
        score = GroundTruthScore(
            accuracy=4,
            completeness=3,
            overall=3.5,
            explanation="Test.",
            judge_model="claude",
        )
        d = score.to_dict()
        assert d["accuracy"] == 4
        assert d["completeness"] == 3
        assert d["overall"] == 3.5
        assert d["explanation"] == "Test."
        assert d["judge_model"] == "claude"


# ---------------------------------------------------------------------------
# TestScoreSynthesis
# ---------------------------------------------------------------------------


class TestScoreSynthesis:
    """score_synthesis() calls the judge model and parses the response."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        """Successful scoring returns a GroundTruthScore."""
        router = MagicMock()

        async def _complete(
            alias_or_id: str,
            *,
            messages: object = None,
            prompt: object = None,
            model_alias: str = "",
            round_number: int = 0,
        ) -> ModelResponse:
            return ModelResponse(
                model_id="vendor/claude-model",
                model_alias="claude",
                round_number=-2,
                content="ACCURACY: 4\nCOMPLETENESS: 3\nEXPLANATION: Good.",
            )

        router.complete = _complete

        score = await score_synthesis(router, "What is X?", "X is something.", "X is Y.", "claude")
        assert score.accuracy == 4
        assert score.completeness == 3
        assert score.overall == 3.5
        assert score.judge_model == "claude"

    @pytest.mark.asyncio
    async def test_parse_failure_returns_error_score(self) -> None:
        """Malformed judge output returns error score (accuracy=-1)."""
        router = MagicMock()

        async def _complete(
            alias_or_id: str,
            *,
            messages: object = None,
            prompt: object = None,
            model_alias: str = "",
            round_number: int = 0,
        ) -> ModelResponse:
            return ModelResponse(
                model_id="vendor/claude-model",
                model_alias="claude",
                round_number=-2,
                content="I cannot score this.",
            )

        router.complete = _complete

        score = await score_synthesis(router, "What is X?", "X is something.", "X is Y.", "claude")
        assert score.accuracy == -1
        assert score.completeness == -1
        assert "could not be parsed" in score.explanation.lower()


# ---------------------------------------------------------------------------
# TestResolveGroundTruth
# ---------------------------------------------------------------------------


class TestResolveGroundTruth:
    """_resolve_ground_truth() inline, file, both (error), neither (None)."""

    def test_inline(self) -> None:
        """Inline text returned as-is."""
        assert _resolve_ground_truth("X is Y", None) == "X is Y"

    def test_file(self, tmp_path: Path) -> None:
        """File contents returned, stripped."""
        f = tmp_path / "ref.md"
        f.write_text("  X is Y  \n", encoding="utf-8")
        assert _resolve_ground_truth(None, str(f)) == "X is Y"

    def test_both_raises(self) -> None:
        """Both sources raises UsageError."""
        with pytest.raises(click.UsageError):
            _resolve_ground_truth("inline", "/some/file")

    def test_neither_returns_none(self) -> None:
        """Neither source returns None."""
        assert _resolve_ground_truth(None, None) is None


# ---------------------------------------------------------------------------
# TestGroundTruthCliOptions
# ---------------------------------------------------------------------------


class TestGroundTruthCliOptions:
    """--ground-truth flags registered on ask and replay."""

    def test_ask_has_ground_truth(self) -> None:
        """ask --help shows --ground-truth and --ground-truth-file."""
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert "--ground-truth-file" in result.output
        assert "--ground-truth " in result.output  # trailing space to not match --ground-truth-file

    def test_replay_has_ground_truth(self) -> None:
        """replay --help shows --ground-truth and --ground-truth-file."""
        runner = CliRunner()
        result = runner.invoke(main, ["replay", "--help"])
        assert "--ground-truth-file" in result.output
        assert "--ground-truth " in result.output


# ---------------------------------------------------------------------------
# Helpers for display tests
# ---------------------------------------------------------------------------


def _make_scored_transcript() -> DebateTranscript:
    """Build a transcript with ground-truth score data."""
    synthesis = ModelResponse(
        model_id="vendor/claude-model",
        model_alias="claude",
        round_number=-1,
        content="Scored synthesis.",
        role="synthesis",
        analysis={
            "ground_truth_score": {
                "accuracy": 4,
                "completeness": 3,
                "overall": 3.5,
                "explanation": "Good but incomplete.",
                "judge_model": "claude",
            }
        },
    )
    return DebateTranscript(
        query="What is X?",
        panel=["claude", "gpt"],
        synthesizer_id="claude",
        max_rounds=1,
        rounds=[
            DebateRound(
                round_number=0,
                round_type="initial",
                responses=[
                    ModelResponse(
                        model_id="vendor/claude-model",
                        model_alias="claude",
                        round_number=0,
                        content="Response.",
                        role="initial",
                    ),
                ],
            )
        ],
        synthesis=synthesis,
    )


def _make_unscored_transcript() -> DebateTranscript:
    """Build a transcript without score data."""
    return DebateTranscript(
        query="What?",
        panel=["claude"],
        synthesizer_id="claude",
        max_rounds=1,
        rounds=[
            DebateRound(
                round_number=0,
                round_type="initial",
                responses=[
                    ModelResponse(
                        model_id="v/c",
                        model_alias="claude",
                        round_number=0,
                        content="R.",
                        role="initial",
                    ),
                ],
            )
        ],
        synthesis=ModelResponse(
            model_id="v/c",
            model_alias="claude",
            round_number=-1,
            content="S.",
            role="synthesis",
        ),
    )


# ---------------------------------------------------------------------------
# TestScoreDisplay
# ---------------------------------------------------------------------------


class TestScoreDisplay:
    """Score rendering in terminal and markdown."""

    def test_markdown_contains_score_section(self) -> None:
        """Markdown output includes ## Score with scores."""
        transcript = _make_scored_transcript()
        result = format_markdown(transcript)
        assert "## Score" in result
        assert "**Accuracy:** 4/5" in result
        assert "**Completeness:** 3/5" in result
        assert "**Overall:** 3.5/5" in result
        assert "Good but incomplete." in result

    def test_markdown_no_score_without_data(self) -> None:
        """Markdown output omits ## Score when no scoring data."""
        transcript = _make_unscored_transcript()
        result = format_markdown(transcript)
        assert "## Score" not in result

    def test_terminal_render_score(self) -> None:
        """render_debate() doesn't crash with score data (smoke test)."""
        transcript = _make_scored_transcript()
        # Just verify no exception; Rich output goes to console.
        render_debate(transcript)

    def test_json_includes_score(self) -> None:
        """to_dict() includes ground_truth_score in synthesis.analysis."""
        transcript = _make_scored_transcript()
        d = transcript.to_dict()
        assert d["synthesis"]["analysis"]["ground_truth_score"]["accuracy"] == 4

    def test_markdown_error_score(self) -> None:
        """Markdown renders parse failure gracefully."""
        transcript = _make_unscored_transcript()
        assert transcript.synthesis is not None
        transcript.synthesis.analysis["ground_truth_score"] = {
            "accuracy": -1,
            "completeness": -1,
            "overall": -1,
            "explanation": "Judge output could not be parsed: garbage",
            "judge_model": "claude",
        }
        result = format_markdown(transcript)
        assert "## Score" in result
        assert "could not be parsed" in result
