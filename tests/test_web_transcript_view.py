"""Tests for the transcript view component."""

from __future__ import annotations

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse


class TestComputeDiff:
    """compute_diff returns structured diff lines."""

    def test_identical_text_returns_no_changes(self) -> None:
        """Identical old and new text produces empty diff."""
        from mutual_dissent.web.components.transcript_view import compute_diff

        lines = compute_diff("Hello world", "Hello world")
        assert lines == []

    def test_addition_marked(self) -> None:
        from mutual_dissent.web.components.transcript_view import compute_diff

        lines = compute_diff("line one", "line one\nline two")
        tags = [line[0] for line in lines]
        assert "+" in tags

    def test_removal_marked(self) -> None:
        from mutual_dissent.web.components.transcript_view import compute_diff

        lines = compute_diff("line one\nline two", "line one")
        tags = [line[0] for line in lines]
        assert "-" in tags

    def test_empty_old_text(self) -> None:
        from mutual_dissent.web.components.transcript_view import compute_diff

        lines = compute_diff("", "new content")
        assert len(lines) > 0

    def test_empty_new_text(self) -> None:
        from mutual_dissent.web.components.transcript_view import compute_diff

        lines = compute_diff("old content", "")
        assert len(lines) > 0


class TestFindPreviousResponse:
    """_find_previous_response locates responses from prior rounds."""

    def test_finds_matching_alias(self) -> None:
        from mutual_dissent.web.components.transcript_view import _find_previous_response

        resp_r0 = ModelResponse(
            model_id="test/model",
            model_alias="claude",
            round_number=0,
            content="round 0",
        )
        round0 = DebateRound(round_number=0, round_type="initial", responses=[resp_r0])
        result = _find_previous_response("claude", 1, [round0])
        assert result is not None
        assert result.content == "round 0"

    def test_returns_none_for_initial_round(self) -> None:
        from mutual_dissent.web.components.transcript_view import _find_previous_response

        result = _find_previous_response("claude", 0, [])
        assert result is None

    def test_returns_none_when_alias_not_found(self) -> None:
        from mutual_dissent.web.components.transcript_view import _find_previous_response

        resp_r0 = ModelResponse(
            model_id="test/model",
            model_alias="gpt",
            round_number=0,
            content="round 0",
        )
        round0 = DebateRound(round_number=0, round_type="initial", responses=[resp_r0])
        result = _find_previous_response("claude", 1, [round0])
        assert result is None


class TestFormatTimingWeb:
    """format_timing_web renders latency and token count."""

    def test_with_latency_and_tokens(self) -> None:
        from mutual_dissent.web.components.transcript_view import format_timing_web

        resp = ModelResponse(
            model_id="test/model",
            model_alias="claude",
            round_number=0,
            content="test",
            latency_ms=2100,
            token_count=450,
        )
        result = format_timing_web(resp)
        assert "2.1s" in result
        assert "450" in result

    def test_with_no_data(self) -> None:
        from mutual_dissent.web.components.transcript_view import format_timing_web

        resp = ModelResponse(
            model_id="test/model",
            model_alias="claude",
            round_number=0,
            content="test",
        )
        result = format_timing_web(resp)
        assert result == ""


class TestFormatCost:
    """_format_cost reads cost from transcript metadata."""

    def test_with_cost(self) -> None:
        from mutual_dissent.web.components.transcript_view import _format_cost

        transcript = DebateTranscript(
            metadata={"stats": {"total_cost_usd": 0.0234}},
        )
        result = _format_cost(transcript)
        assert result == "$0.0234"

    def test_without_cost(self) -> None:
        from mutual_dissent.web.components.transcript_view import _format_cost

        transcript = DebateTranscript()
        result = _format_cost(transcript)
        assert result == ""


class TestTotalTokens:
    """_total_tokens sums tokens across rounds and synthesis."""

    def test_sums_round_and_synthesis_tokens(self) -> None:
        from mutual_dissent.web.components.transcript_view import _total_tokens

        resp1 = ModelResponse(
            model_id="m1",
            model_alias="claude",
            round_number=0,
            content="a",
            token_count=100,
        )
        resp2 = ModelResponse(
            model_id="m2",
            model_alias="gpt",
            round_number=0,
            content="b",
            token_count=200,
        )
        synth = ModelResponse(
            model_id="m1",
            model_alias="claude",
            round_number=-1,
            content="s",
            token_count=50,
        )
        transcript = DebateTranscript(
            rounds=[DebateRound(round_number=0, round_type="initial", responses=[resp1, resp2])],
            synthesis=synth,
        )
        assert _total_tokens(transcript) == 350

    def test_returns_zero_when_no_token_data(self) -> None:
        from mutual_dissent.web.components.transcript_view import _total_tokens

        transcript = DebateTranscript()
        assert _total_tokens(transcript) == 0
