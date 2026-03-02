"""Tests for convergence chart data computation."""

from __future__ import annotations

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse


def _make_transcript_with_rounds(
    round_contents: dict[str, list[str]],
) -> DebateTranscript:
    """Build a transcript where each model has content per round.

    Args:
        round_contents: Mapping of alias to list of content strings,
            one per round. E.g. {"claude": ["v1", "v2"], "gpt": ["v1", "v2"]}.

    Returns:
        DebateTranscript with populated rounds.
    """
    num_rounds = max(len(v) for v in round_contents.values())
    rounds: list[DebateRound] = []
    for i in range(num_rounds):
        responses = []
        for alias, contents in round_contents.items():
            if i < len(contents):
                responses.append(
                    ModelResponse(
                        model_id=f"test/{alias}",
                        model_alias=alias,
                        round_number=i,
                        content=contents[i],
                    )
                )
        round_type = "initial" if i == 0 else "reflection"
        rounds.append(DebateRound(round_number=i, round_type=round_type, responses=responses))
    return DebateTranscript(rounds=rounds)


class TestComputeConvergence:
    """compute_convergence returns per-model change ratios across rounds."""

    def test_identical_responses_yield_zero_change(self) -> None:
        from mutual_dissent.web.components.charts.convergence import compute_convergence

        transcript = _make_transcript_with_rounds({"claude": ["same text", "same text"]})
        result = compute_convergence(transcript)
        assert result["models"] == ["claude"]
        assert result["rounds"] == ["R0→R1"]
        assert len(result["series"]["claude"]) == 1
        assert result["series"]["claude"][0] == 0.0

    def test_completely_different_responses(self) -> None:
        from mutual_dissent.web.components.charts.convergence import compute_convergence

        transcript = _make_transcript_with_rounds(
            {"claude": ["alpha beta gamma", "delta epsilon zeta"]}
        )
        result = compute_convergence(transcript)
        assert result["series"]["claude"][0] > 0.5

    def test_multiple_models(self) -> None:
        from mutual_dissent.web.components.charts.convergence import compute_convergence

        transcript = _make_transcript_with_rounds(
            {
                "claude": ["hello world", "hello world modified"],
                "gpt": ["foo bar baz", "completely new response here"],
            }
        )
        result = compute_convergence(transcript)
        assert set(result["models"]) == {"claude", "gpt"}
        assert result["series"]["gpt"][0] > result["series"]["claude"][0]

    def test_single_round_returns_empty(self) -> None:
        from mutual_dissent.web.components.charts.convergence import compute_convergence

        transcript = _make_transcript_with_rounds({"claude": ["only one round"]})
        result = compute_convergence(transcript)
        assert result["rounds"] == []
        assert result["series"]["claude"] == []

    def test_three_rounds_two_transitions(self) -> None:
        from mutual_dissent.web.components.charts.convergence import compute_convergence

        transcript = _make_transcript_with_rounds(
            {"claude": ["round zero", "round one", "round two"]}
        )
        result = compute_convergence(transcript)
        assert result["rounds"] == ["R0→R1", "R1→R2"]
        assert len(result["series"]["claude"]) == 2
