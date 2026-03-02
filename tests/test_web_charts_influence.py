"""Tests for influence heatmap data computation."""

from __future__ import annotations

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse


def _two_model_debate(
    initial: dict[str, str],
    reflection: dict[str, str],
) -> DebateTranscript:
    """Build a 2-round transcript (initial + reflection) for two models.

    Args:
        initial: Alias-to-content mapping for round 0.
        reflection: Alias-to-content mapping for round 1.

    Returns:
        DebateTranscript with two rounds.
    """
    r0_responses = [
        ModelResponse(model_id=f"test/{a}", model_alias=a, round_number=0, content=c)
        for a, c in initial.items()
    ]
    r1_responses = [
        ModelResponse(model_id=f"test/{a}", model_alias=a, round_number=1, content=c)
        for a, c in reflection.items()
    ]
    return DebateTranscript(
        rounds=[
            DebateRound(round_number=0, round_type="initial", responses=r0_responses),
            DebateRound(round_number=1, round_type="reflection", responses=r1_responses),
        ],
    )


class TestComputeInfluence:
    """compute_influence builds an NxN influence matrix from reflection shifts."""

    def test_returns_model_list(self) -> None:
        from mutual_dissent.web.components.charts.influence import compute_influence

        transcript = _two_model_debate(
            {"claude": "hello", "gpt": "world"},
            {"claude": "hello revised", "gpt": "world revised"},
        )
        result = compute_influence([transcript])
        assert set(result["models"]) == {"claude", "gpt"}

    def test_matrix_shape(self) -> None:
        from mutual_dissent.web.components.charts.influence import compute_influence

        transcript = _two_model_debate(
            {"claude": "aaa", "gpt": "bbb"},
            {"claude": "aaa changed", "gpt": "bbb changed"},
        )
        result = compute_influence([transcript])
        n = len(result["models"])
        assert len(result["matrix"]) == n
        assert all(len(row) == n for row in result["matrix"])

    def test_no_change_yields_zero_influence(self) -> None:
        from mutual_dissent.web.components.charts.influence import compute_influence

        transcript = _two_model_debate(
            {"claude": "same text", "gpt": "same text"},
            {"claude": "same text", "gpt": "same text"},
        )
        result = compute_influence([transcript])
        for row in result["matrix"]:
            for val in row:
                assert val == 0.0

    def test_empty_transcript_list(self) -> None:
        from mutual_dissent.web.components.charts.influence import compute_influence

        result = compute_influence([])
        assert result["models"] == []
        assert result["matrix"] == []

    def test_single_round_no_influence(self) -> None:
        from mutual_dissent.web.components.charts.influence import compute_influence

        transcript = DebateTranscript(
            rounds=[
                DebateRound(
                    round_number=0,
                    round_type="initial",
                    responses=[
                        ModelResponse(
                            model_id="test/claude",
                            model_alias="claude",
                            round_number=0,
                            content="only round",
                        ),
                    ],
                ),
            ],
        )
        result = compute_influence([transcript])
        assert result["models"] == ["claude"]
        assert result["matrix"] == [[0.0]]
