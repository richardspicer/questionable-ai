"""Tests for cost chart data computation."""

from __future__ import annotations

from typing import Any

from mutual_dissent.models import DebateTranscript


class TestPerDebateCost:
    """per_debate_cost extracts per-model cost breakdown from a transcript."""

    def test_extracts_model_costs(self) -> None:
        from mutual_dissent.web.components.charts.cost import per_debate_cost

        transcript = DebateTranscript(
            metadata={
                "stats": {
                    "per_model": {
                        "claude": {"cost_usd": 0.03},
                        "gpt": {"cost_usd": 0.05},
                    },
                    "total_cost_usd": 0.08,
                },
            },
        )
        result = per_debate_cost(transcript)
        assert result["models"] == ["claude", "gpt"]
        assert result["costs"] == [0.03, 0.05]

    def test_returns_empty_when_no_stats(self) -> None:
        from mutual_dissent.web.components.charts.cost import per_debate_cost

        transcript = DebateTranscript()
        result = per_debate_cost(transcript)
        assert result["models"] == []
        assert result["costs"] == []

    def test_returns_empty_when_no_cost_data(self) -> None:
        from mutual_dissent.web.components.charts.cost import per_debate_cost

        transcript = DebateTranscript(
            metadata={"stats": {"total_cost_usd": None, "per_model": {}}},
        )
        result = per_debate_cost(transcript)
        assert result["models"] == []


class TestCumulativeCostSeries:
    """cumulative_cost_series computes running total across transcript summaries."""

    def test_cumulative_over_dates(self) -> None:
        from mutual_dissent.web.components.charts.cost import cumulative_cost_series

        summaries: list[dict[str, Any]] = [
            {"date": "2026-01-01", "cost": 0.10},
            {"date": "2026-01-02", "cost": 0.20},
            {"date": "2026-01-03", "cost": 0.15},
        ]
        result = cumulative_cost_series(summaries)
        assert result["dates"] == ["2026-01-01", "2026-01-02", "2026-01-03"]
        assert result["cumulative"] == [0.10, 0.30, 0.45]

    def test_skips_none_cost(self) -> None:
        from mutual_dissent.web.components.charts.cost import cumulative_cost_series

        summaries: list[dict[str, Any]] = [
            {"date": "2026-01-01", "cost": 0.10},
            {"date": "2026-01-02", "cost": None},
            {"date": "2026-01-03", "cost": 0.20},
        ]
        result = cumulative_cost_series(summaries)
        assert result["dates"] == ["2026-01-01", "2026-01-03"]
        assert result["cumulative"] == [0.10, 0.30]

    def test_empty_input(self) -> None:
        from mutual_dissent.web.components.charts.cost import cumulative_cost_series

        result = cumulative_cost_series([])
        assert result["dates"] == []
        assert result["cumulative"] == []

    def test_all_none_costs(self) -> None:
        from mutual_dissent.web.components.charts.cost import cumulative_cost_series

        summaries: list[dict[str, Any]] = [
            {"date": "2026-01-01", "cost": None},
        ]
        result = cumulative_cost_series(summaries)
        assert result["dates"] == []
