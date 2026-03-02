"""Tests for transcript browser filter and sort logic."""

from __future__ import annotations

from typing import Any


def _make_summary(
    *,
    query: str = "Test query",
    date: str = "2026-02-28",
    panel: str = "claude, gpt",
    tokens: int = 300,
    cost: float | None = 0.05,
    rounds: int = 2,
    experiment_id: str | None = None,
    short_id: str = "abcd1234",
) -> dict[str, Any]:
    """Build a transcript summary dict matching list_transcripts() format."""
    return {
        "id": f"{short_id}-full-uuid",
        "short_id": short_id,
        "date": date,
        "query": query,
        "file": f"{date}_{short_id}.json",
        "panel": panel,
        "synthesizer": "claude",
        "tokens": tokens,
        "cost": cost,
        "rounds": rounds,
        "experiment_id": experiment_id,
    }


class TestFilterTranscripts:
    """filter_transcripts applies query, model, date, and experiment filters."""

    def test_no_filters_returns_all(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [_make_summary(), _make_summary(query="Other")]
        result = filter_transcripts(summaries, {})
        assert len(result) == 2

    def test_query_filter_case_insensitive(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [
            _make_summary(query="What is AI?"),
            _make_summary(query="How does gravity work?"),
        ]
        result = filter_transcripts(summaries, {"query": "ai"})
        assert len(result) == 1
        assert result[0]["query"] == "What is AI?"

    def test_model_filter(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [
            _make_summary(panel="claude, gpt"),
            _make_summary(panel="gemini, grok"),
        ]
        result = filter_transcripts(summaries, {"models": ["claude"]})
        assert len(result) == 1

    def test_date_range_filter(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [
            _make_summary(date="2026-01-15"),
            _make_summary(date="2026-02-20"),
            _make_summary(date="2026-03-10"),
        ]
        result = filter_transcripts(summaries, {"date_from": "2026-02-01", "date_to": "2026-02-28"})
        assert len(result) == 1
        assert result[0]["date"] == "2026-02-20"

    def test_experiment_filter(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [
            _make_summary(experiment_id="EXP-001"),
            _make_summary(experiment_id=None),
        ]
        result = filter_transcripts(summaries, {"experiment_id": "EXP-001"})
        assert len(result) == 1

    def test_combined_filters(self) -> None:
        from mutual_dissent.web.components.transcript_browser import filter_transcripts

        summaries = [
            _make_summary(query="AI ethics", panel="claude, gpt", date="2026-02-15"),
            _make_summary(query="AI safety", panel="gemini", date="2026-02-20"),
            _make_summary(query="Physics question", panel="claude", date="2026-02-18"),
        ]
        result = filter_transcripts(
            summaries,
            {
                "query": "ai",
                "models": ["claude"],
                "date_from": "2026-02-01",
                "date_to": "2026-02-28",
            },
        )
        assert len(result) == 1
        assert result[0]["query"] == "AI ethics"


class TestSortTranscripts:
    """sort_transcripts orders by date, tokens, cost, or rounds."""

    def test_sort_by_date_desc(self) -> None:
        from mutual_dissent.web.components.transcript_browser import sort_transcripts

        summaries = [
            _make_summary(date="2026-01-01"),
            _make_summary(date="2026-03-01"),
            _make_summary(date="2026-02-01"),
        ]
        result = sort_transcripts(summaries, "date", descending=True)
        assert [s["date"] for s in result] == ["2026-03-01", "2026-02-01", "2026-01-01"]

    def test_sort_by_tokens_asc(self) -> None:
        from mutual_dissent.web.components.transcript_browser import sort_transcripts

        summaries = [
            _make_summary(tokens=500),
            _make_summary(tokens=100),
            _make_summary(tokens=300),
        ]
        result = sort_transcripts(summaries, "tokens", descending=False)
        assert [s["tokens"] for s in result] == [100, 300, 500]

    def test_sort_by_cost_with_none(self) -> None:
        from mutual_dissent.web.components.transcript_browser import sort_transcripts

        summaries = [
            _make_summary(cost=0.10),
            _make_summary(cost=None),
            _make_summary(cost=0.05),
        ]
        result = sort_transcripts(summaries, "cost", descending=True)
        assert result[0]["cost"] == 0.10
        assert result[1]["cost"] == 0.05
        assert result[2]["cost"] is None

    def test_sort_by_rounds(self) -> None:
        from mutual_dissent.web.components.transcript_browser import sort_transcripts

        summaries = [
            _make_summary(rounds=3),
            _make_summary(rounds=1),
        ]
        result = sort_transcripts(summaries, "rounds", descending=True)
        assert [s["rounds"] for s in result] == [3, 1]
