"""Tests for the status bar component."""

from __future__ import annotations

from mutual_dissent.web.components.status_bar import format_completion_text, format_status_text


class TestFormatStatusText:
    """format_status_text returns correct text for each debate phase."""

    def test_initial_round(self) -> None:
        """Shows 'Initial round...' for round_type='initial'."""
        result = format_status_text(round_type="initial", round_number=0, total_rounds=2)
        assert result == "Initial round..."

    def test_reflection_round(self) -> None:
        """Shows 'Reflection N of M...' for reflection rounds."""
        result = format_status_text(round_type="reflection", round_number=1, total_rounds=2)
        assert result == "Reflection 1 of 2..."

    def test_synthesis_round(self) -> None:
        """Shows 'Synthesizing...' for synthesis round."""
        result = format_status_text(round_type="synthesis", round_number=-1, total_rounds=2)
        assert result == "Synthesizing..."

    def test_unknown_type_falls_back(self) -> None:
        """Unknown round types show generic text."""
        result = format_status_text(round_type="unknown", round_number=5, total_rounds=2)
        assert "Round 5" in result


class TestFormatCompletionText:
    """format_completion_text shows final stats."""

    def test_with_tokens_and_cost(self) -> None:
        """Shows tokens and cost when available."""
        result = format_completion_text(total_tokens=1500, cost_usd=0.0234)
        assert "1,500 tokens" in result
        assert "$0.0234" in result
        assert result.startswith("Complete")

    def test_with_tokens_only(self) -> None:
        """Shows tokens without cost when cost is None."""
        result = format_completion_text(total_tokens=500, cost_usd=None)
        assert "500 tokens" in result
        assert "$" not in result

    def test_with_no_data(self) -> None:
        """Shows just 'Complete' when no stats available."""
        result = format_completion_text(total_tokens=0, cost_usd=None)
        assert result == "Complete"

    def test_aborted(self) -> None:
        """Shows 'Aborted' text."""
        result = format_completion_text(total_tokens=300, cost_usd=None, aborted=True)
        assert "Aborted" in result
