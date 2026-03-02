"""Cost charts — per-debate breakdown and cumulative spend over time.

Provides pure-Python data extraction for cost visualizations. The per-debate
chart reads from ``transcript.metadata["stats"]["per_model"]``; the cumulative
chart aggregates across transcript summaries sorted by date.
"""

from __future__ import annotations

from typing import Any

from mutual_dissent.models import DebateTranscript

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without NiceGUI)
# ---------------------------------------------------------------------------


def per_debate_cost(transcript: DebateTranscript) -> dict[str, Any]:
    """Extract per-model cost breakdown from a single transcript.

    Args:
        transcript: Debate transcript with stats metadata.

    Returns:
        Dict with ``models`` (list of aliases) and ``costs`` (list of floats).
        Empty lists if cost data is unavailable.
    """
    stats = transcript.metadata.get("stats", {})
    per_model = stats.get("per_model", {})
    if not per_model:
        return {"models": [], "costs": []}

    models = sorted(per_model.keys())
    costs = [per_model[m].get("cost_usd", 0.0) for m in models]

    if all(c == 0.0 for c in costs):
        return {"models": [], "costs": []}

    return {"models": models, "costs": costs}


def cumulative_cost_series(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute cumulative cost over time from transcript summaries.

    Summaries are sorted by date. Entries with ``None`` cost are skipped.

    Args:
        summaries: Transcript summary dicts with ``date`` and ``cost`` keys.

    Returns:
        Dict with ``dates`` (list of date strings) and ``cumulative``
        (list of running totals).
    """
    sorted_sums = sorted(
        [s for s in summaries if s.get("cost") is not None],
        key=lambda s: s.get("date", ""),
    )

    dates: list[str] = []
    cumulative: list[float] = []
    running = 0.0
    for s in sorted_sums:
        running += s["cost"]
        dates.append(s["date"])
        cumulative.append(round(running, 6))

    return {"dates": dates, "cumulative": cumulative}
