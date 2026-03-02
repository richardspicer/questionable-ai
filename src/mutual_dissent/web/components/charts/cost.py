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


# ---------------------------------------------------------------------------
# NiceGUI rendering
# ---------------------------------------------------------------------------

# ECharts hex colors matching MODEL_CSS_COLORS.
_ECHART_COLORS: dict[str, str] = {
    "claude": "#d946ef",
    "gpt": "#22c55e",
    "gemini": "#06b6d4",
    "grok": "#eab308",
}
_DEFAULT_COLOR = "#6b7280"


def render_per_debate_chart(transcript: DebateTranscript) -> None:
    """Render a bar chart of per-model cost for a single transcript.

    Args:
        transcript: Debate transcript with cost metadata.
    """
    from nicegui import ui

    data = per_debate_cost(transcript)

    if not data["models"]:
        ui.label("Cost data unavailable.").classes("text-gray-500 italic")
        return

    colors = [_ECHART_COLORS.get(m, _DEFAULT_COLOR) for m in data["models"]]

    ui.echart(
        {
            "tooltip": {
                "trigger": "axis",
                ":formatter": r"params => params[0].name + ': $' + params[0].value.toFixed(4)",
            },
            "xAxis": {
                "type": "category",
                "data": data["models"],
                "axisLabel": {"color": "#9ca3af"},
            },
            "yAxis": {
                "type": "value",
                "name": "Cost (USD)",
                "axisLabel": {
                    "color": "#9ca3af",
                    ":formatter": r"value => '$' + value.toFixed(4)",
                },
                "nameTextStyle": {"color": "#9ca3af"},
            },
            "series": [
                {
                    "type": "bar",
                    "data": [
                        {"value": c, "itemStyle": {"color": col}}
                        for c, col in zip(data["costs"], colors, strict=True)
                    ],
                }
            ],
        }
    ).classes("w-full h-64")


def render_cumulative_chart(summaries: list[dict[str, Any]]) -> None:
    """Render a line chart of cumulative cost over time.

    Args:
        summaries: Transcript summary dicts with ``date`` and ``cost`` keys.
    """
    from nicegui import ui

    data = cumulative_cost_series(summaries)

    if not data["dates"]:
        ui.label("No cost data available for cumulative chart.").classes("text-gray-500 italic")
        return

    ui.echart(
        {
            "tooltip": {
                "trigger": "axis",
                ":formatter": r"params => params[0].axisValue + ': $' + params[0].value.toFixed(4)",
            },
            "xAxis": {
                "type": "category",
                "data": data["dates"],
                "axisLabel": {"color": "#9ca3af", "rotate": 45},
            },
            "yAxis": {
                "type": "value",
                "name": "Cumulative USD",
                "axisLabel": {
                    "color": "#9ca3af",
                    ":formatter": r"value => '$' + value.toFixed(2)",
                },
                "nameTextStyle": {"color": "#9ca3af"},
            },
            "series": [
                {
                    "type": "line",
                    "data": data["cumulative"],
                    "smooth": True,
                    "areaStyle": {"opacity": 0.15},
                    "itemStyle": {"color": "#3b82f6"},
                }
            ],
        }
    ).classes("w-full h-64")
