"""Convergence chart -- per-model response change across debate rounds.

Measures how much each model's response changes between consecutive rounds
using word-level similarity via ``difflib.SequenceMatcher``. The change
ratio (1 - similarity) ranges from 0 (identical) to 1 (completely different).
"""

from __future__ import annotations

import difflib
from typing import Any

from mutual_dissent.models import DebateTranscript

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without NiceGUI)
# ---------------------------------------------------------------------------


def _change_ratio(old_text: str, new_text: str) -> float:
    """Compute the change ratio between two texts.

    Uses ``difflib.SequenceMatcher`` on word sequences for a meaningful
    similarity score that is robust to minor formatting changes.

    Args:
        old_text: Previous version of the response.
        new_text: Current version of the response.

    Returns:
        Float between 0.0 (identical) and 1.0 (completely different).
    """
    old_words = old_text.split()
    new_words = new_text.split()
    if not old_words and not new_words:
        return 0.0
    similarity = difflib.SequenceMatcher(None, old_words, new_words).ratio()
    return round(1.0 - similarity, 4)


def compute_convergence(transcript: DebateTranscript) -> dict[str, Any]:
    """Compute per-model change ratios across consecutive rounds.

    Args:
        transcript: Debate transcript with one or more rounds.

    Returns:
        Dict with keys:
            - ``models``: List of model aliases.
            - ``rounds``: List of transition labels (e.g. ``"R0->R1"``).
            - ``series``: Dict mapping alias to list of change ratios.
    """
    sorted_rounds = sorted(transcript.rounds, key=lambda r: r.round_number)

    model_contents: dict[str, dict[int, str]] = {}
    for rnd in sorted_rounds:
        for resp in rnd.responses:
            alias = resp.model_alias
            if alias not in model_contents:
                model_contents[alias] = {}
            model_contents[alias][rnd.round_number] = resp.content

    models = sorted(model_contents.keys())
    round_numbers = [r.round_number for r in sorted_rounds]
    transitions: list[str] = []
    for i in range(len(round_numbers) - 1):
        transitions.append(f"R{round_numbers[i]}\u2192R{round_numbers[i + 1]}")

    series: dict[str, list[float]] = {m: [] for m in models}
    for i in range(len(round_numbers) - 1):
        rn_prev = round_numbers[i]
        rn_curr = round_numbers[i + 1]
        for model in models:
            old = model_contents[model].get(rn_prev, "")
            new = model_contents[model].get(rn_curr, "")
            series[model].append(_change_ratio(old, new))

    return {
        "models": models,
        "rounds": transitions,
        "series": series,
    }


# ---------------------------------------------------------------------------
# NiceGUI rendering
# ---------------------------------------------------------------------------

# ECharts hex colors matching MODEL_CSS_COLORS from web/colors.py.
_ECHART_COLORS: dict[str, str] = {
    "claude": "#d946ef",  # fuchsia-500
    "gpt": "#22c55e",  # green-500
    "gemini": "#06b6d4",  # cyan-500
    "grok": "#eab308",  # yellow-500
}
_DEFAULT_COLOR = "#6b7280"  # gray-500


def render_convergence_chart(transcript: DebateTranscript) -> None:
    """Render a grouped bar chart of per-model convergence across rounds.

    Args:
        transcript: Debate transcript to visualize.
    """
    from nicegui import ui

    data = compute_convergence(transcript)

    if not data["rounds"]:
        ui.label("Not enough rounds for convergence chart.").classes("text-gray-500 italic")
        return

    series = []
    for model in data["models"]:
        series.append(
            {
                "name": model,
                "type": "bar",
                "data": data["series"][model],
                "itemStyle": {"color": _ECHART_COLORS.get(model, _DEFAULT_COLOR)},
            }
        )

    ui.echart(
        {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": data["models"], "textStyle": {"color": "#9ca3af"}},
            "xAxis": {
                "type": "category",
                "data": data["rounds"],
                "axisLabel": {"color": "#9ca3af"},
            },
            "yAxis": {
                "type": "value",
                "name": "Change Ratio",
                "min": 0,
                "max": 1,
                "axisLabel": {"color": "#9ca3af"},
                "nameTextStyle": {"color": "#9ca3af"},
            },
            "series": series,
        }
    ).classes("w-full h-64")
