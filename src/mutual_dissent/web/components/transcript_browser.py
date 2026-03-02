"""Transcript browser — filter, sort, and render transcript list.

Provides pure-Python helpers for filtering and sorting transcript summaries,
plus NiceGUI rendering for the browsable transcript table with search and
detail view.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without NiceGUI)
# ---------------------------------------------------------------------------


def filter_transcripts(
    summaries: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter transcript summaries by query text, models, date range, and experiment.

    Args:
        summaries: Transcript summary dicts from ``list_transcripts()``.
        filters: Dict with optional keys:
            - ``query``: Case-insensitive substring match on transcript query.
            - ``models``: List of model aliases; transcript must include at least one.
            - ``date_from``: ISO date string (inclusive lower bound).
            - ``date_to``: ISO date string (inclusive upper bound).
            - ``experiment_id``: Exact match on experiment ID.

    Returns:
        Filtered list of summaries (new list, originals unchanged).
    """
    result = list(summaries)

    query = filters.get("query", "").strip().lower()
    if query:
        result = [s for s in result if query in s.get("query", "").lower()]

    models = filters.get("models")
    if models:
        model_set = {m.lower() for m in models}
        result = [
            s
            for s in result
            if any(m.strip().lower() in model_set for m in s.get("panel", "").split(","))
        ]

    date_from = filters.get("date_from", "")
    if date_from:
        result = [s for s in result if s.get("date", "") >= date_from]

    date_to = filters.get("date_to", "")
    if date_to:
        result = [s for s in result if s.get("date", "") <= date_to]

    experiment_id = filters.get("experiment_id", "").strip()
    if experiment_id:
        result = [s for s in result if s.get("experiment_id") == experiment_id]

    return result


def sort_transcripts(
    summaries: list[dict[str, Any]],
    sort_key: str,
    *,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Sort transcript summaries by a field.

    Args:
        summaries: Transcript summary dicts.
        sort_key: Field name to sort by (``"date"``, ``"tokens"``,
            ``"cost"``, ``"rounds"``).
        descending: Sort descending if True, ascending if False.

    Returns:
        New sorted list.
    """

    has_value = [s for s in summaries if s.get(sort_key) is not None]
    no_value = [s for s in summaries if s.get(sort_key) is None]

    has_value.sort(key=lambda s: s[sort_key], reverse=descending)
    return has_value + no_value
