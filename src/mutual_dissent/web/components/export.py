"""Export -- JSON and CSV download of transcript summaries.

Provides pure-Python serialization of filtered transcript summary lists
into JSON and CSV formats for download from the dashboard.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

# CSV columns in display order.
_CSV_COLUMNS = [
    "short_id",
    "date",
    "query",
    "panel",
    "synthesizer",
    "rounds",
    "tokens",
    "cost",
    "experiment_id",
]


def export_json(summaries: list[dict[str, Any]]) -> str:
    """Serialize transcript summaries to a JSON string.

    Args:
        summaries: Transcript summary dicts from ``list_transcripts()``.

    Returns:
        JSON string containing an array of summary objects.
    """
    return json.dumps(summaries, indent=2, ensure_ascii=False)


def export_csv(summaries: list[dict[str, Any]]) -> str:
    """Serialize transcript summaries to a CSV string.

    One row per transcript with columns: short_id, date, query, panel,
    synthesizer, rounds, tokens, cost, experiment_id.

    Args:
        summaries: Transcript summary dicts from ``list_transcripts()``.

    Returns:
        CSV string with header row and one data row per transcript.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_CSV_COLUMNS)
    for s in summaries:
        writer.writerow([s.get(col, "") for col in _CSV_COLUMNS])
    return output.getvalue()
