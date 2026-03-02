"""Influence heatmap -- NxN model influence matrix from reflection shifts.

Measures how much model j's response changed after seeing model i's response
during reflection rounds. Uses word-level similarity via
``difflib.SequenceMatcher``: influence is the change ratio of model j between
rounds where model i was also present.

For v1, the influence matrix treats each model's total shift during reflection
as potentially influenced by all other models equally. A more sophisticated
approach would require explicit attention tracking.
"""

from __future__ import annotations

import difflib
from typing import Any

from mutual_dissent.models import DebateTranscript

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without NiceGUI)
# ---------------------------------------------------------------------------


def _word_change_ratio(old_text: str, new_text: str) -> float:
    """Compute change ratio between two texts using word-level similarity.

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


def _accumulate_transcript(
    transcript: DebateTranscript,
    model_idx: dict[str, int],
    influence_sums: list[list[float]],
    influence_counts: list[list[int]],
) -> None:
    """Accumulate influence data from a single transcript in-place.

    For each consecutive pair of rounds, computes the word-level change
    ratio for every model and attributes it equally to all panel members.

    Args:
        transcript: A single debate transcript.
        model_idx: Mapping from model alias to matrix index.
        influence_sums: Accumulated influence sums (mutated in-place).
        influence_counts: Accumulated influence counts (mutated in-place).
    """
    sorted_rounds = sorted(transcript.rounds, key=lambda r: r.round_number)

    model_contents: dict[str, dict[int, str]] = {}
    for rnd in sorted_rounds:
        for resp in rnd.responses:
            model_contents.setdefault(resp.model_alias, {})[rnd.round_number] = resp.content

    round_numbers = [r.round_number for r in sorted_rounds]
    panel_aliases = set(model_contents.keys())

    for k in range(len(round_numbers) - 1):
        rn_prev = round_numbers[k]
        rn_curr = round_numbers[k + 1]

        for target_alias in panel_aliases:
            old = model_contents[target_alias].get(rn_prev, "")
            new = model_contents[target_alias].get(rn_curr, "")
            change = _word_change_ratio(old, new)

            j = model_idx[target_alias]
            for source_alias in panel_aliases:
                i = model_idx[source_alias]
                influence_sums[i][j] += change
                influence_counts[i][j] += 1


def compute_influence(transcripts: list[DebateTranscript]) -> dict[str, Any]:
    """Compute an NxN influence matrix across one or more transcripts.

    Cell (i, j) represents the average change in model j's response across
    reflection rounds in transcripts where model i was also on the panel.
    Diagonal cells show self-change (how much a model revised itself).

    Args:
        transcripts: List of debate transcripts to analyze.

    Returns:
        Dict with keys:
            - ``models``: Sorted list of all model aliases seen.
            - ``matrix``: 2D list of floats, ``matrix[i][j]`` is model i's
              influence on model j.
    """
    if not transcripts:
        return {"models": [], "matrix": []}

    all_aliases: set[str] = set()
    for t in transcripts:
        for rnd in t.rounds:
            for resp in rnd.responses:
                all_aliases.add(resp.model_alias)
    models = sorted(all_aliases)

    if not models:
        return {"models": [], "matrix": []}

    n = len(models)
    model_idx = {m: i for i, m in enumerate(models)}
    influence_sums = [[0.0] * n for _ in range(n)]
    influence_counts = [[0] * n for _ in range(n)]

    for t in transcripts:
        _accumulate_transcript(t, model_idx, influence_sums, influence_counts)

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if influence_counts[i][j] > 0:
                matrix[i][j] = round(influence_sums[i][j] / influence_counts[i][j], 4)

    return {"models": models, "matrix": matrix}
