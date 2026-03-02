"""Transcript view component — diff computation and debate rendering.

Provides pure-Python helpers for computing line-level diffs and formatting
timing/cost metadata, plus NiceGUI rendering functions that display debate
transcripts as styled, interactive panels.

Typical usage::

    from mutual_dissent.web.components.transcript_view import render_transcript

    render_transcript(transcript, show_diff=True)
"""

from __future__ import annotations

import difflib

from mutual_dissent.models import DebateRound, DebateTranscript, ExperimentMetadata, ModelResponse
from mutual_dissent.web.colors import get_css_colors

# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without NiceGUI)
# ---------------------------------------------------------------------------


def compute_diff(old_text: str, new_text: str) -> list[tuple[str, str]]:
    """Compute a line-level diff between two texts.

    Uses ``difflib.unified_diff`` and returns structured tuples suitable
    for rendering.  Header lines (``---``, ``+++``, ``@@``) are stripped.

    Args:
        old_text: Previous version of the text.
        new_text: Current version of the text.

    Returns:
        List of ``(tag, line)`` tuples where *tag* is ``" "`` (context),
        ``"+"`` (addition), or ``"-"`` (removal).
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(old_lines, new_lines, lineterm="")
    result: list[tuple[str, str]] = []

    for line in diff_lines:
        # Skip unified-diff header lines.
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            result.append(("+", line[1:]))
        elif line.startswith("-"):
            result.append(("-", line[1:]))
        elif line.startswith(" "):
            result.append((" ", line[1:]))
        # Lines that don't match any prefix (e.g. "\ No newline at end of file")
        # are silently skipped.

    return result


def _find_previous_response(
    alias: str,
    current_round: int,
    rounds: list[DebateRound],
) -> ModelResponse | None:
    """Find a model's response from the previous round.

    Args:
        alias: Model alias to search for (e.g. "claude").
        current_round: The round number we are currently viewing.
        rounds: All debate rounds completed so far.

    Returns:
        The matching ``ModelResponse`` from the previous round, or ``None``
        if *current_round* is 0 or the alias was not found.
    """
    if current_round <= 0:
        return None

    prev_round_number = current_round - 1
    for debate_round in rounds:
        if debate_round.round_number == prev_round_number:
            for resp in debate_round.responses:
                if resp.model_alias == alias:
                    return resp
    return None


def format_timing_web(resp: ModelResponse) -> str:
    """Format latency and token count for web display.

    Args:
        resp: Model response with optional timing data.

    Returns:
        Formatted string like ``"2.1s · 450 tokens"``, or empty string
        if neither latency nor token count is available.
    """
    parts: list[str] = []
    if resp.latency_ms is not None:
        parts.append(f"{resp.latency_ms / 1000:.1f}s")
    if resp.token_count is not None:
        parts.append(f"{resp.token_count:,} tokens")
    return " \u00b7 ".join(parts)


def format_cost(transcript: DebateTranscript) -> str:
    """Format total cost from transcript stats metadata.

    Reads ``total_cost_usd`` from ``transcript.metadata["stats"]``.

    Args:
        transcript: Completed debate transcript with stats metadata.

    Returns:
        Formatted cost string like ``"$0.0234"``, or empty string if
        cost data is not available.
    """
    stats = transcript.metadata.get("stats", {})
    total_cost = stats.get("total_cost_usd")
    if total_cost is None:
        return ""
    return f"${total_cost:.4f}"


def total_tokens(transcript: DebateTranscript) -> int:
    """Sum token_count across all rounds and synthesis.

    Args:
        transcript: Completed debate transcript.

    Returns:
        Total token count, or 0 if no token data available.
    """
    total = 0
    for debate_round in transcript.rounds:
        for resp in debate_round.responses:
            if resp.token_count is not None:
                total += resp.token_count
    if transcript.synthesis and transcript.synthesis.token_count is not None:
        total += transcript.synthesis.token_count
    return total


# ---------------------------------------------------------------------------
# NiceGUI rendering functions
# ---------------------------------------------------------------------------


def _render_response_card(
    resp: ModelResponse,
    *,
    show_diff: bool = False,
    previous_resp: ModelResponse | None = None,
) -> None:
    """Render a single model response as a styled card.

    Uses ``get_css_colors()`` from ``web/colors.py`` for model-specific
    coloring.  Error responses get red styling.  When *show_diff* is
    ``True`` and a *previous_resp* is available, an inline diff is shown
    instead of the full content.

    Args:
        resp: The model response to render.
        show_diff: Whether to show a diff against the previous response.
        previous_resp: The same model's response from the prior round.
    """
    from nicegui import ui

    colors = get_css_colors(resp.model_alias)

    if resp.error:
        border_class = "border-red-500"
        bg_class = "bg-red-500/10"
        text_class = "text-red-400"
    else:
        border_class = colors["border"]
        bg_class = colors["bg"]
        text_class = colors["text"]

    with ui.card().classes(f"w-full border {border_class} {bg_class} p-4"):
        # Header row: alias + timing.
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(resp.model_alias.upper()).classes(f"font-bold text-lg {text_class}")
            timing = format_timing_web(resp)
            if timing:
                ui.label(timing).classes("text-sm text-gray-400")

        if resp.error:
            ui.label(f"Error: {resp.error}").classes("text-red-400 mt-2")
        elif show_diff and previous_resp and not previous_resp.error:
            _render_diff(previous_resp.content, resp.content)
        else:
            ui.markdown(resp.content).classes("mt-2")


def _render_diff(old_text: str, new_text: str) -> None:
    """Render an inline diff as colored preformatted text.

    Additions are green, removals are red, and context lines use the
    default text color.

    Args:
        old_text: Previous version of the text.
        new_text: Current version of the text.
    """
    from nicegui import ui

    diff_lines = compute_diff(old_text, new_text)

    if not diff_lines:
        ui.label("No changes").classes("text-gray-500 italic mt-2")
        return

    html_parts: list[str] = ['<pre class="text-sm mt-2 whitespace-pre-wrap">']
    for tag, line in diff_lines:
        # Escape HTML entities in the line content.
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if tag == "+":
            html_parts.append(f'<span style="color: #4ade80;">{escaped}</span>')
        elif tag == "-":
            html_parts.append(f'<span style="color: #f87171;">{escaped}</span>')
        else:
            html_parts.append(escaped)
    html_parts.append("</pre>")

    ui.html("".join(html_parts))


def render_round_panel(
    debate_round: DebateRound,
    all_rounds: list[DebateRound],
    *,
    show_diff: bool = False,
    default_open: bool = False,
) -> None:
    """Render one debate round as an expansion panel.

    Round 0 is labelled "Round 0: Initial"; subsequent rounds are
    labelled "Round N: Reflection".

    Args:
        debate_round: The round to render.
        all_rounds: All completed rounds (needed for diff lookup).
        show_diff: Whether to show diffs against previous responses.
        default_open: Whether the panel starts expanded.
    """
    from nicegui import ui

    if debate_round.round_type == "initial":
        label = f"Round {debate_round.round_number}: Initial"
    else:
        label = f"Round {debate_round.round_number}: Reflection"

    with ui.expansion(label, value=default_open).classes("w-full"):
        for resp in debate_round.responses:
            previous_resp = _find_previous_response(
                resp.model_alias, debate_round.round_number, all_rounds
            )
            _render_response_card(
                resp,
                show_diff=show_diff,
                previous_resp=previous_resp,
            )


def render_synthesis_section(synthesis: ModelResponse) -> None:
    """Render the synthesis response with distinct styling.

    Uses a thicker border and darker background to visually separate
    the synthesized answer from individual round responses.

    Args:
        synthesis: The synthesizer's final response.
    """
    from nicegui import ui

    colors = get_css_colors(synthesis.model_alias)

    with ui.card().classes(f"w-full border-2 {colors['border']} bg-gray-800 p-6"):
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(f"Synthesis by {synthesis.model_alias}").classes(
                f"font-bold text-xl {colors['text']}"
            )
            timing = format_timing_web(synthesis)
            if timing:
                ui.label(timing).classes("text-sm text-gray-400")

        if synthesis.error:
            ui.label(f"Synthesis failed: {synthesis.error}").classes("text-red-400 mt-2")
        else:
            ui.markdown(synthesis.content).classes("mt-4")


def render_metadata_bar(transcript: DebateTranscript) -> None:
    """Render transcript metadata as a compact info bar.

    Displays transcript ID, panel models, round count, total tokens,
    cost, creation date, and experiment metadata when available.

    Args:
        transcript: The completed debate transcript.
    """
    from nicegui import ui

    with ui.row().classes("w-full flex-wrap gap-4 text-sm text-gray-400"):
        ui.label(f"ID: {transcript.short_id}")

        panel_str = ", ".join(transcript.panel) if transcript.panel else "none"
        ui.label(f"Panel: {panel_str}")

        ui.label(f"Rounds: {transcript.max_rounds}")

        total = total_tokens(transcript)
        if total > 0:
            ui.label(f"Tokens: {total:,}")

        cost = format_cost(transcript)
        if cost:
            ui.label(f"Cost: {cost}")

        date_str = transcript.created_at.strftime("%Y-%m-%d %H:%M UTC")
        ui.label(f"Date: {date_str}")

        experiment = transcript.metadata.get("experiment")
        if isinstance(experiment, ExperimentMetadata):
            ui.label(f"Experiment: {experiment.experiment_id} ({experiment.source_tool})")
        elif isinstance(experiment, dict):
            exp_id = experiment.get("experiment_id", "")
            source = experiment.get("source_tool", "")
            if exp_id:
                ui.label(f"Experiment: {exp_id} ({source})")


def render_score_section(transcript: DebateTranscript) -> None:
    """Render ground-truth score if available.

    Reads scoring data from ``transcript.synthesis.analysis["ground_truth_score"]``
    and displays accuracy, completeness, and overall scores.

    Args:
        transcript: Transcript with optional scoring data.
    """
    from nicegui import ui

    if not transcript.synthesis:
        return
    score_data = transcript.synthesis.analysis.get("ground_truth_score")
    if not score_data:
        return
    if score_data.get("accuracy", -1) < 0:
        ui.label("Score: Judge output could not be parsed").classes("text-gray-500 italic")
        return

    with ui.card().classes("w-full border border-gray-600 bg-gray-800/50 p-4"):
        ui.label("Score").classes("font-bold text-lg text-gray-200")
        with ui.row().classes("gap-6 mt-2"):
            ui.label(f"Accuracy: {score_data['accuracy']}/5").classes("text-gray-300")
            ui.label(f"Completeness: {score_data['completeness']}/5").classes("text-gray-300")
            ui.label(f"Overall: {score_data['overall']}/5").classes("text-gray-300")

        explanation = score_data.get("explanation", "")
        if explanation:
            ui.label(explanation).classes("text-sm text-gray-400 mt-2")


def render_transcript(
    transcript: DebateTranscript,
    *,
    show_diff: bool = False,
) -> None:
    """Render a full debate transcript in the web UI.

    Main entry point for the transcript view component.  Renders the
    original query, each debate round as an expansion panel (with the
    last round open by default), synthesis, ground-truth score, and
    metadata.

    Args:
        transcript: The completed debate transcript to display.
        show_diff: Whether to show diffs between rounds for each model.
    """
    from nicegui import ui

    # Query.
    ui.label("Query").classes("font-bold text-lg text-gray-300")
    ui.label(transcript.query).classes("text-gray-200 mb-4")

    # Rounds — last one defaults to open.
    for i, debate_round in enumerate(transcript.rounds):
        is_last = i == len(transcript.rounds) - 1
        render_round_panel(
            debate_round,
            transcript.rounds,
            show_diff=show_diff,
            default_open=is_last,
        )

    # Synthesis.
    if transcript.synthesis:
        ui.separator().classes("my-4")
        render_synthesis_section(transcript.synthesis)

    # Score.
    render_score_section(transcript)

    # Metadata.
    ui.separator().classes("my-4")
    render_metadata_bar(transcript)
