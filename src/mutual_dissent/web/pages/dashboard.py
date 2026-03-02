"""Research dashboard — transcript browser with charts and export.

Two-panel layout: left panel has filter controls and sort selector; right
panel shows the transcript list, and switches to a detail view with charts
when a transcript is selected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nicegui import ui

from mutual_dissent.transcript import list_transcripts, load_transcript
from mutual_dissent.web.components.charts.convergence import render_convergence_chart
from mutual_dissent.web.components.charts.cost import (
    render_cumulative_chart,
    render_per_debate_chart,
)
from mutual_dissent.web.components.charts.influence import render_influence_heatmap
from mutual_dissent.web.components.export import export_csv, export_json
from mutual_dissent.web.components.transcript_browser import filter_transcripts, sort_transcripts
from mutual_dissent.web.components.transcript_view import render_transcript

logger = logging.getLogger(__name__)

# Known model aliases for the filter multi-select.
_MODEL_ALIASES = ["claude", "gpt", "gemini", "grok"]

_TABLE_COLUMNS: list[dict[str, Any]] = [
    {"name": "date", "label": "Date", "field": "date", "sortable": True},
    {"name": "short_id", "label": "ID", "field": "short_id"},
    {"name": "query", "label": "Query", "field": "query"},
    {"name": "panel", "label": "Panel", "field": "panel"},
    {"name": "rounds", "label": "Rounds", "field": "rounds", "sortable": True},
    {"name": "tokens", "label": "Tokens", "field": "tokens", "sortable": True},
    {"name": "cost_display", "label": "Cost", "field": "cost_display", "sortable": True},
    {"name": "experiment_id", "label": "Experiment", "field": "experiment_id"},
]


@dataclass
class _FilterControls:
    """References to the filter/sort UI elements in the left panel."""

    query_input: Any = None
    model_select: Any = None
    date_from: Any = None
    date_to: Any = None
    experiment_input: Any = None
    sort_select: Any = None
    sort_desc: Any = None
    export_json_btn: Any = None
    export_csv_btn: Any = None
    summary_label: Any = None


@dataclass
class _DashboardState:
    """Mutable state shared across dashboard callbacks."""

    all_summaries: list[dict[str, Any]] = field(default_factory=list)
    selected_id: str | None = None


def _build_filter_panel() -> _FilterControls:
    """Build the left-panel filter, sort, and export controls.

    Returns:
        References to all interactive UI elements for binding.
    """
    controls = _FilterControls()

    ui.label("Filters").classes("font-mono font-bold text-lg")

    controls.query_input = (
        ui.input(label="Search query", placeholder="Filter by query text...")
        .classes("w-full font-mono text-sm")
        .props("outlined dark dense clearable")
    )

    controls.model_select = (
        ui.select(
            _MODEL_ALIASES,
            multiple=True,
            label="Models",
            value=[],
        )
        .classes("w-full font-mono text-sm")
        .props("outlined dark dense")
    )

    with ui.row().classes("w-full gap-2"):
        controls.date_from = (
            ui.input(label="From date", placeholder="YYYY-MM-DD")
            .classes("flex-1 font-mono text-sm")
            .props("outlined dark dense clearable")
        )
        controls.date_to = (
            ui.input(label="To date", placeholder="YYYY-MM-DD")
            .classes("flex-1 font-mono text-sm")
            .props("outlined dark dense clearable")
        )

    controls.experiment_input = (
        ui.input(label="Experiment ID", placeholder="e.g. EXP-001")
        .classes("w-full font-mono text-sm")
        .props("outlined dark dense clearable")
    )

    ui.separator()
    ui.label("Sort").classes("font-mono text-sm text-gray-400")

    controls.sort_select = (
        ui.select(
            {"date": "Date", "tokens": "Tokens", "cost": "Cost", "rounds": "Rounds"},
            value="date",
            label="Sort by",
        )
        .classes("w-full font-mono text-sm")
        .props("outlined dark dense")
    )
    controls.sort_desc = ui.switch("Descending", value=True).classes("font-mono text-xs")

    ui.separator()
    ui.label("Export").classes("font-mono text-sm text-gray-400")

    with ui.row().classes("w-full gap-2"):
        controls.export_json_btn = (
            ui.button("JSON", icon="download").classes("flex-1").props("outline dense")
        )
        controls.export_csv_btn = (
            ui.button("CSV", icon="download").classes("flex-1").props("outline dense")
        )

    ui.separator()
    controls.summary_label = ui.label("").classes("text-xs text-gray-500 font-mono")

    return controls


def _collect_filters(controls: _FilterControls) -> dict[str, Any]:
    """Read current values from filter controls into a filters dict.

    Args:
        controls: References to the filter UI elements.

    Returns:
        Dictionary of active filter parameters.
    """
    filters: dict[str, Any] = {}
    if controls.query_input.value:
        filters["query"] = controls.query_input.value
    if controls.model_select.value:
        filters["models"] = controls.model_select.value
    if controls.date_from.value:
        filters["date_from"] = controls.date_from.value
    if controls.date_to.value:
        filters["date_to"] = controls.date_to.value
    if controls.experiment_input.value:
        filters["experiment_id"] = controls.experiment_input.value
    return filters


def _get_filtered(
    all_summaries: list[dict[str, Any]],
    controls: _FilterControls,
) -> list[dict[str, Any]]:
    """Apply current filters and sort to all summaries.

    Args:
        all_summaries: Complete list of transcript summaries.
        controls: References to filter/sort UI elements.

    Returns:
        Filtered and sorted list of summaries.
    """
    filters = _collect_filters(controls)
    filtered = filter_transcripts(all_summaries, filters)
    return sort_transcripts(
        filtered,
        controls.sort_select.value,
        descending=controls.sort_desc.value,
    )


def _render_detail(
    transcript_id: str,
    right_panel: Any,
    ds: _DashboardState,
    render_table_fn: Any,
) -> None:
    """Render detail view for a selected transcript.

    Args:
        transcript_id: UUID of the transcript to display.
        right_panel: The right-panel UI container.
        ds: Dashboard state for tracking selection.
        render_table_fn: Callback to return to the table view.
    """
    right_panel.clear()
    ds.selected_id = transcript_id

    transcript = load_transcript(transcript_id)
    if transcript is None:
        with right_panel:
            ui.label(f"Could not load transcript {transcript_id[:8]}.").classes(
                "text-red-400 font-mono"
            )
        return

    with right_panel:
        ui.button("\u2190 Back to list", on_click=lambda: render_table_fn()).classes("mb-4").props(
            "flat dense"
        )

        with ui.expansion("Transcript", value=True).classes("w-full"):
            render_transcript(transcript, show_diff=False)

        ui.separator().classes("my-4")
        ui.label("Analytics").classes("font-bold text-lg text-gray-300")

        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-1"):
                ui.label("Convergence").classes("font-mono text-sm text-gray-400")
                render_convergence_chart(transcript)

            with ui.column().classes("flex-1"):
                ui.label("Cost Breakdown").classes("font-mono text-sm text-gray-400")
                render_per_debate_chart(transcript)

        ui.label("Influence").classes("font-mono text-sm text-gray-400 mt-4")
        render_influence_heatmap([transcript])

        ui.separator().classes("my-4")
        ui.label("Cumulative Cost (All Transcripts)").classes("font-mono text-sm text-gray-400")
        render_cumulative_chart(ds.all_summaries)


def render() -> None:
    """Render the research dashboard page.

    Left panel: search, filter controls (date range, model aliases,
    experiment ID), sort selector, and export buttons.

    Right panel: transcript table that switches to a detail view (with
    convergence, cost, and influence charts) when a row is selected.
    """
    ds = _DashboardState(all_summaries=list_transcripts(limit=0))

    with ui.row().classes("w-full h-full gap-0"):
        with ui.column().classes(
            "w-1/4 min-w-[260px] max-w-[340px] p-4 bg-gray-900 "
            "border-r border-gray-700 gap-3 h-[calc(100vh-120px)] overflow-y-auto"
        ):
            controls = _build_filter_panel()

        right_panel = ui.column().classes("flex-1 p-6 gap-4 h-[calc(100vh-120px)] overflow-y-auto")

    def _render_table() -> None:
        """Render the transcript list in the right panel."""
        right_panel.clear()
        ds.selected_id = None
        filtered = _get_filtered(ds.all_summaries, controls)
        controls.summary_label.text = f"{len(filtered)} of {len(ds.all_summaries)} transcripts"
        controls.summary_label.update()

        with right_panel:
            if not filtered:
                ui.label("No transcripts match the current filters.").classes(
                    "text-gray-500 font-mono"
                )
                return

            rows = []
            for s in filtered:
                cost_val = s.get("cost")
                cost_str = f"${cost_val:.4f}" if cost_val is not None else "\u2014"
                rows.append({**s, "cost_display": cost_str})

            table = (
                ui.table(
                    columns=_TABLE_COLUMNS,
                    rows=rows,
                    row_key="short_id",
                    selection="single",
                )
                .classes("w-full")
                .props("dark dense flat")
            )

            def on_select(e: Any) -> None:
                """Handle transcript row selection."""
                if e.selection:
                    _render_detail(e.selection[0]["id"], right_panel, ds, _render_table)

            table.on_select(on_select)

    def on_filter_change(_: Any = None) -> None:
        """Re-render the table when any filter or sort control changes."""
        if ds.selected_id is None:
            _render_table()

    # Bind filter/sort changes.
    for ctrl in (
        controls.query_input,
        controls.model_select,
        controls.date_from,
        controls.date_to,
        controls.experiment_input,
        controls.sort_select,
    ):
        ctrl.on("update:model-value", on_filter_change)
    controls.sort_desc.on_value_change(on_filter_change)

    # Export handlers.
    def on_export_json() -> None:
        """Download filtered transcripts as JSON."""
        content = export_json(_get_filtered(ds.all_summaries, controls))
        ui.download(content.encode("utf-8"), "transcripts.json")

    def on_export_csv() -> None:
        """Download filtered transcripts as CSV."""
        content = export_csv(_get_filtered(ds.all_summaries, controls))
        ui.download(content.encode("utf-8"), "transcripts.csv")

    controls.export_json_btn.on_click(on_export_json)
    controls.export_csv_btn.on_click(on_export_csv)

    # Initial render.
    _render_table()
