"""Debate view — query input and transcript display.

Renders the primary debate interface: a query form on the left and
a scrollable transcript view on the right. Submitting a query runs
a full debate via the orchestrator and displays the result.
"""

from __future__ import annotations

import asyncio
import logging

from nicegui import ui

from mutual_dissent.config import load_config
from mutual_dissent.models import DebateTranscript
from mutual_dissent.orchestrator import run_debate
from mutual_dissent.transcript import save_transcript
from mutual_dissent.web.components.transcript_view import render_transcript

logger = logging.getLogger(__name__)


async def _run_debate(
    *,
    query: str,
    panel: list[str],
    synthesizer: str,
    rounds: int,
    ground_truth: str | None,
) -> DebateTranscript:
    """Execute a debate and return the transcript.

    Args:
        query: The user's question.
        panel: List of model aliases for the panel.
        synthesizer: Model alias for synthesis.
        rounds: Number of reflection rounds.
        ground_truth: Known-correct answer for scoring, or None.

    Returns:
        Completed DebateTranscript.
    """
    config = load_config()
    transcript = await run_debate(
        query,
        config,
        panel=panel,
        synthesizer=synthesizer,
        rounds=rounds,
        ground_truth=ground_truth or None,
    )
    save_transcript(transcript)
    return transcript


def render() -> None:
    """Render the debate view page.

    Left panel: query form with panel/synthesizer/rounds controls.
    Right panel: scrollable transcript display after debate completion.
    """
    config = load_config()
    available_aliases = list(config._model_aliases_v2.keys())

    # Mutable state for diff toggle.
    show_diff: dict[str, bool] = {"value": False}

    with ui.row().classes("w-full h-full gap-0"):
        # --- Left panel: query form ---
        with ui.column().classes(
            "w-1/4 min-w-[280px] max-w-[360px] p-4 bg-gray-900 "
            "border-r border-gray-700 gap-4 h-[calc(100vh-120px)] overflow-y-auto"
        ):
            ui.label("New Debate").classes("font-mono font-bold text-lg")

            # Query textarea.
            query_input = (
                ui.textarea(
                    label="Query",
                    placeholder="Enter your question...",
                )
                .classes("w-full font-mono")
                .props("autogrow outlined dark")
            )

            # Panel selector: checkboxes.
            ui.label("Panel").classes("font-mono text-sm text-gray-400 mt-2")
            panel_checks: dict[str, ui.checkbox] = {}
            for alias in available_aliases:
                checked = alias in config.default_panel
                panel_checks[alias] = ui.checkbox(alias, value=checked).classes("font-mono text-sm")

            # Synthesizer dropdown.
            synth_select = (
                ui.select(
                    available_aliases,
                    value=config.default_synthesizer,
                    label="Synthesizer",
                )
                .classes("w-full font-mono")
                .props("outlined dark")
            )

            # Round count.
            round_input = (
                ui.number(
                    label="Rounds",
                    value=config.default_rounds,
                    min=1,
                    max=3,
                    step=1,
                )
                .classes("w-full font-mono")
                .props("outlined dark")
            )

            # Ground truth (collapsed).
            with ui.expansion("Ground Truth", icon="fact_check").classes("w-full text-sm"):
                gt_input = (
                    ui.textarea(
                        label="Expected answer",
                        placeholder="Optional reference answer for scoring...",
                    )
                    .classes("w-full font-mono text-xs")
                    .props("autogrow outlined dark")
                )

            # Submit button.
            submit_btn = (
                ui.button("Run Debate", icon="play_arrow")
                .classes("w-full mt-2")
                .props("color=primary")
            )

            # Diff toggle.
            diff_toggle = ui.switch("Show diff", value=False).classes("font-mono text-xs mt-2")

        # --- Right panel: response display ---
        with ui.column().classes("flex-1 p-6 gap-4 h-[calc(100vh-120px)] overflow-y-auto"):
            response_container = ui.column().classes("w-full gap-3")
            with response_container:
                ui.label("Submit a query to start a debate.").classes("text-gray-500 font-mono")

    # --- Event handlers ---
    async def on_submit() -> None:
        """Handle debate submission."""
        query_text = query_input.value
        if not query_text or not query_text.strip():
            ui.notify("Please enter a query.", type="warning")
            return

        selected_panel = [alias for alias, cb in panel_checks.items() if cb.value]
        if not selected_panel:
            ui.notify("Select at least one panel model.", type="warning")
            return

        # Disable form during execution.
        submit_btn.disable()
        submit_btn.text = "Running..."

        # Clear response area and show spinner.
        response_container.clear()
        with response_container:
            with ui.row().classes("items-center gap-3 py-8"):
                ui.spinner("dots", size="lg")
                ui.label("Debate in progress...").classes("text-gray-400 font-mono")

        try:
            transcript = await _run_debate(
                query=query_text.strip(),
                panel=selected_panel,
                synthesizer=synth_select.value,
                rounds=int(round_input.value),
                ground_truth=gt_input.value if gt_input.value else None,
            )

            # Render results.
            response_container.clear()
            with response_container:
                render_transcript(transcript, show_diff=show_diff["value"])

        except Exception as exc:
            logger.exception("Debate failed")
            response_container.clear()
            with response_container:
                ui.label(f"Debate failed: {exc}").classes("text-red-400 font-mono")
        finally:
            submit_btn.enable()
            submit_btn.text = "Run Debate"

    submit_btn.on_click(on_submit)

    def on_diff_toggle(e: object) -> None:
        """Track diff toggle state for next render."""
        show_diff["value"] = diff_toggle.value

    diff_toggle.on_value_change(on_diff_toggle)

    # --- Keyboard shortcuts ---
    def on_keyboard(e: object) -> None:
        """Handle global keyboard shortcuts.

        Args:
            e: NiceGUI keyboard event with key, action, and modifiers.
        """
        key = getattr(e, "key", None)
        action = getattr(e, "action", None)
        modifiers = getattr(e, "modifiers", None) or {}

        if action and getattr(action, "keydown", False):
            if key and getattr(key, "enter", False) and modifiers.get("ctrl", False):
                asyncio.create_task(on_submit())

    ui.keyboard(on_key=on_keyboard)
