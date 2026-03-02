"""Debate view — query input and progressive transcript display.

Renders the primary debate interface: a query form on the left and
a scrollable transcript view on the right. Rounds render progressively
as they complete via the orchestrator's ``on_round_complete`` callback,
with a status bar showing progress and an abort button for cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nicegui import ui

from mutual_dissent.config import Config, load_config
from mutual_dissent.models import DebateRound, DebateTranscript
from mutual_dissent.orchestrator import run_debate
from mutual_dissent.transcript import save_transcript
from mutual_dissent.web.components.status_bar import (
    format_completion_text,
    format_status_text,
    render_status_bar,
)
from mutual_dissent.web.components.transcript_view import (
    render_metadata_bar,
    render_round_panel,
    render_score_section,
    render_synthesis_section,
    total_tokens,
)

logger = logging.getLogger(__name__)


@dataclass
class _FormWidgets:
    """Holds references to all form widgets for event handler access.

    Attributes:
        query_input: Query textarea element.
        panel_checks: Mapping of alias to checkbox element.
        synth_select: Synthesizer dropdown element.
        round_input: Round count number input element.
        gt_input: Ground truth textarea element.
        submit_btn: Submit button element.
        abort_btn: Abort button element.
        diff_toggle: Diff toggle switch element.
    """

    query_input: Any
    panel_checks: dict[str, Any]
    synth_select: Any
    round_input: Any
    gt_input: Any
    submit_btn: Any
    abort_btn: Any
    diff_toggle: Any


@dataclass
class _StatusWidgets:
    """Holds references to status bar and response container elements.

    Attributes:
        status_icon: Icon element in the status bar.
        status_label: Label element in the status bar.
        response_container: Column container for progressive round output.
    """

    status_icon: Any
    status_label: Any
    response_container: Any


@dataclass
class _DebateState:
    """Mutable state shared between render and event handlers.

    Attributes:
        show_diff: Whether to show diffs between rounds.
        task: Reference to the running asyncio task, or None.
        completed_rounds: Rounds accumulated during progressive rendering.
    """

    show_diff: bool = False
    task: asyncio.Task[None] | None = None
    completed_rounds: list[DebateRound] = field(default_factory=list)


def _render_form_panel(config: Config) -> _FormWidgets:
    """Render the left-side query form panel.

    Args:
        config: Application configuration for defaults and aliases.

    Returns:
        FormWidgets dataclass with references to all interactive elements.
    """
    available_aliases = list(config._model_aliases_v2.keys())

    ui.label("New Debate").classes("font-mono font-bold text-lg")

    query_input = (
        ui.textarea(label="Query", placeholder="Enter your question...")
        .classes("w-full font-mono")
        .props("autogrow outlined dark")
    )

    ui.label("Panel").classes("font-mono text-sm text-gray-400 mt-2")
    panel_checks: dict[str, Any] = {}
    for alias in available_aliases:
        checked = alias in config.default_panel
        panel_checks[alias] = ui.checkbox(alias, value=checked).classes("font-mono text-sm")

    synth_select = (
        ui.select(available_aliases, value=config.default_synthesizer, label="Synthesizer")
        .classes("w-full font-mono")
        .props("outlined dark")
    )

    round_input = (
        ui.number(label="Rounds", value=config.default_rounds, min=1, max=3, step=1)
        .classes("w-full font-mono")
        .props("outlined dark")
    )

    with ui.expansion("Ground Truth", icon="fact_check").classes("w-full text-sm"):
        gt_input = (
            ui.textarea(
                label="Expected answer",
                placeholder="Optional reference answer for scoring...",
            )
            .classes("w-full font-mono text-xs")
            .props("autogrow outlined dark")
        )

    submit_btn = (
        ui.button("Run Debate", icon="play_arrow").classes("w-full mt-2").props("color=primary")
    )

    abort_btn = (
        ui.button("Abort", icon="stop").classes("w-full mt-1").props("color=negative outline")
    )
    abort_btn.visible = False

    diff_toggle = ui.switch("Show diff", value=False).classes("font-mono text-xs mt-2")

    return _FormWidgets(
        query_input=query_input,
        panel_checks=panel_checks,
        synth_select=synth_select,
        round_input=round_input,
        gt_input=gt_input,
        submit_btn=submit_btn,
        abort_btn=abort_btn,
        diff_toggle=diff_toggle,
    )


def _update_status(
    icon: Any,
    label: Any,
    *,
    icon_name: str,
    icon_class: str,
    text: str,
) -> None:
    """Update the status bar icon and label.

    Args:
        icon: NiceGUI icon element to update.
        label: NiceGUI label element to update.
        icon_name: Material icon name (e.g. "check_circle").
        icon_class: Tailwind class for icon color (e.g. "text-green-400").
        text: Status text to display.
    """
    icon._props["name"] = icon_name
    icon.classes(icon_class, remove="text-gray-400 text-blue-400 text-green-400 text-red-400")
    label.text = text
    icon.update()
    label.update()


async def _scroll_to_bottom() -> None:
    """Scroll the right panel to the bottom of the debate results."""
    await ui.run_javascript(
        "const el = document.getElementById('debate-results');"
        "if (el) el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});"
    )


def render() -> None:
    """Render the debate view page.

    Left panel: query form with panel/synthesizer/rounds controls.
    Right panel: scrollable transcript with progressive round rendering,
    status bar, and abort button.
    """
    ui.add_css("""\
.animate-fade-in {
    animation: fadeIn 0.3s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}""")

    config = load_config()
    state = _DebateState()

    with ui.row().classes("w-full h-full gap-0"):
        with ui.column().classes(
            "w-1/4 min-w-[280px] max-w-[360px] p-4 bg-gray-900 "
            "border-r border-gray-700 gap-4 h-[calc(100vh-120px)] overflow-y-auto"
        ):
            form = _render_form_panel(config)

        with (
            ui.column()
            .classes("flex-1 p-6 gap-4 h-[calc(100vh-120px)] overflow-y-auto")
            .props('id="debate-results"')
        ):
            _status_container, status_icon, status_label = render_status_bar()
            response_container = ui.column().classes("w-full gap-3")
            with response_container:
                ui.label("Submit a query to start a debate.").classes("text-gray-500 font-mono")

    status = _StatusWidgets(
        status_icon=status_icon,
        status_label=status_label,
        response_container=response_container,
    )

    async def on_submit() -> None:
        """Handle debate submission with progressive round rendering."""
        query_text = form.query_input.value
        if not query_text or not query_text.strip():
            ui.notify("Please enter a query.", type="warning")
            return

        selected_panel = [alias for alias, cb in form.panel_checks.items() if cb.value]
        if not selected_panel:
            ui.notify("Select at least one panel model.", type="warning")
            return

        state.task = asyncio.current_task()
        form.submit_btn.disable()
        form.submit_btn.text = "Running..."
        form.abort_btn.visible = True

        status.response_container.clear()
        with status.response_container:
            ui.label("Query").classes("font-bold text-lg text-gray-300 animate-fade-in")
            ui.label(query_text.strip()).classes("text-gray-200 mb-4 animate-fade-in")

        _update_status(
            status.status_icon,
            status.status_label,
            icon_name="hourglass_top",
            icon_class="text-blue-400",
            text="Starting debate...",
        )

        num_rounds = int(form.round_input.value)
        start_time = time.monotonic()
        state.completed_rounds = []

        async def on_round_complete(debate_round: DebateRound) -> None:
            """Render a round progressively and update the status bar.

            Args:
                debate_round: The just-completed debate round.
            """
            try:
                elapsed = time.monotonic() - start_time
                elapsed_str = f" ({elapsed:.1f}s)"
                phase_text = (
                    format_status_text(
                        round_type=debate_round.round_type,
                        round_number=debate_round.round_number,
                        total_rounds=num_rounds,
                    )
                    + elapsed_str
                )
                status.status_label.text = phase_text
                status.status_label.update()

                if debate_round.round_type == "synthesis":
                    assert isinstance(debate_round.responses, list)
                    synthesis = debate_round.responses[0]
                    with status.response_container:
                        ui.separator().classes("my-4 animate-fade-in")
                        with ui.column().classes("animate-fade-in"):
                            render_synthesis_section(synthesis)
                else:
                    state.completed_rounds.append(debate_round)
                    with status.response_container:
                        with ui.column().classes("animate-fade-in"):
                            render_round_panel(
                                debate_round,
                                list(state.completed_rounds),
                                show_diff=state.show_diff,
                                default_open=True,
                            )

                await _scroll_to_bottom()
            except Exception:
                logger.exception("Failed to render round %s", debate_round.round_type)
                ui.notify("Failed to render round.", type="warning")

        try:
            config_fresh = load_config()
            transcript = await run_debate(
                query_text.strip(),
                config_fresh,
                panel=selected_panel,
                synthesizer=form.synth_select.value,
                rounds=num_rounds,
                ground_truth=form.gt_input.value if form.gt_input.value else None,
                on_round_complete=on_round_complete,
            )
            save_transcript(transcript)

            with status.response_container:
                with ui.column().classes("animate-fade-in"):
                    render_score_section(transcript)
                ui.separator().classes("my-4 animate-fade-in")
                with ui.column().classes("animate-fade-in"):
                    render_metadata_bar(transcript)

            elapsed = time.monotonic() - start_time
            token_total = total_tokens(transcript)
            stats = transcript.metadata.get("stats", {})
            cost_usd = stats.get("total_cost_usd")
            completion_text = format_completion_text(total_tokens=token_total, cost_usd=cost_usd)
            _update_status(
                status.status_icon,
                status.status_label,
                icon_name="check_circle",
                icon_class="text-green-400",
                text=f"{completion_text} ({elapsed:.1f}s)",
            )
            await _scroll_to_bottom()

        except asyncio.CancelledError:
            _handle_abort(state, status, form, start_time, selected_panel, num_rounds, query_text)

        except Exception as exc:
            logger.exception("Debate failed")
            elapsed = time.monotonic() - start_time
            _update_status(
                status.status_icon,
                status.status_label,
                icon_name="error",
                icon_class="text-red-400",
                text=f"Failed ({elapsed:.1f}s)",
            )
            with status.response_container:
                ui.label(f"Debate failed: {exc}").classes("text-red-400 font-mono animate-fade-in")
        finally:
            form.submit_btn.enable()
            form.submit_btn.text = "Run Debate"
            form.abort_btn.visible = False
            state.task = None

    form.submit_btn.on_click(on_submit)

    def on_abort() -> None:
        """Cancel the running debate task."""
        if state.task is not None and not state.task.done():
            state.task.cancel()
            ui.notify("Aborting debate...", type="info")

    form.abort_btn.on_click(on_abort)

    def on_diff_toggle(e: object) -> None:
        """Track diff toggle state for next render."""
        state.show_diff = form.diff_toggle.value

    form.diff_toggle.on_value_change(on_diff_toggle)

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


def _handle_abort(
    state: _DebateState,
    status: _StatusWidgets,
    form: _FormWidgets,
    start_time: float,
    selected_panel: list[str],
    num_rounds: int,
    query_text: str,
) -> None:
    """Handle debate cancellation — save partial transcript and update UI.

    Args:
        state: Mutable debate state with completed rounds.
        status: Status bar and response container widgets.
        form: Form widgets for reading synthesizer value.
        start_time: Monotonic timestamp when the debate started.
        selected_panel: Panel model aliases selected for this debate.
        num_rounds: Configured number of reflection rounds.
        query_text: Original query text.
    """
    logger.info("Debate aborted by user")

    transcript = DebateTranscript(
        query=query_text.strip(),
        panel=selected_panel,
        synthesizer_id=form.synth_select.value,
        max_rounds=num_rounds,
        rounds=list(state.completed_rounds),
        metadata={"aborted": True},
    )
    try:
        save_transcript(transcript)
    except Exception:
        logger.exception("Failed to save partial transcript")

    elapsed = time.monotonic() - start_time
    total_tokens = sum(r.token_count or 0 for rnd in state.completed_rounds for r in rnd.responses)
    completion_text = format_completion_text(total_tokens=total_tokens, cost_usd=None, aborted=True)
    _update_status(
        status.status_icon,
        status.status_label,
        icon_name="cancel",
        icon_class="text-red-400",
        text=f"{completion_text} ({elapsed:.1f}s)",
    )

    with status.response_container:
        ui.label("Debate aborted.").classes("text-red-400 font-mono mt-4 animate-fade-in")
