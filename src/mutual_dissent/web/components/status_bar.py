"""Status bar component — debate progress indicator.

Provides pure-Python formatting functions and a NiceGUI rendering
function that displays debate progress as a compact status bar.
"""

from __future__ import annotations


def format_status_text(
    *,
    round_type: str,
    round_number: int,
    total_rounds: int,
) -> str:
    """Format status text for a debate phase.

    Args:
        round_type: One of "initial", "reflection", "synthesis".
        round_number: Current round number (0 for initial, 1+ for reflection).
        total_rounds: Total configured reflection rounds.

    Returns:
        Human-readable status string like "Reflection 1 of 2...".
    """
    if round_type == "initial":
        return "Initial round..."
    if round_type == "reflection":
        return f"Reflection {round_number} of {total_rounds}..."
    if round_type == "synthesis":
        return "Synthesizing..."
    return f"Round {round_number}..."


def format_completion_text(
    *,
    total_tokens: int,
    cost_usd: float | None,
    aborted: bool = False,
) -> str:
    """Format completion status text with optional stats.

    Args:
        total_tokens: Total tokens used across all rounds.
        cost_usd: Total cost in USD, or None if unavailable.
        aborted: Whether the debate was cancelled early.

    Returns:
        Status string like "Complete — 1,500 tokens, $0.0234".
    """
    prefix = "Aborted" if aborted else "Complete"
    parts: list[str] = []
    if total_tokens > 0:
        parts.append(f"{total_tokens:,} tokens")
    if cost_usd is not None:
        parts.append(f"${cost_usd:.4f}")
    if not parts:
        return prefix
    return f"{prefix} — {', '.join(parts)}"


def render_status_bar() -> tuple:
    """Create a status bar widget and return its updatable elements.

    Returns:
        Tuple of (container, icon_element, label_element) for external updates.
    """
    from nicegui import ui

    with ui.row().classes(
        "w-full items-center gap-2 px-4 py-2 bg-gray-800/50 "
        "border border-gray-700 rounded font-mono text-sm"
    ) as container:
        icon = ui.icon("hourglass_empty").classes("text-gray-400")
        label = ui.label("Ready").classes("text-gray-400")

    return container, icon, label
