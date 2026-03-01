"""Debate view placeholder page.

Renders a minimal placeholder for the debate interface. Full implementation
in Brief 2.
"""

from nicegui import ui


def render() -> None:
    """Render the debate view placeholder."""
    ui.label("Debate View").classes("text-2xl font-mono")
    ui.label("Coming in Brief 2").classes("text-gray-500")
