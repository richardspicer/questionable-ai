"""Research dashboard placeholder page.

Renders a minimal placeholder for the research dashboard. Full
implementation in Brief 4.
"""

from nicegui import ui


def render() -> None:
    """Render the research dashboard placeholder."""
    ui.label("Research Dashboard").classes("text-2xl font-mono")
    ui.label("Coming in Brief 4").classes("text-gray-500")
