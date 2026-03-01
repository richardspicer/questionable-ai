"""Shared layout components for the Mutual Dissent web interface.

Provides the navigation shell (header + left drawer) used by all pages.
Dark mode is set globally via ui.run(dark=True) in app.py; the layout
provides a toggle for runtime switching.
"""

from __future__ import annotations

from nicegui import ui

from mutual_dissent import __version__


def create_layout() -> None:
    """Create the shared navigation shell.

    Adds a header with the app title and dark mode toggle, a left drawer
    with page navigation links, and a footer with the version string.
    Call this at the top of every @ui.page function.
    """
    dark = ui.dark_mode()

    with ui.header().classes("items-center justify-between px-4"):
        ui.label("Mutual Dissent").classes("text-lg font-mono font-bold")
        with ui.row().classes("items-center gap-2"):
            ui.switch("Dark mode", value=True).bind_value(dark).classes("text-sm")

    with ui.left_drawer(top_corner=True, bottom_corner=True).classes("bg-gray-900 text-white"):
        ui.link("Debate", "/").classes("text-white no-underline block py-2 px-4 hover:bg-gray-700")
        ui.link("Dashboard", "/dashboard").classes(
            "text-white no-underline block py-2 px-4 hover:bg-gray-700"
        )

    with ui.footer().classes("bg-gray-900 text-gray-500 text-xs py-2 px-4"):
        ui.label(f"Mutual Dissent v{__version__}")
