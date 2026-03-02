"""NiceGUI web application for Mutual Dissent.

Defines page routing and server configuration. Started via the
``mutual-dissent serve`` CLI command.
"""

from __future__ import annotations

from nicegui import ui

from mutual_dissent.web.layout import create_layout
from mutual_dissent.web.pages import config as config_page
from mutual_dissent.web.pages import dashboard, debate


def create_app(*, host: str = "127.0.0.1", port: int = 8080, show: bool = True) -> None:
    """Configure and run the NiceGUI application.

    Registers page routes, applies the shared layout, and starts the
    NiceGUI server. This function blocks until the server is stopped.

    Args:
        host: Bind address. Defaults to localhost.
        port: Port number. Defaults to 8080.
        show: Open browser automatically. Defaults to True.
    """

    @ui.page("/")
    def index() -> None:
        create_layout()
        debate.render()

    @ui.page("/dashboard")
    def dashboard_page() -> None:
        create_layout()
        dashboard.render()

    @ui.page("/config")
    def config_page_route() -> None:
        create_layout()
        config_page.render()

    ui.run(
        host=host,
        port=port,
        title="Mutual Dissent",
        dark=True,
        show=show,
        reload=False,
    )
