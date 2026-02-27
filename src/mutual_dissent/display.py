"""Terminal display — Rich-based formatting for debate output.

Renders debate transcripts to the terminal with color-coded model
responses, progress indicators, and structured synthesis output.

Typical usage::

    from mutual_dissent.display import render_debate

    render_debate(transcript, verbose=True)
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mutual_dissent.models import DebateTranscript, ModelResponse
from mutual_dissent.types import RoutingDecision

console = Console()

# Model alias → color mapping for visual distinction.
MODEL_COLORS: dict[str, str] = {
    "claude": "magenta",
    "gpt": "green",
    "gemini": "cyan",
    "grok": "yellow",
}

DEFAULT_COLOR = "white"


def _get_color(alias: str) -> str:
    """Get the display color for a model alias.

    Args:
        alias: Model short name (e.g. "claude", "gpt").

    Returns:
        Rich color string for the model.
    """
    return MODEL_COLORS.get(alias.lower(), DEFAULT_COLOR)


def render_debate(transcript: DebateTranscript, *, verbose: bool = False) -> None:
    """Render a complete debate transcript to the terminal.

    In default mode, shows a compact summary with just the synthesis.
    In verbose mode, shows all rounds with individual model responses.

    Args:
        transcript: Completed debate transcript to display.
        verbose: If True, show all rounds. If False, show synthesis only.
    """
    console.print()

    if verbose:
        for debate_round in transcript.rounds:
            _render_round(
                debate_round.round_type, debate_round.round_number, debate_round.responses
            )
        console.print()

    if transcript.synthesis:
        _render_synthesis(transcript.synthesis)
    else:
        console.print("[red]No synthesis available.[/red]")

    _render_metadata(transcript)


def _render_round(round_type: str, round_number: int, responses: list[ModelResponse]) -> None:
    """Render one debate round with all model responses.

    Args:
        round_type: "initial" or "reflection".
        round_number: 0 for initial, 1+ for reflection.
        responses: Model responses from this round.
    """
    label = "Initial Round" if round_type == "initial" else f"Reflection Round {round_number}"
    console.rule(f"[bold]{label}[/bold]")
    console.print()

    for resp in responses:
        _render_response(resp)


def _render_response(resp: ModelResponse) -> None:
    """Render a single model response as a colored panel.

    Args:
        resp: The model response to display.
    """
    color = _get_color(resp.model_alias)

    if resp.error:
        panel = Panel(
            f"[red]Error: {resp.error}[/red]",
            title=f"[{color} bold]{resp.model_alias}[/{color} bold]",
            border_style="red",
            padding=(0, 1),
        )
    else:
        panel = Panel(
            Markdown(resp.content),
            title=f"[{color} bold]{resp.model_alias}[/{color} bold]",
            border_style=color,
            padding=(0, 1),
        )

    timing = _format_timing(resp)
    if timing:
        panel.subtitle = timing

    console.print(panel)
    console.print()


def _render_synthesis(synthesis: ModelResponse) -> None:
    """Render the synthesis response prominently.

    Args:
        synthesis: The synthesizer's final response.
    """
    color = _get_color(synthesis.model_alias)

    if synthesis.error:
        console.print(
            Panel(
                f"[red]Synthesis failed: {synthesis.error}[/red]",
                title="[bold red]Synthesis[/bold red]",
                border_style="red",
            )
        )
        return

    console.rule("[bold]Synthesis[/bold]")
    console.print()
    console.print(
        Panel(
            Markdown(synthesis.content),
            title=f"[{color} bold]Synthesized by {synthesis.model_alias}[/{color} bold]",
            border_style=color,
            padding=(1, 2),
        )
    )


def _render_metadata(transcript: DebateTranscript) -> None:
    """Render debate metadata as a compact summary table.

    Args:
        transcript: The completed debate transcript.
    """
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value")

    table.add_row("Transcript", transcript.short_id)
    table.add_row(
        "Panel",
        ", ".join(_format_alias(r.model_alias) for r in transcript.rounds[0].responses)
        if transcript.rounds
        else "none",
    )

    synth_alias = transcript.synthesis.model_alias if transcript.synthesis else "none"
    table.add_row("Synthesizer", synth_alias)
    table.add_row("Rounds", str(transcript.max_rounds))

    # Total tokens if available.
    total_tokens = _total_tokens(transcript)
    if total_tokens > 0:
        table.add_row("Tokens", f"{total_tokens:,}")

    console.print(table)
    console.print()


def _format_timing(resp: ModelResponse) -> str:
    """Format latency and token count for display.

    Args:
        resp: Model response with optional timing data.

    Returns:
        Formatted string like "2.1s · 450 tokens", or empty string.
    """
    parts: list[str] = []
    if resp.latency_ms is not None:
        parts.append(f"{resp.latency_ms / 1000:.1f}s")
    if resp.token_count is not None:
        parts.append(f"{resp.token_count:,} tokens")
    return " · ".join(parts)


def _format_alias(alias: str) -> str:
    """Format a model alias with its color.

    Args:
        alias: Model short name.

    Returns:
        Rich markup string with color applied.
    """
    color = _get_color(alias)
    return f"[{color}]{alias}[/{color}]"


def _total_tokens(transcript: DebateTranscript) -> int:
    """Sum total tokens across all responses in a transcript.

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


def render_config_test(
    results: list[dict[str, RoutingDecision | ModelResponse | str]],
) -> None:
    """Render config test results as a Rich table.

    Displays a table with routing decision and response status for each
    model alias tested, showing vendor, route type, resolved model ID,
    latency, and success/error status.

    Args:
        results: List of result dicts, each containing:
            - ``alias`` (str): Model alias tested.
            - ``decision`` (RoutingDecision): Routing decision for the alias.
            - ``response`` (ModelResponse): Response from the test prompt.
    """
    table = Table(show_header=True, padding=(0, 1))
    table.add_column("Alias", style="bold")
    table.add_column("Vendor")
    table.add_column("Route")
    table.add_column("Model ID", style="dim")
    table.add_column("Latency", justify="right")
    table.add_column("Status")

    for result in results:
        alias = str(result["alias"])
        decision: RoutingDecision = result["decision"]  # type: ignore[assignment]
        response: ModelResponse = result["response"]  # type: ignore[assignment]

        color = _get_color(alias)
        alias_str = f"[{color}]{alias}[/{color}]"
        vendor_str = decision.vendor.value
        route_str = "openrouter" if decision.via_openrouter else "direct"
        model_id_str = response.model_id

        if response.error:
            latency_str = "\u2014"
            status_str = f"[red]\u2717 {response.error}[/red]"
        else:
            if response.latency_ms is not None:
                latency_str = f"{response.latency_ms / 1000:.1f}s"
            else:
                latency_str = "\u2014"
            status_str = "[green]\u2713[/green]"

        table.add_row(alias_str, vendor_str, route_str, model_id_str, latency_str, status_str)

    console.print()
    console.print(table)
    console.print()
