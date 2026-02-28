"""Terminal display — Rich-based formatting for debate output.

Renders debate transcripts to the terminal with color-coded model
responses, progress indicators, and structured synthesis output.

Typical usage::

    from mutual_dissent.display import render_debate

    render_debate(transcript, verbose=True)
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
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

    _render_score(transcript)
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


def _render_score(transcript: DebateTranscript) -> None:
    """Render ground-truth score as a compact summary table.

    Args:
        transcript: Transcript with scoring data in synthesis.analysis.
    """
    if not transcript.synthesis:
        return
    score_data = transcript.synthesis.analysis.get("ground_truth_score")
    if not score_data:
        return
    if score_data.get("accuracy", -1) < 0:
        console.print("\n[dim]Score: Judge output could not be parsed[/dim]")
        return

    console.print()
    console.rule("[bold]Score[/bold]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value")

    table.add_row("Accuracy", f"{score_data['accuracy']}/5")
    table.add_row("Completeness", f"{score_data['completeness']}/5")
    table.add_row("Overall", f"{score_data['overall']}/5")

    console.print(table)

    explanation = score_data.get("explanation", "")
    if explanation:
        console.print(f"\n[dim]{explanation}[/dim]")


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


def format_markdown(transcript: DebateTranscript, *, verbose: bool = False) -> str:
    """Format a debate transcript as a plain Markdown string.

    Pure function — no Rich markup, no console, no side effects. Returns
    a Markdown document suitable for piping to a file or stdout.

    In default mode, shows synthesis and metadata. In verbose mode,
    includes all round responses before synthesis.

    Args:
        transcript: Completed debate transcript to format.
        verbose: If True, include all round responses.

    Returns:
        Plain Markdown string.
    """
    suffix = "..." if len(transcript.query) > 80 else ""
    lines: list[str] = [f"# Debate: {transcript.query[:80]}{suffix}", ""]

    if verbose:
        for debate_round in transcript.rounds:
            lines.extend(_format_round_markdown(debate_round))
            lines.append("")

    lines.extend(_format_synthesis_markdown(transcript))
    lines.extend(_format_score_markdown(transcript))
    lines.append("")
    lines.extend(_format_metadata_markdown(transcript))
    lines.append("")

    return "\n".join(lines)


def _format_round_markdown(debate_round: DebateRound) -> list[str]:
    """Format one debate round as Markdown lines.

    Args:
        debate_round: A single debate round with responses.

    Returns:
        List of Markdown lines for the round.
    """
    if debate_round.round_type == "initial":
        label = "Initial Round"
    else:
        label = f"Reflection Round {debate_round.round_number}"

    lines: list[str] = [f"## {label}", ""]

    for resp in debate_round.responses:
        lines.extend(_format_response_markdown(resp))
        lines.append("")

    return lines


def _format_response_markdown(resp: ModelResponse) -> list[str]:
    """Format a single model response as Markdown lines.

    Args:
        resp: The model response to format.

    Returns:
        List of Markdown lines for the response.
    """
    lines: list[str] = [f"### {resp.model_alias}", ""]
    timing = _format_timing(resp)
    meta_parts = [resp.model_id]
    if timing:
        meta_parts.append(timing)
    lines.append(f"*{' · '.join(meta_parts)}*")
    lines.append("")

    if resp.error:
        lines.append(f"**Error:** {resp.error}")
    else:
        lines.append(resp.content)

    return lines


def _format_synthesis_markdown(transcript: DebateTranscript) -> list[str]:
    """Format the synthesis section as Markdown lines.

    Args:
        transcript: The completed debate transcript.

    Returns:
        List of Markdown lines for the synthesis section.
    """
    lines: list[str] = ["## Synthesis", ""]

    if not transcript.synthesis:
        lines.append("No synthesis available.")
        return lines

    synth = transcript.synthesis

    lines.append(f"**Synthesized by:** {synth.model_alias}")
    lines.append(f"**Model:** {synth.model_id}")
    timing = _format_timing(synth)
    if timing:
        lines.append(f"*{timing}*")
    lines.append("")

    if synth.error:
        lines.append(f"**Error:** {synth.error}")
    else:
        lines.append(synth.content)

    return lines


def _format_score_markdown(transcript: DebateTranscript) -> list[str]:
    """Format ground-truth score as Markdown lines.

    Args:
        transcript: Transcript with scoring data in synthesis.analysis.

    Returns:
        List of Markdown lines for the score section, or empty list.
    """
    if not transcript.synthesis:
        return []
    score_data = transcript.synthesis.analysis.get("ground_truth_score")
    if not score_data:
        return []
    if score_data.get("accuracy", -1) < 0:
        return ["", "## Score", "", "*Judge output could not be parsed.*"]

    lines: list[str] = ["", "## Score", ""]
    lines.append(f"**Accuracy:** {score_data['accuracy']}/5")
    lines.append(f"**Completeness:** {score_data['completeness']}/5")
    lines.append(f"**Overall:** {score_data['overall']}/5")

    explanation = score_data.get("explanation", "")
    if explanation:
        lines.append("")
        lines.append(explanation)

    return lines


def _format_metadata_markdown(transcript: DebateTranscript) -> list[str]:
    """Format transcript metadata as Markdown lines.

    Args:
        transcript: The completed debate transcript.

    Returns:
        List of Markdown lines for the metadata footer.
    """
    lines: list[str] = ["---", ""]

    lines.append(f"**Transcript:** {transcript.short_id}")

    panel_str = (
        ", ".join(r.model_alias for r in transcript.rounds[0].responses)
        if transcript.rounds
        else ", ".join(transcript.panel)
    )
    lines.append(f"**Panel:** {panel_str}")

    synth_alias = transcript.synthesis.model_alias if transcript.synthesis else "none"
    lines.append(f"**Synthesizer:** {synth_alias}")
    lines.append(f"**Rounds:** {transcript.max_rounds}")

    total_tokens = _total_tokens(transcript)
    if total_tokens > 0:
        lines.append(f"**Tokens:** {total_tokens:,}")

    date_str = transcript.created_at.strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"**Date:** {date_str}")

    return lines


def render_transcript_list(transcripts: list[dict[str, Any]]) -> None:
    """Render a list of transcript summaries as a Rich table.

    Displays transcript metadata including ID, date, panel models,
    synthesizer, token count, and query preview. Color-codes model
    aliases using the standard model color scheme.

    Args:
        transcripts: List of transcript summary dicts as returned by
            ``list_transcripts()``. Each dict has keys: short_id, date,
            panel, synthesizer, tokens, query.
    """
    if not transcripts:
        console.print("[dim]No transcripts found.[/dim]")
        return

    table = Table(show_header=True, padding=(0, 1))
    table.add_column("ID", style="bold")
    table.add_column("Date")
    table.add_column("Panel")
    table.add_column("Synthesizer")
    table.add_column("Tokens", justify="right")
    table.add_column("Query", style="dim")

    for t in transcripts:
        # Color-code each panel alias.
        panel_parts = [a.strip() for a in t["panel"].split(",") if a.strip()]
        panel_str = ", ".join(_format_alias(a) for a in panel_parts)

        # Color-code synthesizer, or em dash if empty.
        synth = t["synthesizer"]
        synth_str = _format_alias(synth) if synth else "\u2014"

        # Format tokens with comma separator, or em dash if zero.
        tokens = t["tokens"]
        tokens_str = f"{tokens:,}" if tokens else "\u2014"

        table.add_row(
            t["short_id"],
            t["date"],
            panel_str,
            synth_str,
            tokens_str,
            t["query"],
        )

    console.print()
    console.print(table)
    console.print()


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
