"""CLI entry point for Mutual Dissent.

Provides the ``mutual-dissent`` command with subcommands for running
multi-model debates, replaying transcripts, and managing configuration.

Typical usage::

    mutual-dissent ask "What is MCP security?"
    mutual-dissent ask "Compare REST vs GraphQL" --verbose --rounds 2
    mutual-dissent replay abcd1234 --synthesizer gpt --rounds 1
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console

from mutual_dissent import __version__
from mutual_dissent.config import CONFIG_PATH, Config, load_config
from mutual_dissent.display import (
    format_markdown,
    render_config_show,
    render_config_test,
    render_debate,
    render_transcript_list,
)
from mutual_dissent.models import DebateTranscript, ModelResponse
from mutual_dissent.orchestrator import run_debate, run_replay
from mutual_dissent.providers.router import ProviderRouter
from mutual_dissent.transcript import list_transcripts, load_transcript, save_transcript
from mutual_dissent.types import RoutingDecision

console = Console(stderr=True)


def _resolve_ground_truth(
    ground_truth: str | None,
    ground_truth_file: str | None,
) -> str | None:
    """Resolve ground-truth text from inline or file source.

    Args:
        ground_truth: Inline reference answer text.
        ground_truth_file: Path to file containing reference answer.

    Returns:
        Resolved ground-truth string, or None if neither provided.

    Raises:
        click.UsageError: If both sources are provided.
    """
    if ground_truth and ground_truth_file:
        raise click.UsageError("Cannot use both --ground-truth and --ground-truth-file. Pick one.")
    if ground_truth_file:
        return Path(ground_truth_file).read_text(encoding="utf-8").strip()
    return ground_truth


def _emit_output(
    transcript: DebateTranscript,
    *,
    output: str,
    output_file: str | None,
    verbose: bool,
) -> None:
    """Build and emit formatted output for a debate transcript.

    Dispatches to the appropriate formatter based on ``output`` format.
    When ``output_file`` is set, writes to disk instead of stdout.
    Terminal format with ``--file`` degrades to markdown with a stderr note.

    Args:
        transcript: Completed debate transcript to render.
        output: Output format — "terminal", "json", or "markdown".
        output_file: Path to write output to, or None for stdout.
        verbose: If True, include all round responses in output.
    """
    # Terminal without --file: render Rich panels directly and return.
    if output == "terminal" and output_file is None:
        render_debate(transcript, verbose=verbose)
        return

    # Build the output string.
    if output == "json":
        content = json.dumps(transcript.to_dict(), indent=2)
    else:
        if output == "terminal":
            console.print(
                "[dim]Note: --output terminal not supported with --file, writing as markdown.[/dim]"
            )
        content = format_markdown(transcript, verbose=verbose)

    # Normalize trailing newline.
    content = content.rstrip("\n") + "\n"

    if output_file is not None:
        resolved = Path(output_file).resolve()
        try:
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            console.print(f"[red bold]Error:[/red bold] Cannot write to {resolved}: {exc}")
            sys.exit(1)
        console.print(f"[dim]Output written to {resolved}[/dim]")
    else:
        click.echo(content, nl=False)


@click.group()
@click.version_option(version=__version__, prog_name="mutual-dissent")
def main() -> None:
    """Cross-vendor multi-model debate and consensus engine.

    Sends a query to multiple AI models, shares competing responses back
    for reflection and critique, then synthesizes a final answer through
    a user-selected model.
    """


@main.command()
@click.argument("query")
@click.option(
    "--panel",
    default=None,
    help="Comma-separated model aliases (e.g. claude,gpt,gemini).",
)
@click.option(
    "--synthesizer",
    default=None,
    help="Model alias for final synthesis (default: claude).",
)
@click.option(
    "--rounds",
    default=None,
    type=click.IntRange(1, 3),
    help="Reflection rounds, 1-3 (default: 1).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all round responses, not just synthesis.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Don't save transcript to disk.",
)
@click.option(
    "--output",
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    default="terminal",
    help="Output format (default: terminal).",
)
@click.option(
    "--file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write output to FILE instead of stdout.",
)
@click.option(
    "--ground-truth",
    default=None,
    help="Reference answer to score synthesis against (adds 1 API call).",
)
@click.option(
    "--ground-truth-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="File containing reference answer for scoring.",
)
def ask(
    query: str,
    panel: str | None,
    synthesizer: str | None,
    rounds: int | None,
    verbose: bool,
    no_save: bool,
    output: str,
    output_file: str | None,
    ground_truth: str | None,
    ground_truth_file: str | None,
) -> None:
    """Send a query to the debate panel.

    Fans out QUERY to all panel models, runs reflection rounds, and
    synthesizes a final answer.

    Args:
        query: The question or prompt to debate.
        panel: Comma-separated model aliases.
        synthesizer: Model alias for final synthesis.
        rounds: Number of reflection rounds.
        verbose: Show all round responses.
        no_save: Skip transcript saving.
        output: Output format choice.
        output_file: Path to write output to.
        ground_truth: Inline reference answer for scoring.
        ground_truth_file: Path to file containing reference answer.
    """
    config = load_config()

    # Validate that at least one provider key is configured.
    if not config.api_key and not any(config.providers.values()):
        console.print(
            "[red bold]Error:[/red bold] No API key found.\n"
            "Set OPENROUTER_API_KEY (or another provider key) environment variable\n"
            "or configure keys in ~/.mutual-dissent/config.toml"
        )
        sys.exit(1)

    # Parse panel if provided.
    panel_list = panel.split(",") if panel else None

    # Resolve ground truth.
    resolved_gt = _resolve_ground_truth(ground_truth, ground_truth_file)

    # Run the debate.
    try:
        transcript = asyncio.run(
            run_debate(
                query,
                config,
                panel=panel_list,
                synthesizer=synthesizer,
                rounds=rounds,
                ground_truth=resolved_gt,
            )
        )
    except Exception as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    # Save transcript unless --no-save.
    if not no_save:
        filepath = save_transcript(transcript)
        console.print(f"[dim]Transcript saved: {filepath.name}[/dim]")

    _emit_output(transcript, output=output, output_file=output_file, verbose=verbose)


@main.command("list")
@click.option(
    "--limit",
    default=20,
    type=click.IntRange(1),
    help="Maximum transcripts to show (default: 20).",
)
def list_cmd(limit: int) -> None:
    """List saved debate transcripts.

    Shows a table of recent transcripts with IDs, dates, panel models,
    and queries. Use ``dissent show <id>`` to view a specific transcript.

    Args:
        limit: Maximum number of transcripts to list.
    """
    transcripts = list_transcripts(limit=limit)
    render_transcript_list(transcripts)


@main.command()
@click.argument("transcript_id")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all rounds, not just synthesis.",
)
@click.option(
    "--output",
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    default="terminal",
    help="Output format (default: terminal).",
)
@click.option(
    "--file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write output to FILE instead of stdout.",
)
def show(transcript_id: str, verbose: bool, output: str, output_file: str | None) -> None:
    """Display a saved debate transcript.

    Loads and renders a transcript by its full or partial ID (minimum 4
    characters). In default mode, shows synthesis and metadata. With
    --verbose, shows all rounds.

    Args:
        transcript_id: Full UUID or prefix (min 4 chars) to match.
        verbose: Show all round responses.
        output: Output format choice.
        output_file: Path to write output to.
    """
    if len(transcript_id) < 4:
        console.print("[red bold]Error:[/red bold] Transcript ID must be at least 4 characters.")
        sys.exit(1)

    try:
        transcript = load_transcript(transcript_id)
    except ValueError as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    if transcript is None:
        console.print(
            f"[red bold]Error:[/red bold] No transcript found matching '{transcript_id}'."
        )
        sys.exit(1)

    _emit_output(transcript, output=output, output_file=output_file, verbose=verbose)


@main.command()
@click.argument("transcript_id")
@click.option(
    "--synthesizer",
    default=None,
    help="Model alias to override the original synthesizer.",
)
@click.option(
    "--rounds",
    default=0,
    type=click.IntRange(0, 3),
    help=(
        "Additional reflection rounds before re-synthesis (default: 0). "
        "Each round costs N API calls per panel model."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show all round responses, not just synthesis.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Don't save replay transcript to disk.",
)
@click.option(
    "--output",
    type=click.Choice(["terminal", "json", "markdown"], case_sensitive=False),
    default="terminal",
    help="Output format (default: terminal).",
)
@click.option(
    "--file",
    "output_file",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write output to FILE instead of stdout.",
)
@click.option(
    "--ground-truth",
    default=None,
    help="Reference answer to score synthesis against (adds 1 API call).",
)
@click.option(
    "--ground-truth-file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="File containing reference answer for scoring.",
)
def replay(
    transcript_id: str,
    synthesizer: str | None,
    rounds: int,
    verbose: bool,
    no_save: bool,
    output: str,
    output_file: str | None,
    ground_truth: str | None,
    ground_truth_file: str | None,
) -> None:
    """Re-synthesize or extend an existing debate transcript.

    Loads the transcript matching TRANSCRIPT_ID (full or partial, min 4
    chars), optionally adds reflection rounds, then runs synthesis with
    the original or overridden synthesizer. Produces a new transcript --
    the original is never modified.

    Args:
        transcript_id: Full UUID or prefix (min 4 chars) to match.
        synthesizer: Model alias override for synthesis.
        rounds: Additional reflection rounds to run.
        verbose: Show all round responses.
        no_save: Skip saving replay transcript.
        output: Output format choice.
        output_file: Path to write output to.
        ground_truth: Inline reference answer for scoring.
        ground_truth_file: Path to file containing reference answer.
    """
    # Validate transcript ID length (cheap check first).
    if len(transcript_id) < 4:
        console.print("[red bold]Error:[/red bold] Transcript ID must be at least 4 characters.")
        sys.exit(1)

    config = load_config()

    # Validate API key -- replay makes live API calls.
    if not config.api_key and not any(config.providers.values()):
        console.print(
            "[red bold]Error:[/red bold] No API key found.\n"
            "Set OPENROUTER_API_KEY (or another provider key) environment variable\n"
            "or configure keys in ~/.mutual-dissent/config.toml"
        )
        sys.exit(1)

    # Load source transcript.
    try:
        source = load_transcript(transcript_id)
    except ValueError as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    if source is None:
        console.print(
            f"[red bold]Error:[/red bold] No transcript found matching '{transcript_id}'."
        )
        sys.exit(1)

    # Resolve ground truth.
    resolved_gt = _resolve_ground_truth(ground_truth, ground_truth_file)

    # Run replay.
    try:
        transcript = asyncio.run(
            run_replay(
                source,
                config,
                synthesizer=synthesizer,
                additional_rounds=rounds,
                ground_truth=resolved_gt,
            )
        )
    except Exception as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    # Save unless --no-save.
    if not no_save:
        filepath = save_transcript(transcript)
        console.print(f"[dim]Replay transcript saved: {filepath.name}[/dim]")

    _emit_output(transcript, output=output, output_file=output_file, verbose=verbose)


@main.command()
@click.option("--port", default=8080, help="Port to bind to.")
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--no-open", is_flag=True, help="Don't open browser automatically.")
def serve(port: int, host: str, no_open: bool) -> None:
    """Start the web UI server."""
    from mutual_dissent.web.app import create_app

    create_app(host=host, port=port, show=not no_open)


@main.group()
def config() -> None:
    """Manage configuration."""


@config.command()
def path() -> None:
    """Print the configuration file path."""
    click.echo(CONFIG_PATH)


@config.command("show")
@click.option(
    "--check-models",
    is_flag=True,
    default=False,
    help="Fetch model context lengths from OpenRouter (requires network).",
)
def config_show(check_models: bool) -> None:
    """Display effective configuration."""
    cfg = load_config()

    context_lengths: dict[str, int] | None = None
    if check_models:
        from mutual_dissent.pricing import PricingCache

        cache = PricingCache(alias_map=cfg._model_aliases_v2)

        async def _fetch_lengths() -> dict[str, int]:
            await cache.prefetch()
            lengths: dict[str, int] = {}
            for alias in cfg._model_aliases_v2:
                or_id = cfg._model_aliases_v2[alias].get("openrouter", "")
                if or_id:
                    ctx = await cache.get_context_length(or_id)
                    if ctx is not None:
                        lengths[alias] = ctx
            return lengths

        context_lengths = asyncio.run(_fetch_lengths())

    render_config_show(cfg, context_lengths=context_lengths)


async def _run_config_test(
    cfg: Config,
    aliases: list[str],
) -> list[dict[str, RoutingDecision | ModelResponse | str]]:
    """Send a test prompt to each alias and collect results.

    Opens a ``ProviderRouter``, resolves routing for each alias, then
    sends ``"Say OK"`` to all models in parallel.

    Args:
        cfg: Application configuration.
        aliases: List of unique model aliases to test.

    Returns:
        List of result dicts with ``alias``, ``decision``, and ``response``.
    """
    async with ProviderRouter(cfg) as router:
        decisions = {alias: router.route(alias) for alias in aliases}

        requests = [
            {"alias_or_id": alias, "prompt": "Say OK", "model_alias": alias} for alias in aliases
        ]
        responses = await router.complete_parallel(requests)

    return [
        {"alias": alias, "decision": decisions[alias], "response": resp}
        for alias, resp in zip(aliases, responses, strict=True)
    ]


@config.command()
def test() -> None:
    """Test provider configuration and model routing.

    Sends a minimal prompt to each model in the default panel and
    synthesizer.  Reports routing decisions, provider used, latency,
    and errors.
    """
    cfg = load_config()

    # Validate that at least one provider key is configured.
    if not cfg.api_key and not any(cfg.providers.values()):
        console.print(
            "[red bold]Error:[/red bold] No API key found.\n"
            "Set OPENROUTER_API_KEY (or another provider key) environment variable\n"
            "or configure keys in ~/.mutual-dissent/config.toml"
        )
        sys.exit(1)

    # Collect unique aliases: default panel + synthesizer.
    aliases = list(dict.fromkeys(cfg.default_panel + [cfg.default_synthesizer]))

    console.print(f"[dim]Testing {len(aliases)} model(s)...[/dim]")

    try:
        results = asyncio.run(_run_config_test(cfg, aliases))
    except Exception as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    render_config_test(results)

    # Exit 1 if any model failed.
    if any(r["response"].error for r in results):  # type: ignore[union-attr]
        sys.exit(1)


if __name__ == "__main__":
    main()
