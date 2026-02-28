"""CLI entry point for Mutual Dissent.

Provides the ``mutual-dissent`` command with subcommands for running
multi-model debates, replaying transcripts, and managing configuration.

Typical usage::

    mutual-dissent ask "What is MCP security?"
    mutual-dissent ask "Compare REST vs GraphQL" --verbose --rounds 2
"""

from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console

from mutual_dissent import __version__
from mutual_dissent.config import Config, load_config
from mutual_dissent.display import render_config_test, render_debate, render_transcript_list
from mutual_dissent.models import ModelResponse
from mutual_dissent.orchestrator import run_debate
from mutual_dissent.providers.router import ProviderRouter
from mutual_dissent.transcript import list_transcripts, load_transcript, save_transcript
from mutual_dissent.types import RoutingDecision

console = Console(stderr=True)


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
    type=click.Choice(["terminal", "json"], case_sensitive=False),
    default="terminal",
    help="Output format (default: terminal).",
)
def ask(
    query: str,
    panel: str | None,
    synthesizer: str | None,
    rounds: int | None,
    verbose: bool,
    no_save: bool,
    output: str,
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

    # Run the debate.
    try:
        transcript = asyncio.run(
            run_debate(
                query,
                config,
                panel=panel_list,
                synthesizer=synthesizer,
                rounds=rounds,
            )
        )
    except Exception as exc:
        console.print(f"[red bold]Error:[/red bold] {exc}")
        sys.exit(1)

    # Save transcript unless --no-save.
    if not no_save:
        filepath = save_transcript(transcript)
        console.print(f"[dim]Transcript saved: {filepath.name}[/dim]")

    # Render output.
    if output == "json":
        click.echo(json.dumps(transcript.to_dict(), indent=2))
    else:
        render_debate(transcript, verbose=verbose)


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
    type=click.Choice(["terminal", "json"], case_sensitive=False),
    default="terminal",
    help="Output format (default: terminal).",
)
def show(transcript_id: str, verbose: bool, output: str) -> None:
    """Display a saved debate transcript.

    Loads and renders a transcript by its full or partial ID (minimum 4
    characters). In default mode, shows synthesis and metadata. With
    --verbose, shows all rounds.

    Args:
        transcript_id: Full UUID or prefix (min 4 chars) to match.
        verbose: Show all round responses.
        output: Output format choice.
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

    if output == "json":
        click.echo(json.dumps(transcript.to_dict(), indent=2))
    else:
        render_debate(transcript, verbose=verbose)


@main.group()
def config() -> None:
    """Manage configuration."""


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
