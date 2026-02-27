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
from mutual_dissent.config import load_config
from mutual_dissent.display import render_debate
from mutual_dissent.orchestrator import run_debate
from mutual_dissent.transcript import save_transcript

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


if __name__ == "__main__":
    main()
