"""Debate orchestrator — the core pipeline engine.

Manages the full debate lifecycle: initial fan-out to panel models,
reflection rounds where each model sees others' responses, and final
synthesis by a designated model. All API calls within a round are
parallel.

Typical usage::

    import asyncio
    from mutual_dissent.config import load_config
    from mutual_dissent.orchestrator import run_debate

    config = load_config()
    transcript = asyncio.run(run_debate("What is MCP?", config))
"""

from __future__ import annotations

from mutual_dissent import __version__
from mutual_dissent.config import Config
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.prompts import (
    RoundSummary,
    format_initial,
    format_reflection,
    format_synthesis,
    format_transcript_for_synthesis,
)
from mutual_dissent.providers.router import ProviderRouter


async def run_debate(
    query: str,
    config: Config,
    *,
    panel: list[str] | None = None,
    synthesizer: str | None = None,
    rounds: int | None = None,
) -> DebateTranscript:
    """Execute a full multi-model debate.

    Orchestrates the complete pipeline: initial round → N reflection rounds →
    synthesis. All model calls within a single round execute in parallel.

    Args:
        query: The user's question or prompt.
        config: Loaded application configuration.
        panel: List of model aliases or IDs to use as panelists. Defaults to
            config.default_panel.
        synthesizer: Model alias or ID for synthesis. Defaults to
            config.default_synthesizer.
        rounds: Number of reflection rounds (1-3). Defaults to
            config.default_rounds.

    Returns:
        Complete DebateTranscript with all rounds and synthesis.

    Raises:
        ValueError: If the panel is empty or no provider is available.
    """
    # Resolve defaults.
    panel_aliases = panel or config.default_panel
    synth_alias = synthesizer or config.default_synthesizer
    num_rounds = min(rounds or config.default_rounds, 3)

    # Initialize transcript — stores aliases, not resolved model IDs.
    # Each ModelResponse carries the resolved model_id set by the router.
    transcript = DebateTranscript(
        query=query,
        panel=list(panel_aliases),
        synthesizer_id=synth_alias,
        max_rounds=num_rounds,
        metadata={"version": __version__},
    )

    async with ProviderRouter(config) as router:
        # --- Initial round ---
        initial_responses = await _run_initial_round(router, query, panel_aliases)
        transcript.rounds.append(
            DebateRound(round_number=0, round_type="initial", responses=initial_responses)
        )

        # --- Reflection rounds ---
        prev_responses = initial_responses
        for round_num in range(1, num_rounds + 1):
            reflection_responses = await _run_reflection_round(
                router, query, panel_aliases, prev_responses, round_num
            )
            transcript.rounds.append(
                DebateRound(
                    round_number=round_num,
                    round_type="reflection",
                    responses=reflection_responses,
                )
            )
            prev_responses = reflection_responses

        # --- Synthesis ---
        synthesis = await _run_synthesis(router, query, synth_alias, transcript)
        transcript.synthesis = synthesis

    return transcript


async def _run_initial_round(
    router: ProviderRouter,
    query: str,
    panel_aliases: list[str],
) -> list[ModelResponse]:
    """Fan out the initial query to all panel models in parallel.

    Args:
        router: Active provider router.
        query: User's original query.
        panel_aliases: List of model aliases.

    Returns:
        List of ModelResponse objects from all panel members.
    """
    prompt = format_initial(query)
    requests = [
        {
            "alias_or_id": alias,
            "prompt": prompt,
            "model_alias": alias,
            "round_number": 0,
        }
        for alias in panel_aliases
    ]
    return await router.complete_parallel(requests)


async def _run_reflection_round(
    router: ProviderRouter,
    query: str,
    panel_aliases: list[str],
    prev_responses: list[ModelResponse],
    round_number: int,
) -> list[ModelResponse]:
    """Run one reflection round where each model sees others' responses.

    Each model receives its own previous response plus all other models'
    previous responses, and is asked to reflect and refine.

    Args:
        router: Active provider router.
        query: User's original query.
        panel_aliases: List of model aliases.
        prev_responses: Responses from the previous round.
        round_number: Current reflection round number (1-indexed).

    Returns:
        List of ModelResponse objects from all panel members.
    """
    # Index previous responses by model_alias for lookup.
    response_map: dict[str, ModelResponse] = {r.model_alias: r for r in prev_responses}

    requests = []
    for alias in panel_aliases:
        own = response_map.get(alias)
        own_text = own.content if own and not own.error else "[No response available]"

        others = [
            (r.model_alias, r.content)
            for r in prev_responses
            if r.model_alias != alias and not r.error
        ]

        prompt = format_reflection(query, own_text, others)
        requests.append(
            {
                "alias_or_id": alias,
                "prompt": prompt,
                "model_alias": alias,
                "round_number": round_number,
            }
        )

    return await router.complete_parallel(requests)


async def _run_synthesis(
    router: ProviderRouter,
    query: str,
    synth_alias: str,
    transcript: DebateTranscript,
) -> ModelResponse:
    """Run the synthesis step using the designated model.

    The synthesizer receives the full debate transcript and produces a
    consolidated answer.

    Args:
        router: Active provider router.
        query: User's original query.
        synth_alias: Model alias for synthesis.
        transcript: The debate transcript so far (initial + reflections).

    Returns:
        ModelResponse from the synthesizer.
    """
    # Format all rounds for the synthesizer.
    round_data = []
    for debate_round in transcript.rounds:
        round_data.append(
            RoundSummary(
                round_type=debate_round.round_type,
                responses=[
                    (r.model_alias, r.content) for r in debate_round.responses if not r.error
                ],
            )
        )

    formatted = format_transcript_for_synthesis(round_data)
    prompt = format_synthesis(query, formatted)

    return await router.complete(
        synth_alias,
        prompt=prompt,
        model_alias=synth_alias,
        round_number=-1,
    )
