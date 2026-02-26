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
from mutual_dissent.client import OpenRouterClient
from mutual_dissent.config import Config
from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse
from mutual_dissent.prompts import (
    RoundSummary,
    format_initial,
    format_reflection,
    format_synthesis,
    format_transcript_for_synthesis,
)


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
        ValueError: If the API key is missing or panel is empty.
    """
    # Resolve defaults.
    panel_aliases = panel or config.default_panel
    synth_alias = synthesizer or config.default_synthesizer
    num_rounds = min(rounds or config.default_rounds, 3)

    # Resolve to OpenRouter model IDs.
    panel_ids = config.resolve_panel(panel_aliases)
    synth_id = config.resolve_model(synth_alias)

    # Build alias lookup: model_id → alias.
    alias_map = _build_alias_map(panel_aliases, panel_ids, synth_alias, synth_id)

    # Initialize transcript.
    transcript = DebateTranscript(
        query=query,
        panel=panel_ids,
        synthesizer_id=synth_id,
        max_rounds=num_rounds,
        metadata={"version": __version__, "openrouter_api": True},
    )

    async with OpenRouterClient(api_key=config.api_key) as client:
        # --- Initial round ---
        initial_responses = await _run_initial_round(client, query, panel_ids, alias_map)
        transcript.rounds.append(
            DebateRound(round_number=0, round_type="initial", responses=initial_responses)
        )

        # --- Reflection rounds ---
        prev_responses = initial_responses
        for round_num in range(1, num_rounds + 1):
            reflection_responses = await _run_reflection_round(
                client, query, panel_ids, alias_map, prev_responses, round_num
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
        synthesis = await _run_synthesis(client, query, synth_id, alias_map, transcript)
        transcript.synthesis = synthesis

    return transcript


async def _run_initial_round(
    client: OpenRouterClient,
    query: str,
    panel_ids: list[str],
    alias_map: dict[str, str],
) -> list[ModelResponse]:
    """Fan out the initial query to all panel models in parallel.

    Args:
        client: Active OpenRouter client.
        query: User's original query.
        panel_ids: List of OpenRouter model IDs.
        alias_map: model_id → alias mapping.

    Returns:
        List of ModelResponse objects from all panel members.
    """
    prompt = format_initial(query)
    requests = [
        {
            "model_id": mid,
            "prompt": prompt,
            "model_alias": alias_map.get(mid, mid),
            "round_number": 0,
        }
        for mid in panel_ids
    ]
    return await client.complete_parallel(requests)


async def _run_reflection_round(
    client: OpenRouterClient,
    query: str,
    panel_ids: list[str],
    alias_map: dict[str, str],
    prev_responses: list[ModelResponse],
    round_number: int,
) -> list[ModelResponse]:
    """Run one reflection round where each model sees others' responses.

    Each model receives its own previous response plus all other models'
    previous responses, and is asked to reflect and refine.

    Args:
        client: Active OpenRouter client.
        query: User's original query.
        panel_ids: List of OpenRouter model IDs.
        alias_map: model_id → alias mapping.
        prev_responses: Responses from the previous round.
        round_number: Current reflection round number (1-indexed).

    Returns:
        List of ModelResponse objects from all panel members.
    """
    # Index previous responses by model_id for lookup.
    response_map: dict[str, ModelResponse] = {r.model_id: r for r in prev_responses}

    requests = []
    for mid in panel_ids:
        own = response_map.get(mid)
        own_text = own.content if own and not own.error else "[No response available]"

        others = [
            (alias_map.get(r.model_id, r.model_id), r.content)
            for r in prev_responses
            if r.model_id != mid and not r.error
        ]

        prompt = format_reflection(query, own_text, others)
        requests.append(
            {
                "model_id": mid,
                "prompt": prompt,
                "model_alias": alias_map.get(mid, mid),
                "round_number": round_number,
            }
        )

    return await client.complete_parallel(requests)


async def _run_synthesis(
    client: OpenRouterClient,
    query: str,
    synth_id: str,
    alias_map: dict[str, str],
    transcript: DebateTranscript,
) -> ModelResponse:
    """Run the synthesis step using the designated model.

    The synthesizer receives the full debate transcript and produces a
    consolidated answer.

    Args:
        client: Active OpenRouter client.
        query: User's original query.
        synth_id: OpenRouter model ID for synthesis.
        alias_map: model_id → alias mapping.
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
                    (alias_map.get(r.model_id, r.model_id), r.content)
                    for r in debate_round.responses
                    if not r.error
                ],
            )
        )

    formatted = format_transcript_for_synthesis(round_data)
    prompt = format_synthesis(query, formatted)

    return await client.complete(
        model_id=synth_id,
        prompt=prompt,
        model_alias=alias_map.get(synth_id, synth_id),
        round_number=-1,
    )


def _build_alias_map(
    panel_aliases: list[str],
    panel_ids: list[str],
    synth_alias: str,
    synth_id: str,
) -> dict[str, str]:
    """Build a model_id → alias mapping for display purposes.

    Args:
        panel_aliases: Short names used for panel members.
        panel_ids: Resolved OpenRouter model IDs for panel.
        synth_alias: Short name for the synthesizer.
        synth_id: Resolved OpenRouter model ID for synthesizer.

    Returns:
        Dictionary mapping model IDs to their human-readable aliases.
    """
    alias_map: dict[str, str] = {}
    for alias, mid in zip(panel_aliases, panel_ids, strict=True):
        alias_map[mid] = alias
    alias_map[synth_id] = synth_alias
    return alias_map
