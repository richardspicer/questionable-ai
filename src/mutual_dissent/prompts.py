"""Prompt templates for debate rounds.

Contains the prompt templates used for initial queries, reflection rounds,
and final synthesis. Templates use Python string formatting with named
placeholders.

Design note: Templates are intentionally minimal and model-agnostic. They
don't include system prompts or model-specific formatting — OpenRouter
normalizes that across providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INITIAL_PROMPT = """\
You are participating in a multi-model panel discussion. Answer the following \
query to the best of your ability. Be thorough but concise.

Query: {query}"""

REFLECTION_PROMPT = """\
You previously answered a query as part of a multi-model panel. Below is your \
original response, followed by how other models on the panel responded.

Your previous response:
{own_response}

Other panel members' responses:
{other_responses}

Reflect on the other responses. Where do you agree? Where do you disagree? \
What did they identify that you missed? What did you get right that they missed? \
Provide your refined answer to the original query.

Original query: {query}"""

SYNTHESIS_PROMPT = """\
You are the designated synthesizer for a multi-model panel discussion. Below \
is the full debate transcript including initial responses and any reflection \
rounds from all panel members.

Original query: {query}

{formatted_transcript}

Synthesize the strongest elements from all panel members into a single, \
well-reasoned response. Note where the panel reached consensus and where \
significant disagreements remain. Do not simply concatenate — produce a \
coherent, unified answer."""


def format_initial(query: str) -> str:
    """Format the initial round prompt.

    Args:
        query: The user's original question.

    Returns:
        Formatted prompt string for the initial round.
    """
    return INITIAL_PROMPT.format(query=query)


def format_reflection(
    query: str,
    own_response: str,
    other_responses: list[tuple[str, str]],
) -> str:
    """Format a reflection round prompt.

    Args:
        query: The user's original question.
        own_response: This model's response from the previous round.
        other_responses: List of (model_alias, response_text) tuples from
            other panel members.

    Returns:
        Formatted prompt string for the reflection round.
    """
    formatted_others = "\n\n".join(f"[{alias}]:\n{text}" for alias, text in other_responses)
    return REFLECTION_PROMPT.format(
        query=query,
        own_response=own_response,
        other_responses=formatted_others,
    )


def format_synthesis(
    query: str,
    formatted_transcript: str,
) -> str:
    """Format the synthesis prompt.

    Args:
        query: The user's original question.
        formatted_transcript: Pre-formatted string of all rounds and responses.

    Returns:
        Formatted prompt string for the synthesis step.
    """
    return SYNTHESIS_PROMPT.format(
        query=query,
        formatted_transcript=formatted_transcript,
    )


@dataclass
class RoundSummary:
    """Summary of a debate round for synthesis formatting.

    Attributes:
        round_type: One of "initial", "reflection".
        responses: List of (model_alias, response_text) tuples.
    """

    round_type: str
    responses: list[tuple[str, str]] = field(default_factory=list)


def format_transcript_for_synthesis(
    rounds: list[RoundSummary],
) -> str:
    """Format debate rounds into a readable transcript for the synthesizer.

    Args:
        rounds: List of RoundSummary objects containing round type and responses.

    Returns:
        Formatted multi-round transcript string.
    """
    sections = []
    for round_info in rounds:
        header = f"=== {round_info.round_type.upper()} ROUND ==="
        entries = "\n\n".join(f"[{alias}]:\n{text}" for alias, text in round_info.responses)
        sections.append(f"{header}\n\n{entries}")
    return "\n\n".join(sections)
