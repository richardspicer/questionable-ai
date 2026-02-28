"""Ground-truth scoring — LLM-as-judge evaluation of synthesis quality.

Scores a debate synthesis against a known-correct reference answer using
a judge model. Produces structured accuracy and completeness scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mutual_dissent.prompts import format_scoring

if TYPE_CHECKING:
    from mutual_dissent.providers.router import ProviderRouter


@dataclass
class GroundTruthScore:
    """Result of scoring a synthesis against a ground-truth reference.

    Attributes:
        accuracy: 1-5 accuracy score (-1 on parse failure).
        completeness: 1-5 completeness score (-1 on parse failure).
        overall: Average of accuracy and completeness.
        explanation: Judge's explanation of the scores.
        judge_model: Model alias that performed the scoring.
    """

    accuracy: int
    completeness: int
    overall: float
    explanation: str
    judge_model: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with all score fields.
        """
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "overall": self.overall,
            "explanation": self.explanation,
            "judge_model": self.judge_model,
        }


def parse_score_response(content: str) -> tuple[int, int, str]:
    """Parse a judge model's scoring response.

    Extracts ACCURACY, COMPLETENESS, and EXPLANATION from the judge's
    structured text output. Case-insensitive, whitespace-tolerant.

    Args:
        content: Raw text response from the judge model.

    Returns:
        Tuple of (accuracy, completeness, explanation).

    Raises:
        ValueError: If ACCURACY or COMPLETENESS fields are missing or
            contain non-numeric values.
    """
    # Extract accuracy.
    acc_match = re.search(r"(?i)accuracy\s*:\s*(\S+)", content)
    if not acc_match:
        raise ValueError("Missing ACCURACY field in judge response")
    try:
        accuracy = int(acc_match.group(1))
    except ValueError:
        raise ValueError(f"Non-numeric ACCURACY value: {acc_match.group(1)}") from None

    # Extract completeness.
    comp_match = re.search(r"(?i)completeness\s*:\s*(\S+)", content)
    if not comp_match:
        raise ValueError("Missing COMPLETENESS field in judge response")
    try:
        completeness = int(comp_match.group(1))
    except ValueError:
        raise ValueError(f"Non-numeric COMPLETENESS value: {comp_match.group(1)}") from None

    # Clamp to 1-5.
    accuracy = max(1, min(5, accuracy))
    completeness = max(1, min(5, completeness))

    # Extract explanation (everything after EXPLANATION:, may be multiline).
    expl_match = re.search(r"(?i)explanation\s*:\s*(.*)", content, re.DOTALL)
    explanation = expl_match.group(1).strip() if expl_match else ""

    return accuracy, completeness, explanation


async def score_synthesis(
    router: ProviderRouter,
    query: str,
    synthesis_content: str,
    ground_truth: str,
    judge_alias: str,
) -> GroundTruthScore:
    """Score a synthesis against a ground-truth reference using an LLM judge.

    Sends a scoring prompt to the judge model and parses the structured
    response. If parsing fails, returns an error score rather than raising.

    Args:
        router: Active provider router.
        query: The original debate query.
        synthesis_content: The synthesized answer to evaluate.
        ground_truth: Known-correct reference answer.
        judge_alias: Model alias for the judge (typically the synthesizer).

    Returns:
        GroundTruthScore with parsed scores or error indicators.
    """
    prompt = format_scoring(query, ground_truth, synthesis_content)

    response = await router.complete(
        judge_alias,
        prompt=prompt,
        model_alias=judge_alias,
        round_number=-2,
    )

    try:
        accuracy, completeness, explanation = parse_score_response(response.content)
        overall = (accuracy + completeness) / 2
        return GroundTruthScore(
            accuracy=accuracy,
            completeness=completeness,
            overall=overall,
            explanation=explanation,
            judge_model=judge_alias,
        )
    except ValueError:
        return GroundTruthScore(
            accuracy=-1,
            completeness=-1,
            overall=-1,
            explanation=f"Judge output could not be parsed: {response.content}",
            judge_model=judge_alias,
        )
