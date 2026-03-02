"""Transcript logging — JSON serialization and file storage.

Writes complete debate transcripts as structured JSON files to the
transcript directory (~/.mutual-dissent/transcripts/). Supports
saving, listing, and loading transcripts by ID.

File naming convention: {date}_{short-id}.json
Example: 2026-02-21_a1b2c3d4.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mutual_dissent.config import TRANSCRIPT_DIR, ensure_dirs
from mutual_dissent.models import DebateRound, DebateTranscript, ExperimentMetadata, ModelResponse


def save_transcript(transcript: DebateTranscript) -> Path:
    """Save a debate transcript as a JSON file.

    Args:
        transcript: The completed debate transcript to save.

    Returns:
        Path to the saved JSON file.

    Example::

        path = save_transcript(transcript)
        print(f"Saved to {path}")
    """
    ensure_dirs()
    date_str = transcript.created_at.strftime("%Y-%m-%d")
    filename = f"{date_str}_{transcript.short_id}.json"
    filepath = TRANSCRIPT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def load_transcript(transcript_id: str) -> DebateTranscript | None:
    """Load a transcript by full or partial ID.

    Searches the transcript directory for files matching the given ID
    prefix. Partial IDs (minimum 4 characters) are supported.

    Args:
        transcript_id: Full UUID or prefix (min 4 chars) to match.

    Returns:
        DebateTranscript if found, None if no match.

    Raises:
        ValueError: If multiple transcripts match the prefix.
    """
    ensure_dirs()
    matches = _find_transcript_files(transcript_id)

    if not matches:
        return None
    if len(matches) > 1:
        filenames = [m.name for m in matches]
        raise ValueError(
            f"Ambiguous transcript ID '{transcript_id}'. "
            f"Matches: {', '.join(filenames)}. Use a longer prefix."
        )

    return _parse_transcript_file(matches[0])


def list_transcripts(limit: int = 20) -> list[dict[str, Any]]:
    """List saved transcripts, most recent first.

    Args:
        limit: Maximum number of transcripts to return. Use 0 for no limit.

    Returns:
        List of dicts with 'id', 'short_id', 'date', 'query', 'file',
        'panel', 'synthesizer', 'tokens', 'cost', 'rounds', and
        'experiment_id' keys.
    """
    ensure_dirs()
    files = sorted(TRANSCRIPT_DIR.glob("*.json"), reverse=True)

    if limit > 0:
        files = files[:limit]

    results: list[dict[str, Any]] = []
    for filepath in files:
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            panel_list: list[str] = data.get("panel", [])
            metadata = data.get("metadata", {})
            stats = metadata.get("stats", {})
            experiment = metadata.get("experiment")
            results.append(
                {
                    "id": data.get("transcript_id", ""),
                    "short_id": data.get("transcript_id", "")[:8],
                    "date": data.get("created_at", "")[:10],
                    "query": _truncate(data.get("query", ""), 80),
                    "file": filepath.name,
                    "panel": ", ".join(panel_list),
                    "synthesizer": data.get("synthesizer_id", ""),
                    "tokens": _count_tokens_from_dict(data),
                    "cost": stats.get("total_cost_usd"),
                    "rounds": len(data.get("rounds", [])),
                    "experiment_id": experiment.get("experiment_id")
                    if isinstance(experiment, dict)
                    else None,
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def _count_tokens_from_dict(data: dict[str, Any]) -> int:
    """Sum all token_count values from a transcript data dict.

    Iterates through all rounds and their responses, plus the synthesis
    response if present, to compute a total token count.

    Args:
        data: Parsed transcript JSON dict with 'rounds' and optional
            'synthesis' keys.

    Returns:
        Total token count across all responses. Returns 0 if no token
        data is found.
    """
    total = 0
    for round_data in data.get("rounds", []):
        for response in round_data.get("responses", []):
            token_count = response.get("token_count")
            if token_count is not None:
                total += token_count

    synthesis = data.get("synthesis")
    if synthesis is not None:
        token_count = synthesis.get("token_count")
        if token_count is not None:
            total += token_count

    return total


def _find_transcript_files(transcript_id: str) -> list[Path]:
    """Find transcript files matching an ID prefix.

    Args:
        transcript_id: Full or partial transcript ID.

    Returns:
        List of matching file paths.
    """
    matches = []
    for filepath in TRANSCRIPT_DIR.glob("*.json"):
        # ID is embedded in filename after the date: {date}_{short-id}.json
        # But we also check the full ID inside the file for longer prefixes.
        name_parts = filepath.stem.split("_", 1)
        if len(name_parts) == 2 and name_parts[1].startswith(transcript_id[:8]):
            # Quick filename match — verify against full ID in file.
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                full_id = data.get("transcript_id", "")
                if full_id.startswith(transcript_id):
                    matches.append(filepath)
            except (json.JSONDecodeError, KeyError):
                continue
    return matches


def _parse_datetime(value: str | None) -> datetime:
    """Parse an ISO 8601 timestamp string to a datetime.

    Args:
        value: ISO 8601 formatted timestamp string, or None.

    Returns:
        Parsed datetime object. Falls back to current UTC time if parsing
        fails or the value is empty.
    """
    if not value:
        return datetime.now(UTC)
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return datetime.now(UTC)


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    """Parse a dictionary into a ModelResponse dataclass.

    Handles missing optional fields gracefully for old transcripts
    that predate the role/routing/analysis additions.

    Args:
        data: Dictionary matching the ModelResponse.to_dict() format.

    Returns:
        Fully populated ModelResponse instance.
    """
    return ModelResponse(
        model_id=data["model_id"],
        model_alias=data["model_alias"],
        round_number=data["round_number"],
        content=data["content"],
        timestamp=_parse_datetime(data.get("timestamp", "")),
        token_count=data.get("token_count"),
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        latency_ms=data.get("latency_ms"),
        error=data.get("error"),
        role=data.get("role", ""),
        routing=data.get("routing"),
        analysis=data.get("analysis", {}),
    )


def _parse_transcript_file(filepath: Path) -> DebateTranscript:
    """Parse a JSON file into a fully deserialized DebateTranscript.

    Reconstructs all nested dataclass instances including rounds,
    responses, and synthesis. Handles backward compatibility with
    older transcripts that may lack newer fields.

    Args:
        filepath: Path to the transcript JSON file.

    Returns:
        DebateTranscript with fully populated rounds and synthesis.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    rounds: list[DebateRound] = []
    for round_data in data.get("rounds", []):
        responses = [_parse_response(r) for r in round_data.get("responses", [])]
        rounds.append(
            DebateRound(
                round_number=round_data["round_number"],
                round_type=round_data["round_type"],
                responses=responses,
            )
        )

    synthesis_data = data.get("synthesis")
    synthesis = _parse_response(synthesis_data) if synthesis_data else None

    metadata = data.get("metadata", {})

    # Reconstitute ExperimentMetadata if present.
    experiment_raw = metadata.get("experiment")
    if isinstance(experiment_raw, dict) and "experiment_id" in experiment_raw:
        metadata["experiment"] = ExperimentMetadata.from_dict(experiment_raw)

    transcript = DebateTranscript(
        transcript_id=data["transcript_id"],
        query=data["query"],
        panel=data["panel"],
        synthesizer_id=data["synthesizer_id"],
        max_rounds=data["max_rounds"],
        rounds=rounds,
        synthesis=synthesis,
        created_at=_parse_datetime(data.get("created_at", "")),
        metadata=metadata,
    )
    return transcript


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if it exceeds max_len.

    Args:
        text: Input string.
        max_len: Maximum length before truncation.

    Returns:
        Truncated string with '...' suffix, or original if short enough.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
