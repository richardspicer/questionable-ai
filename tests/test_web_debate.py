"""Tests for the debate page module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mutual_dissent.models import DebateTranscript


class TestRunDebateHelper:
    """_run_debate helper calls orchestrator correctly."""

    @pytest.mark.asyncio
    async def test_calls_run_debate_with_params(self) -> None:
        """Calls orchestrator.run_debate with form parameters."""
        from mutual_dissent.web.pages.debate import _run_debate

        mock_transcript = DebateTranscript(query="test query")
        with (
            patch("mutual_dissent.web.pages.debate.load_config") as mock_config,
            patch(
                "mutual_dissent.web.pages.debate.run_debate",
                new_callable=AsyncMock,
                return_value=mock_transcript,
            ) as mock_run,
            patch("mutual_dissent.web.pages.debate.save_transcript") as mock_save,
        ):
            mock_config.return_value = MagicMock()
            result = await _run_debate(
                query="test query",
                panel=["claude", "gpt"],
                synthesizer="claude",
                rounds=1,
                ground_truth=None,
            )

        mock_run.assert_called_once()
        mock_save.assert_called_once_with(mock_transcript)
        assert result is mock_transcript

    @pytest.mark.asyncio
    async def test_passes_ground_truth_when_provided(self) -> None:
        """Passes ground_truth to run_debate when non-empty."""
        from mutual_dissent.web.pages.debate import _run_debate

        mock_transcript = DebateTranscript(query="test")
        with (
            patch("mutual_dissent.web.pages.debate.load_config") as mock_config,
            patch(
                "mutual_dissent.web.pages.debate.run_debate",
                new_callable=AsyncMock,
                return_value=mock_transcript,
            ) as mock_run,
            patch("mutual_dissent.web.pages.debate.save_transcript"),
        ):
            mock_config.return_value = MagicMock()
            await _run_debate(
                query="test",
                panel=["claude"],
                synthesizer="claude",
                rounds=1,
                ground_truth="the answer is 42",
            )

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["ground_truth"] == "the answer is 42"

    @pytest.mark.asyncio
    async def test_skips_empty_ground_truth(self) -> None:
        """Passes None for ground_truth when empty string."""
        from mutual_dissent.web.pages.debate import _run_debate

        mock_transcript = DebateTranscript(query="test")
        with (
            patch("mutual_dissent.web.pages.debate.load_config") as mock_config,
            patch(
                "mutual_dissent.web.pages.debate.run_debate",
                new_callable=AsyncMock,
                return_value=mock_transcript,
            ) as mock_run,
            patch("mutual_dissent.web.pages.debate.save_transcript"),
        ):
            mock_config.return_value = MagicMock()
            await _run_debate(
                query="test",
                panel=["claude"],
                synthesizer="claude",
                rounds=1,
                ground_truth="",
            )

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["ground_truth"] is None
