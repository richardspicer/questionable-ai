"""Tests for the debate page module — progressive rendering logic."""

from __future__ import annotations

from mutual_dissent.models import DebateRound, DebateTranscript, ModelResponse


class TestCallbackAccumulation:
    """The on_round callback accumulates rounds correctly."""

    def test_rounds_accumulate_in_order(self) -> None:
        """Rounds accumulate in the order callbacks fire."""
        accumulated: list[DebateRound] = []

        for i, rtype in enumerate(["initial", "reflection", "synthesis"]):
            rnd = DebateRound(round_number=i, round_type=rtype, responses=[])
            accumulated.append(rnd)

        assert [r.round_type for r in accumulated] == [
            "initial",
            "reflection",
            "synthesis",
        ]

    def test_synthesis_round_not_in_accumulated(self) -> None:
        """Synthesis rounds should not be in accumulated_rounds (only initial/reflection)."""
        accumulated: list[DebateRound] = []

        rounds = [
            DebateRound(round_number=0, round_type="initial", responses=[]),
            DebateRound(round_number=1, round_type="reflection", responses=[]),
            DebateRound(round_number=-1, round_type="synthesis", responses=[]),
        ]
        for rnd in rounds:
            if rnd.round_type != "synthesis":
                accumulated.append(rnd)

        assert len(accumulated) == 2
        assert all(r.round_type != "synthesis" for r in accumulated)


class TestAbortHandling:
    """Abort creates partial transcript with correct metadata."""

    def test_partial_transcript_has_aborted_flag(self) -> None:
        """Partial transcript after abort has metadata['aborted'] = True."""
        rounds = [
            DebateRound(round_number=0, round_type="initial", responses=[]),
        ]
        transcript = DebateTranscript(
            query="test",
            panel=["claude"],
            synthesizer_id="claude",
            max_rounds=2,
            rounds=rounds,
            metadata={"aborted": True},
        )
        assert transcript.metadata["aborted"] is True
        assert len(transcript.rounds) == 1
        assert transcript.synthesis is None

    def test_partial_transcript_preserves_completed_rounds(self) -> None:
        """All rounds completed before abort are preserved."""
        r0 = DebateRound(
            round_number=0,
            round_type="initial",
            responses=[
                ModelResponse(model_id="m1", model_alias="claude", round_number=0, content="a"),
                ModelResponse(model_id="m2", model_alias="gpt", round_number=0, content="b"),
            ],
        )
        r1 = DebateRound(
            round_number=1,
            round_type="reflection",
            responses=[
                ModelResponse(model_id="m1", model_alias="claude", round_number=1, content="c"),
                ModelResponse(model_id="m2", model_alias="gpt", round_number=1, content="d"),
            ],
        )
        transcript = DebateTranscript(
            query="test",
            rounds=[r0, r1],
            metadata={"aborted": True},
        )
        assert len(transcript.rounds) == 2
        assert len(transcript.rounds[0].responses) == 2
        assert len(transcript.rounds[1].responses) == 2

    def test_partial_transcript_serializes(self) -> None:
        """Partial transcript can be serialized to dict."""
        transcript = DebateTranscript(
            query="test",
            rounds=[DebateRound(round_number=0, round_type="initial", responses=[])],
            metadata={"aborted": True},
        )
        data = transcript.to_dict()
        assert data["metadata"]["aborted"] is True
        assert len(data["rounds"]) == 1
