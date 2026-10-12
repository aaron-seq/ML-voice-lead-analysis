"""Conversation dynamics derived from transcript timings.

Transcripts are built as WebVTT strings rather than model objects so these
tests exercise the real path a customer's export takes: parse, infer roles,
measure. A test that hand-builds perfect segments would not have caught the
cue-splitting problem that `build_turns` exists to solve.
"""

from __future__ import annotations

import pytest

from app.analysis.diarization import infer_roles
from app.analysis.dynamics import (
    analyse_dynamics,
    build_turns,
    describe,
    has_usable_timings,
)
from app.analysis.parsing import parse_transcript
from app.domain.models import SpeakerRole


def vtt(*cues: tuple[str, str, str]) -> str:
    """Build a WebVTT document from (start, end, "Speaker: text") triples."""
    blocks = ["WEBVTT", ""]
    for index, (start, end, text) in enumerate(cues, start=1):
        blocks += [str(index), f"{start} --> {end}", text, ""]
    return "\n".join(blocks)


def analyse(*cues: tuple[str, str, str]):
    return analyse_dynamics(infer_roles(parse_transcript(vtt(*cues))))


# A short two-sided call with known timings, reused across tests.
# Rep speaks 0-10 and 20-26 (16s), prospect 12-18 and 30-45 (21s).
BALANCED = (
    ("00:00:00.000", "00:00:10.000",
     "Rep: Can I ask how you handle your data pipelines at the moment?"),
    ("00:00:12.000", "00:00:18.000",
     "Prospect: Honestly it is a mess, we have three engineers on it."),
    ("00:00:20.000", "00:00:26.000",
     "Rep: What does that cost you in a typical week?"),
    ("00:00:30.000", "00:00:45.000",
     "Prospect: Two days a week, and we have budget approved to fix it."),
)


class TestAvailability:
    def test_untimed_transcript_reports_unavailable_with_a_reason(self):
        transcript = infer_roles(
            parse_transcript("Rep: Hello there.\nProspect: We have budget.")
        )
        result = analyse_dynamics(transcript)
        assert result.available is False
        assert result.unavailable_reason
        assert "WebVTT" in result.unavailable_reason

    def test_untimed_transcript_invents_no_numbers(self):
        """An estimated duration corrupts every per-minute figure downstream."""
        transcript = infer_roles(
            parse_transcript("Rep: Hello there.\nProspect: We have budget.")
        )
        result = analyse_dynamics(transcript)
        for field in (
            "call_span_s", "rep_talk_seconds", "rep_talk_time_ratio",
            "longest_rep_monologue_s", "rep_speech_rate_wpm", "dead_air_ratio",
            "prospect_share_by_third", "rep_response_latency_s",
        ):
            assert getattr(result, field) is None, field

    def test_timed_transcript_is_available(self):
        assert analyse(*BALANCED).available is True

    def test_a_single_stray_timestamp_is_not_a_timed_transcript(self):
        """Two data points must not produce confident-looking metrics."""
        mixed = parse_transcript(
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\nRep: Hi there.\n"
        )
        # One timed cue only: below the majority threshold once padded out.
        mixed.segments.extend(
            s.model_copy(update={"index": i + 1, "start_s": None, "end_s": None})
            for i, s in enumerate([mixed.segments[0]] * 4)
        )
        assert has_usable_timings(mixed) is False

    def test_one_timed_turn_is_not_enough_to_measure_a_conversation(self):
        result = analyse(
            ("00:00:00.000", "00:00:05.000", "Rep: Hello, is this a good time?"),
        )
        assert result.available is False
        assert "two" in result.unavailable_reason


class TestTurnGrouping:
    def test_consecutive_cues_from_one_speaker_become_one_turn(self):
        """Subtitle formats split an utterance across cues. Counting cues as
        turns inflates turn_count and breaks monologue measurement."""
        turns = build_turns(
            infer_roles(parse_transcript(vtt(
                ("00:00:00.000", "00:00:04.000", "Rep: So the way this works is"),
                ("00:00:04.000", "00:00:08.000", "Rep: we watch your pipelines"),
                ("00:00:08.000", "00:00:12.000", "Rep: and alert before they fail."),
                ("00:00:13.000", "00:00:16.000", "Prospect: That sounds useful."),
            )))
        )
        assert len(turns) == 2
        assert turns[0].role is SpeakerRole.REP
        assert turns[0].start_s == 0.0
        assert turns[0].end_s == 12.0

    def test_grouped_turn_sums_the_words_of_its_cues(self):
        turns = build_turns(
            infer_roles(parse_transcript(vtt(
                ("00:00:00.000", "00:00:02.000", "Rep: one two three"),
                ("00:00:02.000", "00:00:04.000", "Rep: four five"),
                ("00:00:05.000", "00:00:07.000", "Prospect: yes indeed"),
            )))
        )
        assert turns[0].word_count == 5

    def test_untimed_segments_are_skipped_not_guessed(self):
        transcript = infer_roles(parse_transcript(vtt(
            ("00:00:00.000", "00:00:04.000", "Rep: Hello there how are you"),
            ("00:00:06.000", "00:00:10.000", "Prospect: Fine thanks, go ahead"),
        )))
        transcript.segments[1].start_s = None
        assert len(build_turns(transcript)) == 1


class TestTalkTime:
    def test_talk_time_is_measured_in_seconds_not_words(self):
        """The whole point of the module: a fast talker and a slow talker with
        the same airtime must report the same talk time."""
        result = analyse(*BALANCED)
        assert result.rep_talk_seconds == pytest.approx(16.0)
        assert result.prospect_talk_seconds == pytest.approx(21.0)

    def test_talk_ratio_is_a_share_of_attributed_speech(self):
        result = analyse(*BALANCED)
        assert result.rep_talk_time_ratio == pytest.approx(16 / 37, abs=1e-3)

    def test_word_share_and_time_share_genuinely_diverge(self):
        """If these always agreed, measuring time would be pointless. The rep
        here speaks few words slowly; the prospect many words quickly."""
        result = analyse(
            ("00:00:00.000", "00:00:20.000", "Rep: So. Tell me. Everything."),
            ("00:00:20.000", "00:00:25.000",
             "Prospect: We have eighty pipelines and two engineers babysitting "
             "them every single week which is why we have budget approved now."),
        )
        assert result.rep_talk_time_ratio is not None
        assert result.rep_talk_time_ratio > 0.75  # rep dominated the airtime

        from app.analysis.outcome import compute_talk_metrics
        words = compute_talk_metrics(
            infer_roles(parse_transcript(vtt(
                ("00:00:00.000", "00:00:20.000", "Rep: So. Tell me. Everything."),
                ("00:00:20.000", "00:00:25.000",
                 "Prospect: We have eighty pipelines and two engineers babysitting "
                 "them every single week which is why we have budget approved now."),
            )))
        )
        assert words.rep_talk_ratio < 0.25  # but spoke a quarter of the words

    def test_call_span_is_wall_clock_not_summed_speech(self):
        result = analyse(*BALANCED)
        assert result.call_span_s == pytest.approx(45.0)

    def test_longest_monologue_is_the_longest_grouped_turn(self):
        result = analyse(
            ("00:00:00.000", "00:00:05.000", "Rep: Quick question for you."),
            ("00:00:06.000", "00:00:09.000", "Prospect: Go on then."),
            ("00:00:10.000", "00:00:40.000", "Rep: So the long version is this."),
            ("00:00:40.000", "00:00:55.000", "Rep: And it continues for a while."),
            ("00:00:56.000", "00:00:59.000", "Prospect: I see, right."),
        )
        # 10-55 is one grouped rep turn, so 45s rather than the 30s first cue.
        assert result.longest_rep_monologue_s == pytest.approx(45.0)


class TestInterruptions:
    def test_overlapping_turns_are_counted_as_an_interruption(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000",
             "Prospect: Our biggest problem is really the manual work involved"),
            ("00:00:08.000", "00:00:14.000",
             "Rep: Right, and that is exactly what we fix for teams."),
        )
        assert result.interruptions_by_rep == 1
        assert result.interruptions_by_prospect == 0
        assert result.overtalk_seconds == pytest.approx(2.0)

    def test_interruptions_are_attributed_to_whoever_started_speaking(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000",
             "Rep: Let me walk you through how the pricing model works here"),
            ("00:00:07.000", "00:00:12.000",
             "Prospect: Sorry, but how much is it actually going to cost us?"),
        )
        assert result.interruptions_by_prospect == 1
        assert result.interruptions_by_rep == 0

    def test_touching_cues_are_not_an_interruption(self):
        """Diarisers routinely emit abutting cues; treating every boundary as
        an interruption would make the metric meaningless."""
        result = analyse(
            ("00:00:00.000", "00:00:10.000",
             "Rep: Does Thursday work for a follow up conversation at all?"),
            ("00:00:10.000", "00:00:15.000",
             "Prospect: Thursday is fine by me, send the invite over."),
        )
        assert result.interruptions_by_rep == 0
        assert result.interruptions_by_prospect == 0
        assert result.overtalk_seconds == 0.0

    def test_rounding_noise_below_the_threshold_is_ignored(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000",
             "Rep: Does Thursday work for a follow up conversation at all?"),
            ("00:00:09.900", "00:00:15.000",
             "Prospect: Thursday is fine by me, send the invite over."),
        )
        assert result.interruptions_by_prospect == 0

    def test_same_speaker_overlap_is_never_an_interruption(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000", "Rep: One part of the answer is this"),
            ("00:00:09.000", "00:00:14.000", "Rep: and the other part is that"),
            ("00:00:20.000", "00:00:24.000", "Prospect: Understood, thank you."),
        )
        assert result.interruptions_by_rep == 0
        assert result.interruptions_by_prospect == 0


class TestResponseLatency:
    def test_latency_is_reported_per_role(self):
        """The two mean different things: the rep's is how much room they leave
        for an answer, the prospect's is how long they think before giving one."""
        result = analyse(
            ("00:00:00.000", "00:00:05.000", "Rep: What is driving the change?"),
            ("00:00:09.000", "00:00:14.000",
             "Prospect: A compliance deadline in the first quarter."),
            ("00:00:15.000", "00:00:18.000", "Rep: Understood, that is helpful."),
        )
        assert result.prospect_response_latency_s == pytest.approx(4.0)
        assert result.rep_response_latency_s == pytest.approx(1.0)

    def test_long_gaps_are_excluded_as_holds(self):
        """A two-minute gap is a hold or a missing chunk, not a turn-taking
        pause, and averaging it in would swamp every real measurement."""
        result = analyse(
            ("00:00:00.000", "00:00:05.000", "Rep: Let me put you on hold."),
            ("00:03:00.000", "00:03:05.000", "Prospect: Still here, no problem."),
            ("00:03:07.000", "00:03:10.000", "Rep: Thanks for waiting there."),
        )
        assert result.prospect_response_latency_s is None
        assert result.rep_response_latency_s == pytest.approx(2.0)

    def test_overlaps_do_not_become_negative_latency(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000", "Rep: So the way the pricing works"),
            ("00:00:08.000", "00:00:12.000", "Prospect: How much does it cost?"),
        )
        assert result.prospect_response_latency_s is None


class TestSpeechRate:
    def test_speech_rate_uses_speaking_time_not_call_span(self):
        """Otherwise one speaker's rate falls simply because the other talked."""
        result = analyse(
            ("00:00:00.000", "00:00:30.000",
             " ".join(["Rep: word"] + ["word"] * 59)),  # 60 words in 30s = 120 wpm
            ("00:01:00.000", "00:02:00.000",
             "Prospect: " + " ".join(["word"] * 120)),
        )
        assert result.rep_speech_rate_wpm == pytest.approx(120.0, abs=1.0)

    def test_implausible_rates_are_dropped_rather_than_reported(self):
        """1200 wpm means the timings are wrong, not that someone spoke fast."""
        result = analyse(
            ("00:00:00.000", "00:00:01.000", "Rep: " + " ".join(["word"] * 200)),
            ("00:00:05.000", "00:00:30.000", "Prospect: " + " ".join(["word"] * 50)),
        )
        assert result.rep_speech_rate_wpm is None
        assert result.prospect_speech_rate_wpm is not None


class TestDeadAir:
    def test_dead_air_is_the_share_nobody_was_speaking(self):
        result = analyse(
            ("00:00:00.000", "00:00:10.000", "Rep: A question for you here."),
            ("00:00:30.000", "00:00:40.000", "Prospect: And here is my answer."),
        )
        # 40s span, 20s of speech, so half the call is silence.
        assert result.dead_air_ratio == pytest.approx(0.5)

    def test_overtalk_does_not_drive_dead_air_negative(self):
        """Overlapping turns double-count airtime; without subtracting the
        overlap, attributed speech can exceed the span."""
        result = analyse(
            ("00:00:00.000", "00:00:10.000", "Rep: Talking right over the top"),
            ("00:00:00.500", "00:00:10.000", "Prospect: And so am I, loudly"),
        )
        assert result.dead_air_ratio is not None
        assert result.dead_air_ratio >= 0.0


class TestEngagementTrajectory:
    def test_share_by_third_tracks_a_prospect_opening_up(self):
        result = analyse(
            ("00:00:00.000", "00:00:28.000", "Rep: Let me introduce why I called"),
            ("00:00:28.000", "00:00:30.000", "Prospect: Okay."),
            ("00:00:30.000", "00:00:45.000", "Rep: And here is the detail of it"),
            ("00:00:45.000", "00:01:00.000", "Prospect: That is interesting to me"),
            ("00:01:00.000", "00:01:05.000", "Rep: What would you need to see?"),
            ("00:01:05.000", "00:01:30.000",
             "Prospect: A proof of concept, and we have budget approved for it"),
        )
        thirds = result.prospect_share_by_third
        assert thirds is not None and len(thirds) == 3
        assert thirds[0] < thirds[1] < thirds[2]

    def test_turns_straddling_a_boundary_are_split_proportionally(self):
        """A 60s turn spanning two thirds belongs to both, not to whichever
        third happens to contain its start."""
        result = analyse(
            ("00:00:00.000", "00:01:00.000", "Rep: One very long opening stretch"),
            ("00:01:00.000", "00:01:30.000", "Prospect: And a shorter reply here"),
        )
        thirds = result.prospect_share_by_third
        assert thirds is not None
        assert thirds[0] == pytest.approx(0.0)
        assert thirds[2] == pytest.approx(1.0)


class TestObservations:
    def test_untimed_transcripts_produce_no_observations(self):
        transcript = infer_roles(parse_transcript("Rep: Hi.\nProspect: Hello."))
        assert describe(analyse_dynamics(transcript)) == []

    def test_a_dominant_rep_is_called_out(self):
        result = analyse(
            ("00:00:00.000", "00:01:30.000", "Rep: Let me tell you all about it"),
            ("00:01:30.000", "00:01:40.000", "Prospect: Right, I see, okay then."),
        )
        notes = " ".join(describe(result))
        assert "talk time" in notes

    def test_every_observation_names_the_number_behind_it(self):
        """A claim without its number cannot be argued with."""
        result = analyse(
            ("00:00:00.000", "00:03:00.000", "Rep: An extremely long monologue"),
            ("00:03:00.000", "00:03:10.000", "Prospect: Right, I see, okay then."),
        )
        notes = describe(result)
        assert notes
        assert all(any(ch.isdigit() for ch in note) for note in notes)

    def test_a_long_monologue_is_reported_in_minutes(self):
        result = analyse(
            ("00:00:00.000", "00:03:00.000", "Rep: An extremely long monologue"),
            ("00:03:00.000", "00:03:10.000", "Prospect: Right, I see, okay then."),
        )
        assert any("minutes" in note for note in describe(result))


class TestOutOfOrderCues:
    """Nothing guarantees a diariser emits cues in time order, and the overlap
    scan stops early at the first turn starting after the current one ends --
    which silently misses overlaps if the input is unsorted."""

    def test_overlap_detection_survives_unordered_cues(self):
        ordered = analyse(
            ("00:00:00.000", "00:00:10.000",
             "Prospect: Our biggest problem is really the manual work involved"),
            ("00:00:08.000", "00:00:14.000",
             "Rep: Right, and that is exactly what we fix for teams like yours."),
            ("00:00:20.000", "00:00:26.000",
             "Prospect: That is interesting, tell me more about the pricing."),
        )
        shuffled = analyse(
            ("00:00:20.000", "00:00:26.000",
             "Prospect: That is interesting, tell me more about the pricing."),
            ("00:00:00.000", "00:00:10.000",
             "Prospect: Our biggest problem is really the manual work involved"),
            ("00:00:08.000", "00:00:14.000",
             "Rep: Right, and that is exactly what we fix for teams like yours."),
        )
        assert shuffled.interruptions_by_rep == ordered.interruptions_by_rep == 1
        assert shuffled.overtalk_seconds == ordered.overtalk_seconds
