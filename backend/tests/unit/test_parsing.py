"""Transcript parsing across every supported source format."""

from __future__ import annotations

import json

import pytest

from app.analysis.parsing import parse_transcript
from app.core.errors import EmptyTranscriptError, UnsupportedFormatError


class TestPlainText:
    def test_diarised_text_splits_into_turns(self):
        transcript = parse_transcript(
            "Rep: Hello there.\nProspect: Hi, who is this?\nRep: Priya from Northwind."
        )
        assert transcript.source_format == "diarized_text"
        assert len(transcript.segments) == 3
        assert transcript.segments[0].speaker == "Rep"
        assert transcript.segments[1].text == "Hi, who is this?"

    def test_prose_is_not_mistaken_for_a_transcript(self):
        """A single stray colon must not turn prose into a diarised call."""
        transcript = parse_transcript(
            "The meeting covered three things: pricing, timeline and scope. "
            "It went well overall. We agreed to reconvene."
        )
        assert transcript.source_format == "plain_text"

    def test_undiarised_text_is_split_into_sentences(self):
        transcript = parse_transcript(
            "This is the first sentence. Here is a second one! And a third?"
        )
        assert transcript.source_format == "plain_text"
        assert len(transcript.segments) == 3

    def test_empty_input_is_rejected(self):
        with pytest.raises(EmptyTranscriptError):
            parse_transcript("   \n  \t ")

    def test_transcript_of_only_artifacts_is_rejected(self):
        """Cleaning must not silently produce a zero-content transcript."""
        with pytest.raises(EmptyTranscriptError):
            parse_transcript("[inaudible] [crosstalk] um uh [music]")


class TestAwsTranscribe:
    def test_speaker_labelled_output(self):
        payload = {
            "results": {
                "transcripts": [{"transcript": "hello there hi who is this"}],
                "speaker_labels": {
                    "segments": [
                        {
                            "speaker_label": "spk_0",
                            "start_time": "0.0",
                            "end_time": "1.5",
                            "items": [
                                {"start_time": "0.0"},
                                {"start_time": "0.5"},
                            ],
                        },
                        {
                            "speaker_label": "spk_1",
                            "start_time": "1.6",
                            "end_time": "3.0",
                            "items": [{"start_time": "1.6"}],
                        },
                    ]
                },
                "items": [
                    {"type": "pronunciation", "start_time": "0.0",
                     "alternatives": [{"content": "hello"}]},
                    {"type": "pronunciation", "start_time": "0.5",
                     "alternatives": [{"content": "there"}]},
                    {"type": "pronunciation", "start_time": "1.6",
                     "alternatives": [{"content": "hi"}]},
                ],
            }
        }
        transcript = parse_transcript(json.dumps(payload))
        assert transcript.source_format == "aws_transcribe"
        assert len(transcript.segments) == 2
        assert transcript.segments[0].text == "hello there"
        assert transcript.segments[0].start_s == 0.0
        assert transcript.duration_s == pytest.approx(3.0)

    def test_falls_back_to_flat_transcript_without_labels(self):
        payload = {
            "results": {"transcripts": [{"transcript": "One sentence. Another one."}]}
        }
        transcript = parse_transcript(json.dumps(payload))
        assert transcript.source_format == "aws_transcribe"
        assert len(transcript.segments) == 2


class TestWhisperAndGenericJson:
    def test_whisper_segments(self):
        payload = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello there."},
                {"start": 2.0, "end": 4.0, "text": "Hi, who is this?"},
            ],
        }
        transcript = parse_transcript(json.dumps(payload))
        assert transcript.source_format == "whisper"
        assert transcript.segments[1].start_s == 2.0

    def test_bare_array_is_treated_as_segments(self):
        payload = [{"speaker": "rep", "text": "Hello"}, {"speaker": "p", "text": "Hi"}]
        transcript = parse_transcript(json.dumps(payload))
        assert len(transcript.segments) == 2

    def test_unrecognised_json_shape_is_rejected(self):
        with pytest.raises(UnsupportedFormatError):
            parse_transcript(json.dumps({"foo": 1, "bar": [2]}))

    def test_malformed_json_reports_a_useful_error(self):
        with pytest.raises(UnsupportedFormatError) as exc:
            parse_transcript('{"segments": [')
        assert "failed to parse" in exc.value.message.lower()


class TestSubtitles:
    VTT = """WEBVTT

00:00:00.000 --> 00:00:02.500
<v Rep>Hello there, thanks for joining.

00:00:02.500 --> 00:00:05.000
<v Prospect>Happy to be here.
"""

    SRT = """1
00:00:00,000 --> 00:00:02,500
Rep: Hello there.

2
00:00:02,500 --> 00:00:05,000
Prospect: Hi.
"""

    def test_webvtt_with_voice_tags(self):
        transcript = parse_transcript(self.VTT)
        assert transcript.source_format == "vtt"
        assert transcript.segments[0].speaker == "Rep"
        assert transcript.segments[0].start_s == 0.0
        assert transcript.segments[1].end_s == 5.0

    def test_srt_with_inline_speakers(self):
        transcript = parse_transcript(self.SRT)
        assert transcript.source_format == "srt"
        assert transcript.segments[0].speaker == "Rep"
        # Cue numbers must not leak into the text.
        assert "1" not in transcript.segments[0].text


class TestNormalisation:
    def test_transcription_artifacts_are_stripped(self):
        transcript = parse_transcript(
            "Rep: So um [inaudible] the pricing is uh competitive.\n"
            "Prospect: Right, [crosstalk] understood."
        )
        text = transcript.text.lower()
        assert "inaudible" not in text
        assert "[" not in text
        assert "pricing" in text

    def test_bytes_input_is_accepted(self):
        transcript = parse_transcript(b"Rep: Hello.\nProspect: Hi there.")
        assert len(transcript.segments) == 2

    def test_duration_is_none_when_untimed(self):
        """Untimed transcripts must not fabricate a duration."""
        transcript = parse_transcript("Rep: Hello.\nProspect: Hi there.")
        assert transcript.duration_s is None
        assert transcript.estimated_duration_s() > 0
