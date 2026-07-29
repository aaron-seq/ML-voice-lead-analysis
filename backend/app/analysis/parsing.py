"""Transcript ingestion.

Accepts the formats teams actually have and normalises them into a
``Transcript``. Format is sniffed from content, not the file extension, because
in practice ``.json`` covers at least four incompatible shapes.

Supported:
  * AWS Transcribe JSON (with or without speaker labels)
  * OpenAI Whisper JSON (``segments`` array)
  * CallSense native JSON (a serialised ``Transcript``)
  * Generic ``{"segments": [...]}`` / ``{"turns": [...]}`` shapes
  * WebVTT and SRT subtitles
  * Diarised plain text (``Speaker 1: hello``)
  * Unlabelled plain text
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.analysis.text import clean_transcript_text, split_sentences
from app.core.errors import EmptyTranscriptError, UnsupportedFormatError
from app.domain.models import SpeakerRole, Transcript, TranscriptSegment

_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<ts1>[\d:.,]+)\]\s*)?"
    r"(?P<speaker>[A-Za-z][\w .'-]{0,40}?)\s*[:>]\s*"
    r"(?P<text>.+)$"
)
_VTT_CUE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)
_VTT_VOICE_RE = re.compile(r"<v\s+([^>]+)>", re.IGNORECASE)


def _timestamp_to_seconds(value: str) -> float:
    """Parse ``HH:MM:SS.mmm`` / ``MM:SS.mmm`` into seconds."""
    value = value.replace(",", ".")
    parts = value.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def _finalise(
    raw_segments: list[dict[str, Any]],
    source_format: str,
    language: str = "en",
) -> Transcript:
    """Clean, drop empties and index a list of raw segment dicts."""
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        text = clean_transcript_text(str(item.get("text", "")))
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                index=len(segments),
                speaker=str(item.get("speaker") or "unknown"),
                role=item.get("role", SpeakerRole.UNKNOWN),
                text=text,
                start_s=item.get("start_s"),
                end_s=item.get("end_s"),
            )
        )
    if not segments:
        raise EmptyTranscriptError(
            "Transcript contains no usable text after cleaning",
            source_format=source_format,
        )
    return Transcript(source_format=source_format, segments=segments, language=language)


# --------------------------------------------------------------------------
# JSON dialects
# --------------------------------------------------------------------------


def _parse_aws_transcribe(payload: dict[str, Any]) -> Transcript:
    """AWS Transcribe output.

    When speaker labels are present, items are grouped into turns by the
    ``speaker_labels`` segment list. Otherwise the flat transcript string is
    split into sentences so downstream timeline analysis still has structure.
    """
    results = payload["results"]
    language = payload.get("LanguageCode", "en").split("-")[0]

    items = results.get("items", [])
    labels = results.get("speaker_labels", {}).get("segments", [])

    if labels and items:
        # Index pronunciation items by start time for O(1) lookup per label item.
        by_start: dict[str, dict[str, Any]] = {
            item["start_time"]: item
            for item in items
            if item.get("type") == "pronunciation" and "start_time" in item
        }
        raw: list[dict[str, Any]] = []
        for seg in labels:
            words = [
                by_start[entry["start_time"]]["alternatives"][0]["content"]
                for entry in seg.get("items", [])
                if entry.get("start_time") in by_start
            ]
            if not words:
                continue
            raw.append({
                "speaker": seg.get("speaker_label", "unknown"),
                "text": " ".join(words),
                "start_s": float(seg["start_time"]),
                "end_s": float(seg["end_time"]),
            })
        if raw:
            return _finalise(raw, "aws_transcribe", language)

    transcripts = results.get("transcripts", [])
    if not transcripts:
        raise UnsupportedFormatError("AWS Transcribe payload has no transcripts")
    text = transcripts[0].get("transcript", "")
    raw = [{"speaker": "unknown", "text": sentence} for sentence in split_sentences(text)]
    return _finalise(raw or [{"speaker": "unknown", "text": text}],
                     "aws_transcribe", language)


def _parse_segment_list(
    payload: dict[str, Any], key: str, source_format: str
) -> Transcript:
    """Whisper-style and generic ``{key: [{start, end, text, speaker}]}``."""
    raw = []
    for item in payload[key]:
        if not isinstance(item, dict):
            continue
        raw.append({
            "speaker": item.get("speaker") or item.get("speaker_label") or "unknown",
            "role": item.get("role", SpeakerRole.UNKNOWN),
            "text": item.get("text") or item.get("content") or "",
            "start_s": _coerce_float(item.get("start", item.get("start_s"))),
            "end_s": _coerce_float(item.get("end", item.get("end_s"))),
        })
    return _finalise(raw, source_format, payload.get("language", "en"))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json(payload: dict[str, Any]) -> Transcript:
    """Dispatch on the shape of a decoded JSON transcript."""
    if "results" in payload and isinstance(payload["results"], dict):
        return _parse_aws_transcribe(payload)

    # Native CallSense transcript round-trips exactly.
    if payload.get("source_format") and isinstance(payload.get("segments"), list):
        return _parse_segment_list(payload, "segments", str(payload["source_format"]))

    if isinstance(payload.get("segments"), list):
        is_whisper = "language" in payload or "duration" in payload
        fmt = "whisper" if is_whisper else "segments_json"
        return _parse_segment_list(payload, "segments", fmt)

    for key in ("turns", "utterances", "messages"):
        if isinstance(payload.get(key), list):
            return _parse_segment_list(payload, key, f"{key}_json")

    if isinstance(payload.get("transcript"), str):
        return parse_text(payload["transcript"])

    if isinstance(payload.get("text"), str):
        return parse_text(payload["text"])

    raise UnsupportedFormatError(
        "Unrecognised JSON transcript shape",
        top_level_keys=sorted(payload.keys())[:12],
    )


# --------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------


def parse_subtitles(content: str) -> Transcript:
    """WebVTT / SRT. Cue numbers and styling blocks are discarded."""
    raw: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        match = _VTT_CUE_RE.search(block)
        if not match:
            continue
        lines = [ln for ln in block.splitlines() if not _VTT_CUE_RE.search(ln)]
        lines = [ln for ln in lines if not ln.strip().isdigit()]
        text = " ".join(lines).strip()
        if not text:
            continue

        speaker = "unknown"
        voice = _VTT_VOICE_RE.search(text)
        if voice:
            speaker = voice.group(1).strip()
            text = _VTT_VOICE_RE.sub("", text)
        text = re.sub(r"</?[a-z][^>]*>", "", text)

        # A leading "Name:" inside the cue is a common diarisation convention.
        inline = _SPEAKER_LINE_RE.match(text)
        if speaker == "unknown" and inline and len(inline.group("speaker")) <= 25:
            speaker = inline.group("speaker").strip()
            text = inline.group("text")

        raw.append({
            "speaker": speaker,
            "text": text,
            "start_s": _timestamp_to_seconds(match.group("start")),
            "end_s": _timestamp_to_seconds(match.group("end")),
        })

    if not raw:
        raise UnsupportedFormatError("No subtitle cues found")
    return _finalise(raw, "vtt" if "WEBVTT" in content[:64].upper() else "srt")


# --------------------------------------------------------------------------
# Plain text
# --------------------------------------------------------------------------


def parse_text(content: str) -> Transcript:
    """Diarised or undiarised plain text.

    A document counts as diarised when at least a third of its non-empty lines
    carry a ``Speaker:`` prefix and at least two distinct speakers appear. The
    threshold avoids misreading prose containing a stray colon as a transcript.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        raise EmptyTranscriptError("Transcript is empty")

    labelled: list[tuple[str, str, float | None]] = []
    for line in lines:
        match = _SPEAKER_LINE_RE.match(line)
        if match and len(match.group("speaker")) <= 40:
            ts = match.group("ts1")
            labelled.append((
                match.group("speaker").strip(),
                match.group("text").strip(),
                _timestamp_to_seconds(ts) if ts else None,
            ))

    speakers = {name.lower() for name, _, _ in labelled}
    if len(labelled) >= max(2, len(lines) // 3) and len(speakers) >= 2:
        raw = [
            {"speaker": name, "text": text, "start_s": ts}
            for name, text, ts in labelled
        ]
        return _finalise(raw, "diarized_text")

    # Undiarised: split into sentences so the timeline still has resolution.
    sentences = split_sentences(content)
    raw = [{"speaker": "unknown", "text": s} for s in sentences]
    return _finalise(raw or [{"speaker": "unknown", "text": content}], "plain_text")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_transcript(content: str | bytes, filename: str | None = None) -> Transcript:
    """Parse any supported transcript. Format is detected from content.

    Args:
        content: Raw transcript bytes or text.
        filename: Optional, used only to improve error messages.

    Raises:
        EmptyTranscriptError: Nothing usable after cleaning.
        UnsupportedFormatError: Content matched no known dialect.
    """
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Never fail an upload on a stray byte; replacement characters are
            # recoverable, a rejected transcript is not.
            text = content.decode("utf-8", errors="replace")
    else:
        text = content

    if not text.strip():
        raise EmptyTranscriptError("Transcript is empty", filename=filename)

    stripped = text.lstrip()

    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # A leading '{' means JSON was clearly intended, so report the
            # parse error. A leading '[' does not: transcripts routinely open
            # with an annotation like "[inaudible]" or "[00:01] Rep: ...", and
            # rejecting those as malformed JSON would be wrong.
            if stripped.startswith("{"):
                raise UnsupportedFormatError(
                    f"Content looks like JSON but failed to parse: {exc.msg}",
                    filename=filename, line=exc.lineno,
                ) from exc
            payload = None

        if payload is not None:
            if isinstance(payload, list):
                payload = {"segments": payload}
            if not isinstance(payload, dict):
                raise UnsupportedFormatError(
                    "JSON transcript must be an object or array", filename=filename
                )
            return parse_json(payload)

    if stripped.upper().startswith("WEBVTT") or _VTT_CUE_RE.search(text[:4000]):
        return parse_subtitles(text)

    return parse_text(text)
