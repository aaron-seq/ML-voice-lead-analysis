"""Tokenisation and phrase matching primitives.

Pure Python and ``re`` only. The engine's base path deliberately carries no
heavyweight NLP dependency: spaCy, torch and transformers together are ~2.5 GB
and a model download, which previously made the pipeline impossible to run on
a fresh clone or in a Lambda. Optional analyzers may upgrade individual steps
(see analysis.sentiment), but every result the product depends on is
reproducible from this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n+")
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'m": " am",
}


def normalise(text: str) -> str:
    """Fold to a comparable form: NFKC, lower-case, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens with apostrophes stripped.

    Apostrophes are removed rather than split so that "don't" -> "dont" matches
    the negator list, which stores the collapsed forms.
    """
    return [t.replace("'", "") for t in _WORD_RE.findall(text.lower())]


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter.

    Good enough for transcripts, which are already short-turn and lightly
    punctuated. Does not attempt abbreviation handling -- a mis-split costs a
    slightly odd summary bullet, not a wrong score.
    """
    parts = (p.strip() for p in _SENTENCE_RE.split(text))
    return [p for p in parts if p]


def clean_transcript_text(text: str) -> str:
    """Strip transcription artifacts that would otherwise skew analysis."""
    text = re.sub(
        r"\[(inaudible|unclear|crosstalk|silence|music|laughter|noise)[^\]]*\]",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\((inaudible|unclear|crosstalk)[^)]*\)", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\b(um+|uh+|erm+|mm+hm*)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_quote(text: str, limit: int = 220) -> str:
    """Trim a quote to a readable length on a word boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


# Tokens a phrase may step over when matching. Restricted to function words
# that carry no independent meaning, so "send it over" still matches "send
# over" but "send the contract over" does not collapse into it.
SKIPPABLE: frozenset[str] = frozenset({
    "it", "that", "this", "the", "a", "an", "you", "me", "us", "them",
    "just", "really", "then", "also", "actually",
})

# At most one skipped token per phrase. Two or more and the match stops being
# the same statement -- the gap is long enough to hold a negation or a
# different object.
MAX_SKIPS = 1


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    """Where a lexicon phrase fired within a token stream."""

    phrase: str
    weight: float
    start: int
    end: int
    key: object = None
    skips: int = 0


class PhraseMatcher:
    """Multi-phrase matcher over token sequences.

    Phrases are indexed by their first token, so matching is O(tokens x
    candidates-per-token) rather than O(tokens x phrases). With ~250 phrases
    across all lexicons this keeps a full pass on a long call under a
    millisecond, which is what allows every analyzer to run its own matcher
    without a shared mutable state.

    Matching tolerates a single skipped function word (see ``SKIPPABLE``),
    because people do not speak in lexicon entries: "send it over" and "send
    over" are the same commitment, and requiring contiguity silently loses the
    signal. Skips are reported on the match so callers can discount them.

    Overlapping matches are resolved longest-first, so "not interested" wins
    over "interested".
    """

    __slots__ = ("_index", "_max_len")

    def __init__(self, entries: list[tuple[str, float, object]]) -> None:
        self._index: dict[str, list[tuple[tuple[str, ...], float, str, object]]] = {}
        self._max_len = 1
        for phrase, weight, key in entries:
            tokens = tuple(tokenize(phrase))
            if not tokens:
                continue
            self._max_len = max(self._max_len, len(tokens))
            self._index.setdefault(tokens[0], []).append((tokens, weight, phrase, key))
        # Longest first so greedy matching prefers the most specific phrase.
        for bucket in self._index.values():
            bucket.sort(key=lambda item: len(item[0]), reverse=True)

    @staticmethod
    def _match_at(
        tokens: list[str], start: int, phrase: tuple[str, ...]
    ) -> tuple[int, int] | None:
        """Try to match ``phrase`` from ``start``. Returns (end, skips)."""
        i = start
        skips = 0
        n = len(tokens)
        for position, expected in enumerate(phrase):
            if i >= n:
                return None
            if tokens[i] == expected:
                i += 1
                continue
            # A skip is only allowed mid-phrase; leading and trailing gaps
            # would let a phrase match text it does not really appear in.
            if (
                position > 0
                and skips < MAX_SKIPS
                and tokens[i] in SKIPPABLE
                and i + 1 < n
                and tokens[i + 1] == expected
            ):
                skips += 1
                i += 2
                continue
            return None
        return i, skips

    def find(self, tokens: list[str]) -> list[PhraseMatch]:
        """Return non-overlapping matches, scanning left to right."""
        matches: list[PhraseMatch] = []
        i = 0
        n = len(tokens)
        while i < n:
            bucket = self._index.get(tokens[i])
            if not bucket:
                i += 1
                continue
            for phrase_tokens, weight, phrase, key in bucket:
                result = self._match_at(tokens, i, phrase_tokens)
                if result is not None:
                    end, skips = result
                    matches.append(PhraseMatch(phrase, weight, i, end, key, skips))
                    i = end
                    break
            else:
                i += 1
        return matches

    def __len__(self) -> int:  # pragma: no cover - trivial
        return sum(len(bucket) for bucket in self._index.values())


def is_question(text: str) -> bool:
    """Detect interrogatives, including the unpunctuated ones transcripts give."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    tokens = tokenize(stripped)
    if not tokens:
        return False
    openers = {
        "what", "when", "where", "who", "whom", "which", "why", "how",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "would", "will", "should", "have", "has", "any",
    }
    return tokens[0] in openers


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
