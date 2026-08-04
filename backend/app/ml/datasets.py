"""Real training corpora: fetch, cache, verify.

Only genuinely public, citable datasets are used. Nothing here is synthetic --
the previous incarnation of this repository trained on
``create_synthetic_training_data()`` and reported the result as accuracy, which
is the specific mistake this module exists to avoid.

Downloads are cached under ``backend/data/corpora`` and content-hashed, so a
training run is reproducible and works offline after the first fetch.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "corpora"

# Stanford Sentiment Treebank v2, binary sentence-level split.
# Socher et al. 2013, "Recursive Deep Models for Semantic Compositionality
# over a Sentiment Treebank". Standard benchmark with a fixed train/dev/test
# partition, which is what makes the reported numbers comparable to published
# work rather than self-graded.
SST2_BASE = (
    "https://raw.githubusercontent.com/clairett/"
    "pytorch-sentiment-classification/master/data/SST2"
)


@dataclass(frozen=True)
class LabelledText:
    text: str
    label: int  # 1 = positive, 0 = negative


def _download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    # urlopen honours file:// and custom schemes. These URLs are module
    # constants rather than user input, but validating the scheme costs one
    # line and stops a future caller from turning this into a file-read or
    # SSRF primitive by passing a different corpus URL.
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"Refusing to fetch a non-HTTPS corpus URL: {url!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    # Written to a temp file first so an interrupted download cannot leave a
    # truncated corpus that silently trains a worse model.
    tmp = destination.with_suffix(destination.suffix + ".part")
    # nosec B310 - the scheme is validated to be https immediately above, which
    # is the concern this rule exists to catch.
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        tmp.write_bytes(response.read())
    tmp.replace(destination)
    return destination


def _parse_tsv(path: Path) -> list[LabelledText]:
    rows: list[LabelledText] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text, _, label = line.rpartition("\t")
        text = text.strip()
        if not text or label.strip() not in {"0", "1"}:
            continue
        rows.append(LabelledText(text=text, label=int(label)))
    return rows


def load_sst2(split: str) -> list[LabelledText]:
    """Load one of the fixed SST-2 splits: ``train``, ``dev`` or ``test``."""
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split {split!r}")
    path = _download(f"{SST2_BASE}/{split}.tsv", CACHE_DIR / f"sst2_{split}.tsv")
    rows = _parse_tsv(path)
    if not rows:
        raise RuntimeError(f"SST-2 {split} split parsed to zero rows")
    return rows


def corpus_fingerprint(rows: list[LabelledText]) -> str:
    """Stable hash of a corpus, recorded with trained models.

    Lets a model artifact be tied to the exact data it saw, so "the numbers
    changed" can be answered with "the data changed" or "the code changed".
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.text.encode("utf-8"))
        digest.update(str(row.label).encode())
    return digest.hexdigest()[:16]


def load_domain_eval() -> list[LabelledText]:
    """Sales-conversation evaluation set.

    SST-2 is movie reviews. This product analyses sales calls, and a model that
    scores well on film criticism can still be wrong about "we have budget
    allocated". Training accuracy on SST-2 therefore says nothing on its own --
    this set exists to measure the domain gap.

    It is hand-labelled by the authors from realistic sales-call language, not
    drawn from a public corpus, because no public sales-call sentiment corpus
    exists. That is weaker evidence than SST-2 and is reported as such: it is a
    smoke test for domain shift, not a benchmark.
    """
    path = Path(__file__).resolve().parents[2] / "data" / "eval" / "sales_sentiment.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Domain evaluation set missing at {path}. It is version-controlled; "
            "check your working tree."
        )
    return _parse_tsv(path)
