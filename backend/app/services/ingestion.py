"""Call ingestion: store the transcript, analyse it, persist the result.

This is the single write path into the product. Keeping it in one place means
the HTTP upload endpoint, the JSON endpoint and the CLI importer cannot drift
apart in how they name, store or analyse a call.
"""

from __future__ import annotations

import logging
import re
import uuid

from app.analysis.engine import AnalysisEngine
from app.core.errors import CallSenseError
from app.db.repository import AuditRepository, CallRepository
from app.domain.models import AnalysisStatus, Call
from app.storage.base import BlobStore

logger = logging.getLogger(__name__)

_TITLE_CLEAN = re.compile(r"[^\w\s.'-]+")


def new_call_id() -> str:
    """Opaque, unguessable id.

    Not derived from the filename: filenames leak customer names into URLs and
    collide across uploads.
    """
    return uuid.uuid4().hex[:16]


def derive_title(filename: str | None, fallback: str = "Untitled call") -> str:
    if not filename:
        return fallback
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = _TITLE_CLEAN.sub(" ", stem).strip()
    cleaned = re.sub(r"[_-]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned[:200] or fallback


class IngestionService:
    """Coordinates blob storage, the analysis engine and the database."""

    def __init__(
        self,
        repository: CallRepository,
        blob_store: BlobStore,
        engine: AnalysisEngine,
        audit: AuditRepository | None = None,
    ) -> None:
        self._repo = repository
        self._blobs = blob_store
        self._engine = engine
        self._audit = audit

    async def ingest(
        self,
        content: bytes | str,
        title: str,
        principal_id: str,
        filename: str | None = None,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> Call:
        """Store and analyse a transcript.

        The raw transcript is persisted *before* analysis so that a parsing or
        engine failure still leaves the source recoverable and re-analysable
        instead of discarding the customer's upload.
        """
        call_id = new_call_id()
        raw = content.encode("utf-8") if isinstance(content, str) else content

        blob_key = f"transcripts/{call_id}"
        source_uri = await self._blobs.put(blob_key, raw, "text/plain; charset=utf-8")

        await self._repo.create(
            call_id=call_id,
            title=title,
            source_format="unknown",
            source_uri=source_uri,
            owner=owner,
            tags=tags,
        )

        try:
            transcript, analysis = self._engine.analyse_text(
                raw, call_id=call_id, filename=filename
            )
        except CallSenseError as exc:
            # A bad transcript is the user's problem to fix, but the record and
            # the stored blob remain so they can see what was rejected and why.
            # Committed explicitly: re-raising rolls the session back otherwise.
            await self._repo.mark_status(
                call_id, AnalysisStatus.FAILED, exc.message, commit=True
            )
            logger.warning(
                "Ingestion failed", extra={"call_id": call_id, "reason": exc.code}
            )
            raise
        except Exception as exc:
            await self._repo.mark_status(
                call_id,
                AnalysisStatus.FAILED,
                f"Unexpected analysis error: {exc}",
                commit=True,
            )
            logger.exception("Unexpected analysis failure", extra={"call_id": call_id})
            raise

        call = await self._repo.attach_analysis(
            call_id, analysis, transcript.word_count, transcript.source_format
        )
        # Persist the normalised transcript so the UI can show role-attributed
        # turns without re-parsing, and so evidence indices stay valid.
        await self._blobs.put(
            f"parsed/{call_id}",
            transcript.model_dump_json().encode("utf-8"),
            "application/json",
        )

        if self._audit:
            await self._audit.record(
                principal_id,
                "call.ingest",
                call_id,
                {"title": title, "format": transcript.source_format,
                 "words": transcript.word_count},
            )

        logger.info(
            "Call ingested",
            extra={
                "call_id": call_id,
                "format": transcript.source_format,
                "lead_score": analysis.lead_score.score,
            },
        )
        return call

    async def reanalyse(self, call_id: str, principal_id: str) -> Call:
        """Re-run the current engine over a stored transcript.

        The reason ``ENGINE_VERSION`` is persisted: after a scoring change,
        this brings an old call up to date without needing the original upload.
        """
        call = await self._repo.get(call_id)
        raw = await self._blobs.get(f"transcripts/{call_id}")

        await self._repo.mark_status(call_id, AnalysisStatus.RUNNING)
        try:
            transcript, analysis = self._engine.analyse_text(raw, call_id=call_id)
        except Exception as exc:
            await self._repo.mark_status(
                call_id, AnalysisStatus.FAILED, str(exc), commit=True
            )
            raise

        updated = await self._repo.attach_analysis(
            call_id, analysis, transcript.word_count, transcript.source_format
        )
        await self._blobs.put(
            f"parsed/{call_id}",
            transcript.model_dump_json().encode("utf-8"),
            "application/json",
        )

        if self._audit:
            await self._audit.record(
                principal_id,
                "call.reanalyse",
                call_id,
                {"engine_version": analysis.metadata.engine_version,
                 "previous_score": call.lead_score,
                 "new_score": analysis.lead_score.score},
            )
        return updated

    async def delete(self, call_id: str, principal_id: str) -> None:
        """Remove a call and its stored blobs.

        Audited, because deleting recorded customer conversations is exactly
        what a compliance review will ask about.
        """
        await self._repo.get(call_id)  # 404 before any destructive work

        if self._audit:
            await self._audit.record(principal_id, "call.delete", call_id, {})

        await self._repo.delete(call_id)
        for key in (f"transcripts/{call_id}", f"parsed/{call_id}"):
            try:
                await self._blobs.delete(key)
            except Exception:
                # The row is gone; an orphaned blob is a cleanup task, not a
                # reason to fail the user's delete.
                logger.warning("Failed to delete blob", extra={"key": key})
