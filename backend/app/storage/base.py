"""Blob storage abstraction.

Raw transcripts and audio live here; structured analysis lives in the database.
The split matters: the previous design used S3 JSON files *as* the database,
which made listing calls an O(n) fan-out of GET requests and made filtering or
aggregating impossible without downloading everything.

The local backend is the default so a fresh clone runs with no cloud account
(issue #9). S3 is a drop-in swap via ``STORAGE_BACKEND=s3``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed-ish blob storage keyed by an opaque string."""

    async def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        """Store ``data`` and return a URI that ``get`` accepts."""
        ...

    async def get(self, key: str) -> bytes:
        """Fetch a blob. Raises ``NotFoundError`` when absent."""
        ...

    async def delete(self, key: str) -> None:
        """Remove a blob. Succeeds silently when already absent."""
        ...

    async def exists(self, key: str) -> bool: ...

    async def health(self) -> tuple[bool, str]:
        """Return ``(ok, detail)`` for the health endpoint."""
        ...


def build_store(settings) -> BlobStore:
    """Construct the configured backend."""
    from app.core.config import StorageBackend

    if settings.storage_backend is StorageBackend.S3:
        from app.storage.s3 import S3BlobStore

        return S3BlobStore(
            bucket=settings.s3_bucket,
            prefix=settings.s3_prefix,
            region=settings.aws_region,
        )

    from app.storage.local import LocalBlobStore

    return LocalBlobStore(settings.storage_local_path)
