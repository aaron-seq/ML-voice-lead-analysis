"""Filesystem blob storage -- the default backend.

Writes go through a temporary file and an atomic rename so a crashed or
concurrent write can never leave a half-written transcript that later parses
into a silently truncated analysis.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from app.core.errors import NotFoundError, StorageError

# Keys become path fragments, so they are restricted to characters that cannot
# escape the root. Traversal is additionally blocked by the resolve() check in
# _path(); this is defence in depth, not the only guard.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


class LocalBlobStore:
    """Stores blobs under a single root directory."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not _SAFE_KEY.match(key) or ".." in key:
            raise StorageError("Invalid storage key", key=key)
        path = (self._root / key).resolve()
        # Even with a validated key, confirm the resolved path stayed inside
        # the root -- symlinks in the tree could otherwise redirect it.
        if not path.is_relative_to(self._root):
            raise StorageError("Storage key escapes the storage root", key=key)
        return path

    async def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        path = self._path(key)
        await asyncio.to_thread(self._write_atomic, path, data)
        return f"file://{path}"

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise NotFoundError("Blob not found", key=key) from exc
        except OSError as exc:
            raise StorageError(f"Failed to read blob: {exc}", key=key) from exc

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(path.unlink, True)

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def health(self) -> tuple[bool, str]:
        try:
            probe = self._root / ".health"
            await asyncio.to_thread(probe.write_bytes, b"ok")
            await asyncio.to_thread(probe.unlink, True)
            return True, f"local:{self._root}"
        except OSError as exc:
            return False, f"local storage not writable: {exc}"
