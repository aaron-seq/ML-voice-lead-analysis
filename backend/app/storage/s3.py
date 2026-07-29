"""S3 blob storage.

boto3 is synchronous, so every call is dispatched to a worker thread rather
than blocking the event loop. Credentials come from the standard provider chain
(instance role, environment, shared config) -- this module never accepts an
access key as a parameter, so keys cannot be logged or persisted by accident.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re

from app.core.errors import NotFoundError, StorageError

logger = logging.getLogger(__name__)

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


class S3BlobStore:
    """Blob storage backed by an S3 bucket."""

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1") -> None:
        if not bucket:
            raise StorageError("S3_BUCKET must be set when STORAGE_BACKEND=s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                region_name=self._region,
                config=Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    connect_timeout=5,
                    read_timeout=30,
                    max_pool_connections=25,
                ),
            )
        return self._client

    def _full_key(self, key: str) -> str:
        if not _SAFE_KEY.match(key) or ".." in key:
            raise StorageError("Invalid storage key", key=key)
        return f"{self._prefix}{key}"

    async def _call(self, method: str, **kwargs):
        client = self._get_client()
        func = functools.partial(getattr(client, method), **kwargs)
        return await asyncio.to_thread(func)

    async def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        full_key = self._full_key(key)
        try:
            await self._call(
                "put_object",
                Bucket=self._bucket,
                Key=full_key,
                Body=data,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )
        except Exception as exc:
            raise StorageError(f"S3 write failed: {exc}", key=key) from exc
        return f"s3://{self._bucket}/{full_key}"

    async def get(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            response = await self._call("get_object", Bucket=self._bucket, Key=full_key)
            return await asyncio.to_thread(response["Body"].read)
        except Exception as exc:
            if _error_code(exc) in {"NoSuchKey", "404"}:
                raise NotFoundError("Blob not found", key=key) from exc
            raise StorageError(f"S3 read failed: {exc}", key=key) from exc

    async def delete(self, key: str) -> None:
        try:
            await self._call(
                "delete_object", Bucket=self._bucket, Key=self._full_key(key)
            )
        except Exception as exc:
            raise StorageError(f"S3 delete failed: {exc}", key=key) from exc

    async def exists(self, key: str) -> bool:
        try:
            await self._call("head_object", Bucket=self._bucket, Key=self._full_key(key))
            return True
        except Exception as exc:
            if _error_code(exc) in {"NoSuchKey", "404", "NotFound"}:
                return False
            raise StorageError(f"S3 head failed: {exc}", key=key) from exc

    async def health(self) -> tuple[bool, str]:
        try:
            await self._call("head_bucket", Bucket=self._bucket)
            return True, f"s3://{self._bucket}"
        except Exception as exc:
            return False, f"s3 bucket unreachable: {exc}"


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""
