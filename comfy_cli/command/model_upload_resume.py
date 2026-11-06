"""Private, expiring model-upload sessions and the bucket's acknowledged byte range."""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import stat
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import requests

CHUNK_BYTES = 64 * 1024**2
LIFETIME = 7 * 24 * 60 * 60
STALL_SECONDS = 30 * 60
_MAX_ENTRIES = 32
_RANGE = re.compile(r"bytes=0-([0-9]+)$")


class SessionExpired(Exception):
    """Storage or complete no longer knows this upload; start again."""


class TransferFailed(Exception):
    """Only a status is safe to expose; never keep a transport exception."""


class TlsSetupFailed(Exception):
    """A permanent TLS failure; never retain transport exception text."""

    def __init__(self) -> None:
        super().__init__("certificate verification or CA certificate bundle setup failed")


class UploadPaused(Exception):
    """The argument is the last byte count acknowledged by storage."""


def _expiry(start: dict) -> float:
    try:
        return datetime.fromisoformat(start["expires_at"].replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return 0.0


def chunk_size(start: dict) -> int:
    value = start.get("chunk_size_bytes")
    if type(value) is not int or value < 256 * 1024 or value > CHUNK_BYTES or value % (256 * 1024):
        raise TransferFailed(None)
    return value


class SessionStore:
    """At most 32 bounded JSON files, written atomically with user-only access.

    No credential is used to identify the account: the caller supplies the ids
    returned by the authenticated user and workspace endpoints. Unknown identity
    disables persistence. Failure to secure or write the directory stops saving,
    but does not stop the current transfer.
    """

    def __init__(self, directory: Path, binding: dict) -> None:
        self.directory, self.binding = directory, binding
        self.path: Path | None = None

    def _entries(self) -> list[Path]:
        if self.directory.is_symlink():
            return []
        try:
            entries = list(self.directory.glob("*.json"))
            temporary = sorted(self.directory.glob(".upload-*"), key=lambda p: p.lstat().st_mtime)
            for index, path in enumerate(temporary):
                if index < len(temporary) - _MAX_ENTRIES or time.time() - path.lstat().st_mtime >= LIFETIME:
                    path.unlink(missing_ok=True)
            valid = []
            now = time.time()
            for path in entries:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                if info.st_size > 16384 or now - info.st_mtime >= LIFETIME:
                    path.unlink(missing_ok=True)
                else:
                    valid.append(path)
            return sorted(valid, key=lambda p: p.stat().st_mtime)
        except OSError:
            return []

    def has_saved(self) -> bool:
        return bool(self._entries())

    def _secure(self) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise OSError("session directory is a symlink")
        self.directory.chmod(0o700)

    def load(self, identity: dict | None) -> dict | None:
        if identity is None:
            return None
        try:
            self._secure()
        except OSError:
            return None
        for path in reversed(self._entries()):
            try:
                info = path.lstat()
                if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                    continue
                with path.open(encoding="utf-8") as stream:
                    saved = json.load(stream)
                if not isinstance(saved, dict) or not isinstance(saved.get("start"), dict):
                    continue
                created = saved.get("created_at")
                if not isinstance(created, (float, int)) or not math.isfinite(created) or created > time.time():
                    continue
                start = saved["start"]
                if min(created + LIFETIME, _expiry(start)) <= time.time():
                    path.unlink(missing_ok=True)
                    continue
                if saved.get("binding") == self.binding and saved.get("identity") == identity:
                    # Validate before any request to a saved credential address.
                    if start.get("upload_mode") != "resumable" or not valid_start(start):
                        continue
                    chunk_size(start)
                    self.path = path
                    return start
            except (OSError, ValueError, KeyError, TypeError, TransferFailed):
                continue
        return None

    def save(self, start: dict, identity: dict | None) -> bool:
        if identity is None or _expiry(start) <= time.time():
            return False
        temporary = None
        try:
            self._secure()
            # Two commands may both start before either has saved a session.
            # Each owns a separate record, including when their bindings match.
            path = self.directory / f"{uuid4().hex}.json"
            entries = self._entries()
            for old in entries[: max(0, len(entries) - _MAX_ENTRIES + 1)]:
                old.unlink(missing_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=self.directory)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    {"binding": self.binding, "identity": identity, "created_at": time.time(), "start": start}, stream
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self.path = path
            return True
        except (OSError, ValueError):
            return False
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def clear(self) -> None:
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self.path = None


def valid_start(start: dict) -> bool:
    if not isinstance(start.get("upload_url"), str):
        return False
    try:
        url = urlsplit(start.get("upload_url", ""))
        return (
            isinstance(start.get("upload_id"), str)
            and bool(start["upload_id"])
            and bool(start["upload_url"])
            and (
                start.get("upload_mode") is None
                or (url.scheme == "https" and bool(url.hostname) and not url.username and not url.password)
            )
            and start.get("method") == "PUT"
            and isinstance(start.get("headers"), dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in start["headers"].items())
            and start.get("upload_mode") in (None, "resumable")
        )
    except (TypeError, ValueError):
        return False


def _certificate_failure(error: BaseException) -> bool:
    """Recognize native and wrapped certificate errors without reporting their text."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        # Some adapters flatten the native error into a message.
        text = str(current).lower()
        if "certificate_verify_failed" in text or "certificate verify failed" in text:
            return True
        pending.extend(
            value
            for value in (*current.args, getattr(current, "reason", None), current.__cause__, current.__context__)
            if isinstance(value, BaseException)
        )
    return False


def _request(start: dict, data: bytes, content_range: str) -> tuple[int | None, str | None]:
    try:
        response = requests.put(
            start["upload_url"],
            data=data,
            headers={**start["headers"], "Content-Length": str(len(data)), "Content-Range": content_range},
            timeout=(10, 60),
            allow_redirects=False,
        )
    except requests.exceptions.SSLError as error:
        if _certificate_failure(error):
            raise TlsSetupFailed() from None
        return None, None
    except requests.RequestException:
        # Transport exceptions may quote the session credential.
        return None, None
    except OSError:
        raise TlsSetupFailed() from None
    try:
        return response.status_code, response.headers.get("Range")
    finally:
        response.close()


def transfer(
    start: dict, size: int, read_range, progress, *, resumed: bool, clock=time.monotonic, sleep=time.sleep
) -> None:
    """Query after every uncertain PUT; only acknowledged bytes advance progress.

    ``read_range`` checks each whole hashing block before returning a piece.
    It may start inside a block, because a connection can drop at any byte.
    """
    piece_size = chunk_size(start)
    offset, query, delay, last_progress = 0, resumed, 1, clock()
    while True:
        if clock() - last_progress >= STALL_SECONDS:
            raise UploadPaused(offset)
        end = offset if query else min(offset + piece_size, size)
        body = b"" if query else read_range(offset, end - offset)
        content_range = f"bytes */{size}" if query else f"bytes {offset}-{end - 1}/{size}"
        status, held = _request(start, body, content_range)
        if status in (200, 201, 412):
            progress(size - offset)
            return
        if status in (404, 410):
            raise SessionExpired()
        if status == 308:
            match = _RANGE.fullmatch(held) if isinstance(held, str) else None
            if held is not None and match is None:
                raise TransferFailed(308)
            acknowledged = int(match.group(1)) + 1 if match else 0
            if acknowledged < offset or acknowledged > size or (not query and acknowledged > end):
                raise TransferFailed(308)
            if acknowledged > offset:
                progress(acknowledged - offset)
                offset, last_progress, delay = acknowledged, clock(), 1
                if offset == size:
                    return
                query = False
                continue
            # A fresh empty status is normal; a piece with no progress backs off.
            if query:
                query = False
                continue
        elif status is not None and status not in (408, 429) and status < 500:
            raise TransferFailed(status)
        sleep(min(delay, max(0, STALL_SECONDS - (clock() - last_progress))))
        delay, query = min(delay * 2, 60), True
