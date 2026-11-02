"""``comfy assets library`` — browse, borrow and upload assets in Comfy
Cloud's asset library.

Mirrors the cloud-saved-workflow subcommands in ``workflow.py`` (``list``,
``get``, ...): thin Typer commands over ``cloud_http``'s shared helpers,
emitting a JSON envelope via the renderer. Cloud-only — there is no local
``/api/assets`` surface.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.command.cloud_http import (
    ResponseUnparseable,
    cloud_target_or_local_error,
    handle_cloud_http_error,
    http_request,
)
from comfy_cli.output.renderer import get_renderer

app = typer.Typer(help="Browse your Comfy Cloud asset library (list, borrow, upload).")

# A Comfy Cloud content hash: BLAKE3, 32-byte digest, lowercase hex — bare or in
# the canonical ``blake3:<hex>`` wire form. Comfy Cloud's from-hash lookup
# matches every stored form of the content only for ``blake3:<hex>``.
_HASH_WITH_EXT_RE = re.compile(r"^(?:blake3:)?([0-9a-fA-F]{64})\.[A-Za-z0-9]+$")


# A value that reads as a hex hash even when mangled: optional ``blake3:``,
# 16-80 hex chars (a garbled copy can gain or lose a few), optional extension.
_HEXISH_RE = re.compile(r"^(?:blake3:)?([0-9a-fA-F]{16,80})(?:\.[A-Za-z0-9]+)?$")
_HEX_DIGEST_RE = re.compile(r"^(?:blake3:)?([0-9a-f]{64})(?:\.[A-Za-z0-9]+)?$")
# Evidence a missing hash is a mangled copy of a stored one: a shared prefix
# this long (chance of a random match across a page of hashes is ~1e-16; a
# 4-char prefix, by contrast, matches by chance in a sizable library), or a
# single inserted / dropped / changed character anywhere.
_MIN_SHARED_PREFIX = 16
_MAX_SUGGESTIONS = 3
# One page at the API's maximum, newest first. Never paged: a suggestion is a
# best-effort hint on an error path, so it gets exactly one bounded request.
_SUGGESTION_SCAN_LIMIT = 500
# The listing only feeds an optional hint, so it must not hold the error for
# the default 30s request timeout.
_SUGGESTION_TIMEOUT_SECONDS = 5.0


def _within_one_edit(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ by at most one inserted, dropped or
    changed character (the common ways a copied hash picks up a typo)."""
    import os.path

    if abs(len(a) - len(b)) > 1:
        return False
    prefix = len(os.path.commonprefix([a, b]))
    suffix = len(os.path.commonprefix([a[::-1], b[::-1]]))
    if len(a) == len(b):
        return prefix + suffix >= len(a) - 1
    return prefix + suffix >= min(len(a), len(b))


def _near_hash_suggestions(value: str, target) -> list[dict]:
    """Library assets whose hash the not-found ``value`` is almost certainly a
    mangled copy of: within one character edit of it, or sharing a prefix of
    at least :data:`_MIN_SHARED_PREFIX` hex chars.

    ``[]`` without a request unless ``value`` looks like a MANGLED hex hash: a
    well-formed digest (64 lowercase hex, bare, canonical or with an
    extension) that is not found is simply absent, and an ordinary file name
    is not a hash. Otherwise one request lists the newest
    :data:`_SUGGESTION_SCAN_LIMIT` assets you own; any failure there yields
    ``[]`` so the not-found envelope is never lost to it.
    """
    import os.path
    import urllib.parse

    m = _HEXISH_RE.match(value)
    if not m or _HEX_DIGEST_RE.match(value):
        return []
    wanted = m.group(1).lower()
    query = urllib.parse.urlencode(
        {"limit": _SUGGESTION_SCAN_LIMIT, "sort": "created_at", "order": "desc", "include_public": "false"}
    )
    try:
        _, body = http_request(target.url("assets") + "?" + query, target, timeout=_SUGGESTION_TIMEOUT_SECONDS)
        rows = (body or {}).get("assets") or []
    except Exception:  # noqa: BLE001 — best-effort hint on an error path
        return []

    scored = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or not isinstance(r.get("hash"), str):
            continue
        hm = _HEX_DIGEST_RE.match(r["hash"])
        if not hm:
            continue
        stored = hm.group(1)
        shared = len(os.path.commonprefix([wanted, stored]))
        one_edit = _within_one_edit(wanted, stored)
        if one_edit or shared >= _MIN_SHARED_PREFIX:
            scored.append(((one_edit, shared), shared, r))
    scored.sort(key=lambda t: t[0], reverse=True)  # stable: ties keep newest-first order
    return [
        {"hash": r["hash"], "name": r.get("name"), "id": r.get("id"), "shared_prefix": shared}
        for _, shared, r in scored[:_MAX_SUGGESTIONS]
    ]


def _normalize_content_hash(value: str) -> str:
    """Map ``<hash>.<ext>`` — the stored file name an agent tends to pass — to
    the canonical ``blake3:<hex>`` hash; leave anything else untouched.

    A direct upload stores its blob under ``<hex>.<ext>``, so that is the name
    the agent sees, but ``/api/assets/from-hash`` matches its input exactly
    unless it is the canonical ``blake3:<hex>``, which matches every storage
    shape of that content (bare hex, ``<hex>.<ext>``, canonical). Only a value
    whose remainder is a full 64-hex digest is rewritten, so an ordinary file
    name (``photo.png``) still reaches the server as given.
    """
    m = _HASH_WITH_EXT_RE.match(value)
    return f"blake3:{m.group(1).lower()}" if m else value


@app.command("ls", help="List your assets on Comfy Cloud.")
@tracking.track_command("assets")
def ls_cmd(
    name: Annotated[
        str | None,
        typer.Option("--name", show_default=False, help="Case-insensitive substring match on asset name."),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option(
            "--tags", show_default=False, help="Comma-separated tags; assets must have ALL of them (e.g. input,output)."
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap rows returned (max 500).")] = 20,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import http.client
    import urllib.error
    import urllib.parse

    renderer = get_renderer()
    target = cloud_target_or_local_error(where, renderer)

    params: list[tuple[str, Any]] = [("limit", min(max(limit, 1), 500))]
    if name:
        params.append(("name_contains", name))
    for t in tags.split(",") if tags else []:
        t = t.strip()
        if t:
            params.append(("include_tags", t))
    url = target.url("assets") + "?" + urllib.parse.urlencode(params)

    try:
        # `strict_json` so a 200 carrying a proxy/captive-portal error page is
        # reported as malformed rather than collapsed to `None`, which is
        # indistinguishable here from a genuinely empty body — and would render
        # a broken response as a successful EMPTY library, the same masquerade
        # the shape guards below refuse to make.
        _, body = http_request(url, target, strict_json=True)
    # HTTPException: a connection closed partway through the body (IncompleteRead).
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise handle_cloud_http_error(renderer, e, operation="list") from e
    except ResponseUnparseable as e:
        renderer.error(
            code="cloud_http_error",
            message="unexpected response from /api/assets (body was not valid JSON)",
            hint="check whether a proxy or captive portal is intercepting the request",
            details={"operation": "list"},
        )
        raise typer.Exit(code=1) from e

    # A non-empty body decodes here only if it was valid JSON, but valid JSON is
    # not necessarily the shape we asked for: a proxy or error page can answer 200
    # with an array or a scalar, and `b.get(...)` would then raise a raw
    # `AttributeError` past the `except` above — a traceback instead of an error
    # envelope. Guard it the way `workflow list` guards the same failure on the
    # sibling endpoint. An empty body stays a legitimate `None` (→ no rows).
    if body is not None and not isinstance(body, dict):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/assets (expected a JSON object)",
            details={"got_type": type(body).__name__},
        )
        raise typer.Exit(code=1)

    b = body or {}
    # A missing/empty `assets` is a legitimately-empty listing; a present non-list
    # `assets` is malformed the same way, and `len(rows)` would raise `TypeError`.
    rows = b.get("assets")
    if rows is None:
        rows = []
    elif not isinstance(rows, list):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/assets (assets must be a JSON array)",
            details={"got_type": type(rows).__name__},
        )
        raise typer.Exit(code=1)
    assets = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "hash": r.get("hash"),
            "mime_type": r.get("mime_type"),
            "size": r.get("size"),
            "tags": r.get("tags"),
            "preview_url": r.get("preview_url"),
            "job_id": r.get("job_id"),
            "created_at": r.get("created_at"),
        }
        for r in rows
        if isinstance(r, dict)
    ]
    payload = {
        # `count` describes the array actually emitted, not the raw server rows.
        # A non-dict row is dropped from `assets` above, so counting raw rows
        # reported more items than the payload carries — and a knowingly-skewed
        # `count` sitting next to a forwarded `total` undercuts the point of
        # forwarding an authoritative count at all. Identical to the old value
        # for every well-formed response; it differs only where the old value
        # was simply wrong. Matches the repo convention in
        # `comfy_cli/command/nodes.py`, where `count` is over the emitted rows.
        "count": len(assets),
        "assets": assets,
    }
    # Forward the server's own truncation signal instead of making callers infer
    # it from an exactly-full page: `has_more`/`total` are both `required` on the
    # cloud API's `ListAssetsResponse`, and `has_more` comes off a limit+1
    # sentinel row rather than off the returned row count, so it stays correct
    # on a page that comes back short. Only forward what the server actually
    # sent — an older or local server may omit them, and a JSON `null` in the
    # envelope would poison a consumer's type assertion, so omit the key rather
    # than emitting None.
    #
    # Each field is checked against its OWN type, not a shared bool-or-int test:
    # `bool` is a subclass of `int` in Python, so one shared check would forward
    # `has_more: 0` and `total: false` and emit an envelope that violates
    # schemas/assets_library.json — the very contract this command publishes.
    if isinstance(b.get("has_more"), bool):
        payload["has_more"] = b["has_more"]
    total = b.get("total")
    # `>= 0` because the schema publishes `total` as a non-negative integer, and a
    # guard looser than the declared contract is how the envelope ends up violating
    # its own schema.
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        payload["total"] = total
    renderer.emit(payload, command="assets library ls", where="cloud")


@app.command("ensure", help="Ensure you own an asset by content hash (borrows public/shared bytes, no re-upload).")
@tracking.track_command("assets")
def ensure_cmd(
    hash: Annotated[str, typer.Option("--hash", help="Asset content hash (as returned by `assets library ls`).")],
    tags: Annotated[
        str,
        typer.Option("--tags", help="Comma-separated tags to attach (>=1 required by the API)."),
    ] = "input",
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import http.client
    import urllib.error

    renderer = get_renderer()
    target = cloud_target_or_local_error(where, renderer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or ["input"]
    url = target.url("assets/from-hash")
    normalized_hash = _normalize_content_hash(hash)
    try:
        status, body = http_request(url, target, method="POST", body={"hash": normalized_hash, "tags": tag_list})
    # HTTPException: a connection closed partway through the body (IncompleteRead).
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
        # The parameterized helper, not `cloud_http`'s: that one hardcodes the
        # saved-workflow vocabulary, so a 404 here read "workflow not found
        # (ensure)" with a hint to list workflows — for a request that never
        # named a workflow — e.g. when a caller passed a file name where the
        # content hash belongs.
        from comfy_cli.command._cloud_errors import handle_cloud_http_error as _handle_cloud_http_error

        not_found_hint = (
            "pass the `hash` from `comfy --json assets library ls --name <file>` "
            "(a file name is not a hash), or upload the file first with `comfy upload <file> --where cloud`"
        )
        extra = None
        if isinstance(e, urllib.error.HTTPError) and e.code == 404:
            suggestions = _near_hash_suggestions(hash, target)
            if suggestions:
                best = suggestions[0]
                extra = {"suggestions": suggestions}
                not_found_hint = (
                    f"did you mean {best['hash']} ({best['name']})? copy the hash exactly from "
                    "`comfy --json assets library ls` — see `details.suggestions`"
                )
        raise _handle_cloud_http_error(
            renderer,
            e,
            operation="ensure",
            not_found_code="asset_not_found",
            not_found_message=f"no asset with content hash {hash!r} in your Comfy Cloud library",
            not_found_hint=not_found_hint,
            id_label="hash",
            resource_id=hash,
            not_found_details=extra,
        ) from e

    # Unlike `ls`, an EMPTY body is an error here, not an empty result: a POST
    # cannot confirm the borrow without one, and the server contract
    # (`AssetCreated`) always returns an object with an `id`. `http_request`
    # also collapses an unparseable body (an HTML error page from a proxy, say)
    # to `None`, so `NoneType` covers both. An object without an `id` (`{}`, or
    # a gateway error body returned with a 2xx) confirms nothing either —
    # otherwise each emits `{"ok": true}` with a null id and the caller's own
    # hash echoed back as if the borrow had happened. A truthy non-string `id`
    # (`true`, `[1]`) is no asset id either.
    if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not body["id"]:
        renderer.error(
            code="cloud_http_error",
            message="unexpected response from /api/assets/from-hash (expected a JSON object describing the asset)",
            hint="the borrow could not be confirmed; retry, and check whether a proxy is intercepting the request",
            details={
                "operation": "ensure",
                "hash": hash,
                "status": status,
                "got_type": type(body).__name__,
                "missing": "id" if isinstance(body, dict) else None,
            },
        )
        raise typer.Exit(code=1)

    payload = {
        "id": body["id"],
        "hash": body.get("hash") or normalized_hash,
        "created_new": status == 201,
    }
    renderer.emit(payload, command="assets library ensure", where="cloud")


# What Comfy Cloud takes from `upload` (POST /api/assets/uploads): one
# `.safetensors` file of 1 byte to 2 GiB under a bare name of at most 255 bytes.
# A pickle format (.ckpt, .pt, .bin) is refused: it can run code when loaded.
_MAX_UPLOAD_BYTES = 2 * 1024**3
_MAX_FILENAME_BYTES = 255
# The first upload plus one fresh one after a PUT that got no answer or a 5xx.
# Fresh, not a second PUT to the same URL: a signed URL lasts 15 minutes, which
# a slow 2 GB transfer can use up.
_UPLOAD_STARTS = 2
# Storage answers 412 to a PUT for an object that already exists. Each upload
# id names a fresh object, so it can only be this upload's: the bytes landed.
_ALREADY_WRITTEN = 412
_PUT_TIMEOUT = (10, 600)
# Storing the file (hashing up to 2 GiB) normally takes a minute or two.
_TASK_POLL_SECONDS = 2.0
_TASK_WAIT_SECONDS = 15 * 60.0
# A task read that gets a 5xx or no answer is retried for this long from the
# first failure in a row; a 4xx ends the wait at once.
_UNANSWERED_WINDOW_SECONDS = 60.0
_TASK_DONE = frozenset({"completed", "failed", "cancelled"})
_REFUSAL_HINTS = {
    "FEATURE_DISABLED": "this workspace cannot take model uploads yet",
    "MODEL_STORAGE_LIMIT_REACHED": "delete models you no longer need, or move to a plan with more model storage",
    "UNSUPPORTED_FILE_TYPE": "convert the model to .safetensors",
    "UNSUPPORTED_MODEL_TYPE": "pass a --type from `model_upload_folders` in GET /api/features",
    "FILE_TOO_LARGE": "files can be up to 2 GB",
}
_sleep = time.sleep
_clock = time.monotonic


@dataclass(frozen=True)
class _UploadItem:
    """What ``UploadProgressReporter`` reads about the file."""

    filename: str
    size_bytes: int
    kind: str = "model"


class _PutFailed(Exception):
    """The PUT did not land; ``args[0]`` is the status, or ``None`` when no answer came."""


class _FileUnreadable(Exception):
    """The file could not be opened, sized or read while it was being sent; ``args[0]`` is the ``OSError``."""


class _TlsSetupFailed(Exception):
    """This machine's TLS settings stopped the PUT before it connected; ``args[0]`` is the message to show."""


def _file_problem(path: Path, e: OSError) -> tuple[str, str]:
    """``(reason, message)`` for an ``OSError`` met while checking the file."""
    if isinstance(e, FileNotFoundError):
        return "file_not_found", f"{path} does not exist or is not a file"
    return "file_unreadable", f"{path} cannot be read ({e.strerror or e})"


def _local_refusal(path: Path) -> tuple[str, str] | None:
    """``(reason, message)`` when Comfy Cloud would refuse this file, else ``None``."""
    name = path.name
    if not name.lower().endswith(".safetensors"):
        return "unsupported_file_type", (
            f"{name} is not a .safetensors file; only .safetensors model files can be uploaded "
            "(a pickle format such as .ckpt, .pt or .bin can run code when it is loaded)"
        )
    try:
        bad = len(name.encode("utf-8")) > _MAX_FILENAME_BYTES
    except UnicodeEncodeError:  # not valid UTF-8
        bad = True
    if (
        bad
        or name.startswith(".")
        or ".." in name
        or "\\" in name
        or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name)
    ):
        return "bad_filename", (
            f"{name!r} is not a file name Comfy Cloud accepts: at most {_MAX_FILENAME_BYTES} bytes of UTF-8, "
            "no leading dot, no '..', backslash or control characters"
        )
    try:
        if not path.is_file():
            return "file_not_found", f"{path} does not exist or is not a file"
        # Opened here so a file this user cannot read is refused before the
        # start call creates an upload on the server.
        with path.open("rb"):
            pass
        size = path.stat().st_size
    except OSError as e:
        return _file_problem(path, e)
    if size < 1:
        return "empty_file", f"{path} is empty"
    if size > _MAX_UPLOAD_BYTES:
        return "file_too_large", f"{name} is {size} bytes; files can be up to 2 GB ({_MAX_UPLOAD_BYTES} bytes)"
    return None


def _http_failure(renderer, e: Exception, operation: str, sign_in=None) -> typer.Exit:
    """Emit the envelope for a failed cloud API call; return the Exit to raise.

    A 400/403/404 carrying Comfy Cloud's ``{"code", "message"}`` body is a
    refusal: its message is shown as it is, under ``asset_upload_refused``.
    """
    import json
    import urllib.error

    from comfy_cli.command._cloud_errors import emit_status_error, retry_after_from_headers

    if not isinstance(e, urllib.error.HTTPError):
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy cloud whoami`",
            details={"operation": operation},
        )
        return typer.Exit(code=1)
    try:
        body = (e.read(1000) or b"").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — a failed read must not pre-empt the envelope
        body = ""
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError):
        parsed = None
    server_code = parsed.get("code") if isinstance(parsed, dict) else None
    server_message = parsed.get("message") if isinstance(parsed, dict) else None
    if (
        e.code in (400, 403, 404)
        and isinstance(server_code, str)
        and isinstance(server_message, str)
        and server_message
    ):
        renderer.error(
            code="asset_upload_refused",
            message=server_message,
            hint=_REFUSAL_HINTS.get(server_code, "`details.server_code` names the refusal; a retry gets the same"),
            details={"status": e.code, "server_code": server_code, "operation": operation},
        )
    elif e.code == 401 and sign_in is not None and sign_in.account_replaced:
        renderer.error(
            code="cloud_unauthorized",
            message=(
                f"HTTP 401 during {operation}: the upload's sign-in expired, and this machine is now "
                "signed in to another account or workspace, which cannot finish it"
            ),
            hint=(
                "sign in again as the account that started the upload (`comfy cloud login`), then check "
                "`comfy assets library ls --tags models` or upload the file again"
            ),
            details={"status": 401, "operation": operation, "reason": "account_replaced"},
        )
    elif e.code == 401:
        renderer.error(
            code="cloud_unauthorized",
            message=f"HTTP 401 during {operation}",
            hint="re-run `comfy cloud login`",
            details={"status": 401, "operation": operation},
        )
    else:
        emit_status_error(
            renderer,
            status=e.code,
            retry_after=retry_after_from_headers(getattr(e, "headers", None)),
            operation=operation,
            message=f"HTTP {e.code} during {operation}",
            hint="check `details.body` for the server's message",
            details={"status": e.code, "body": body, "operation": operation},
        )
    return typer.Exit(code=1)


def _unexpected(renderer, route: str, problem: str, **details: Any) -> typer.Exit:
    renderer.error(
        code="cloud_http_error",
        message=f"unexpected response from {route} ({problem})",
        hint="check whether a proxy or captive portal is intercepting the request",
        details=details,
    )
    return typer.Exit(code=1)


def _send(target, sign_in, url: str, **kwargs) -> tuple[int, dict | None]:
    """``http_request``, keeping a stored sign-in's token fresh through a long upload.

    The token lasts fifteen minutes and a 2 GB PUT can outlast it. It is re-read
    before each call; after a 401 the shared refresh is forced and the call sent
    once more (the server refuses before acting, so a resent complete queues no
    second task). A sign-in to another account made meanwhile is never used: the
    server files the upload and its task under the account that started it.
    ``sign_in`` is ``None`` for an API key or a token handed in by env: those are
    sent unchanged and their 401 surfaces.
    """
    import urllib.error

    if sign_in is None:
        return http_request(url, target, **kwargs)
    try:
        return http_request(url, replace(target, auth_token=sign_in.current()), **kwargs)
    except urllib.error.HTTPError as e:
        if e.code != 401 or not sign_in.after_rejection():
            raise
    return http_request(url, replace(target, auth_token=sign_in.current()), **kwargs)


def _call(renderer, target, operation: str, parts: tuple[str, ...], body: dict | None = None, *, sign_in=None) -> dict:
    """One JSON call to the cloud API; anything but a JSON object is a mapped error."""
    import http.client
    import urllib.error

    route = "/api/" + "/".join(parts)
    method = "GET" if body is None else "POST"
    try:
        _, parsed = _send(target, sign_in, target.url(*parts), method=method, body=body, strict_json=True)
    # HTTPException: a connection closed partway through the body (IncompleteRead).
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise _http_failure(renderer, e, operation, sign_in) from e
    except ResponseUnparseable as e:
        raise _unexpected(renderer, route, "body was not valid JSON", operation=operation) from e
    if not isinstance(parsed, dict):
        raise _unexpected(renderer, route, "expected a JSON object", operation=operation)
    return parsed


def _put(upload_url: str, headers: dict, path: Path, progress) -> None:
    """Stream the file to its signed URL, or raise ``_PutFailed``.

    A transport exception is dropped, never shown: its text quotes the signed
    URL, whose query string is a live credential. A file that cannot be opened,
    sized or read raises ``_FileUnreadable`` instead: a fresh upload would fail
    the same way, so it is not a transport failure to retry. Any other
    ``OSError`` comes from setting up TLS (a CA bundle path that no longer
    exists) and raises ``_TlsSetupFailed`` with its message; a retry would
    fail the same way too.
    """
    import urllib.parse

    import requests

    from comfy_cli.builder_api import _CountingReader

    class _Reader(_CountingReader):
        # urllib3 reports a body that fails to read as a dropped connection,
        # so the read error is kept here to tell the two apart.
        failed: OSError | None = None

        def __init__(self, raw, progress, size: int) -> None:
            super().__init__(raw, progress)
            self._size = size

        def __len__(self) -> int:
            # Requests takes the length from here rather than its own
            # `os.fstat`, so the file is sized only below, where an error is
            # known to be the file's. Same Content-Length either way.
            return self._size

        def read(self, size: int = -1) -> bytes:
            try:
                return super().read(size)
            except OSError as e:
                self.failed = e
                raise

    try:
        f = path.open("rb")
    except OSError as e:
        raise _FileUnreadable(e) from None
    with f:
        try:
            size = os.fstat(f.fileno()).st_size
        except OSError as e:  # e.g. a stale handle on a network filesystem
            raise _FileUnreadable(e) from None
        reader = _Reader(f, progress, size)
        try:
            resp = requests.put(
                upload_url,
                data=reader,
                headers=headers,
                timeout=_PUT_TIMEOUT,
                allow_redirects=False,  # the URL names one object; never divert the file
            )
        except OSError as e:  # `requests.RequestException` is an OSError too
            if reader.failed is not None:
                raise _FileUnreadable(reader.failed) from None
            if isinstance(e, requests.RequestException):
                raise _PutFailed(None) from None
            # Neither the file nor the transport (Requests wraps those): the
            # TLS setup, e.g. REQUESTS_CA_BUNDLE naming a removed file. Its
            # text names a certificate path, but is never shown if it quotes
            # the signed URL.
            text, query = str(e) or type(e).__name__, urllib.parse.urlsplit(upload_url).query
            if upload_url in text or (query and query in text):
                text = type(e).__name__
            raise _TlsSetupFailed(text) from None
    if not (200 <= resp.status_code < 300 or resp.status_code == _ALREADY_WRITTEN):
        raise _PutFailed(resp.status_code)


def _wait_for_task(renderer, target, task_id: str, sign_in=None) -> dict:
    """Read the task until it ends, riding out a brief 5xx, 429 or dropped connection."""
    import http.client
    import urllib.error
    import urllib.parse

    from comfy_cli.command._cloud_errors import retry_after_from_headers

    url = target.url("tasks", urllib.parse.quote(task_id, safe=""))
    deadline = _clock() + _TASK_WAIT_SECONDS
    unanswered_since: float | None = None
    status = None
    while True:
        pause = _TASK_POLL_SECONDS
        try:
            _, task = _send(target, sign_in, url, strict_json=True)
        # HTTPException: a connection closed partway through the body (IncompleteRead).
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            now = _clock()
            unanswered_since = now if unanswered_since is None else unanswered_since
            code = e.code if isinstance(e, urllib.error.HTTPError) else None
            refused = code is not None and code < 500 and code != 429
            if refused or now - unanswered_since >= _UNANSWERED_WINDOW_SECONDS:
                raise _http_failure(renderer, e, "task", sign_in) from e
            retry_after = retry_after_from_headers(getattr(e, "headers", None)) if code == 429 else None
            if retry_after is not None:
                # Never past the end of the window or of the whole wait.
                left = min(unanswered_since + _UNANSWERED_WINDOW_SECONDS, deadline) - now
                pause = max(_TASK_POLL_SECONDS, min(retry_after, left))
        except ResponseUnparseable as e:
            raise _unexpected(renderer, "/api/tasks", "body was not valid JSON", task_id=task_id) from e
        else:
            unanswered_since = None
            if not isinstance(task, dict) or not isinstance(task.get("status"), str):
                raise _unexpected(renderer, "/api/tasks", "no task status", task_id=task_id)
            status = task["status"]
            if status in _TASK_DONE:
                return task
        if _clock() >= deadline:
            renderer.error(
                code="asset_upload_timeout",
                message=f"Comfy Cloud was still storing the file after {int(_TASK_WAIT_SECONDS)}s",
                hint="storing goes on without this command; check `comfy assets library ls --tags models` later",
                details={"task_id": task_id, "status": status},
            )
            raise typer.Exit(code=1)
        _sleep(pause)


@app.command("upload", help="Upload a .safetensors model file (up to 2 GB) into your Comfy Cloud model library.")
@tracking.track_command("assets")
def upload_cmd(
    file: Annotated[Path, typer.Argument(help="The .safetensors file to upload.", show_default=False)],
    folder: Annotated[
        str, typer.Option("--type", help="Model folder to file it under (e.g. loras, checkpoints, vae).")
    ],
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.parse

    from comfy_cli.command.build_upload_progress import UploadProgressReporter
    from comfy_cli.credentials import DataPlaneToken

    def refuse(reason: str, message: str) -> typer.Exit:
        renderer.error(code="asset_upload_invalid", message=message, details={"reason": reason, "file": str(file)})
        return typer.Exit(code=1)

    renderer = get_renderer()
    refusal = _local_refusal(file)
    if refusal is not None:
        raise refuse(*refusal)
    target = cloud_target_or_local_error(where, renderer)
    # A stored sign-in's token can expire during the PUT, so completion and
    # polling keep it fresh, but only from the same sign-in: Comfy Cloud files the
    # staged bytes and the task under the user and workspace that started them.
    # An API key, or a token handed in by env, is sent as is.
    sign_in = (
        DataPlaneToken.for_target_token(
            target.auth_token, base_url=target.base_url, allow_clear=True, same_account=True
        )
        if target.auth_token
        else None
    )

    # Absent while uploads are off for the workspace, or when the server could
    # not evaluate its flags. Either way the start call decides, and it refuses
    # before any byte is sent.
    folders = _call(renderer, target, "features", ("features",), sign_in=sign_in).get("model_upload_folders")
    if folders is not None:
        if not isinstance(folders, list) or not all(isinstance(f, str) for f in folders):
            raise _unexpected(renderer, "/api/features", "model_upload_folders is not a list of strings")
        if folder not in folders:
            renderer.error(
                code="asset_upload_invalid",
                message=f"{folder!r} is not a model type Comfy Cloud takes uploads for; pick one of: {', '.join(folders)}",
                details={"reason": "unsupported_model_type", "type": folder, "valid_types": folders},
            )
            raise typer.Exit(code=1)

    try:
        size = file.stat().st_size
    except OSError as e:  # removed or locked since it was checked; nothing is sent yet
        raise refuse(*_file_problem(file, e)) from None
    item = _UploadItem(filename=file.name, size_bytes=size)
    tags = ["models", folder]
    reporter = UploadProgressReporter(renderer)
    reporter.plan([item], already_held=0)
    for attempt in range(1, _UPLOAD_STARTS + 1):
        start = _call(
            renderer,
            target,
            "start",
            ("assets", "uploads"),
            {"filename": item.filename, "size_bytes": item.size_bytes, "tags": tags},
            sign_in=sign_in,
        )
        upload_id, upload_url, headers = start.get("upload_id"), start.get("upload_url"), start.get("headers")
        if not (
            isinstance(upload_id, str)
            and upload_id
            and isinstance(upload_url, str)
            and upload_url
            and start.get("method") == "PUT"
            and isinstance(headers, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items())
        ):
            # Never the URL itself in the error: it is a credential.
            raise _unexpected(renderer, "/api/assets/uploads", "expected upload_id, upload_url, method PUT and headers")
        try:
            with reporter.uploading(item, 1, 1) as progress:
                _put(upload_url, headers, file, progress)
            break
        except _FileUnreadable as e:
            renderer.error(
                code="asset_upload_failed",
                message=f"could not read {file} while sending it to storage ({e.args[0].strerror or e.args[0]})",
                hint="check that the file is still there and readable, then upload it again; "
                "nothing was added to the library",
                details={"reason": "file_unreadable", "upload_id": upload_id, "attempts": attempt},
            )
            raise typer.Exit(code=1) from None
        except _TlsSetupFailed as e:
            renderer.error(
                code="asset_upload_failed",
                message=f"could not set up a secure connection to storage ({e.args[0]})",
                hint="point REQUESTS_CA_BUNDLE (or CURL_CA_BUNDLE) at a certificate bundle that exists, or unset it, "
                "then upload again; nothing was added to the library",
                details={"reason": "tls_setup", "upload_id": upload_id, "attempts": attempt},
            )
            raise typer.Exit(code=1) from None
        except _PutFailed as e:
            status = e.args[0]
            if (status is None or status >= 500) and attempt < _UPLOAD_STARTS:
                renderer.warn("The upload to storage did not go through; starting a fresh upload.")
                continue
            answer = "no response" if status is None else f"HTTP {status}"
            renderer.error(
                code="asset_upload_failed",
                message=f"sending {item.filename} to storage failed ({answer})",
                hint="retry; nothing was added to the library",
                details={"status": status, "upload_id": upload_id, "attempts": attempt},
            )
            raise typer.Exit(code=1) from None

    complete = ("assets", "uploads", urllib.parse.quote(upload_id, safe=""), "complete")
    task_id = _call(
        renderer, target, "complete", complete, {"filename": item.filename, "tags": tags}, sign_in=sign_in
    ).get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise _unexpected(renderer, "/api/assets/uploads/complete", "no task_id", upload_id=upload_id)
    if renderer.is_pretty():
        renderer.info("Uploaded; Comfy Cloud is checking and storing the file.")
    task = _wait_for_task(renderer, target, task_id, sign_in)

    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    if task["status"] != "completed" or result.get("success") is False:
        reason = task.get("error_message") or result.get("error") or f"the task ended {task['status']}"
        renderer.error(
            code="asset_upload_failed",
            message=f"Comfy Cloud did not store {item.filename}: {reason}",
            hint="the message says why; fix the file and upload it again",
            details={"task_id": task_id, "status": task["status"]},
        )
        raise typer.Exit(code=1)
    asset_id = result.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise _unexpected(renderer, "/api/tasks", "a completed task with no asset_id", task_id=task_id)
    payload = {
        "id": asset_id,
        "name": result["filename"] if isinstance(result.get("filename"), str) else item.filename,
        "folder": folder,
        "hash": result["hash"] if isinstance(result.get("hash"), str) else None,
        "size": item.size_bytes,
        "task_id": task_id,
    }
    if renderer.is_pretty():
        renderer.success(f"Added {payload['name']} to {folder}")
        renderer.info(f"asset id: {asset_id}\nhash: {payload['hash'] or '-'}")
    renderer.emit(payload, command="assets library upload", where="cloud", changed=True)
