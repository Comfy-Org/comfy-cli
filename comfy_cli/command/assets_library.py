"""``comfy assets library`` — browse, borrow and upload assets in Comfy
Cloud's asset library.

Mirrors the cloud-saved-workflow subcommands in ``workflow.py`` (``list``,
``get``, ...): thin Typer commands over ``cloud_http``'s shared helpers,
emitting a JSON envelope via the renderer. Cloud-only — there is no local
``/api/assets`` surface.
"""

from __future__ import annotations

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
from comfy_cli.command.model_upload_resume import TlsSetupFailed as _TlsSetupFailed
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
# `.safetensors` file under a bare name of at most 255 bytes. Start enforces
# the size limit (up to 25 GiB when the server supports resumable uploads).
# A pickle format (.ckpt, .pt, .bin) is refused: it can run code when loaded.
_MAX_FILENAME_BYTES = 255
# The first upload plus one fresh one after a PUT that got no answer or a 5xx.
# Fresh, not a second PUT to the same URL: a signed URL lasts 15 minutes, which
# a slow 2 GB transfer can use up.
_UPLOAD_STARTS = 2
# Storage answers 412 to a PUT for an object that already exists. Each upload
# id names a fresh object, so it can only be this upload's: the bytes landed.
_ALREADY_WRITTEN = 412
_PUT_TIMEOUT = (10, 600)
# Sent with start and complete; a server without upload analytics ignores it.
# DO_NOT_TRACK or COMFY_NO_TELEMETRY drops it, `resumed` and the stop report.
_UPLOAD_CLIENT = {"client": "comfy_cli"}
# A report that a transfer stopped gets one request and no retry, and the
# command waits this long at most for all of it, DNS and slow answers included.
_REPORT_TIMEOUT_SECONDS = 3.0
# The wait on that report goes in steps this long, so a Ctrl+C lands within one
# step: on Windows a timed join does not see Ctrl+C until it ends.
_REPORT_WAIT_STEP_SECONDS = 0.1
# Large uploads can take more than fifteen minutes to check and store.
_TASK_POLL_SECONDS = 2.0
# A task read that gets a 5xx or no answer is retried for this long from the
# first failure in a row; a 4xx ends the wait at once.
_UNANSWERED_WINDOW_SECONDS = 60.0
_TASK_DONE = frozenset({"completed", "failed", "cancelled"})
_REFUSAL_HINTS = {
    "FEATURE_DISABLED": "this workspace cannot take model uploads yet",
    "MODEL_STORAGE_LIMIT_REACHED": "delete models you no longer need, or move to a plan with more model storage",
    "UNSUPPORTED_FILE_TYPE": "convert the model to .safetensors",
    "UNSUPPORTED_MODEL_TYPE": "pass a --type from `model_upload_folders` in GET /api/features",
    "FILE_TOO_LARGE": "use a file within the size limit in the server message",
}
# Any form Comfy Cloud reports a model's content hash in: bare hex, canonical
# ``blake3:<hex>`` or the stored ``<hex>.<ext>``, any case.
_ANY_HASH_RE = re.compile(r"^(?:blake3:)?([0-9a-fA-F]{64})(?:\.[A-Za-z0-9]+)?$")
# One page at the API's maximum when looking up the models a name already
# stands for; a name matching more than this is refused, not paged.
_LOOKUP_LIMIT = 500
# An approval's SHA-256: 64 hex, bare or ``sha256:``-prefixed, any case.
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
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
    """The file could not be opened or read while it was being sent; ``args[0]`` is the ``OSError``."""


class _FileChanged(Exception):
    """A block of the file read for the PUT is not the block hashed before it; ``args[0]`` is the block's index."""


class _BlockChanged(OSError):
    """Raised from inside the PUT's body read, which Requests and urllib3 let through only as an ``OSError``."""


class _HashedBytes:
    """The file, read for the PUT, giving out only the bytes that were hashed.

    Each block is read whole and checked against its digest from the hash
    taken before the upload (``FileDigests.blocks``) before any of it is
    handed on, so a byte that was not hashed, and approved by its hash, never
    reaches storage. Reading stops at the hashed size: bytes added to the end
    since are not sent.
    """

    def __init__(self, raw, digests, offset: int = 0) -> None:
        self._raw, self._digests = raw, digests
        self._pending = b""
        self._next_block = offset // digests.block_size
        self._skip = offset % digests.block_size
        self._given = offset
        if offset:
            raw.seek(self._next_block * digests.block_size)

    def read(self, size: int = -1) -> bytes:
        from blake3 import blake3

        if not self._pending:
            d = self._digests
            if self._next_block >= len(d.blocks):
                return b""
            block = self._raw.read(min(d.block_size, d.size - self._next_block * d.block_size))
            if blake3(block).digest() != d.blocks[self._next_block]:
                raise _BlockChanged(f"block {self._next_block} of the file is not the one hashed")
            self._pending, self._next_block = block[self._skip :], self._next_block + 1
            self._skip = 0
        n = len(self._pending) if size is None or size < 0 else size
        out, self._pending = self._pending[:n], self._pending[n:]
        self._given += len(out)
        return out

    @property
    def mode(self) -> str:
        return self._raw.mode

    def fileno(self) -> int:
        return self._raw.fileno()

    def tell(self) -> int:
        return self._given


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
    return None


class _ResumableRefused(Exception):
    """Ingest answered a resumable start with 403 ``FEATURE_DISABLED``.

    Raised by ``_call`` only for the start that asks for it, so the caller can
    start once more as a single PUT, as the editor does.
    """


def _error_text(e) -> str:
    try:
        return (e.read(1000) or b"").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — a failed read must not pre-empt the envelope
        return ""


def _http_failure(renderer, e: Exception, operation: str, sign_in=None, body: str | None = None) -> typer.Exit:
    """Emit the envelope for a failed cloud API call; return the Exit to raise.

    A 400/403/404 carrying Comfy Cloud's ``{"code", "message"}`` body is a
    refusal: its message is shown as it is, under ``asset_upload_refused``.
    ``body`` is the error body when the caller has already read it.
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
    if body is None:
        body = _error_text(e)
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


def _call(
    renderer,
    target,
    operation: str,
    parts: tuple[str, ...],
    body: dict | None = None,
    *,
    sign_in=None,
    query: list[tuple[str, Any]] | None = None,
    resumable_start: bool = False,
) -> dict:
    """One JSON call to the cloud API; anything but a JSON object is a mapped error.

    With ``resumable_start``, a 403 ``FEATURE_DISABLED`` raises
    ``_ResumableRefused`` instead: ingest answers so to every resumable start
    while its resumable flag is off, whatever the size, and takes the same
    file as a single PUT. A workspace without the model library gets the same
    code; its single-PUT start is refused too and shown as the refusal.
    """
    import http.client
    import json
    import urllib.error
    import urllib.parse

    route = "/api/" + "/".join(parts)
    method = "GET" if body is None else "POST"
    url = target.url(*parts) + ("?" + urllib.parse.urlencode(query) if query else "")
    try:
        _, parsed = _send(target, sign_in, url, method=method, body=body, strict_json=True)
    except urllib.error.HTTPError as e:
        if not (resumable_start and e.code == 403):
            raise _http_failure(renderer, e, operation, sign_in) from e
        text = _error_text(e)
        try:
            server = json.loads(text)
        except (ValueError, RecursionError):
            server = None
        if isinstance(server, dict) and server.get("code") == "FEATURE_DISABLED":
            raise _ResumableRefused from e
        raise _http_failure(renderer, e, operation, sign_in, body=text) from e
    # HTTPException: a connection closed partway through the body (IncompleteRead).
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise _http_failure(renderer, e, operation, sign_in) from e
    except ResponseUnparseable as e:
        raise _unexpected(renderer, route, "body was not valid JSON", operation=operation) from e
    if not isinstance(parsed, dict):
        raise _unexpected(renderer, route, "expected a JSON object", operation=operation)
    return parsed


def _put(upload_url: str, headers: dict, path: Path, progress, digests) -> None:
    """Stream the bytes ``digests`` was taken from to the signed URL, or raise ``_PutFailed``.

    Each block is checked against ``digests`` before any of it is sent
    (``_HashedBytes``); a block that changed since raises ``_FileChanged``,
    with only the earlier, unchanged blocks sent.

    A transport exception is dropped, never shown: its text quotes the signed
    URL, whose query string is a live credential. A file that cannot be opened
    or read raises ``_FileUnreadable`` instead: a fresh upload would fail
    the same way, so it is not a transport failure to retry. Certificate
    verification/loading errors and other TLS setup ``OSError`` failures
    raise ``_TlsSetupFailed`` with fixed TLS guidance; a retry would fail the
    same way too. Transient TLS errors still permit a fresh upload.
    """
    import requests

    from comfy_cli.builder_api import _CountingReader
    from comfy_cli.command.model_upload_resume import _tls_setup_failure

    class _Reader(_CountingReader):
        # urllib3 reports a body that fails to read as a dropped connection,
        # so the read error is kept here to tell the two apart.
        failed: OSError | None = None

        def __init__(self, raw, progress, size: int) -> None:
            super().__init__(raw, progress)
            self._size = size

        def __len__(self) -> int:
            # Requests takes the length from here: the hashed size, the only
            # bytes `_HashedBytes` gives out.
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
        # The size hashed, not the file's size now: only the hashed bytes are
        # sent, and a file cut short since fails its last block's check.
        size = digests.size
        reader = _Reader(_HashedBytes(f, digests), progress, size)
        try:
            resp = requests.put(
                upload_url,
                data=reader,
                headers=headers,
                timeout=_PUT_TIMEOUT,
                allow_redirects=False,  # the URL names one object; never divert the file
            )
        except OSError as e:  # `requests.RequestException` is an OSError too
            if isinstance(reader.failed, _BlockChanged):
                raise _FileChanged(str(reader.failed)) from None
            if reader.failed is not None:
                raise _FileUnreadable(reader.failed) from None
            if isinstance(e, (requests.exceptions.SSLError, requests.exceptions.ProxyError)) and _tls_setup_failure(e):
                raise _TlsSetupFailed() from None
            if isinstance(e, requests.RequestException):
                raise _PutFailed(None) from None
            # Neither the file nor the transport (Requests wraps those): the
            # TLS setup, e.g. REQUESTS_CA_BUNDLE naming a removed file. Never
            # retain its text: Requests may normalize a credential in the URL.
            raise _TlsSetupFailed() from None
    if not (200 <= resp.status_code < 300 or resp.status_code == _ALREADY_WRITTEN):
        raise _PutFailed(resp.status_code)


def _read_range(path, digests, offset: int, length: int) -> bytes:
    try:
        with path.open("rb") as raw:
            reader = _HashedBytes(raw, digests, offset)
            blocks = []
            while length:
                block = reader.read(length)
                if not block:
                    raise _FileChanged()
                blocks.append(block)
                length -= len(block)
            return b"".join(blocks)
    except _BlockChanged:
        raise _FileChanged() from None
    except OSError as e:
        raise _FileUnreadable(e) from None


def _upload_identity(target, sign_in) -> dict | None:
    """Only server-confirmed account and workspace ids permit saved-session reuse."""
    import http.client
    import urllib.error

    try:
        _, user = _send(target, sign_in, target.url("user"), strict_json=True)
        workspace = _current_workspace(target, sign_in)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ResponseUnparseable):
        return None
    account = user.get("id") if isinstance(user, dict) else None
    if not isinstance(account, str) or not account or not workspace:
        return None
    return {"account": account, "workspace": workspace}


def _complete_upload(renderer, target, sign_in, upload_id: str, body: dict, *, resumable: bool) -> dict:
    import http.client
    import urllib.error
    import urllib.parse

    from comfy_cli.command.model_upload_resume import STALL_SECONDS, SessionExpired

    parts = ("assets", "uploads", urllib.parse.quote(upload_id, safe=""), "complete")
    if not resumable:
        return _call(renderer, target, "complete", parts, body, sign_in=sign_in)
    started, delay = _clock(), 1
    while True:
        try:
            _, parsed = _send(target, sign_in, target.url(*parts), method="POST", body=body, strict_json=True)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            code = e.code if isinstance(e, urllib.error.HTTPError) else None
            if code == 404:
                raise SessionExpired() from None
            if (code is not None and code != 429 and code < 500) or _clock() - started >= STALL_SECONDS:
                raise _http_failure(renderer, e, "complete", sign_in) from None
            _sleep(min(delay, STALL_SECONDS - (_clock() - started)))
            delay = min(delay * 2, 60)
        except ResponseUnparseable:
            raise _unexpected(renderer, "/api/assets/uploads/complete", "body was not valid JSON") from None
        else:
            if not isinstance(parsed, dict):
                raise _unexpected(renderer, "/api/assets/uploads/complete", "expected a JSON object")
            return parsed


def _upload_analytics(**extra) -> dict:
    """The ``upload_analytics`` field for a start or complete body.

    Empty under DO_NOT_TRACK or COMFY_NO_TELEMETRY; the server treats a missing
    value as unknown. It goes to Comfy Cloud with the upload request itself, so
    the opt-in tracking setting (product analytics) does not gate it.
    """
    return {} if tracking._telemetry_disabled_by_env() else {"upload_analytics": {**_UPLOAD_CLIENT, **extra}}


def _stop_reason(e: BaseException) -> str:
    """The events route's ``reason`` for what ended a transfer for good."""
    from comfy_cli.command.model_upload_resume import SessionExpired, TransferFailed, UploadPaused

    if isinstance(e, KeyboardInterrupt):
        return "canceled"
    if isinstance(e, SessionExpired):
        return "session_expired"
    if isinstance(e, _FileChanged):
        return "file_changed"
    if isinstance(e, _FileUnreadable):
        return "file_unreadable"
    if isinstance(e, _TlsSetupFailed):
        return "network"
    # Every caller passes one of the types above or _PutFailed or TransferFailed.
    status = e.args[1] if isinstance(e, UploadPaused) else e.args[0]
    if status is not None:
        return "http"
    # Storage gave no answer; a resumable transfer fails without one only on a
    # chunk size in the start answer that it cannot use.
    return "unknown" if isinstance(e, TransferFailed) else "network"


def _report_stopped(target, sign_in, start: dict, report: dict, *, canceling: bool = False) -> None:
    """Tell Comfy Cloud a transfer stopped for good, when start says it takes such reports.

    A stored sign-in sends the token the upload already holds: reading the
    store could wait on its lock or refresh the token, and the report runs on Ctrl+C.
    """
    import threading
    import urllib.parse

    version = start.get("upload_analytics_version")
    if type(version) is not int or version != 1 or tracking._telemetry_disabled_by_env():
        return
    if sign_in is not None and not sign_in.held:
        return
    url = target.url("assets", "uploads", urllib.parse.quote(start["upload_id"], safe=""), "events")
    authed = target if sign_in is None else replace(target, auth_token=sign_in.held)

    def send() -> None:
        try:
            http_request(url, authed, method="POST", body={**_UPLOAD_CLIENT, **report}, timeout=_REPORT_TIMEOUT_SECONDS)
        except BaseException:  # noqa: BLE001 - nothing the report raises reaches the upload
            pass

    worker = threading.Thread(target=send, name="comfy-upload-stop-report", daemon=True)
    try:
        worker.start()
        deadline = _clock() + _REPORT_TIMEOUT_SECONDS
        while worker.is_alive():
            left = deadline - _clock()
            if left <= 0:
                break
            worker.join(min(_REPORT_WAIT_STEP_SECONDS, left))
    except KeyboardInterrupt:
        if not canceling:
            raise
    # A thread that cannot start sends nothing.
    except Exception:
        pass


def _wait_for_task(renderer, target, task_id: str, sign_in=None, *, reporter) -> dict:
    """Read the task until it ends, riding out a brief 5xx, 429 or dropped connection."""
    import http.client
    import urllib.error
    import urllib.parse

    from comfy_cli.command._cloud_errors import retry_after_from_headers

    url = target.url("tasks", urllib.parse.quote(task_id, safe=""))
    started, last_notice = _clock(), _clock()
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
                # Never past the end of the unanswered window.
                left = unanswered_since + _UNANSWERED_WINDOW_SECONDS - now
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
        if renderer.is_pretty() and _clock() - last_notice >= 30:
            reporter.waiting(int(_clock() - started))
            last_notice = _clock()
        _sleep(pause)


def _content_hash(value: Any) -> str | None:
    """``blake3:<hex>`` for any form Comfy Cloud reports a content hash in
    (bare hex, canonical ``blake3:<hex>``, or the stored ``<hex>.<ext>``);
    ``None`` for anything else, so an absent or garbled hash never matches."""
    m = _ANY_HASH_RE.match(value) if isinstance(value, str) else None
    return f"blake3:{m.group(1).lower()}" if m else None


def _models_named(renderer, target, name: str, folder: str, sign_in, *, strict: bool) -> list[dict] | None:
    """The models in ``folder`` named exactly ``name`` in this workspace's model
    library: every member's, and never the public catalog.

    These are the rows a run on Comfy Cloud loads a model from, by folder and
    name (the cloud's ``runtimemodels.Lookup`` reads the workspace without a
    user and without public models), so they are the rows the name already
    stands for. ``workspace_library=true`` asks for exactly that set; without
    it Comfy Cloud lists only the caller's own models plus public ones. A
    listing that does not fit on one page cannot show that none of them holds
    other bytes: with ``strict`` it refuses rather than guess, and without it
    the answer is ``None``, not known.
    """
    body = _call(
        renderer,
        target,
        "lookup",
        ("assets",),
        sign_in=sign_in,
        query=[
            # Comma-separated: Comfy Cloud refuses a repeated include_tags with a 400.
            ("include_tags", f"models,{folder}"),
            ("name_contains", name),
            ("workspace_library", "true"),
            # Comfy Cloud drops public models for a workspace library anyway;
            # said here too, so a server that ignores workspace_library never
            # answers with public models the run does not load.
            ("include_public", "false"),
            ("limit", _LOOKUP_LIMIT),
        ],
    )
    rows = body.get("assets")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise _unexpected(renderer, "/api/assets", "assets must be a list of objects", operation="lookup")
    if body.get("has_more") is True:
        if not strict:
            return None
        renderer.error(
            code="asset_upload_invalid",
            message=f"more than {_LOOKUP_LIMIT} models in {folder} have names containing {name!r}, "
            "so it cannot be checked that none of them is already named that with other bytes",
            hint="rename the file to a more distinctive name and upload it again",
            details={"reason": "name_unverified", "file": name, "type": folder},
        )
        raise typer.Exit(code=1)
    return [r for r in rows if r.get("name") == name]


def _stored_asset(target, sign_in, asset_id: str) -> dict | None:
    """Comfy Cloud's record of ``asset_id``, or ``None`` when it cannot be read."""
    import http.client
    import urllib.error
    import urllib.parse

    try:
        _, asset = _send(target, sign_in, target.url("assets", urllib.parse.quote(asset_id, safe="")), strict_json=True)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ResponseUnparseable):
        return None
    return asset if isinstance(asset, dict) else None


def _current_workspace(target, sign_in) -> str | None:
    """The id of the workspace this sign-in uploads into, or ``None`` when Comfy Cloud does not say."""
    import http.client
    import urllib.error

    try:
        _, body = _send(target, sign_in, target.url("workspaces", "current"), strict_json=True)
    except (urllib.error.URLError, OSError, http.client.HTTPException, ResponseUnparseable):
        return None
    ws = body.get("id") if isinstance(body, dict) else None
    return ws if isinstance(ws, str) and ws else None


def _loader_path(asset: dict | None) -> str | None:
    """The value a loader widget takes for the model, as Comfy Cloud's record
    gives it; ``None`` when there is no record or it gives none (Comfy Cloud
    could not derive one from the name)."""
    value = asset.get("loader_path") if isinstance(asset, dict) else None
    return value if isinstance(value, str) and value else None


@app.command(
    "upload",
    help="Upload one named .safetensors model file (up to 25 GB) into your Comfy Cloud model library. Run the same command again to resume a stopped upload. "
    "Succeeds only once Comfy Cloud's hash of the stored file matches this file's; --dry-run checks without uploading.",
)
@tracking.track_command("assets")
def upload_cmd(
    file: Annotated[Path, typer.Argument(help="The .safetensors file to upload.", show_default=False)],
    folder: Annotated[
        str, typer.Option("--type", help="Model folder to file it under (e.g. loras, checkpoints, vae).")
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Check the file, the folder and the library, and report the file's hashes and whether "
            "Comfy Cloud already has it; upload nothing.",
        ),
    ] = False,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    run_upload(file, folder, where=where, dry_run=dry_run, command="assets library upload")


def run_upload(
    file: Path,
    folder: str,
    *,
    where: str | None,
    dry_run: bool,
    command: str,
    workspace: str | None = None,
    expected_sha256: str | None = None,
    yes: bool = True,
    ctx: typer.Context | None = None,
    refuse_name_reuse: bool = False,
) -> None:
    """Upload ``file``, and only ``file``, into ``folder`` of the Comfy Cloud model library.

    Shared by ``comfy assets library upload`` and ``comfy model upload``. It
    succeeds only once Comfy Cloud's hash of the stored bytes is the file's
    own. ``workspace`` and ``expected_sha256`` pin an approval: when given, a
    sign-in to another workspace, or a file whose bytes are not the ones
    approved, uploads nothing. ``yes=False`` asks before the upload starts.
    The same bytes already under the name are never sent again.
    ``refuse_name_reuse`` refuses a name that already holds other bytes in
    the folder; without it the file is uploaded under that name, as a new
    version of it.
    """
    from comfy_cli.command.build_upload_progress import UploadProgressReporter
    from comfy_cli.command.model_upload_resume import (
        SessionExpired,
        SessionStore,
        TransferFailed,
        UploadPaused,
        transfer,
        valid_start,
    )
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.credentials import DataPlaneToken
    from comfy_cli.hashing import file_digests
    from comfy_cli.interaction import confirm

    def refuse(reason: str, message: str, **details: Any) -> typer.Exit:
        renderer.error(
            code="asset_upload_invalid", message=message, details={"reason": reason, "file": str(file), **details}
        )
        return typer.Exit(code=1)

    renderer = get_renderer()
    refusal = _local_refusal(file)
    if refusal is not None:
        raise refuse(*refusal)
    approved = None
    if expected_sha256 is not None:
        m = _SHA256_RE.match(expected_sha256)
        if m is None:
            raise refuse("bad_expected_sha256", f"--expected-sha256 {expected_sha256!r} is not a 64-hex SHA-256")
        approved = m.group(1).lower()
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

    # The person approved an upload into one workspace; a sign-in that has
    # since moved to another must not put the file there instead.
    workspace_id = None
    if workspace is not None:
        workspace_id = _current_workspace(target, sign_in)
        if workspace_id != workspace:
            renderer.error(
                code="asset_upload_wrong_workspace",
                message=(
                    f"this sign-in uploads into workspace {workspace_id}, not {workspace}"
                    if workspace_id is not None
                    else f"Comfy Cloud did not say which workspace this sign-in uploads into, "
                    f"so it cannot be checked that it is {workspace}"
                ),
                hint="check the workspace with `comfy cloud status`; sign in to the approved one with "
                "`comfy cloud login`, then run the upload again",
                details={
                    "reason": "workspace_mismatch" if workspace_id is not None else "workspace_unknown",
                    "workspace": workspace,
                    "current_workspace": workspace_id,
                },
            )
            raise typer.Exit(code=1)

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
        if renderer.is_pretty():
            renderer.info(f"Hashing {file.name}...")
        # One read gives the hashes, the size and each block's digest; the PUT
        # sends only bytes that match these blocks (`_HashedBytes`).
        digests = file_digests(file)
        modified_ns = file.stat().st_mtime_ns
    except OSError as e:  # removed or locked since it was checked; nothing is sent yet
        raise refuse(*_file_problem(file, e)) from None
    local_hash, sha256, size = digests.blake3, digests.sha256, digests.size
    if approved is not None and approved != sha256:
        raise refuse(
            "file_changed",
            f"{file.name}'s bytes are not the ones --expected-sha256 names; the file changed since it was checked",
            expected_sha256=approved,
            sha256=sha256,
        )

    # A run loads the newest model of a name in a folder, so a second file
    # under a name already in use changes what that name loads for every
    # workflow in the workspace: `comfy model upload` refuses it, while
    # `comfy assets library upload` keeps uploading it as a new version. The
    # same bytes under the name are already there and need no upload.
    named = _models_named(renderer, target, file.name, folder, sign_in, strict=refuse_name_reuse) or []
    same = [r for r in named if _content_hash(r.get("hash")) == local_hash]
    others = [r for r in named if _content_hash(r.get("hash")) != local_hash]
    if others and refuse_name_reuse:
        raise refuse(
            "name_in_use",
            f"{folder}/{file.name} is already in your Comfy Cloud library with different bytes; "
            "uploading would change what that name loads",
            hash=local_hash,
            existing=[{"id": r.get("id"), "hash": r.get("hash")} for r in others],
        )
    # Uploading again is what makes these bytes the newest under the name,
    # so a name that also holds other bytes, or a lookup too long to read,
    # is not "already present" for the older command.
    present = same[0] if same and not others else None

    def report(asset_id, *, task_id=None, uploaded=False, name=file.name, asset=None) -> dict:
        return {
            "id": asset_id,
            "name": name,
            "folder": folder,
            "loader_path": _loader_path(asset),
            "hash": local_hash,
            "sha256": sha256,
            "size": size,
            "workspace_id": workspace_id,
            "task_id": task_id,
            "uploaded": uploaded,
            "already_present": present is not None,
            "dry_run": dry_run,
        }

    if dry_run or present is not None:
        payload = report(present.get("id") if present is not None else None, asset=present)
        if renderer.is_pretty():
            if present is not None:
                renderer.success(f"{folder}/{file.name} is already in your library with these bytes; nothing to upload")
            else:
                renderer.info(f"Would upload {file.name} ({size} bytes) to {folder}")
            renderer.info(f"sha256: {sha256}\nhash: {local_hash}")
        renderer.emit(payload, command=command, where="cloud", changed=False)
        return

    into = f" in workspace {workspace_id}" if workspace_id else ""
    if not confirm(
        f"Upload {file} ({size} bytes, sha256 {sha256}) to {folder} on Comfy Cloud{into}?",
        yes=yes,
        error_code="model_upload_needs_confirm",
        details={"file": str(file), "type": folder, "sha256": sha256, "workspace_id": workspace_id},
        ctx=ctx,
    ):
        # Only a prompt can decline, and one is allowed only in pretty mode,
        # where `emit` writes nothing; a machine caller is refused above.
        renderer.info("Nothing uploaded.")
        return

    item = _UploadItem(filename=file.name, size_bytes=size)
    tags = ["models", folder]
    reporter = UploadProgressReporter(renderer)
    reporter.plan([item], already_held=0)
    store = SessionStore(
        Path(ConfigManager().get_config_path()) / "model-uploads",
        {
            "server": target.url(),
            "path": str(file.resolve()),
            "size": size,
            "mtime_ns": modified_ns,
            "hash": local_hash,
            "folder": folder,
        },
    )
    identity = _upload_identity(target, sign_in) if store.has_saved() else None
    saved = store.load(identity)

    def fresh_start(resumable: bool) -> dict:
        body = {
            "filename": item.filename,
            "size_bytes": item.size_bytes,
            "tags": tags,
            "resumable": resumable,
            **_upload_analytics(),
        }
        return _call(renderer, target, "start", ("assets", "uploads"), body, sign_in=sign_in, resumable_start=resumable)

    def stopped(e: BaseException) -> None:
        report = {"size_bytes": size, "model_folder": folder, "reason": _stop_reason(e)}
        _report_stopped(target, sign_in, start, report, canceling=isinstance(e, KeyboardInterrupt))

    # Once ingest refuses a resumable start, the rest of the run asks for a single PUT.
    ask_resumable = True
    for attempt in range(1, _UPLOAD_STARTS + 1):
        resumed = saved is not None
        session_saved = resumed
        start = saved
        if start is None:
            try:
                start = fresh_start(ask_resumable)
            except _ResumableRefused:
                # A refusal of this start too, such as ingest's 2 GiB
                # FILE_TOO_LARGE for a single PUT, is shown as any other.
                ask_resumable = False
                start = fresh_start(False)
        saved = None
        if not valid_start(start):
            raise _unexpected(renderer, "/api/assets/uploads", "expected upload_id, upload_url, method PUT and headers")
        upload_id, upload_url, headers = start["upload_id"], start["upload_url"], start["headers"]
        resumable = start.get("upload_mode") == "resumable"
        if resumable and not resumed:
            identity = identity or _upload_identity(target, sign_in)
            session_saved = store.save(start, identity)
            if not session_saved:
                renderer.warn("This upload cannot be saved for a later run; connection retries still work.")
        # Whether the bytes went on from a stored offset; None until the transfer ends.
        recovered = None
        try:
            with reporter.uploading(item, 1, 1) as progress:
                if resumable:
                    recovered = transfer(
                        start,
                        size,
                        lambda offset, length: _read_range(file, digests, offset, length),
                        progress,
                        resumed=resumed,
                        clock=_clock,
                        sleep=_sleep,
                    )
                else:
                    _put(upload_url, headers, file, progress, digests)
                    recovered = False
            task_id = _complete_upload(
                renderer,
                target,
                sign_in,
                upload_id,
                {"filename": item.filename, "tags": tags, **_upload_analytics(resumed=recovered)},
                resumable=resumable,
            ).get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise _unexpected(renderer, "/api/assets/uploads/complete", "no task_id", upload_id=upload_id)
            store.clear()
            break
        except UploadPaused as e:
            from comfy_cli.output.progress import human_bytes

            next_run = (
                "Run the same command again to continue while the saved session remains valid."
                if session_saved
                else "This session was not saved. Run the same command again to start a new upload."
            )
            renderer.error(
                code="asset_upload_failed",
                message=f"Upload paused at {human_bytes(e.args[0])} of {human_bytes(size)}. {next_run}",
                details={
                    "reason": "upload_paused",
                    "upload_id": upload_id,
                    "bytes_held": e.args[0],
                    "session_saved": session_saved,
                },
            )
            stopped(e)
            raise typer.Exit(code=1) from None
        except _FileUnreadable as e:
            renderer.error(
                code="asset_upload_failed",
                message=f"could not read {file} while sending it to storage ({e.args[0].strerror or e.args[0]})",
                hint="check that the file is still there and readable, then upload it again; "
                "nothing was added to the library",
                details={"reason": "file_unreadable", "upload_id": upload_id, "attempts": attempt},
            )
            stopped(e)
            raise typer.Exit(code=1) from None
        except _FileChanged as e:
            store.clear()
            # Only the blocks before the changed one were sent, and the
            # upload is never completed, so the staged bytes never become an asset.
            renderer.error(
                code="asset_upload_failed",
                message=f"{file.name} changed after it was hashed; none of the changed bytes were sent",
                hint="wait until nothing is writing the file, then upload it again; nothing was added to the library",
                details={"reason": "file_changed", "upload_id": upload_id, "hash": local_hash, "attempts": attempt},
            )
            stopped(e)
            raise typer.Exit(code=1) from None
        except _TlsSetupFailed as e:
            renderer.error(
                code="asset_upload_failed",
                message=f"could not set up a secure connection to storage ({e.args[0]})",
                hint="check the storage certificate and proxy, and point REQUESTS_CA_BUNDLE (or CURL_CA_BUNDLE) "
                "at a trusted certificate bundle that exists, or unset it; then upload again; "
                "nothing was added to the library",
                details={"reason": "tls_setup", "upload_id": upload_id, "attempts": attempt},
            )
            stopped(e)
            raise typer.Exit(code=1) from None
        except (_PutFailed, TransferFailed, SessionExpired) as e:
            expired = isinstance(e, SessionExpired)
            status = 404 if expired else e.args[0]
            refused = (
                resumable
                and isinstance(e, TransferFailed)
                and status is not None
                and 400 <= status < 500
                and status not in (408, 429)
            )
            if expired or refused:
                store.clear()
            if (
                expired or refused or (not resumable and (status is None or status >= 500))
            ) and attempt < _UPLOAD_STARTS:
                renderer.warn("The upload to storage did not go through; starting a fresh upload.")
                # Each upload id is one attempt, so the one given up here reports
                # before the fresh upload; complete's own failures are the server's to see.
                if recovered is None:
                    stopped(e)
                continue
            answer = "no response" if status is None else f"HTTP {status}"
            renderer.error(
                code="asset_upload_failed",
                message=f"sending {item.filename} to storage failed ({answer})",
                hint="retry; nothing was added to the library",
                details={"status": status, "upload_id": upload_id, "attempts": attempt},
            )
            # Complete's own failures (a session it no longer knows) are the server's to see.
            if recovered is None:
                stopped(e)
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt as e:
            if recovered is None:
                stopped(e)
            raise

    reporter.storing(task_id)
    task = _wait_for_task(renderer, target, task_id, sign_in, reporter=reporter)

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

    # The upload counts only when Comfy Cloud's own hash of what it stored is
    # this file's. The task reports it; the asset itself is read when not.
    stored = _content_hash(result.get("hash"))
    asset = None
    if stored is None:
        asset = _stored_asset(target, sign_in, asset_id)
        stored = _content_hash(asset.get("hash")) if asset is not None else None
    if stored != local_hash:
        mismatch = stored is not None
        renderer.error(
            code="asset_upload_unverified",
            message=(
                f"Comfy Cloud stored asset {asset_id} with hash {stored}, not {item.filename}'s {local_hash}"
                if mismatch
                else f"Comfy Cloud stored asset {asset_id} but its hash could not be read to check it"
            ),
            hint=(
                f"delete asset {asset_id} from your Comfy Cloud model library before running a workflow that "
                f"loads {item.filename}, then upload it again"
                if mismatch
                else f"check asset {asset_id}'s hash with `comfy assets library ls --name {item.filename} "
                "--tags models` before running a workflow that loads it"
            ),
            details={
                "reason": "hash_mismatch" if mismatch else "hash_unavailable",
                "asset_id": asset_id,
                "task_id": task_id,
                "hash": local_hash,
                "stored_hash": stored,
            },
        )
        raise typer.Exit(code=1)

    # The bytes are verified; the record is read only for its loader_path,
    # which stays null when it cannot be read.
    if asset is None:
        asset = _stored_asset(target, sign_in, asset_id)
    name = result["filename"] if isinstance(result.get("filename"), str) else item.filename
    payload = report(asset_id, task_id=task_id, uploaded=True, name=name, asset=asset)
    if renderer.is_pretty():
        renderer.success(f"Added {name} to {folder}")
        renderer.info(f"asset id: {asset_id}\nsha256: {sha256}\nhash: {local_hash} (matches the stored file)")
    renderer.emit(payload, command=command, where="cloud", changed=True)
