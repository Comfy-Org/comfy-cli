"""``comfy assets library`` — browse and borrow assets from Comfy Cloud's
asset library.

Mirrors the cloud-saved-workflow subcommands in ``workflow.py`` (``list``,
``get``, ...): thin Typer commands over ``cloud_http``'s shared helpers,
emitting a JSON envelope via the renderer. Cloud-only — there is no local
``/api/assets`` surface.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.command.cloud_http import (
    cloud_target_or_local_error,
    handle_cloud_http_error,
    http_request,
)
from comfy_cli.output.renderer import get_renderer

app = typer.Typer(help="Browse your Comfy Cloud asset library (list, borrow).")

# A Comfy Cloud content hash: BLAKE3, 32-byte digest, lowercase hex — bare or in
# the canonical ``blake3:<hex>`` wire form. Comfy Cloud's from-hash lookup
# matches every stored form of the content only for ``blake3:<hex>``.
_HASH_WITH_EXT_RE = re.compile(r"^(?:blake3:)?([0-9a-f]{64})\.[A-Za-z0-9]+$")


# A value that reads as a hex hash even when mangled: optional ``blake3:``,
# 16-80 hex chars (a garbled copy can gain or lose a few), optional extension.
_HEXISH_RE = re.compile(r"^(?:blake3:)?([0-9a-fA-F]{16,80})(?:\.[A-Za-z0-9]+)?$")
_HEX_DIGEST_RE = re.compile(r"^(?:blake3:)?([0-9a-f]{64})(?:\.[A-Za-z0-9]+)?$")
_MIN_SHARED_PREFIX = 4
_MAX_SUGGESTIONS = 3
# One page at the API's maximum, newest first. Never paged: a suggestion is a
# best-effort hint on an error path, so it gets exactly one bounded request.
_SUGGESTION_SCAN_LIMIT = 500
# The listing only feeds an optional hint, so it must not hold the error for
# the default 30s request timeout.
_SUGGESTION_TIMEOUT_SECONDS = 5.0


def _near_hash_suggestions(value: str, target) -> list[dict]:
    """Library assets whose hash shares the longest prefix (>= 4 hex chars) with
    a hash that was not found — an agent copying a 64-char hex string tends to
    keep the first few characters and garble the rest.

    ``[]`` unless ``value`` looks like a hex hash; that check runs first, so an
    ordinary file name costs no request. The listing is one request for the
    newest :data:`_SUGGESTION_SCAN_LIMIT` assets you own; any failure there
    yields ``[]`` so the not-found envelope is never lost to it.
    """
    import os.path
    import urllib.parse

    m = _HEXISH_RE.match(value)
    if not m:
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
        shared = len(os.path.commonprefix([wanted, hm.group(1)]))
        if shared >= _MIN_SHARED_PREFIX:
            scored.append((shared, r))
    scored.sort(key=lambda t: -t[0])  # stable: ties keep newest-first order
    return [
        {"hash": r["hash"], "name": r.get("name"), "id": r.get("id"), "shared_prefix": shared}
        for shared, r in scored[:_MAX_SUGGESTIONS]
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
    return f"blake3:{m.group(1)}" if m else value


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
        _, body = http_request(url, target)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise handle_cloud_http_error(renderer, e, operation="list") from e

    rows = (body or {}).get("assets") or []
    payload = {
        "count": len(rows),
        "assets": [
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
        ],
    }
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
    import urllib.error

    renderer = get_renderer()
    target = cloud_target_or_local_error(where, renderer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or ["input"]
    url = target.url("assets/from-hash")
    try:
        status, body = http_request(
            url, target, method="POST", body={"hash": _normalize_content_hash(hash), "tags": tag_list}
        )
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        # The parameterized helper, not `cloud_http`'s: that one hardcodes the
        # saved-workflow vocabulary, so a 404 here read "workflow not found
        # (ensure)" with a hint to list workflows — for a request that never
        # named a workflow. Seen in prod when an agent passed a file name where
        # the content hash belongs.
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

    b = body or {}
    payload = {
        "id": b.get("id"),
        "hash": b.get("hash", hash),
        "created_new": status == 201,
    }
    renderer.emit(payload, command="assets library ensure", where="cloud")
