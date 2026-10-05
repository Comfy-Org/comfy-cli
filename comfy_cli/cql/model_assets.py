"""Cloud model assets: a model file the catalog does not list but Cloud can load.

On Comfy Cloud, ``/api/object_info`` lists the curated model library only. A
model in the Cloud asset library (a file the user imported, or a public model
imported from Hugging Face) is absent from it, yet a job that names it runs:
the server resolves a model widget by looking the filename up among the
caller's ``models`` assets, public ones included, by exact name. Without this
module the catalog check rejects such a value with ``unknown_enum_value``, and
an edit is refused outright, although the workflow would run.

:func:`model_asset_exists` answers the question the server answers, the same
way: is there a ``models`` asset, owned by the caller or public, whose name is
exactly this value. It is consulted only AFTER a catalog miss on a model-file
value, so a workflow that validates today costs no request.

It is deterministic and conservative:

- Only the cloud target asks. A local ComfyUI has no asset library; its catalog
  is the whole truth, so nothing changes there.
- Only exact name matches count, mirroring the server. A near match does not.
- Any failure (no credential, network, HTTP error, unexpected body) answers
  ``False``, which keeps today's finding. A lookup that cannot run never turns
  a rejection into an acceptance.
- Answers are cached for the process, so a value is looked up at most once.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
from collections.abc import Callable
from typing import Any

#: A value that names a model file. Matches the loader extensions
#: :mod:`comfy_cli.model_variants` treats as model files.
MODEL_FILE = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)$", re.IGNORECASE)

#: Set to ``1`` to switch the lookup off (offline runs, tests).
DISABLE_ENV = "COMFY_CLI_NO_MODEL_ASSET_LOOKUP"

#: Rows requested per page. ``name_contains`` narrows the listing to names
#: containing the value; longer names that also contain it share the listing.
_PAGE = 100

#: Pages read before giving up. ``name_contains`` with a full filename rarely
#: needs a second page; the bound only stops a pathological listing.
_MAX_PAGES = 10

#: Response cap for one lookup page.
_MAX_BYTES = 8 << 20

#: Looks one name up; returns whether a ``models`` asset has exactly that name.
Lookup = Callable[[str], bool]

_lookup: Lookup | None = None
_lookup_resolved = False
_cache: dict[str, bool] = {}
#: The command's own ``--where`` (``None`` = the default route). Set by the
#: graph loaders through :func:`use_where`, so a command routed to cloud by its
#: flag looks assets up on cloud even when the default route is local.
_where: str | None = None
_installed = False


def set_lookup(lookup: Lookup | None) -> None:
    """Install ``lookup`` (``None`` = never found) and clear the cache. For tests
    and embedders; the CLI resolves its own lookup lazily. An installed lookup
    is kept across :func:`use_where`."""
    global _lookup, _lookup_resolved, _installed
    _lookup = lookup
    _lookup_resolved = True
    _installed = True
    _cache.clear()


def reset() -> None:
    """Forget the installed lookup, route and cache; the next call resolves afresh."""
    global _lookup, _lookup_resolved, _installed, _where
    _lookup = None
    _lookup_resolved = False
    _installed = False
    _where = None
    _cache.clear()


def use_where(where: str | None) -> None:
    """Route the lookup like the command that loads the catalog: its ``--where``
    flag, or the default route when ``None``. A change of route drops the
    resolved lookup and its cached answers."""
    global _where, _lookup, _lookup_resolved
    if where == _where:
        return
    _where = where
    if not _installed:
        _lookup = None
        _lookup_resolved = False
        _cache.clear()


def model_asset_exists(value: Any) -> bool:
    """Whether Cloud can load ``value`` as a model although the catalog lacks it."""
    # The server compares the widget value as is, so the lookup does too: it
    # is never trimmed, so " x.safetensors" is not the asset "x.safetensors".
    if not isinstance(value, str) or not MODEL_FILE.search(value):
        return False
    name = value
    if name in _cache:
        return _cache[name]
    lookup = _resolve_lookup()
    found = False
    if lookup is not None:
        try:
            found = bool(lookup(name))
        except Exception:  # noqa: BLE001 — any failure keeps today's finding
            found = False
    _cache[name] = found
    return found


def _resolve_lookup() -> Lookup | None:
    global _lookup, _lookup_resolved
    if not _lookup_resolved:
        _lookup_resolved = True
        _lookup = None if os.environ.get(DISABLE_ENV) == "1" else _cloud_lookup()
    return _lookup


def _cloud_lookup() -> Lookup | None:
    """The asset lookup for the routed target, or ``None`` off cloud."""
    try:
        from comfy_cli.target import resolve_target

        target = resolve_target(where=_where, allow_clear=False)
    except Exception:  # noqa: BLE001 — routing trouble means no lookup, not a crash
        return None
    if not target.is_cloud or not (target.auth_token or target.api_key):
        return None
    return lambda name: asset_name_exists(target, name)


def asset_name_exists(target, name: str) -> bool:
    """Ask ``target``'s ``/api/assets`` for a ``models`` asset named exactly ``name``.

    The same rule the server uses to load a model widget: the caller's assets
    and public ones, matched on the asset's name. ``name_contains`` narrows the
    listing server-side; the exact comparison is made here.
    """
    from comfy_cli.http import ResponseTooLarge, request_json

    offset = 0
    for _ in range(_MAX_PAGES):
        params = {
            "include_tags": "models",
            "include_public": "true",
            "name_contains": name,
            "limit": _PAGE,
            "offset": offset,
        }
        url = target.url("assets") + "?" + urllib.parse.urlencode(params)
        try:
            _, body = request_json(url, target, timeout=15.0, max_bytes=_MAX_BYTES)
        except (urllib.error.URLError, OSError, ValueError, ResponseTooLarge):
            return False
        if not isinstance(body, dict):
            return False
        rows = body.get("assets") or []
        if any(isinstance(a, dict) and a.get("name") == name for a in rows):
            return True
        offset += len(rows)
        total = body.get("total")
        if not rows or not isinstance(total, int) or offset >= total:
            return False
    return False
