"""Parse and classify ComfyUI execution failures into envelope-ready fields.

The cloud's ``/api/jobs/<id>`` detail response carries a failed job's cause as
``error_message`` — a JSON-encoded string holding ``exception_message``,
``exception_type``, ``node_id``, ``node_type`` and a full Python traceback.
Embedding that string verbatim in an error envelope buries the one line a
consumer needs under kilobytes of server traceback, and historically it was
duplicated into both ``error.message`` and ``details``. The local WebSocket
``execution_error`` event carries the same fields as an already-decoded dict.

:func:`classify` normalizes either shape into a single envelope-ready verdict
and routes known-transient failures to their own error code so callers (and
agents) can tell "resubmit" apart from "your workflow is broken". The full
raw traceback intentionally stays out of the verdict — it remains available
on the server record via ``comfy jobs status <prompt_id>``.
"""

from __future__ import annotations

import json
from typing import Any

# Marker for the cloud API-node session-token expiry ("Unauthorized: Please
# login first to use this node", sometimes wrapped in "Polling aborted due to
# error: ..."). The token lives server-side and expires mid-execution on
# long-running API-node jobs; resubmitting the same workflow succeeds, while
# re-running `comfy cloud login` changes nothing.
_TRANSIENT_AUTH_MARKER = "please login first to use this node"

# How many trailing traceback frames survive into envelope details.
_TRACEBACK_TAIL_FRAMES = 2


def parse_error_message(raw: Any) -> dict[str, Any]:
    """Normalize a job-failure payload into flat fields.

    Accepts the cloud's JSON-encoded ``error_message`` string, an
    already-decoded dict (local WebSocket event), plain text, or None.
    Always returns at least ``{"exception_message": str}``.
    """
    data: dict[str, Any]
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return {"exception_message": raw.strip()}
        if not isinstance(decoded, dict):
            return {"exception_message": raw.strip()}
        data = decoded
    elif raw is None:
        return {"exception_message": ""}
    else:
        return {"exception_message": str(raw)}

    node_id = data.get("node_id")
    out: dict[str, Any] = {
        "exception_message": str(data.get("exception_message") or "").strip(),
        "exception_type": data.get("exception_type"),
        # Contract: node_id is always a string in envelope details, even if
        # the server sends an int.
        "node_id": str(node_id) if node_id is not None else None,
        "node_type": data.get("node_type"),
    }
    tb = data.get("traceback") or []
    if isinstance(tb, str):
        tb = [tb]
    if tb:
        out["traceback_tail"] = [str(frame) for frame in tb[-_TRACEBACK_TAIL_FRAMES:]]
    return out


def is_transient_auth(text: Any) -> bool:
    """True when the failure text is the server-side API-node token expiry."""
    return _TRANSIENT_AUTH_MARKER in str(text or "").lower()


def classify(raw: Any) -> dict[str, Any]:
    """Build an envelope-ready ``{"code", "message", "hint", "details"}`` verdict.

    ``message`` is the one-line cause prefixed with the failing node;
    ``details`` carries the structured fields plus a short traceback tail —
    never the full raw server traceback.
    """
    parsed = parse_error_message(raw)
    cause = parsed.get("exception_message") or "ComfyUI reported an execution error."
    node_type = parsed.get("node_type")
    node_id = parsed.get("node_id")
    if node_type:
        where = f"{node_type} (node {node_id})" if node_id is not None else str(node_type)
        message = f"{where}: {cause}"
    else:
        message = cause
    details = {k: v for k, v in parsed.items() if v not in (None, "", [])}
    if is_transient_auth(cause):
        return {
            "code": "transient_auth",
            "message": message,
            "hint": (
                "an API node's server-side session token expired mid-execution — transient; "
                "resubmit the same workflow and it will succeed on retry. "
                "Local credentials are fine: `comfy cloud login` will not help"
            ),
            "details": details,
        }
    return {
        "code": "execution_error",
        "message": message,
        "hint": (
            "inspect details (node_type/node_id/exception_type); "
            "full server traceback: `comfy --json jobs status <prompt_id>`"
        ),
        "details": details,
    }
