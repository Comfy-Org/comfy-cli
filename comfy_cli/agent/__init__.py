"""``comfy agent`` — read the local comfy agent's approvals.

The local agent (the ``comfy-agent`` binary on the user's own machine) runs
every command in a sandbox and every file tool behind a fence. What the user
lets it reach beyond the defaults lives in two files in its data dir:

- ``permissions.json``  — folders (``{"paths": [{"path", "reason", "approved_at"}]}``)
  and the agent's own requests waiting for a human
  (``{"pending": [{"id", "kind", "target", "reason", "requested_at"}]}``)
- ``egress-allow.json`` — hosts   (``{"hosts": [{"host", "reason", "approved_at"}]}``)

The agent's ``request_path`` / ``request_host`` tools only append to
``pending``: the model can ask, and injected content in a workflow can make it
ask, but nothing it can call moves a target into an approved list. The user
approves or denies a request in the agent panel, a channel the model cannot
reach, and the agent vets the target and records the approval itself. This
module only reads those files: the model can run this CLI, so nothing here
writes an approval. The agent is the only writer of these files: it loads them
at start and saves its whole state on every change. The data dir is
``AGENT_DATA_DIR`` when set, else ``~/.comfy-agent`` — the same rule the agent
and its launcher use.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

PERMISSIONS_FILE = "permissions.json"
EGRESS_ALLOW_FILE = "egress-allow.json"
DISCOVERY_FILE = "agent.json"


class StateError(ValueError):
    """A file in the agent's data dir cannot be read, or is not the shape the agent writes."""


# A request id as the agent mints it: 16 lower-case hex characters.
_REQUEST_ID = re.compile(r"^[0-9a-f]{16}$")
_REQUEST_KINDS = ("path", "host")


def data_dir(explicit: str | None = None) -> Path:
    """The agent's data dir: ``--data-dir``, else ``AGENT_DATA_DIR``, else ``~/.comfy-agent``."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("AGENT_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".comfy-agent"


def _read_json(path: Path) -> dict:
    """The file as a JSON object, ``{}`` when absent; :class:`StateError` when unreadable or not an object."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StateError(f"{path}: cannot read: {exc}") from exc
    if not isinstance(data, dict):
        raise StateError(f"{path}: expected a JSON object, found {type(data).__name__}")
    return data


def _load(path: Path, key: str) -> tuple[dict, list[dict]]:
    """The file's object and its ``key`` list of entries; a non-list or a non-object entry is a broken file."""
    data = _read_json(path)
    raw = data.get(key)
    if raw is None:
        return data, []
    if not isinstance(raw, list) or not all(isinstance(e, dict) for e in raw):
        raise StateError(f"{path}: {key!r} must be a list of objects")
    return data, list(raw)


def _is_request(e: object) -> bool:
    return (
        isinstance(e, dict)
        and isinstance(e.get("id"), str)
        and bool(_REQUEST_ID.match(e["id"]))
        and e.get("kind") in _REQUEST_KINDS
        and isinstance(e.get("target"), str)
    )


def _pending(data: dict) -> list[dict]:
    """The well-formed requests in a permissions object; a malformed entry or a non-list is skipped, not an error.

    A request the agent wrote badly is nothing a human can act on, and a
    broken ``pending`` must not lock ``comfy agent permissions`` out of the
    approvals the same file holds.
    """
    raw = data.get("pending")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if _is_request(e)]


@dataclass(frozen=True)
class AgentState:
    data_dir: Path
    running: bool
    port: int | None
    paths: list[dict]
    hosts: list[dict]
    pending: list[dict]


def read_state(root: Path) -> AgentState:
    """What the data dir says: whether an agent published itself, the approvals, and the requests waiting.

    Raises :class:`StateError` when any of the three files is unreadable or
    not the shape the agent writes.
    """
    disc = _read_json(root / DISCOVERY_FILE)
    raw_port = disc.get("port")
    # bool is an int subclass; a JSON true must not become port 1.
    port = raw_port if type(raw_port) is int and 1 <= raw_port <= 65535 else None
    perms, paths = _load(root / PERMISSIONS_FILE, "paths")
    _, hosts = _load(root / EGRESS_ALLOW_FILE, "hosts")
    return AgentState(data_dir=root, running=bool(disc), port=port, paths=paths, hosts=hosts, pending=_pending(perms))
