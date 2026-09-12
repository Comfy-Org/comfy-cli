"""``comfy agent`` — read and extend the local comfy agent's approvals.

The local agent (the ``comfy-agent`` binary on the user's own machine) runs
every command in a sandbox and every file tool behind a fence. What the user
lets it reach beyond the defaults lives in two files in its data dir, written
by the agent's own ``allow_path`` / ``allow_host`` tools when the user says yes
in chat:

- ``permissions.json``  — folders (``{"paths": [{"path", "reason", "approved_at"}]}``)
- ``egress-allow.json`` — hosts   (``{"hosts": [{"host", "reason", "approved_at"}]}``)

This module writes the same files from a terminal. A running agent re-reads
them every ~20 s; otherwise they apply at its next start. The data dir is
``AGENT_DATA_DIR`` when set, else ``~/.comfy-agent`` — the same rule the agent
and its launcher use.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from comfy_cli.file_utils import atomic_write_text

PERMISSIONS_FILE = "permissions.json"
EGRESS_ALLOW_FILE = "egress-allow.json"
DISCOVERY_FILE = "agent.json"

# Folders a terminal grant may never name, in either direction: the same
# credential-store rule the agent's store enforces, so a grant that the agent
# would refuse from chat cannot be smuggled in from the CLI.
_DENIED_HOME_SUBDIRS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".azure",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
    ".comfy-agent",
    "Library/Keychains",
    "Library/Application Support/comfy-cli",
    ".config/comfy-cli",
    "AppData/Local/comfy-cli",
    "AppData/Roaming/Microsoft/Credentials",
)


def data_dir(explicit: str | None = None) -> Path:
    """The agent's data dir: ``--data-dir``, else ``AGENT_DATA_DIR``, else ``~/.comfy-agent``."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("AGENT_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".comfy-agent"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: cannot read: {exc}") from exc
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class AgentState:
    data_dir: Path
    running: bool
    port: int | None
    paths: list[dict]
    hosts: list[dict]


def read_state(root: Path) -> AgentState:
    """What the data dir says: whether an agent published itself, and the approvals."""
    disc = _read_json(root / DISCOVERY_FILE)
    raw_port = disc.get("port")
    # bool is an int subclass; a JSON true must not become port 1.
    port = raw_port if type(raw_port) is int and 1 <= raw_port <= 65535 else None
    paths = _read_json(root / PERMISSIONS_FILE).get("paths") or []
    hosts = _read_json(root / EGRESS_ALLOW_FILE).get("hosts") or []
    return AgentState(
        data_dir=root,
        running=bool(disc),
        port=port,
        paths=[p for p in paths if isinstance(p, dict)],
        hosts=[h for h in hosts if isinstance(h, dict)],
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def vet_path(raw: str) -> Path:
    """An approvable folder: absolute, existing, not the disk, not a credential store or a parent of one.

    Returns the folder with symlinks resolved: the checks below run on the real
    location, the way the agent's own fence does, so a link named elsewhere
    that points into a credential store is refused and what gets recorded is
    the folder the agent will actually open.
    """
    p = Path(raw.strip()).expanduser()
    if not raw.strip():
        raise ValueError("a folder is required")
    if not p.is_absolute():
        raise ValueError(f"{raw!r} is not an absolute path")
    p = Path(os.path.normpath(p))
    if not p.is_dir():
        raise ValueError(f"{p} is not an existing folder")
    try:
        p = p.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{p} cannot be resolved: {exc}") from exc
    if p.parent == p:
        raise ValueError(f"{p} is the whole filesystem and stays closed")
    home = Path.home().resolve()
    for rel in _DENIED_HOME_SUBDIRS:
        denied = (home / rel).resolve()
        if _same_or_under(p, denied):
            raise ValueError(f"{p} stays closed: it is a credential or agent-state folder")
        if _same_or_under(denied, p) and denied.exists():
            raise ValueError(f"{p} stays closed: it contains {denied}, a credential or agent-state folder")
    return p


def _same_or_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def vet_host(raw: str) -> str:
    """A bare host name, lower-case, no scheme, path or port."""
    h = raw.strip().lower()
    if not h:
        raise ValueError("a host is required")
    if "://" in h:
        h = urlsplit(h).hostname or ""
    h = h.split("/", 1)[0].split(":", 1)[0].rstrip(".")
    if not h or any(c.isspace() for c in h):
        raise ValueError(f"{raw!r} is not a host name")
    return h


def allow_path(root: Path, folder: str, reason: str) -> tuple[Path, bool]:
    """Record an approved folder. Returns (folder, added); added is False when it was already there."""
    p = vet_path(folder)
    file = root / PERMISSIONS_FILE
    data = _read_json(file)
    entries = [e for e in (data.get("paths") or []) if isinstance(e, dict)]
    if any(_same_path(str(e.get("path", "")), str(p)) for e in entries):
        return p, False
    entries.append({"path": str(p), "reason": reason, "approved_at": _now()})
    _write(file, {"paths": entries})
    return p, True


def allow_host(root: Path, host: str, reason: str) -> tuple[str, bool]:
    """Record an approved host. Returns (host, added)."""
    h = vet_host(host)
    file = root / EGRESS_ALLOW_FILE
    data = _read_json(file)
    entries = [e for e in (data.get("hosts") or []) if isinstance(e, dict)]
    if any(str(e.get("host", "")).lower() == h for e in entries):
        return h, False
    entries.append({"host": h, "reason": reason, "approved_at": _now()})
    _write(file, {"hosts": entries})
    return h, True


def _same_path(a: str, b: str) -> bool:
    if os.name == "nt":
        return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))
    return os.path.normpath(a) == os.path.normpath(b)


def _write(file: Path, payload: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(file, json.dumps(payload, indent=2) + "\n")
    try:
        os.chmod(file, 0o600)
    except OSError:
        pass
