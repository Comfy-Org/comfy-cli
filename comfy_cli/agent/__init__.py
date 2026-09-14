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

The checks here are a courtesy, not the fence: the agent's sandbox deny list
and egress proxy decide what a grant can open, and they win over anything
written to these files. Vetting here means a grant the agent would never
honour is refused with a reason instead of recorded and silently ignored.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from comfy_cli.file_utils import atomic_write_text

PERMISSIONS_FILE = "permissions.json"
EGRESS_ALLOW_FILE = "egress-allow.json"
DISCOVERY_FILE = "agent.json"


class StateError(ValueError):
    """A file in the agent's data dir cannot be read, or is not the shape the agent writes."""


# Folders a terminal grant may never name, in either direction. This mirrors
# the agent's own deny list (``CredentialDenyDirs`` in the cloud repo,
# services/agent/internal/runner/sandbox_seatbelt_profile.go) — the credential
# and browser-profile stores of every platform, denied on every platform so a
# grant vetted on one OS means the same on another — plus ``.azure`` and the
# whole of ``.docker``, where the CLI is stricter than the agent. A folder that
# contains one of these (the home folder, above all) is refused whether or not
# the store exists yet: ``ssh-keygen`` or ``aws configure`` would otherwise
# create keys inside a folder the agent already reaches.
_DENIED_HOME_SUBDIRS = (
    # Every platform: dotfile stores and tokens for services the agent's egress
    # allow list lets a child reach (Hugging Face, GitHub, PyPI, npm, Docker).
    ".ssh",
    ".aws",
    ".gnupg",
    ".azure",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
    ".git-credentials",
    ".pypirc",
    ".npmrc",
    ".config/gh",
    ".cache/huggingface/token",
    # The agent's and the CLI's own state (tokens, sessions, the bearer token).
    ".comfy-agent",
    ".comfy-cli",
    ".config/comfy-cli",
    "Library/Application Support/comfy-cli",
    "AppData/Roaming/comfy-cli",
    "AppData/Local/comfy-cli",
    # macOS keychain and browser profiles.
    "Library/Keychains",
    "Library/Application Support/Google/Chrome",
    "Library/Application Support/Firefox",
    "Library/Safari",
    "Library/Containers/com.apple.Safari",
    # Linux browser profiles and keyrings.
    ".mozilla/firefox",
    ".config/google-chrome",
    ".config/chromium",
    ".local/share/keyrings",
    # Windows browser profiles, gcloud and the DPAPI credential vaults.
    "AppData/Roaming/Mozilla/Firefox/Profiles",
    "AppData/Local/Google/Chrome/User Data",
    "AppData/Local/Microsoft/Edge/User Data",
    "AppData/Roaming/gcloud",
    "AppData/Roaming/Microsoft/Credentials",
    "AppData/Local/Microsoft/Credentials",
)

# A name the agent denies wherever it lives (its ``.env`` rule, applied to
# every path component): a folder called ``.env`` cannot be granted, and a
# ``.env`` file inside an approved folder stays unreadable regardless.
_DENIED_NAME = re.compile(r"^\.env(\..*)?$|^\.envrc$", re.IGNORECASE)

# Hosts the agent's egress proxy refuses even when they are in the allow file
# (``sandboxKnownRefusedHosts`` in the cloud repo): Comfy's telemetry endpoints
# and the shared object-storage roots anyone can create a bucket on. Recording
# one would change nothing at the proxy, so it is refused here with the reason.
_REFUSED_HOSTS = {
    "t.comfy.org": "a telemetry endpoint",
    "api.mixpanel.com": "a telemetry endpoint",
    "r2.cloudflarestorage.com": "shared object storage anyone can create a bucket on",
    "storage.googleapis.com": "shared object storage anyone can create a bucket on",
}

# One DNS label: letters, digits and inner hyphens, at most 63 characters.
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


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


@dataclass(frozen=True)
class AgentState:
    data_dir: Path
    running: bool
    port: int | None
    paths: list[dict]
    hosts: list[dict]


def read_state(root: Path) -> AgentState:
    """What the data dir says: whether an agent published itself, and the approvals.

    Raises :class:`StateError` when any of the three files is unreadable or
    not the shape the agent writes.
    """
    disc = _read_json(root / DISCOVERY_FILE)
    raw_port = disc.get("port")
    # bool is an int subclass; a JSON true must not become port 1.
    port = raw_port if type(raw_port) is int and 1 <= raw_port <= 65535 else None
    _, paths = _load(root / PERMISSIONS_FILE, "paths")
    _, hosts = _load(root / EGRESS_ALLOW_FILE, "hosts")
    return AgentState(data_dir=root, running=bool(disc), port=port, paths=paths, hosts=hosts)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fold(p: Path) -> str:
    """A path in the form its filesystem compares: case folded on Windows and macOS.

    The default macOS volume (APFS, HFS+) is case-insensitive, and ``resolve``
    does not correct case on POSIX, so ``~/.SSH`` names the same folder as
    ``~/.ssh`` there. Folding on darwin closes that; on a case-sensitive volume
    it only refuses a little more.
    """
    s = os.path.normcase(str(p))
    if sys.platform == "darwin":
        s = s.lower()
    return s


def _same_or_under(child: Path, parent: Path) -> bool:
    c, p = _fold(child), _fold(parent)
    if c == p:
        return True
    return c.startswith(p.rstrip(os.sep) + os.sep)


def vet_path(raw: str, root: Path | None = None) -> Path:
    """An approvable folder: absolute, existing, not the disk, not a credential store or a parent of one.

    Returns the folder with symlinks resolved: the checks below run on the real
    location, the way the agent's own fence does, so a link named elsewhere
    that points into a credential store is refused and what gets recorded is
    the folder the agent will actually open. ``root`` is the agent's data dir:
    it, anything under it, and any folder that contains it are refused, so a
    grant can never cover the agent's own state (``--data-dir /tmp/x --path /tmp``).
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("a folder is required")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError(f"{raw!r} is not an absolute path")
    p = Path(os.path.normpath(p))
    try:
        # is_dir() raises on EACCES on Python 3.10–3.12 rather than answering False.
        if not p.is_dir():
            raise ValueError(f"{p} is not an existing folder")
        p = p.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{p} cannot be opened: {exc}") from exc
    if p.parent == p:
        raise ValueError(f"{p} is the whole filesystem and stays closed")
    for part in p.parts:
        if _DENIED_NAME.match(part):
            raise ValueError(f"{p} stays closed: a .env name is denied wherever it lives")
    home = Path.home()
    for rel in _DENIED_HOME_SUBDIRS:
        # Both spellings: the lexical one and, when the store is a symlink to
        # somewhere else, the folder it really is.
        lexical = Path(os.path.normpath(home / rel))
        for denied in {lexical, lexical.resolve()}:
            if _same_or_under(p, denied):
                raise ValueError(f"{p} stays closed: it is a credential or agent-state folder")
            if _same_or_under(denied, p):
                raise ValueError(f"{p} stays closed: it contains {denied}, a credential or agent-state folder")
    if root is not None:
        agent_root = Path(os.path.normpath(root.expanduser()))
        try:
            agent_root = agent_root.resolve()
        except OSError:
            pass
        if _same_or_under(p, agent_root) or _same_or_under(agent_root, p):
            raise ValueError(f"{p} stays closed: it is, contains, or is inside the agent's own data dir {agent_root}")
    return p


def _as_ip(h: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        return None


def vet_host(raw: str) -> str:
    """One host the agent's proxy could match: a full host name or a routable address, lower-case, bare.

    The proxy matches an approved host exactly (no wildcards, no suffixes), so
    ``*`` and a bare top-level domain would record nothing it could ever use;
    loopback and link-local addresses would open this machine's own services
    and cloud metadata endpoints through the proxy; and the known-refused hosts
    are refused there whatever the file says. Each is refused here with why.
    """
    h = raw.strip()
    if not h:
        raise ValueError("a host is required")
    if "://" in h:
        parts = urlsplit(h)
        if parts.username is not None or parts.password is not None:
            raise ValueError(f"{raw!r} carries credentials; give the host name only")
        h = parts.hostname or ""
    else:
        if "@" in h:
            raise ValueError(f"{raw!r} carries credentials (user@host); give the host name only")
        h = h.split("/", 1)[0]
        if h.startswith("["):
            h = h[1:].split("]", 1)[0]
        elif h.count(":") == 1:
            h = h.split(":", 1)[0]
    h = h.strip().lower().rstrip(".")
    if not h:
        raise ValueError(f"{raw!r} is not a host name")
    if "," in h:
        raise ValueError(f"{raw!r} names more than one host; pass one --host per host")
    if "*" in h:
        raise ValueError(f"{raw!r} is a wildcard; the agent matches an approved host exactly, so name the host")
    ip = _as_ip(h)
    if ip is not None:
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved:
            raise ValueError(
                f"{h} is a loopback, link-local or reserved address: this machine's own services "
                "and cloud metadata endpoints are never opened through the agent's proxy"
            )
        return str(ip)
    try:
        h = h.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"{raw!r} is not a host name") from exc
    labels = h.split(".")
    if len(labels) < 2 or h == "localhost":
        raise ValueError(
            f"{h!r} has no domain: a bare name or top-level domain cannot be allowed; give the full host name"
        )
    if len(h) > 253 or not all(_LABEL.match(label) for label in labels):
        raise ValueError(f"{raw!r} is not a host name")
    if h in _REFUSED_HOSTS:
        raise ValueError(f"{h} stays refused: it is {_REFUSED_HOSTS[h]}, and the agent never opens it")
    return h


def allow_path(root: Path, folder: str, reason: str) -> tuple[Path, bool]:
    """Record an approved folder. Returns (folder, added); added is False when it was already there.

    Raises ``ValueError`` for a folder that cannot be approved,
    :class:`StateError` for a permissions file the agent would not load, and
    ``OSError`` when the data dir cannot be written.
    """
    p = vet_path(folder, root=root)
    file = root / PERMISSIONS_FILE
    data, entries = _load(file, "paths")
    if any(_same_path(str(e.get("path", "")), str(p)) for e in entries):
        return p, False
    entries.append({"path": str(p), "reason": reason, "approved_at": _now()})
    # Keep whatever else the agent put at the top level; only the list changes.
    _write(file, {**data, "paths": entries})
    return p, True


def allow_host(root: Path, host: str, reason: str) -> tuple[str, bool]:
    """Record an approved host. Returns (host, added). Raises as :func:`allow_path` does."""
    h = vet_host(host)
    file = root / EGRESS_ALLOW_FILE
    data, entries = _load(file, "hosts")
    if any(str(e.get("host", "")).lower() == h for e in entries):
        return h, False
    entries.append({"host": h, "reason": reason, "approved_at": _now()})
    _write(file, {**data, "hosts": entries})
    return h, True


def _same_path(a: str, b: str) -> bool:
    return _fold(Path(os.path.normpath(a))) == _fold(Path(os.path.normpath(b)))


def _write(file: Path, payload: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(file, json.dumps(payload, indent=2) + "\n")
    try:
        os.chmod(file, 0o600)
    except OSError:
        pass
