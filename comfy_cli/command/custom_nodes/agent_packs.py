"""Find custom node packs the local Comfy agent says it created.

The local agent writes a `.comfy-agent-install` JSON record inside each folder
pack it creates. The record is signed with a key only the local agent holds, so comfy-cli can read
it but cannot verify it: a listed pack "says it was made by the agent".

The record is untrusted input: a cloned or shared pack can carry one. This
reader applies the same shape rules as the agent's own reader: a regular file
(no links), a size cap, a known format version, the identifying fields, and a
`pack` that names the folder it sits in.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

INSTALL_RECORD_NAME = ".comfy-agent-install"
RECORD_VERSION = 1
MAX_RECORD_BYTES = 8 << 10
# ComfyUI skips a pack whose folder name ends in `.disabled` (the agent's own
# off switch); ComfyUI-Manager moves a switched-off pack to custom_nodes/.disabled.
DISABLED_SUFFIX = ".disabled"
MANAGER_DISABLED_FOLDER = ".disabled"


@dataclass(frozen=True)
class AgentPack:
    pack: str
    folder: str
    state: str
    route: str
    created_at: str
    thread_id: str
    tool_call_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_record(pack_path: Path) -> dict | None:
    record_path = pack_path / INSTALL_RECORD_NAME
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        checked = os.lstat(record_path)
        if not stat.S_ISREG(checked.st_mode):
            return None
        fd = os.open(record_path, flags)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as f:
            opened = os.fstat(f.fileno())
            # Without O_NOFOLLOW (Windows) the name can be swapped for a link
            # or another file between lstat and open. Read only the file that
            # was checked.
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (checked.st_dev, checked.st_ino):
                return None
            raw = f.read(MAX_RECORD_BYTES + 1)
        if not hasattr(os, "O_NOFOLLOW"):
            # No atomic no-follow open here: confirm the name still points at
            # the same regular file, not at a link to it, after the read.
            after = os.lstat(record_path)
            if not stat.S_ISREG(after.st_mode) or (after.st_dev, after.st_ino) != (checked.st_dev, checked.st_ino):
                return None
    except OSError:
        return None
    if len(raw) > MAX_RECORD_BYTES:
        return None
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    version = record.get("v")
    if type(version) is not int or version != RECORD_VERSION:
        return None
    for field in ("pack", "route", "created_at", "mac"):
        if not isinstance(record.get(field), str) or not record[field]:
            return None
    return record


def _optional_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _names_pack(recorded: str, entry: str, under_manager_folder: bool) -> bool:
    if recorded == entry or recorded + DISABLED_SUFFIX == entry:
        return True
    if under_manager_folder:
        version = entry.removeprefix(recorded + "@")
        return version != entry and version != ""
    return False


def _scan(folder: Path, under_manager_folder: bool, prefix: str) -> list[AgentPack]:
    packs: list[AgentPack] = []
    try:
        if under_manager_folder and not stat.S_ISDIR(os.lstat(folder).st_mode):
            return packs
        entries = sorted(folder.iterdir(), key=lambda p: p.name)
    except OSError:
        return packs
    for entry in entries:
        if entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir():
            continue
        record = _read_record(entry)
        if record is None or not _names_pack(record["pack"], entry.name, under_manager_folder):
            continue
        if under_manager_folder:
            state = "disabled-by-manager"
        elif entry.name.endswith(DISABLED_SUFFIX):
            state = "disabled"
        else:
            state = "enabled"
        packs.append(
            AgentPack(
                pack=record["pack"],
                folder=prefix + entry.name,
                state=state,
                route=record["route"],
                created_at=record["created_at"],
                thread_id=_optional_str(record.get("thread_id")),
                tool_call_id=_optional_str(record.get("tool_call_id")),
            )
        )
    return packs


def find_agent_packs(custom_nodes: Path) -> list[AgentPack]:
    """Return the packs under `custom_nodes` whose record names them, sorted by pack."""
    packs = _scan(custom_nodes, False, "")
    packs += _scan(custom_nodes / MANAGER_DISABLED_FOLDER, True, MANAGER_DISABLED_FOLDER + "/")
    return sorted(packs, key=lambda p: (p.pack, p.folder))
