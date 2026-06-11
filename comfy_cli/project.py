"""project/1 convention — pure discovery, layout, and journaling helpers.

A PROJECT is any directory holding a ``comfy.yaml`` marker whose parsed
``schema`` is ``"project/1"``::

    schema: project/1
    defaults:
      where: cloud

The convention is the contract (agent-first, like the output-schema and
error-code ratchets): five conventional top-level dirs —
``assets/ fragments/ blueprints/ outputs/ .comfy/`` (machine-owned) — and an
append-only run journal at ``.comfy/runs.jsonl`` so provenance is queryable
state, not hand-written manifests.

This module is pure domain logic — no Typer, no renderer (mirror of
``fragments.py``). Hard rules:

- :func:`find_project` NEVER raises. A malformed/unreadable ``comfy.yaml``
  or one with the wrong schema is treated as "no project at that level" and
  the walk continues — a stray marker file must not crash unrelated commands.
- :func:`journal` is best-effort: every exception is swallowed internally.
  A journaling failure can never fail a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_MARKER = "comfy.yaml"
PROJECT_SCHEMA = "project/1"

# The five conventional top-level dirs. Anything else (non-hidden) is
# surfaced as a warning by `comfy project status` — warnings only, never
# enforcement. Files are fine anywhere.
CONVENTIONAL_DIRS = ("assets", "fragments", "blueprints", "outputs", ".comfy")


@dataclass
class Project:
    root: Path
    config: dict


def find_project(start: Path | None = None) -> Project | None:
    """Walk up from ``start`` (default: cwd) to the filesystem root; the
    first directory whose ``comfy.yaml`` parses to a dict with
    ``schema: project/1`` wins. Returns ``None`` when nothing governs
    ``start``. Never raises."""
    try:
        here = (Path(start) if start is not None else Path.cwd()).resolve()
    except OSError:
        return None
    for candidate in (here, *here.parents):
        config = _load_marker(candidate / PROJECT_MARKER)
        if config is not None:
            return Project(root=candidate, config=config)
    return None


def _load_marker(marker: Path) -> dict | None:
    """Parse one candidate marker. Anything short of a well-formed project/1
    dict — missing file, unreadable file, YAML that doesn't parse, non-dict
    document, wrong/absent schema — is ``None`` (keep walking)."""
    try:
        if not marker.is_file():
            return None
        parsed = yaml.safe_load(marker.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — any failure means "not a project here"
        return None
    if isinstance(parsed, dict) and parsed.get("schema") == PROJECT_SCHEMA:
        return parsed
    return None


def unknown_dirs(project: Project) -> list[str]:
    """Top-level directories outside the convention, sorted. Hidden dirs are
    skipped (``.comfy`` is conventional anyway); files are fine anywhere.
    Consumed by ``comfy project status`` as warnings — nothing enforces."""
    try:
        names = sorted(p.name for p in project.root.iterdir() if p.is_dir())
    except OSError:
        return []
    return [n for n in names if not n.startswith(".") and n not in CONVENTIONAL_DIRS]


def journal(project: Project, **event) -> None:
    """Append one JSON line to ``<root>/.comfy/runs.jsonl`` (creating
    ``.comfy/`` if needed). A ``ts`` (UTC ISO-8601, seconds) is auto-added.

    Best-effort by contract: ALL exceptions are swallowed — a read-only
    directory, a full disk, or an unserializable value must never fail the
    command being journaled."""
    try:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        comfy_dir = project.root / ".comfy"
        comfy_dir.mkdir(parents=True, exist_ok=True)
        with (comfy_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — journaling can never fail a run
        pass


def read_journal(project: Project, limit: int = 20) -> list[dict]:
    """Return the last ``limit`` journal events, newest last. Corrupt or
    non-object lines are skipped; a missing/unreadable journal is ``[]``."""
    path = project.root / ".comfy" / "runs.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events[-limit:] if limit and limit > 0 else events
