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

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from comfy_cli.fragments import AssetError, VarError

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


ASSETS_LOCK_SCHEMA = "assets-lock/1"
_ASSETS_PUSH_HINT = "run: comfy assets push"


def read_assets_lock(project: Project) -> dict:
    """The ``assets-lock/1`` map ``{name: {sha256, cloud_name, …}}`` written
    by ``comfy assets push``; ``{}`` when absent or malformed. Never raises."""
    path = project.root / ".comfy" / "assets.lock.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if (
        isinstance(parsed, dict)
        and parsed.get("schema") == ASSETS_LOCK_SCHEMA
        and isinstance(parsed.get("assets"), dict)
    ):
        return parsed["assets"]
    return {}


def make_asset_resolver(project: Project) -> Callable[[str], str]:
    """Resolver for ``$asset.<name>`` blueprint refs, backed by the push lock.

    Loads ``.comfy/assets.lock.json`` once; per referenced name (hashing is
    lazy — only names actually referenced are hashed):

    - no lock entry, or the file is missing under ``assets/`` →
      :class:`~comfy_cli.fragments.AssetError` ``asset_not_pushed``
    - on-disk sha256 differs from the lock → ``asset_stale``
    - otherwise → the lock entry's ``cloud_name`` (the server-side filename
      on the push target).
    """
    lock = read_assets_lock(project)

    def resolve(name: str) -> str:
        entry = lock.get(name)
        if not isinstance(entry, dict) or not entry.get("cloud_name"):
            raise AssetError(
                f"asset {name!r} has not been pushed (no entry in .comfy/assets.lock.json)",
                code="asset_not_pushed",
                hint=_ASSETS_PUSH_HINT,
            )
        path = project.root / "assets" / name
        if not path.is_file():
            raise AssetError(
                f"asset {name!r} is in the lock but missing from assets/",
                code="asset_not_pushed",
                hint=_ASSETS_PUSH_HINT,
            )
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry.get("sha256"):
            raise AssetError(
                f"asset {name!r} changed on disk after its last push (sha256 mismatch)",
                code="asset_stale",
                hint=_ASSETS_PUSH_HINT,
            )
        return str(entry["cloud_name"])

    return resolve


def make_var_resolver(project: Project) -> Callable[[str], Any]:
    """Resolver for ``$var.<name>`` blueprint refs, backed by the project's
    ``comfy.yaml`` top-level ``vars:`` block (name → scalar str/int/float/bool).

    Returns the RAW scalar (never ``str()``'d) so non-STRING params keep
    their widget types. A missing/non-mapping ``vars:`` block is treated as
    empty; an undefined name raises
    :class:`~comfy_cli.fragments.VarError` ``var_not_defined`` with a hint
    pointing at the project's ``comfy.yaml``.
    """
    vars_block = project.config.get("vars")
    if not isinstance(vars_block, dict):
        vars_block = {}

    def resolve(name: str) -> Any:
        if name not in vars_block:
            raise VarError(
                f"var {name!r} is not defined in the project's `vars:` block",
                code="var_not_defined",
                hint=f"add it under `vars:` in {project.root / PROJECT_MARKER}",
            )
        return vars_block[name]

    return resolve


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
