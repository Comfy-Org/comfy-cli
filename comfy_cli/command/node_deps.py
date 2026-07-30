"""``comfy node deps`` — read-only per-pack Python dependency report.

For every installed custom node pack (or just the ones named on the command
line), reports the pack's *declared* Python requirements — from its
``requirements.txt`` and/or its ``pyproject.toml`` ``[project] dependencies``
— alongside the version actually installed in the workspace venv and a
computed status per requirement.

Read-only by construction: it never installs anything, never shells out to
``cm-cli``, and never touches the network. The only subprocess it runs is a
single ``<workspace python> -m pip list --format=json`` for the whole report
(not once per pack).

Statuses:

- ``satisfied``   — installed, and the declared specifier accepts that version
  (a bare name with no specifier is satisfied by any installed version);
- ``mismatch``    — installed, but the specifier rejects that version (both
  versions are reported);
- ``missing``     — not present in the workspace venv at all;
- ``unparseable`` — a pip option (``--extra-index-url``, ``-r other.txt``), a
  bare VCS/URL line, or anything else :class:`packaging.requirements.Requirement`
  rejects. Kept in the report rather than dropped silently;
- ``unknown``     — the comparison could not be made: either the ``pip list``
  probe failed (accompanied by an ``installed_versions_unavailable`` warning on
  the report) or the installed dist records a version that isn't PEP 440.

Known caveat: pyproject dependencies are read through the shared registry
parser (:func:`comfy_cli.command.pack_scan.read_pyproject`), which drops the
``comfyui-frontend-package`` pin from ``[project] dependencies``. Such a pin is
a core-ComfyUI concern rather than a pack dependency, so it will not appear in
this report even though it is declared.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from comfy_cli.command.pack_scan import iter_pack_dirs, read_pyproject

PIP_LIST_TIMEOUT_SECONDS = 60

# Same comment rules as ``uv.py::parse_req_file``: a ``#`` at line start or
# preceded by whitespace begins a comment. VCS URL fragments like
# ``#subdirectory=pkg`` / ``#egg=foo`` survive because they are not preceded by
# whitespace.
_INLINE_COMMENT_RE = re.compile(r"(^|\s+)#.*$")

# Every status a requirement row can carry; the per-pack summary always reports
# all of them so consumers can index without a KeyError.
STATUSES = ("satisfied", "mismatch", "missing", "unparseable", "unknown")

SOURCE_REQUIREMENTS_TXT = "requirements.txt"
SOURCE_PYPROJECT = "pyproject.toml"


# ---------------------------------------------------------------------------
# Installed versions
# ---------------------------------------------------------------------------


def pip_list(python: str) -> tuple[dict[str, str] | None, str | None]:
    """Return ``(canonical_name -> version, error)`` for the workspace venv.

    Runs ``<python> -m pip list --format=json`` exactly once. On any failure
    the map is ``None`` and the second element carries a human-readable reason
    — a report that can't see the venv is still a useful report.
    """
    from packaging.utils import canonicalize_name

    try:
        proc = subprocess.run(
            [python, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=PIP_LIST_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        return None, f"`pip list` exited {e.returncode}" + (f": {detail[-1]}" if detail else "")
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"could not run `pip list`: {e}"

    try:
        rows = json.loads(proc.stdout)
    except ValueError as e:
        return None, f"could not parse `pip list --format=json` output: {e}"
    if not isinstance(rows, list):
        return None, "`pip list --format=json` did not return a list"

    versions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, version = row.get("name"), row.get("version")
        if isinstance(name, str) and name:
            versions[canonicalize_name(name)] = str(version) if version is not None else ""
    return versions, None


# ---------------------------------------------------------------------------
# Declared requirements
# ---------------------------------------------------------------------------


def _requirements_txt_lines(path: Path) -> tuple[list[str], str | None]:
    """Return ``(lines, error)`` for a pack ``requirements.txt``.

    Mirrors ``uv.py::parse_req_file``'s comment/blank handling, but keeps pip
    option lines (``--extra-index-url``, ``-r base.txt``) instead of splitting
    them off — this report surfaces them as ``unparseable`` rather than
    dropping them.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [], f"could not read {path}: {e}"

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _INLINE_COMMENT_RE.sub("", raw_line).strip()
        if line:
            lines.append(line)
    return lines, None


def _pyproject_lines(path: Path) -> list[str]:
    cfg = read_pyproject(str(path))
    if cfg is None:
        return []
    deps = getattr(cfg.project, "dependencies", None) or []
    return [d.strip() for d in deps if isinstance(d, str) and d.strip()]


def _classify(raw: str, installed_versions: dict[str, str] | None) -> dict[str, Any]:
    """Turn one declared requirement line into a report row."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    row: dict[str, Any] = {
        "raw": raw,
        "name": None,
        "specifier": None,
        "installed": None,
        "status": "unparseable",
    }

    # A leading ``-`` is a pip option (``-r``, ``-e``, ``--extra-index-url``),
    # never a requirement — `Requirement` would either reject it or, worse,
    # misread it. Short-circuit so the reason is unambiguous.
    if raw.startswith("-"):
        return row

    try:
        req = Requirement(raw)
    except InvalidRequirement:
        # Bare VCS/URL lines (``git+https://…``, ``https://…/x.whl``) land here.
        return row

    row["name"] = req.name
    row["specifier"] = str(req.specifier)
    if req.marker is not None:
        # Markers are NOT evaluated: the relevant environment is the workspace
        # venv's, not this CLI process's. Surfaced so a consumer can tell why a
        # platform-specific pin reads as `missing`.
        row["marker"] = str(req.marker)

    if installed_versions is None:
        row["status"] = "unknown"
        return row

    installed = installed_versions.get(canonicalize_name(req.name))
    row["installed"] = installed
    if installed is None:
        row["status"] = "missing"
    elif not req.specifier:
        # A bare name is satisfied by any installed version.
        row["status"] = "satisfied"
    else:
        from packaging.version import InvalidVersion

        try:
            contains = req.specifier.contains(installed, prereleases=True)
        except InvalidVersion:
            # A dist whose recorded version isn't PEP 440 (rare, but a broken
            # `.dist-info` will do it) can't be compared. Say so rather than
            # letting the exception abort the whole report.
            row["status"] = "unknown"
            return row
        row["status"] = "satisfied" if contains else "mismatch"
    return row


def _pack_report(pack_dir: Path, workspace: Path, installed_versions: dict[str, str] | None) -> tuple[dict, list[str]]:
    """Build one pack's report row. Returns ``(row, warnings)``."""
    warnings: list[str] = []
    requirement_files: list[str] = []
    # ``raw`` line -> row, so a line declared in BOTH files is reported once
    # (requirements.txt wins the ``source`` attribution, being read first).
    rows: dict[str, dict[str, Any]] = {}

    req_txt = pack_dir / "requirements.txt"
    if req_txt.is_file():
        requirement_files.append(SOURCE_REQUIREMENTS_TXT)
        lines, err = _requirements_txt_lines(req_txt)
        if err:
            warnings.append(err)
        for line in lines:
            if line not in rows:
                rows[line] = {**_classify(line, installed_versions), "source": SOURCE_REQUIREMENTS_TXT}

    pyproject = pack_dir / "pyproject.toml"
    if pyproject.is_file():
        lines = _pyproject_lines(pyproject)
        # Only advertise pyproject as a requirement file when it actually
        # declares dependencies — most packs ship one purely for registry
        # metadata.
        if lines:
            requirement_files.append(SOURCE_PYPROJECT)
        for line in lines:
            if line not in rows:
                rows[line] = {**_classify(line, installed_versions), "source": SOURCE_PYPROJECT}

    requirements = list(rows.values())
    summary = {status: 0 for status in STATUSES}
    for row in requirements:
        summary[row["status"]] += 1

    try:
        rel_path = str(pack_dir.relative_to(workspace))
    except ValueError:
        rel_path = str(pack_dir)

    return (
        {
            "pack": pack_dir.name,
            "path": rel_path,
            "status": "installed",
            "requirement_files": requirement_files,
            "requirements": requirements,
            "summary": summary,
        },
        warnings,
    )


def _not_installed_row(name: str) -> dict[str, Any]:
    return {
        "pack": name,
        "path": None,
        "status": "not_installed",
        "requirement_files": [],
        "requirements": [],
        "summary": dict.fromkeys(STATUSES, 0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    comfy_path: str | None,
    pack_names: list[str] | None = None,
    *,
    python: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Build the per-pack dependency report.

    Returns ``(report, warnings)``. ``report`` is ``None`` when *comfy_path*
    resolves to no usable workspace — the caller turns that into a
    ``not_in_workspace`` error envelope. Each warning is
    ``{"code": ..., "message": ...}``.
    """
    from comfy_cli.resolve_python import resolve_workspace_python

    warnings: list[dict[str, str]] = []
    if not comfy_path or not os.path.isdir(comfy_path):
        return None, warnings

    workspace = Path(comfy_path)
    python = python or resolve_workspace_python(comfy_path)

    installed_versions, pip_error = pip_list(python)
    if installed_versions is None:
        warnings.append(
            {
                "code": "installed_versions_unavailable",
                "message": (
                    f"could not read installed packages from {python}"
                    + (f" ({pip_error})" if pip_error else "")
                    + " — every parseable requirement is reported with status 'unknown'"
                ),
            }
        )

    pack_dirs = iter_pack_dirs(workspace / "custom_nodes")
    packs: list[dict[str, Any]] = []
    if pack_names:
        by_lower = {p.name.lower(): p for p in pack_dirs}
        for requested in pack_names:
            match = by_lower.get(requested.strip().lower())
            if match is None:
                packs.append(_not_installed_row(requested))
                continue
            row, pack_warnings = _pack_report(match, workspace, installed_versions)
            packs.append(row)
            warnings.extend({"code": "pack_read_error", "message": w} for w in pack_warnings)
    else:
        for pack_dir in pack_dirs:
            row, pack_warnings = _pack_report(pack_dir, workspace, installed_versions)
            packs.append(row)
            warnings.extend({"code": "pack_read_error", "message": w} for w in pack_warnings)

    compiled = workspace / "requirements.compiled"
    compiled_present = compiled.is_file()
    report = {
        "workspace": str(workspace),
        "python": python,
        "compiled_lock": {
            "present": compiled_present,
            # Presence + path only in v1 — the lock is never parsed or diffed.
            "path": str(compiled) if compiled_present else None,
        },
        "packs": packs,
    }
    return report, warnings


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "satisfied": "[green]satisfied[/green]",
    "mismatch": "[bold red]mismatch[/bold red]",
    "missing": "[bold yellow]missing[/bold yellow]",
    "unparseable": "[dim]unparseable[/dim]",
    "unknown": "[dim]unknown[/dim]",
}


def _render_pretty(renderer, report: dict[str, Any]) -> None:
    from rich.markup import escape
    from rich.table import Table

    console = renderer.console()
    for pack in report["packs"]:
        title = escape(pack["pack"])
        if pack["status"] == "not_installed":
            renderer.print(f"[bold]{title}[/bold] — [yellow]not installed[/yellow]")
            continue
        if not pack["requirements"]:
            renderer.print(f"[bold]{title}[/bold] — [dim]no declared requirements[/dim]")
            continue

        tbl = Table(show_header=True, header_style="bold", title=title, title_justify="left")
        tbl.add_column("requirement")
        tbl.add_column("source")
        tbl.add_column("installed")
        tbl.add_column("status")
        for req in pack["requirements"]:
            # Values come from pack-controlled files; escape so a spec like
            # ``foo[/]`` can't raise a rich MarkupError and crash the report.
            installed = escape(req["installed"]) if req["installed"] is not None else "[dim]-[/dim]"
            tbl.add_row(
                escape(req["raw"]),
                escape(req["source"]),
                installed,
                _STATUS_STYLE.get(req["status"], escape(req["status"])),
            )
        console.print(tbl)

    totals = dict.fromkeys(STATUSES, 0)
    for pack in report["packs"]:
        for status, count in pack["summary"].items():
            totals[status] = totals.get(status, 0) + count
    renderer.print(
        "[bold]totals[/bold]: "
        + ", ".join(f"{count} {status}" for status, count in totals.items() if count)
        + (" — nothing declared" if not any(totals.values()) else "")
    )
    if report["compiled_lock"]["present"]:
        renderer.print(f"[dim]compiled lock: {escape(report['compiled_lock']['path'])}[/dim]")


def execute(renderer, comfy_path: str | None, pack_names: list[str] | None = None) -> None:
    """Entry point wired from ``comfy node deps``."""
    from rich.markup import escape

    report, warnings = build_report(comfy_path, pack_names)
    if report is None:
        import typer

        renderer.error(
            code="not_in_workspace",
            message=(
                "ComfyUI workspace not found"
                + (f" at {comfy_path!r}" if comfy_path else "")
                + ". Run 'comfy install', run 'comfy' from a ComfyUI directory, or pass '--workspace'."
            ),
            hint="run: comfy install   (or pass --workspace /path/to/ComfyUI)",
            command="node deps",
        )
        # Mirrors the `comfy which` convention: renderer.error records the code,
        # typer.Exit is what actually makes the process exit non-zero.
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        _render_pretty(renderer, report)
    for warning in warnings:
        renderer.warn(escape(warning["message"]))
    # Warnings ride along in the payload too: a JSON consumer reads only stdout.
    report["warnings"] = warnings
    renderer.emit(report, command="node deps")
