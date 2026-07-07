"""Agent skill installer — multi-skill bundle.

Ships a small set of SKILL.md files (the agent-facing description of how to
drive ``comfy``) and writes them into every supported agent host on the
user's machine:

- Claude Code        — ``<scope>/.claude/skills/<name>/SKILL.md``
- Cursor             — ``<scope>/.cursor/rules/<name>.mdc``
- AGENTS.md (any)    — append a fenced ``<name>`` block

The point: instead of running an MCP server, one command teaches every agent
on the box how to drive ``comfy`` directly. This is the productized form of
the agent-first CLI.

Bundled skills (4 total):

- ``comfy``          — the primary driver skill (command surface, output
                       contract, routing, discovery, execution, and all
                       domain patterns: image, video, audio, cloud, edit,
                       condition, pipeline)
- ``comfy-fragments``— typed reusable workflow fragments + YAML blueprint
                       composition (build large pipelines from small pieces)
- ``comfy-debug``    — debugging skill for when workflows fail or jobs hang
- ``comfy-relay``    — what to put in chat while driving the CLI
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

# Where the bundled skills live. Each tuple is (skill_name, package_subdir).
# ``skill_name`` is the public identifier used in AGENTS.md fences and as the
# subdir name in installed targets. ``package_subdir`` is the local resource
# directory.
_SKILL_PACKAGE_ROOT = "comfy_cli.skills"
_SKILL_FILE = "SKILL.md"


# (name, subdir) — the subdir is named after the skill, same as the rule
# `skills validate` enforces on third-party skills (directory == frontmatter
# name). The bundled skills must satisfy their own convention.
BUNDLED_SKILLS: tuple[tuple[str, str], ...] = (
    ("comfy", "comfy"),
    ("comfy-fragments", "comfy-fragments"),
    ("comfy-debug", "comfy-debug"),
    ("comfy-relay", "comfy-relay"),
    ("comfy-director", "comfy-director"),
)


# Skills we used to bundle and have since folded into the consolidated `comfy`
# skill. `install()` prunes any of these left behind on disk so machines that
# installed an older bundle converge to the current 4 on the next install.
RETIRED_SKILLS: tuple[str, ...] = (
    "comfy-image",
    "comfy-video",
    "comfy-audio",
    "comfy-edit",
    "comfy-condition",
    "comfy-pipeline",
    "comfy-cloud",
)


def bundled_skill_names() -> tuple[str, ...]:
    return tuple(name for name, _ in BUNDLED_SKILLS)


def _resolve_subdir(skill_name: str) -> str:
    for name, subdir in BUNDLED_SKILLS:
        if name == skill_name:
            return subdir
    raise ValueError(f"unknown bundled skill {skill_name!r}; choices: {', '.join(n for n, _ in BUNDLED_SKILLS)}")


TargetKind = Literal["claude-code", "cursor", "agents-md"]
Scope = Literal["user", "project"]


@dataclass(frozen=True)
class TargetPlan:
    """A planned write to a single agent host for a single skill."""

    skill: str
    kind: TargetKind
    scope: Scope
    path: Path
    exists: bool  # whether the file already exists at `path`


@dataclass
class TargetResult:
    """Outcome of executing a TargetPlan."""

    skill: str
    kind: TargetKind
    scope: Scope
    path: Path
    action: Literal["wrote", "skipped", "removed", "absent", "would_write", "would_remove"]
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "kind": self.kind,
            "scope": self.scope,
            "path": str(self.path),
            "action": self.action,
            "reason": self.reason,
        }


def skill_content(name: str = "comfy") -> str:
    """Return the SKILL.md text for the named bundled skill."""
    subdir = _resolve_subdir(name)
    pkg = resources.files(_SKILL_PACKAGE_ROOT) / subdir
    return (pkg / _SKILL_FILE).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Third-party / path-based skill source resolution
# ---------------------------------------------------------------------------

_FRONTMATTER_NAME_RE = re.compile(r"^name:\s*(\S+)\s*$", re.MULTILINE)
_FRONTMATTER_DESC_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class SkillSource:
    name: str  # canonical skill name (frontmatter ``name:``)
    content: str  # full SKILL.md content
    bundled: bool  # False for path-loaded skills


def load_skill_source(token: str) -> SkillSource:
    """Resolve a --skill token: a bundled name, or a path to a skill dir/SKILL.md.

    Path rules: the path may be a directory containing SKILL.md or the SKILL.md
    itself. Frontmatter must declare ``name:`` (matching the directory name when
    a directory is given) and a non-empty ``description:``. Raises ValueError
    with a human-readable reason on any violation.
    """
    p = Path(token).expanduser()
    looks_like_path = os.sep in token or token.startswith((".", "~")) or p.exists()
    if not looks_like_path:
        return SkillSource(name=token, content=skill_content(token), bundled=True)

    md = p / "SKILL.md" if p.is_dir() else p
    if not md.is_file():
        raise ValueError(f"no SKILL.md found at {p}")
    content = md.read_text(encoding="utf-8")
    if not content.startswith("---"):
        raise ValueError(f"{md}: missing frontmatter block")
    name_m = _FRONTMATTER_NAME_RE.search(content)
    desc_m = _FRONTMATTER_DESC_RE.search(content)
    if not name_m:
        raise ValueError(f"{md}: frontmatter must declare `name:`")
    if not desc_m or not desc_m.group(1).strip():
        raise ValueError(f"{md}: frontmatter must declare a non-empty `description:`")
    name = name_m.group(1)
    # The name becomes a directory component of the install target — restrict it
    # to a simple slug so a hostile SKILL.md can't traverse out of the skills dir.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(
            f"{md}: frontmatter name {name!r} must be a simple slug ([A-Za-z0-9_-], no leading dot/separators)"
        )
    if p.is_dir() and p.name != name:
        raise ValueError(f"{md}: frontmatter name {name!r} must match directory name {p.name!r}")
    return SkillSource(name=name, content=content, bundled=False)


# ---------------------------------------------------------------------------
# Install manifest — provenance and staleness tracking
# ---------------------------------------------------------------------------


def manifest_path() -> Path:
    """Return the path to the skills install manifest (in the CLI config dir)."""
    from comfy_cli.config_manager import ConfigManager

    return Path(ConfigManager().get_config_path()) / "skills-manifest.json"


def read_manifest() -> dict:
    """Read the manifest; returns {} on missing or corrupt file."""
    try:
        return json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_manifest(manifest: dict) -> None:
    """Atomically write the manifest (tmp + rename so a SIGINT can't corrupt it)."""
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_installed(target_path: Path, skill_name: str, content: str) -> None:
    """Add or update the manifest entry for a successfully installed skill file."""
    from comfy_cli.config_manager import ConfigManager

    try:
        cli_version = ConfigManager().get_cli_version()
    except Exception:
        cli_version = "0.0.0"

    manifest = read_manifest()
    manifest[str(target_path)] = {
        "skill": skill_name,
        "sha256": _sha256(content),
        "cli_version": cli_version,
    }
    write_manifest(manifest)


def _remove_manifest_entry(target_path: Path) -> None:
    """Remove the manifest entry for an uninstalled skill file (if present)."""
    manifest = read_manifest()
    key = str(target_path)
    if key in manifest:
        del manifest[key]
        write_manifest(manifest)


SkillState = Literal["current", "stale", "modified", "missing", "unmanaged"]


def _compute_skill_state(path: Path, skill_name: str, manifest: dict) -> SkillState:
    """Compute the provenance state for a single installed skill file.

    States:
    - missing    — file does not exist on disk
    - current    — file matches the current bundled content exactly
    - stale      — file matches the manifest sha (user hasn't edited) but bundled moved on
    - modified   — file differs from manifest sha (user has edited it)
    - unmanaged  — file exists, no manifest entry, and differs from bundled content
    """
    if not path.exists():
        return "missing"

    try:
        file_content = path.read_text(encoding="utf-8")
    except OSError:
        return "missing"

    file_sha = _sha256(file_content)

    # Try to get the current bundled content hash for comparison.
    try:
        bundled_content = skill_content(skill_name)
        bundled_sha = _sha256(bundled_content)
    except ValueError:
        # Not a bundled skill (path-based) — only manifest tells us if it's unmodified.
        bundled_sha = None

    if bundled_sha is not None and file_sha == bundled_sha:
        return "current"

    entry = manifest.get(str(path))
    if entry is None:
        # File exists, no manifest, differs from bundled — unmanaged.
        return "unmanaged"

    manifest_sha = entry.get("sha256", "")
    if file_sha == manifest_sha:
        # File matches what was installed (user hasn't edited), but bundled moved on.
        return "stale"

    # File differs from both manifest and bundled — user edited it.
    return "modified"


def _agents_fence(skill_name: str) -> tuple[str, str]:
    return (f"<!-- {skill_name}:start -->", f"<!-- {skill_name}:end -->")


def _resolve_paths(*, skill_name: str, scope: Scope, project_root: Path) -> dict[TargetKind, Path]:
    """Map each target kind to the path it would be written to under `scope`.

    For ``user`` scope, paths are under ``~``.
    For ``project`` scope, paths are under ``project_root`` (typically cwd).
    """
    if scope == "user":
        home = Path.home()
        return {
            "claude-code": home / ".claude" / "skills" / skill_name / "SKILL.md",
            "cursor": home / ".cursor" / "rules" / f"{skill_name}.mdc",
            "agents-md": home / "AGENTS.md",
        }
    return {
        "claude-code": project_root / ".claude" / "skills" / skill_name / "SKILL.md",
        "cursor": project_root / ".cursor" / "rules" / f"{skill_name}.mdc",
        "agents-md": project_root / "AGENTS.md",
    }


def _normalize_skills(skills: Sequence[str] | None) -> list[SkillSource]:
    """Resolve a sequence of skill tokens into SkillSource objects.

    Tokens that look like paths are resolved via ``load_skill_source``; plain
    names are validated against the bundled set.  ``None`` / empty defaults to
    all bundled skills.
    """
    if not skills:
        return [SkillSource(name=name, content=skill_content(name), bundled=True) for name, _ in BUNDLED_SKILLS]
    out: list[SkillSource] = []
    for s in skills:
        p = Path(s).expanduser()
        looks_like_path = os.sep in s or s.startswith((".", "~")) or p.exists()
        if looks_like_path:
            # Raises ValueError on invalid path skills — caller handles.
            out.append(load_skill_source(s))
        else:
            _resolve_subdir(s)  # validates bundled name; raises ValueError on unknown
            out.append(SkillSource(name=s, content=skill_content(s), bundled=True))
    return out


def plan_install(
    *,
    scope: Scope = "user",
    targets: Sequence[TargetKind] | None = None,
    skills: Sequence[str] | None = None,
    project_root: Path | None = None,
) -> list[TargetPlan]:
    """Return a TargetPlan per (skill, target) pair.

    Default skills: all bundled. Default targets: all three. Default scope: user.
    """
    root = project_root or Path.cwd()
    plans: list[TargetPlan] = []
    for source in _normalize_skills(skills):
        all_paths = _resolve_paths(skill_name=source.name, scope=scope, project_root=root)
        kinds: list[TargetKind] = list(targets) if targets else list(all_paths.keys())
        for kind in kinds:
            path = all_paths[kind]
            plans.append(TargetPlan(skill=source.name, kind=kind, scope=scope, path=path, exists=path.exists()))
    return plans


def install(
    *,
    scope: Scope = "user",
    targets: Sequence[TargetKind] | None = None,
    skills: Sequence[str] | None = None,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> list[TargetResult]:
    """Install (or preview) the chosen skills across the chosen targets.

    AGENTS.md gets one fenced block per skill (idempotent re-installs replace
    the block). Other targets are full file writes per skill.
    """
    root = project_root or Path.cwd()
    results: list[TargetResult] = []
    sources = _normalize_skills(skills)
    # Build a per-name content map so path-based skills carry their own content.
    content_map: dict[str, str] = {src.name: src.content for src in sources}
    plans = plan_install(scope=scope, targets=targets, skills=skills, project_root=root)
    for plan in plans:
        if dry_run:
            results.append(
                TargetResult(
                    skill=plan.skill,
                    kind=plan.kind,
                    scope=plan.scope,
                    path=plan.path,
                    action="would_write",
                )
            )
            continue
        try:
            content = content_map[plan.skill]
            if plan.kind == "claude-code":
                _write_claude_skill(plan.path, content)
                _record_installed(plan.path, plan.skill, content)
            elif plan.kind == "cursor":
                _write_cursor_rule(plan.path, content, skill_name=plan.skill)
                _record_installed(plan.path, plan.skill, content)
            elif plan.kind == "agents-md":
                _upsert_agents_md_block(plan.path, content, skill_name=plan.skill)
            results.append(
                TargetResult(skill=plan.skill, kind=plan.kind, scope=plan.scope, path=plan.path, action="wrote")
            )
        except OSError as e:
            results.append(
                TargetResult(
                    skill=plan.skill,
                    kind=plan.kind,
                    scope=plan.scope,
                    path=plan.path,
                    action="skipped",
                    reason=str(e),
                )
            )
    return results


def uninstall(
    *,
    scope: Scope = "user",
    targets: Sequence[TargetKind] | None = None,
    skills: Sequence[str] | None = None,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> list[TargetResult]:
    """Remove the chosen skills from each target. AGENTS.md keeps its other content."""
    results: list[TargetResult] = []
    for plan in plan_install(scope=scope, targets=targets, skills=skills, project_root=project_root):
        if dry_run:
            action: Literal["would_remove", "absent"] = "would_remove" if plan.exists else "absent"
            results.append(
                TargetResult(skill=plan.skill, kind=plan.kind, scope=plan.scope, path=plan.path, action=action)
            )
            continue
        try:
            if plan.kind == "agents-md":
                removed = _remove_agents_md_block(plan.path, skill_name=plan.skill)
                results.append(
                    TargetResult(
                        skill=plan.skill,
                        kind=plan.kind,
                        scope=plan.scope,
                        path=plan.path,
                        action="removed" if removed else "absent",
                    )
                )
            else:
                if plan.path.exists():
                    plan.path.unlink()
                    _remove_manifest_entry(plan.path)
                    results.append(
                        TargetResult(
                            skill=plan.skill,
                            kind=plan.kind,
                            scope=plan.scope,
                            path=plan.path,
                            action="removed",
                        )
                    )
                else:
                    results.append(
                        TargetResult(
                            skill=plan.skill,
                            kind=plan.kind,
                            scope=plan.scope,
                            path=plan.path,
                            action="absent",
                        )
                    )
        except OSError as e:
            results.append(
                TargetResult(
                    skill=plan.skill,
                    kind=plan.kind,
                    scope=plan.scope,
                    path=plan.path,
                    action="skipped",
                    reason=str(e),
                )
            )
    return results


def _agents_block_present(path: Path, skill_name: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    start, end = _agents_fence(skill_name)
    return start in text and end in text


def _prune_one(name: str, kind: TargetKind, scope: Scope, path: Path, dry_run: bool) -> TargetResult:
    present = _agents_block_present(path, name) if kind == "agents-md" else path.exists()
    if not present:
        return TargetResult(skill=name, kind=kind, scope=scope, path=path, action="absent")

    # Manifest guard: only delete a file when:
    #   (a) the manifest records it as comfy-managed, OR
    #   (b) no manifest file exists (legacy pre-manifest installs — converge anyway).
    if kind != "agents-md":
        mfile = manifest_path()
        manifest_exists = mfile.exists()
        if manifest_exists:
            manifest = read_manifest()
            if str(path) not in manifest:
                # File exists but is not comfy-managed — skip to preserve user files.
                return TargetResult(skill=name, kind=kind, scope=scope, path=path, action="absent")

    if dry_run:
        return TargetResult(
            skill=name,
            kind=kind,
            scope=scope,
            path=path,
            action="would_remove",
        )
    try:
        if kind == "agents-md":
            _remove_agents_md_block(path, skill_name=name)
        else:
            path.unlink()
            _remove_manifest_entry(path)
            # Claude skills live in their own <name>/ dir — drop it if now empty.
            if kind == "claude-code" and not any(path.parent.iterdir()):
                path.parent.rmdir()
        return TargetResult(skill=name, kind=kind, scope=scope, path=path, action="removed")
    except OSError as e:
        return TargetResult(skill=name, kind=kind, scope=scope, path=path, action="skipped", reason=str(e))


def prune_retired(
    *,
    scope: Scope = "user",
    targets: Sequence[TargetKind] | None = None,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> list[TargetResult]:
    """Remove any retired skills (see ``RETIRED_SKILLS``) left on disk.

    Idempotent and safe when none exist — every absent target reports
    ``absent``. Lets a machine that installed an older bundle converge to the
    current set on the next ``install``.
    """
    root = project_root or Path.cwd()
    results: list[TargetResult] = []
    for name in RETIRED_SKILLS:
        all_paths = _resolve_paths(skill_name=name, scope=scope, project_root=root)
        kinds: list[TargetKind] = list(targets) if targets else list(all_paths.keys())
        for kind in kinds:
            results.append(_prune_one(name, kind, scope, all_paths[kind], dry_run))
    return results


# ---------------------------------------------------------------------------
# per-target writers
# ---------------------------------------------------------------------------


def _backup_if_user_edited(path: Path, expected_content: str) -> Path | None:
    """If `path` exists with content that differs from what we'd write, save a
    timestamped `.bak` so a user's hand-edits aren't silently destroyed.

    Returns the backup path (or None if no backup was needed).
    """
    if not path.exists():
        return None
    try:
        current = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if current == expected_content:
        return None
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix(path.suffix + f".{ts}.bak")
    try:
        bak.write_text(current, encoding="utf-8")
    except OSError:
        return None
    return bak


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via tmp + rename so a SIGINT mid-write can't leave the file empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _write_claude_skill(path: Path, content: str) -> None:
    _backup_if_user_edited(path, content)
    _atomic_write_text(path, content)


def _cursor_description_for(skill_name: str) -> str:
    return {
        "comfy": "comfy CLI for ComfyUI workflows, models, node-graph queries, image/video/audio generation, cloud, and pipeline orchestration.",
        "comfy-fragments": "Typed reusable workflow fragments + YAML blueprint composition: build large pipelines from small tested pieces via comfy CLI.",
        "comfy-debug": "Debugging skill for the comfy CLI: failed workflows, hung jobs, error envelopes.",
        "comfy-relay": "What to put in chat while driving the comfy CLI: show artifacts, surface results, truncation rules.",
    }.get(skill_name, f"comfy CLI skill: {skill_name}")


def _write_cursor_rule(path: Path, content: str, *, skill_name: str) -> None:
    body = _strip_frontmatter(content)
    rule = f'---\ndescription: {_cursor_description_for(skill_name)}\nglobs: "**/*"\nalwaysApply: false\n---\n\n{body}'
    _backup_if_user_edited(path, rule)
    _atomic_write_text(path, rule)


def _upsert_agents_md_block(path: Path, content: str, *, skill_name: str) -> None:
    start, end = _agents_fence(skill_name)
    block = f"\n{start}\n{content}\n{end}\n"
    if not path.exists():
        _atomic_write_text(path, block.lstrip("\n"))
        return
    existing = path.read_text(encoding="utf-8")
    if start in existing and end in existing:
        before, _, rest = existing.partition(start)
        _, _, after = rest.partition(end)
        new = before.rstrip() + "\n\n" + block.lstrip("\n") + after.lstrip("\n")
    else:
        new = existing.rstrip() + "\n" + block
    _atomic_write_text(path, new)


def _remove_agents_md_block(path: Path, *, skill_name: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    start, end = _agents_fence(skill_name)
    if start not in text or end not in text:
        return False
    before, _, rest = text.partition(start)
    _, _, after = rest.partition(end)
    new = before.rstrip() + ("\n" + after.lstrip("\n") if after.strip() else "\n")
    path.write_text(new, encoding="utf-8")
    return True


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---\n"):
        return content
    _, _, rest = content.partition("---\n")
    _, _, body = rest.partition("---\n")
    return body.lstrip("\n")
