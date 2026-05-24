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

Bundled skills:

- ``comfy``          — the primary driver skill (command surface, output
                       contract, CQL, ``--where``, etc.)
- ``comfy-image``    — image generation patterns: t2i, variations, upscaling,
                       ControlNet, batch sweeps
- ``comfy-video``    — video generation: I2V, T2V, assembly, fps wiring,
                       motion prompts
- ``comfy-audio``    — audio/music generation: ACE-Step, TTS, duration matching
- ``comfy-pipeline`` — multi-stage orchestration: upload/download composition,
                       parallel fan-out, project layout
- ``comfy-fragments``— typed reusable workflow fragments + YAML recipe
                       composition (build large pipelines from small pieces)
- ``comfy-debug``    — debugging skill for when workflows fail or jobs hang
- ``comfy-cloud``    — Comfy Cloud–specific story (auth, base-URL pinning,
                       ``--where cloud`` gotchas)
"""

from __future__ import annotations

import os
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


BUNDLED_SKILLS: tuple[tuple[str, str], ...] = (
    ("comfy", "comfy"),
    ("comfy-image", "image"),
    ("comfy-video", "video"),
    ("comfy-audio", "audio"),
    ("comfy-edit", "edit"),
    ("comfy-condition", "condition"),
    ("comfy-pipeline", "pipeline"),
    ("comfy-fragments", "fragments"),
    ("comfy-debug", "debug"),
    ("comfy-cloud", "cloud"),
    ("comfy-relay", "relay"),
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


def _normalize_skills(skills: Sequence[str] | None) -> list[str]:
    if not skills:
        return [name for name, _ in BUNDLED_SKILLS]
    out: list[str] = []
    for s in skills:
        _resolve_subdir(s)  # validates
        out.append(s)
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
    for skill_name in _normalize_skills(skills):
        all_paths = _resolve_paths(skill_name=skill_name, scope=scope, project_root=root)
        kinds: list[TargetKind] = list(targets) if targets else list(all_paths.keys())
        for kind in kinds:
            path = all_paths[kind]
            plans.append(TargetPlan(skill=skill_name, kind=kind, scope=scope, path=path, exists=path.exists()))
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
    results: list[TargetResult] = []
    plans = plan_install(scope=scope, targets=targets, skills=skills, project_root=project_root)
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
            content = skill_content(plan.skill)
            if plan.kind == "claude-code":
                _write_claude_skill(plan.path, content)
            elif plan.kind == "cursor":
                _write_cursor_rule(plan.path, content, skill_name=plan.skill)
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
        "comfy": "comfy CLI for ComfyUI workflows, models, and node-graph queries.",
        "comfy-image": "Image generation patterns: t2i, variations, upscaling, ControlNet, batch sweeps via comfy CLI.",
        "comfy-video": "Video generation patterns: I2V, T2V, video assembly, fps wiring, motion prompts via comfy CLI.",
        "comfy-audio": "Audio/music generation: ACE-Step, TTS, audio loading, duration matching via comfy CLI.",
        "comfy-edit": "Transform existing assets: upscale, inpaint, style transfer, image editing via comfy CLI.",
        "comfy-condition": "Guide generation: ControlNet, masks, reference images, motion control via comfy CLI.",
        "comfy-pipeline": "Multi-stage pipeline orchestration: upload/download, pipe composition, parallel fan-out via comfy CLI.",
        "comfy-fragments": "Typed reusable workflow fragments + YAML recipe composition: build large pipelines from small tested pieces via comfy CLI.",
        "comfy-debug": "Debugging skill for the comfy CLI: failed workflows, hung jobs, error envelopes.",
        "comfy-cloud": "Comfy Cloud usage with the comfy CLI: OAuth login, --where cloud, base-URL overrides.",
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
