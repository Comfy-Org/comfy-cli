"""Tests for the multi-skill installer (no MCP needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_cli.skills import (
    BUNDLED_SKILLS,
    bundled_skill_names,
    install,
    plan_install,
    skill_content,
    uninstall,
)


# ---------------------------------------------------------------------------
# Bundled skill inventory
# ---------------------------------------------------------------------------


def test_bundles_at_least_comfy_debug_cloud():
    names = bundled_skill_names()
    assert "comfy" in names
    assert "comfy-debug" in names
    assert "comfy-cloud" in names


def test_bundled_skills_have_required_frontmatter():
    for name in bundled_skill_names():
        text = skill_content(name)
        assert text.startswith("---\n"), f"{name}: missing frontmatter"
        assert f"name: {name}" in text, f"{name}: frontmatter name doesn't match"
        assert "description:" in text, f"{name}: frontmatter missing description"


def test_comfy_skill_content_has_required_sections():
    text = skill_content("comfy").lower()
    assert "comfy --json discover" in text
    assert "routing" in text or "--where" in text
    assert "cql" in text
    assert "envelope" in text  # the section that documents the output contract


def test_comfy_debug_skill_covers_common_error_codes():
    text = skill_content("comfy-debug")
    for code in ("server_not_running", "cloud_not_configured", "workflow_not_api_format", "prompt_rejected"):
        assert code in text, f"comfy-debug skill should mention {code}"


def test_comfy_cloud_skill_covers_setup_and_routing():
    text = skill_content("comfy-cloud")
    for needle in ("comfy cloud login", "cloud set-base-url", "--where cloud", "COMFY_CLOUD_BASE_URL"):
        assert needle in text, f"comfy-cloud skill should mention {needle}"


def test_skill_content_rejects_unknown_name():
    with pytest.raises(ValueError) as exc:
        skill_content("not-a-real-skill")
    assert "not-a-real-skill" in str(exc.value)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_plan_install_default_covers_every_skill_and_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = plan_install(scope="user", project_root=tmp_path / "anywhere")
    skill_target_pairs = {(p.skill, p.kind) for p in plans}
    expected = {(name, kind) for name in bundled_skill_names() for kind in ("claude-code", "cursor", "agents-md")}
    assert skill_target_pairs == expected


def test_plan_install_project_scope_paths(tmp_path: Path):
    plans = plan_install(scope="project", project_root=tmp_path, skills=["comfy-debug"])
    paths = {p.kind: p.path for p in plans}
    assert paths["claude-code"] == tmp_path / ".claude" / "skills" / "comfy-debug" / "SKILL.md"
    assert paths["cursor"] == tmp_path / ".cursor" / "rules" / "comfy-debug.mdc"
    assert paths["agents-md"] == tmp_path / "AGENTS.md"


def test_plan_install_filters_by_skill(tmp_path: Path):
    plans = plan_install(scope="project", project_root=tmp_path, skills=["comfy", "comfy-cloud"])
    assert {p.skill for p in plans} == {"comfy", "comfy-cloud"}


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def test_install_writes_every_skill_to_every_target(tmp_path: Path):
    results = install(scope="project", project_root=tmp_path)
    for r in results:
        assert r.action == "wrote", f"unexpected action for {r.skill}/{r.kind}: {r.action}"
        assert r.path.exists()

    for name in bundled_skill_names():
        claude = (tmp_path / f".claude/skills/{name}/SKILL.md").read_text(encoding="utf-8")
        assert claude.startswith("---\n")
        assert f"name: {name}" in claude

        cursor = (tmp_path / f".cursor/rules/{name}.mdc").read_text(encoding="utf-8")
        assert "alwaysApply: false" in cursor

    agents_md = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    for name in bundled_skill_names():
        assert f"<!-- {name}:start -->" in agents_md
        assert f"<!-- {name}:end -->" in agents_md


def test_install_one_skill_only(tmp_path: Path):
    install(scope="project", project_root=tmp_path, skills=["comfy-debug"])
    assert (tmp_path / ".claude/skills/comfy-debug/SKILL.md").exists()
    assert not (tmp_path / ".claude/skills/comfy/SKILL.md").exists()
    assert not (tmp_path / ".claude/skills/comfy-cloud/SKILL.md").exists()
    agents = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "<!-- comfy-debug:start -->" in agents
    assert "<!-- comfy:start -->" not in agents
    assert "<!-- comfy-cloud:start -->" not in agents


def test_install_is_idempotent_across_skills(tmp_path: Path):
    install(scope="project", project_root=tmp_path)
    install(scope="project", project_root=tmp_path)
    agents_md = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    for name in bundled_skill_names():
        assert agents_md.count(f"<!-- {name}:start -->") == 1
        assert agents_md.count(f"<!-- {name}:end -->") == 1


def test_install_preserves_existing_agents_md_content(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("# My project agent notes\n\nKeep this.\n", encoding="utf-8")
    install(scope="project", project_root=tmp_path, targets=["agents-md"])
    text = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "# My project agent notes" in text
    assert "Keep this." in text


def test_dry_run_does_not_touch_disk(tmp_path: Path):
    results = install(scope="project", project_root=tmp_path, dry_run=True)
    for r in results:
        assert r.action == "would_write"
        assert not r.path.exists()


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


def test_uninstall_removes_all_skills_and_keeps_other_agents_md(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("# keep me\n\n", encoding="utf-8")
    install(scope="project", project_root=tmp_path)
    for name in bundled_skill_names():
        assert (tmp_path / f".claude/skills/{name}/SKILL.md").exists()

    uninstall(scope="project", project_root=tmp_path)
    for name in bundled_skill_names():
        assert not (tmp_path / f".claude/skills/{name}/SKILL.md").exists()
        assert not (tmp_path / f".cursor/rules/{name}.mdc").exists()

    agents_md = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "# keep me" in agents_md
    for name in bundled_skill_names():
        assert f"<!-- {name}:start -->" not in agents_md


def test_uninstall_can_target_one_skill(tmp_path: Path):
    install(scope="project", project_root=tmp_path)
    uninstall(scope="project", project_root=tmp_path, skills=["comfy-debug"])
    assert not (tmp_path / ".claude/skills/comfy-debug/SKILL.md").exists()
    assert (tmp_path / ".claude/skills/comfy/SKILL.md").exists()
    assert (tmp_path / ".claude/skills/comfy-cloud/SKILL.md").exists()


def test_uninstall_is_safe_on_clean_tree(tmp_path: Path):
    results = uninstall(scope="project", project_root=tmp_path)
    for r in results:
        assert r.action in {"absent", "removed"}


# ---------------------------------------------------------------------------
# Backup / atomic-write contracts (kept from the original)
# ---------------------------------------------------------------------------


def test_install_backs_up_user_edited_skill_md(tmp_path: Path):
    skill_path = tmp_path / ".claude/skills/comfy/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("# my custom edits, please preserve me\n", encoding="utf-8")

    install(scope="project", project_root=tmp_path, skills=["comfy"], targets=["claude-code"])

    backups = list(skill_path.parent.glob("SKILL.md.*.bak"))
    assert len(backups) == 1, f"expected exactly one backup, got {backups}"
    assert "my custom edits" in backups[0].read_text(encoding="utf-8")
    assert "my custom edits" not in skill_path.read_text(encoding="utf-8")


def test_install_does_not_back_up_identical_content(tmp_path: Path):
    install(scope="project", project_root=tmp_path, skills=["comfy"], targets=["claude-code"])
    skill_path = tmp_path / ".claude/skills/comfy/SKILL.md"
    install(scope="project", project_root=tmp_path, skills=["comfy"], targets=["claude-code"])
    assert list(skill_path.parent.glob("SKILL.md.*.bak")) == []


def test_install_atomic_write_does_not_leave_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    skill_path = tmp_path / ".claude/skills/comfy/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("original\n", encoding="utf-8")

    import os as real_os

    real_replace = real_os.replace

    def boom(src, dst):
        raise OSError("simulated kill -9 between write and rename")

    monkeypatch.setattr("comfy_cli.skills.os.replace", boom)

    results = install(scope="project", project_root=tmp_path, skills=["comfy"], targets=["claude-code"])

    assert results[0].action == "skipped"
    assert skill_path.read_text(encoding="utf-8") == "original\n"
    leftover = list(skill_path.parent.glob("*.tmp"))
    assert leftover == [], f"tmp not cleaned: {leftover}"
    monkeypatch.setattr("comfy_cli.skills.os.replace", real_replace)
