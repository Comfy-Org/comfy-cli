"""Tests for the multi-skill installer (no MCP needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer
from comfy_cli.skills import (
    RETIRED_SKILLS,
    bundled_skill_names,
    install,
    plan_install,
    prune_retired,
    skill_content,
    uninstall,
)
from comfy_cli.skills.command import app


def _force_json_renderer():
    """Pin the renderer to JSON so tests can read envelopes off stdout."""
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


# ---------------------------------------------------------------------------
# Bundled skill inventory
# ---------------------------------------------------------------------------


def test_bundles_expected_skills():
    names = bundled_skill_names()
    assert "comfy" in names
    assert "comfy-fragments" in names
    assert "comfy-debug" in names
    assert "comfy-relay" in names


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
    assert "nodes ls" in text  # the node introspection surface
    assert "envelope" in text  # the section that documents the output contract


def test_comfy_debug_skill_covers_common_error_codes():
    text = skill_content("comfy-debug")
    for code in ("server_not_running", "cloud_not_configured", "workflow_not_api_format", "prompt_rejected"):
        assert code in text, f"comfy-debug skill should mention {code}"


def test_comfy_skill_covers_cloud_setup_and_routing():
    # Cloud guidance was folded into the consolidated `comfy` skill's
    # "Cloud" gotchas section when the standalone comfy-cloud skill was retired.
    text = skill_content("comfy")
    for needle in ("comfy cloud login", "cloud set-base-url", "--where cloud"):
        assert needle in text, f"comfy skill should mention {needle}"


def test_comfy_fragments_skill_covers_composition():
    text = skill_content("comfy-fragments")
    for needle in ("workflow compose", "_fragment", "blueprint"):
        assert needle in text, f"comfy-fragments skill should mention {needle}"


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
    plans = plan_install(scope="project", project_root=tmp_path, skills=["comfy", "comfy-fragments"])
    assert {p.skill for p in plans} == {"comfy", "comfy-fragments"}


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
    assert not (tmp_path / ".claude/skills/comfy-fragments/SKILL.md").exists()
    agents = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "<!-- comfy-debug:start -->" in agents
    assert "<!-- comfy:start -->" not in agents
    assert "<!-- comfy-fragments:start -->" not in agents


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
    assert (tmp_path / ".claude/skills/comfy-relay/SKILL.md").exists()


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


# ---------------------------------------------------------------------------
# Retired-skill pruning (10 → 4 convergence)
# ---------------------------------------------------------------------------


def test_retired_and_bundled_are_disjoint():
    assert set(RETIRED_SKILLS).isdisjoint(set(bundled_skill_names()))


def _seed_retired_orphan(root: Path, name: str) -> tuple[Path, Path, Path]:
    """Write a stale orphan for `name` across all three targets."""
    claude = root / ".claude" / "skills" / name / "SKILL.md"
    cursor = root / ".cursor" / "rules" / f"{name}.mdc"
    agents = root / "AGENTS.md"
    claude.parent.mkdir(parents=True, exist_ok=True)
    cursor.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text("stale\n", encoding="utf-8")
    cursor.write_text("stale\n", encoding="utf-8")
    agents.write_text(f"# keep me\n\n<!-- {name}:start -->\nstale\n<!-- {name}:end -->\n", encoding="utf-8")
    return claude, cursor, agents


def test_prune_retired_removes_orphans(tmp_path: Path):
    name = RETIRED_SKILLS[0]
    claude, cursor, agents = _seed_retired_orphan(tmp_path, name)

    results = prune_retired(scope="project", project_root=tmp_path)

    assert not claude.exists()
    assert not claude.parent.exists()  # empty <name>/ dir cleaned up too
    assert not cursor.exists()
    agents_text = agents.read_text(encoding="utf-8")
    assert f"<!-- {name}:start -->" not in agents_text
    assert "# keep me" in agents_text  # unrelated content preserved
    removed = {(r.skill, r.kind) for r in results if r.action == "removed"}
    assert (name, "claude-code") in removed
    assert (name, "cursor") in removed
    assert (name, "agents-md") in removed


def test_prune_retired_is_absent_on_clean_tree(tmp_path: Path):
    results = prune_retired(scope="project", project_root=tmp_path)
    assert results, "should still report a plan per retired skill/target"
    assert all(r.action == "absent" for r in results)


def test_prune_retired_dry_run_does_not_delete(tmp_path: Path):
    name = RETIRED_SKILLS[0]
    claude, _cursor, _agents = _seed_retired_orphan(tmp_path, name)

    results = prune_retired(scope="project", project_root=tmp_path, dry_run=True)

    assert claude.exists(), "dry-run must not delete"
    assert any(r.action == "would_remove" for r in results)


def test_install_converges_old_machine(tmp_path: Path):
    # Simulate a machine that installed a now-retired skill, then reinstall the
    # current bundle and prune in one pass (as the install command does).
    name = RETIRED_SKILLS[0]
    claude, cursor, _agents = _seed_retired_orphan(tmp_path, name)

    prune_retired(scope="project", project_root=tmp_path)
    install(scope="project", project_root=tmp_path)

    assert not claude.exists()
    assert not cursor.exists()
    for current in bundled_skill_names():
        assert (tmp_path / ".claude" / "skills" / current / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Path-based (third-party) skill install
# ---------------------------------------------------------------------------


def test_install_accepts_directory_path(tmp_path: Path):
    """--skill ./my-skill/ (dir containing SKILL.md) installs a third-party skill."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_content_text = "---\nname: my-skill\ndescription: Test skill.\n---\n\n# My Skill\nBody.\n"
    (skill_dir / "SKILL.md").write_text(skill_content_text, encoding="utf-8")

    target_root = tmp_path / "target"
    target_root.mkdir()

    results = install(
        scope="project",
        project_root=target_root,
        skills=[str(skill_dir)],
        targets=["claude-code"],
    )
    assert results, "expected at least one result"
    assert results[0].action == "wrote"

    installed_path = target_root / ".claude" / "skills" / "my-skill" / "SKILL.md"
    assert installed_path.exists(), f"expected SKILL.md at {installed_path}"
    assert installed_path.read_text(encoding="utf-8") == skill_content_text


def test_install_rejects_path_with_mismatched_name(tmp_path: Path):
    """frontmatter name must match the directory name."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: other-name\ndescription: d.\n---\nBody.\n", encoding="utf-8")
    from comfy_cli.skills import load_skill_source

    with pytest.raises(ValueError, match="other-name"):
        load_skill_source(str(skill_dir))


def test_skills_validate_command(tmp_path: Path):
    """comfy skills validate <path> returns ok:true for a well-formed skill."""
    import json

    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Test skill.\n---\n\n# My Skill\nBody.\n",
        encoding="utf-8",
    )

    _force_json_renderer()
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(skill_dir)])
        assert result.exit_code == 0, f"expected exit 0, got {result.exit_code}\noutput: {result.output}"

        # Last non-empty line is the envelope.
        lines = [line for line in result.output.splitlines() if line.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is True
        assert envelope["data"]["valid"] is True
        assert envelope["data"]["name"] == "my-skill"
    finally:
        reset_renderer_for_testing()


# ---------------------------------------------------------------------------
# Security: slug guard — name must be a simple slug, no path traversal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_name", ["../../evil", "/tmp/evil", "a/b", "a\\b", ".hidden", ".."])
def test_load_skill_source_rejects_non_slug_names(tmp_path, bad_name):
    """Frontmatter name must be a simple slug — traversal/absolute names are rejected
    even when the token is a direct SKILL.md file path (no dir-match check applies)."""
    from comfy_cli.skills import load_skill_source

    md = tmp_path / "SKILL.md"
    md.write_text(f"---\nname: {bad_name}\ndescription: d.\n---\nBody.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simple slug"):
        load_skill_source(str(md))


def test_load_skill_source_accepts_slug_names(tmp_path):
    from comfy_cli.skills import load_skill_source

    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: anime-video_v2\ndescription: d.\n---\nBody.\n", encoding="utf-8")
    assert load_skill_source(str(md)).name == "anime-video_v2"


def test_validate_command_rejects_evil_name(tmp_path: Path):
    """comfy skills validate must exit 1 and emit skill_invalid for a traversal name."""
    import json

    md = tmp_path / "SKILL.md"
    md.write_text("---\nname: ../../evil\ndescription: d.\n---\nBody.\n", encoding="utf-8")

    _force_json_renderer()
    try:
        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(md)])
        assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}\noutput: {result.output}"

        lines = [line for line in result.output.splitlines() if line.strip()]
        envelope = json.loads(lines[-1])
        assert envelope["ok"] is False
        assert envelope["error"]["code"] == "skill_invalid"
    finally:
        reset_renderer_for_testing()


# ---------------------------------------------------------------------------
# Manifest (provenance + staleness) — Task 9
# ---------------------------------------------------------------------------


def test_install_records_manifest(tmp_path: Path):
    """Installing writes a manifest entry: target path -> {skill, sha256, cli_version}."""
    import hashlib

    from comfy_cli.skills import manifest_path, read_manifest, skill_content

    results = install(scope="project", project_root=tmp_path, skills=["comfy-debug"], targets=["claude-code"])
    assert results[0].action == "wrote"

    mpath = manifest_path()
    assert mpath.exists(), f"manifest file not created at {mpath}"
    manifest = read_manifest()

    installed_path = str(tmp_path / ".claude" / "skills" / "comfy-debug" / "SKILL.md")
    assert installed_path in manifest, f"manifest missing entry for {installed_path}"
    entry = manifest[installed_path]
    assert entry["skill"] == "comfy-debug"
    assert "sha256" in entry
    assert "cli_version" in entry

    expected_sha = hashlib.sha256(skill_content("comfy-debug").encode("utf-8")).hexdigest()
    assert entry["sha256"] == expected_sha


def test_prune_skips_unmanaged_files(tmp_path: Path):
    """A user-authored skill at a retired-name path is NOT deleted by prune when manifest exists."""
    from comfy_cli.skills import manifest_path, write_manifest

    # Seed a file at a retired-skill path but do NOT put it in the manifest.
    retired_name = RETIRED_SKILLS[0]
    claude_path = tmp_path / ".claude" / "skills" / retired_name / "SKILL.md"
    claude_path.parent.mkdir(parents=True, exist_ok=True)
    claude_path.write_text("# user-authored content — must survive\n", encoding="utf-8")

    # Write a manifest that exists but does NOT contain this path.
    mpath = manifest_path()
    mpath.parent.mkdir(parents=True, exist_ok=True)
    write_manifest({"some/other/path": {"skill": "comfy", "sha256": "abc", "cli_version": "0.0.0"}})

    # Run prune — manifest exists but doesn't record this file as comfy-managed.
    results = prune_retired(scope="project", project_root=tmp_path, targets=["claude-code"])

    assert claude_path.exists(), "prune must NOT delete unmanaged files when manifest exists"
    result_for_name = [r for r in results if r.skill == retired_name and r.kind == "claude-code"]
    assert result_for_name, "should still report a result for the retired skill"
    assert result_for_name[0].action == "absent"


def test_status_reports_stale_and_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """status distinguishes current / stale (bundled content moved on) / modified (user edited)."""
    import hashlib
    import json as _json

    from comfy_cli.skills import read_manifest, write_manifest
    from comfy_cli.skills.command import app

    # status_cmd resolves project_root from Path.cwd() — point it at tmp_path.
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()

    def _invoke_status():
        """Re-arm the JSON renderer for each CliRunner invocation (each creates a new stdout)."""
        _force_json_renderer()
        try:
            result = runner.invoke(app, ["status", "--scope", "project"])
        finally:
            reset_renderer_for_testing()
        return result

    try:
        # Step 1: install comfy-debug into claude-code only, then check status.
        install(scope="project", project_root=tmp_path, skills=["comfy-debug"], targets=["claude-code"])
        result = _invoke_status()
        assert result.exit_code == 0, result.output
        lines = [ln for ln in result.output.splitlines() if ln.strip()]
        envelope = _json.loads(lines[-1])
        rows = envelope["data"]["targets"]
        debug_row = next((r for r in rows if r["skill"] == "comfy-debug" and r["kind"] == "claude-code"), None)
        assert debug_row is not None
        assert debug_row["state"] == "current"

        # Step 2: edit the installed file — state should become 'modified'.
        installed_path = tmp_path / ".claude" / "skills" / "comfy-debug" / "SKILL.md"
        installed_path.write_text(installed_path.read_text(encoding="utf-8") + "\n# user edit\n", encoding="utf-8")

        result2 = _invoke_status()
        assert result2.exit_code == 0, result2.output
        lines2 = [ln for ln in result2.output.splitlines() if ln.strip()]
        envelope2 = _json.loads(lines2[-1])
        rows2 = envelope2["data"]["targets"]
        debug_row2 = next((r for r in rows2 if r["skill"] == "comfy-debug" and r["kind"] == "claude-code"), None)
        assert debug_row2 is not None
        assert debug_row2["state"] == "modified"

        # Step 3: restore the file to bundled content but with an old (stale) manifest hash.
        # file sha == manifest sha != bundled sha  →  state == 'stale'
        stale_content = "# stale version of the skill\n"
        installed_path.write_text(stale_content, encoding="utf-8")
        manifest = read_manifest()
        manifest[str(installed_path)] = {
            "skill": "comfy-debug",
            "sha256": hashlib.sha256(stale_content.encode("utf-8")).hexdigest(),
            "cli_version": "0.0.0",
        }
        write_manifest(manifest)

        result3 = _invoke_status()
        assert result3.exit_code == 0, result3.output
        lines3 = [ln for ln in result3.output.splitlines() if ln.strip()]
        envelope3 = _json.loads(lines3[-1])
        rows3 = envelope3["data"]["targets"]
        debug_row3 = next((r for r in rows3 if r["skill"] == "comfy-debug" and r["kind"] == "claude-code"), None)
        assert debug_row3 is not None
        assert debug_row3["state"] == "stale"
    finally:
        reset_renderer_for_testing()
