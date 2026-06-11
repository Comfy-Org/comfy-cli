"""Unit tests for ``comfy_cli.project`` — the pure project/1 convention module.

Everything runs against tmp_path. The module must never raise out of
discovery or journaling: a stray/malformed ``comfy.yaml`` anywhere up the
tree must not crash a non-project command, and a journaling failure must
never fail a run.
"""

from __future__ import annotations

import json
import os

import pytest

from comfy_cli import project as project_mod
from comfy_cli.project import (
    CONVENTIONAL_DIRS,
    Project,
    find_project,
    journal,
    read_journal,
    unknown_dirs,
)

MARKER = "schema: project/1\ndefaults:\n  where: cloud\n"


def _make_project(tmp_path) -> Project:
    # A dedicated subdir: the autouse conftest fixtures plant their own
    # dirs (comfy-cli-config/, jobs/) directly in tmp_path.
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "comfy.yaml").write_text(MARKER)
    p = find_project(root)
    assert p is not None
    return p


# ---------------------------------------------------------------------------
# find_project — walk-up discovery
# ---------------------------------------------------------------------------


class TestFindProject:
    def test_finds_marker_in_start_dir(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        p = find_project(tmp_path)
        assert p is not None
        assert p.root == tmp_path.resolve()
        assert p.config["schema"] == "project/1"
        assert p.config["defaults"] == {"where": "cloud"}

    def test_walks_up_from_nested_dir(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        nested = tmp_path / "blueprints" / "deep" / "deeper"
        nested.mkdir(parents=True)
        p = find_project(nested)
        assert p is not None
        assert p.root == tmp_path.resolve()

    def test_stops_at_first_valid_marker(self, tmp_path):
        # Inner project shadows the outer one when starting inside it.
        (tmp_path / "comfy.yaml").write_text(MARKER)
        inner = tmp_path / "sub"
        inner.mkdir()
        (inner / "comfy.yaml").write_text(MARKER)
        p = find_project(inner)
        assert p is not None
        assert p.root == inner.resolve()

    def test_no_marker_returns_none(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_project(nested) is None

    def test_malformed_yaml_skipped_and_walk_continues(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        mid = tmp_path / "mid"
        mid.mkdir()
        (mid / "comfy.yaml").write_text("{::: not yaml :::")
        p = find_project(mid)
        assert p is not None
        assert p.root == tmp_path.resolve()  # malformed level skipped, kept walking

    def test_wrong_schema_skipped_and_walk_continues(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        mid = tmp_path / "mid"
        mid.mkdir()
        (mid / "comfy.yaml").write_text("schema: project/999\n")
        p = find_project(mid)
        assert p is not None
        assert p.root == tmp_path.resolve()

    def test_non_dict_yaml_skipped(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text("- just\n- a\n- list\n")
        assert find_project(tmp_path) is None

    def test_missing_schema_key_skipped(self, tmp_path):
        # A stray comfy.yaml that isn't a project marker (e.g. some other
        # tool's config) must be ignored, never crash.
        (tmp_path / "comfy.yaml").write_text("foo: bar\n")
        assert find_project(tmp_path) is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
    def test_unreadable_marker_skipped_and_walk_continues(self, tmp_path):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        mid = tmp_path / "mid"
        mid.mkdir()
        unreadable = mid / "comfy.yaml"
        unreadable.write_text(MARKER)
        unreadable.chmod(0o000)
        try:
            p = find_project(mid)
        finally:
            unreadable.chmod(0o644)
        assert p is not None
        assert p.root == tmp_path.resolve()

    def test_defaults_to_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "comfy.yaml").write_text(MARKER)
        monkeypatch.chdir(tmp_path)
        p = find_project()
        assert p is not None
        assert p.root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# unknown_dirs — warnings-only convention check
# ---------------------------------------------------------------------------


class TestUnknownDirs:
    def test_conventional_dirs_are_known(self, tmp_path):
        p = _make_project(tmp_path)
        for d in CONVENTIONAL_DIRS:
            (p.root / d).mkdir()
        assert unknown_dirs(p) == []

    def test_flags_unconventional_top_level_dirs(self, tmp_path):
        p = _make_project(tmp_path)
        (p.root / "assets").mkdir()
        (p.root / "scratch").mkdir()
        (p.root / "renders").mkdir()
        assert unknown_dirs(p) == ["renders", "scratch"]

    def test_hidden_dirs_and_files_are_ignored(self, tmp_path):
        p = _make_project(tmp_path)
        (p.root / ".git").mkdir()
        (p.root / ".comfy").mkdir()
        (p.root / "notes.md").write_text("files are fine anywhere")
        assert unknown_dirs(p) == []


# ---------------------------------------------------------------------------
# journal / read_journal — best-effort run provenance
# ---------------------------------------------------------------------------


class TestJournal:
    def test_append_read_round_trip(self, tmp_path):
        p = _make_project(tmp_path)
        journal(p, cmd="run", prompt_id="abc", where="cloud")

        events = read_journal(p)
        assert len(events) == 1
        ev = events[0]
        assert ev["cmd"] == "run"
        assert ev["prompt_id"] == "abc"
        assert ev["where"] == "cloud"
        # ts auto-added: UTC ISO-8601 at seconds precision.
        assert "ts" in ev
        assert "T" in ev["ts"]
        assert "." not in ev["ts"]  # timespec="seconds"

    def test_journal_creates_comfy_dir(self, tmp_path):
        p = _make_project(tmp_path)
        assert not (p.root / ".comfy").exists()
        journal(p, cmd="compose")
        assert (p.root / ".comfy" / "runs.jsonl").is_file()

    def test_newest_last_and_limit(self, tmp_path):
        p = _make_project(tmp_path)
        for i in range(25):
            journal(p, cmd="run", seq=i)
        events = read_journal(p, limit=20)
        assert len(events) == 20
        assert events[-1]["seq"] == 24  # newest last
        assert events[0]["seq"] == 5

    def test_corrupt_journal_line_skipped(self, tmp_path):
        p = _make_project(tmp_path)
        journal(p, cmd="run", seq=0)
        path = p.root / ".comfy" / "runs.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
            fh.write('"a bare string"\n')
        journal(p, cmd="run", seq=1)

        events = read_journal(p)
        assert [e["seq"] for e in events] == [0, 1]

    def test_read_journal_missing_file_returns_empty(self, tmp_path):
        p = _make_project(tmp_path)
        assert read_journal(p) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
    def test_journal_never_raises_on_readonly_dir(self, tmp_path):
        p = _make_project(tmp_path)
        comfy_dir = p.root / ".comfy"
        comfy_dir.mkdir()
        comfy_dir.chmod(0o500)
        try:
            journal(p, cmd="run")  # must swallow the OSError
        finally:
            comfy_dir.chmod(0o755)
        assert read_journal(p) == []

    def test_journal_never_raises_on_unserializable_event(self, tmp_path):
        p = _make_project(tmp_path)
        journal(p, cmd="run", weird=object())  # default=str or swallowed — never raises

    def test_journal_module_is_pure(self):
        # Mirror fragments.py's purity: no typer/renderer imports at module level.
        import sys

        src = (project_mod.__file__ and open(project_mod.__file__).read()) or ""
        assert "import typer" not in src
        assert "comfy_cli.output" not in src
        assert project_mod.__name__ in sys.modules
