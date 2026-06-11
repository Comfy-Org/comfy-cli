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


# ---------------------------------------------------------------------------
# make_asset_resolver — `$asset.<name>` → cloud_name via the push lock
# ---------------------------------------------------------------------------


def _write_lock(root, assets: dict) -> None:
    (root / ".comfy").mkdir(exist_ok=True)
    (root / ".comfy" / "assets.lock.json").write_text(json.dumps({"schema": "assets-lock/1", "assets": assets}))


def _add_asset(root, name: str, data: bytes) -> str:
    path = root / "assets" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    import hashlib

    return hashlib.sha256(data).hexdigest()


class TestMakeAssetResolver:
    def test_asset_error_is_a_blueprint_error(self):
        from comfy_cli.fragments import AssetError, BlueprintError

        assert issubclass(AssetError, BlueprintError)

    def test_resolves_pushed_asset_to_cloud_name(self, tmp_path):
        p = _make_project(tmp_path)
        sha = _add_asset(p.root, "keyframes/s1.png", b"png-bytes")
        _write_lock(p.root, {"keyframes/s1.png": {"sha256": sha, "cloud_name": "ab12.png"}})

        resolve = project_mod.make_asset_resolver(p)
        assert resolve("keyframes/s1.png") == "ab12.png"

    def test_missing_lock_file_raises_not_pushed(self, tmp_path):
        from comfy_cli.fragments import AssetError

        p = _make_project(tmp_path)
        _add_asset(p.root, "a.png", b"x")

        resolve = project_mod.make_asset_resolver(p)
        with pytest.raises(AssetError) as exc:
            resolve("a.png")
        assert exc.value.code == "asset_not_pushed"
        assert exc.value.hint == "run: comfy assets push"

    def test_name_missing_from_lock_raises_not_pushed(self, tmp_path):
        from comfy_cli.fragments import AssetError

        p = _make_project(tmp_path)
        sha = _add_asset(p.root, "a.png", b"x")
        _write_lock(p.root, {"other.png": {"sha256": sha, "cloud_name": "zz.png"}})

        with pytest.raises(AssetError) as exc:
            project_mod.make_asset_resolver(p)("a.png")
        assert exc.value.code == "asset_not_pushed"
        assert exc.value.hint == "run: comfy assets push"

    def test_file_missing_on_disk_raises_not_pushed(self, tmp_path):
        from comfy_cli.fragments import AssetError

        p = _make_project(tmp_path)
        _write_lock(p.root, {"ghost.png": {"sha256": "0" * 64, "cloud_name": "gh.png"}})

        with pytest.raises(AssetError) as exc:
            project_mod.make_asset_resolver(p)("ghost.png")
        assert exc.value.code == "asset_not_pushed"
        assert exc.value.hint == "run: comfy assets push"

    def test_sha_mismatch_raises_stale(self, tmp_path):
        from comfy_cli.fragments import AssetError

        p = _make_project(tmp_path)
        _add_asset(p.root, "a.png", b"new-content")
        _write_lock(p.root, {"a.png": {"sha256": "0" * 64, "cloud_name": "old.png"}})

        with pytest.raises(AssetError) as exc:
            project_mod.make_asset_resolver(p)("a.png")
        assert exc.value.code == "asset_stale"
        assert exc.value.hint == "run: comfy assets push"

    def test_hashing_is_lazy_per_referenced_name(self, tmp_path):
        """A broken lock entry for an UNREFERENCED name must not matter."""
        p = _make_project(tmp_path)
        sha = _add_asset(p.root, "good.png", b"good")
        _write_lock(
            p.root,
            {
                "good.png": {"sha256": sha, "cloud_name": "good-srv.png"},
                "ghost.png": {"sha256": "0" * 64, "cloud_name": "gh.png"},  # no file on disk
            },
        )

        assert project_mod.make_asset_resolver(p)("good.png") == "good-srv.png"


# ---------------------------------------------------------------------------
# make_var_resolver — `$var.<name>` → scalar from comfy.yaml `vars:`
# ---------------------------------------------------------------------------


def _make_project_with_vars(tmp_path, vars_yaml: str) -> Project:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "comfy.yaml").write_text(MARKER + vars_yaml)
    p = find_project(root)
    assert p is not None
    return p


class TestMakeVarResolver:
    def test_var_error_is_a_blueprint_error_with_code(self):
        from comfy_cli.fragments import BlueprintError, VarError

        assert issubclass(VarError, BlueprintError)

    def test_resolves_scalars_with_raw_types(self, tmp_path):
        """$var returns the raw YAML scalar — int stays int, bool stays bool —
        so non-STRING params keep their widget types."""
        p = _make_project_with_vars(
            tmp_path,
            "vars:\n  style: cinematic, golden hour\n  steps: 28\n  cfg: 4.5\n  hires: true\n",
        )
        resolve = project_mod.make_var_resolver(p)
        assert resolve("style") == "cinematic, golden hour"
        assert resolve("steps") == 28 and isinstance(resolve("steps"), int)
        assert resolve("cfg") == 4.5
        assert resolve("hires") is True

    def test_missing_name_raises_var_not_defined(self, tmp_path):
        from comfy_cli.fragments import VarError

        p = _make_project_with_vars(tmp_path, "vars:\n  style: x\n")
        with pytest.raises(VarError) as exc:
            project_mod.make_var_resolver(p)("nope")
        assert exc.value.code == "var_not_defined"
        assert "vars:" in (exc.value.hint or "")
        assert str(p.root / "comfy.yaml") in (exc.value.hint or "")

    def test_no_vars_block_treated_as_empty(self, tmp_path):
        from comfy_cli.fragments import VarError

        p = _make_project(tmp_path)
        with pytest.raises(VarError) as exc:
            project_mod.make_var_resolver(p)("style")
        assert exc.value.code == "var_not_defined"

    def test_non_mapping_vars_block_treated_as_empty(self, tmp_path):
        from comfy_cli.fragments import VarError

        p = _make_project_with_vars(tmp_path, "vars: [a, b]\n")
        with pytest.raises(VarError) as exc:
            project_mod.make_var_resolver(p)("a")
        assert exc.value.code == "var_not_defined"
