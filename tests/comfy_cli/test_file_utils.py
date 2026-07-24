import zipfile

import pytest

from comfy_cli import file_utils
from comfy_cli.file_utils import atomic_write_text


def test_zip_files_respects_comfyignore(tmp_path, monkeypatch):
    project_dir = tmp_path
    (project_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (project_dir / "ignore.log").write_text("ignore", encoding="utf-8")
    ignored_dir = project_dir / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "nested.txt").write_text("nested", encoding="utf-8")

    (project_dir / ".comfyignore").write_text("*.log\nignored_dir/\n", encoding="utf-8")

    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        file_utils,
        "list_git_tracked_files",
        lambda base_path=".": [
            "keep.txt",
            "ignore.log",
            "ignored_dir/nested.txt",
        ],
    )

    file_utils.zip_files(str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "keep.txt" in names
    assert "ignore.log" not in names
    assert not any(name.startswith("ignored_dir/") for name in names)


def test_zip_files_force_include_overrides_ignore(tmp_path, monkeypatch):
    project_dir = tmp_path
    include_dir = project_dir / "include_me"
    include_dir.mkdir()
    (include_dir / "data.json").write_text("{}", encoding="utf-8")

    (project_dir / "other.txt").write_text("ok", encoding="utf-8")
    (project_dir / ".comfyignore").write_text("include_me/\n", encoding="utf-8")

    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        file_utils,
        "list_git_tracked_files",
        lambda base_path=".": [
            "other.txt",
            "include_me/data.json",
        ],
    )

    file_utils.zip_files(str(zip_path), includes=["include_me"])

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "include_me/data.json" in names
    assert "other.txt" in names


def test_zip_files_without_git_falls_back_to_walk(tmp_path, monkeypatch):
    project_dir = tmp_path
    (project_dir / "file.txt").write_text("data", encoding="utf-8")
    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(file_utils, "list_git_tracked_files", lambda base_path=".": [])

    file_utils.zip_files(str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "file.txt" in names
    assert "node.zip" not in names


def test_atomic_write_text_creates_file_and_parents(tmp_path):
    target = tmp_path / "sub" / "dir" / "out.json"
    atomic_write_text(target, '{"a": 1}')

    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    # No stray tmp files left behind.
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_text_fsync_true_still_writes(tmp_path):
    target = tmp_path / "durable.txt"
    atomic_write_text(target, "durable", fsync=True)

    assert target.read_text(encoding="utf-8") == "durable"
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_text_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "out.txt"
    target.write_text("original", encoding="utf-8")

    def boom(src, dst):
        raise OSError("replace failed")

    # Fail at the rename step, after the tmp file has been written.
    monkeypatch.setattr(file_utils.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "new content")

    # The tmp file is cleaned up and the destination is untouched.
    assert list(target.parent.glob("*.tmp")) == []
    assert target.read_text(encoding="utf-8") == "original"
