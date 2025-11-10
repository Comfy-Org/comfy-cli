import os
import zipfile
from pathlib import Path

import pytest

from comfy_cli import file_utils


@pytest.fixture(autouse=True)
def restore_cwd():
    original_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)


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
