import subprocess
import zipfile
from unittest.mock import patch

from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.registry.config_parser import extract_node_configuration

PYPROJECT = """\
[project]
name = "test-node"
version = "1.0.0"
description = "A test node"
license = {text = "MIT"}

[tool.comfy]
PublisherId = "test-publisher"
DisplayName = "Test Node"
includes = ["models"]
"""


def test_pack_creates_zip_with_correct_contents(tmp_path, monkeypatch):
    """Full integration: git repo + pyproject.toml + .comfyignore + includes -> zip.

    Verifies that `comfy node pack`:
    - includes git-tracked files
    - excludes files matched by .comfyignore (even if git-tracked)
    - excludes untracked files
    - force-includes directories listed in [tool.comfy] includes (even if untracked)
    - does not include the zip file itself
    """
    monkeypatch.chdir(tmp_path)
    # extract_node_configuration's default path is frozen at import time;
    # patch it so it reads pyproject.toml from the temp directory.
    monkeypatch.setattr(extract_node_configuration, "__defaults__", (str(tmp_path / "pyproject.toml"),))

    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "__init__.py").write_text("# entry\n")
    (tmp_path / "nodes.py").write_text("class MyNode: pass\n")
    (tmp_path / ".comfyignore").write_text("*.log\n")
    (tmp_path / "debug.log").write_text("log output\n")

    # Non-git-tracked directory listed in includes
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "weights.bin").write_bytes(b"\x00" * 8)

    # Init git and commit (debug.log is git-tracked but .comfyignore'd)
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "add", "pyproject.toml", "__init__.py", "nodes.py", ".comfyignore", "debug.log"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

    # Create untracked file after commit
    (tmp_path / "untracked.txt").write_text("not in git\n")

    with patch("comfy_cli.tracking.prompt_tracking_consent"):
        result = CliRunner().invoke(app, ["node", "pack"])
    assert result.exit_code == 0, result.output

    zip_path = tmp_path / "node.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    # Git-tracked files present
    assert "pyproject.toml" in names
    assert "__init__.py" in names
    assert "nodes.py" in names

    # .comfyignore excludes git-tracked file
    assert "debug.log" not in names

    # Untracked file excluded
    assert "untracked.txt" not in names

    # includes directory added despite not being git-tracked
    assert "models/weights.bin" in names

    # Zip itself not included
    assert "node.zip" not in names
