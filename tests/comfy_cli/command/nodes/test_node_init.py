import subprocess

import tomlkit
from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.command import app

runner = CliRunner()


def test_node_init_strips_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://ghp_FAKESECRET123@github.com/user/ComfyUI-TestNode.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "requirements.txt").write_text("requests\n")

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    with open(tmp_path / "pyproject.toml") as f:
        data = tomlkit.parse(f.read())
    raw = tomlkit.dumps(data)
    assert "ghp_FAKESECRET123" not in raw
    assert data["project"]["urls"]["Repository"] == "https://github.com/user/ComfyUI-TestNode"


def test_node_init_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 1
    assert "already exists" in result.stdout
