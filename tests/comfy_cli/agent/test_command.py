"""``comfy agent permissions`` / ``comfy agent allow`` against a temp data dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.agent import allow_host, allow_path, read_state, vet_host, vet_path
from comfy_cli.agent.command import app
from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def _json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    yield
    reset_renderer_for_testing()


def _envelope(result) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_allow_path_records_and_deduplicates(tmp_path: Path):
    root = tmp_path / "data"
    folder = tmp_path / "Pictures"
    folder.mkdir()
    p, added = allow_path(root, str(folder), "reference photos")
    assert added and p == folder
    _, again = allow_path(root, str(folder), "again")
    assert not again
    state = read_state(root)
    assert [e["path"] for e in state.paths] == [str(folder)]
    assert state.paths[0]["reason"] == "reference photos"
    assert state.paths[0]["approved_at"].endswith("Z")
    assert not state.running


def test_vet_path_refuses_what_the_agent_refuses(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / "Documents").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    with pytest.raises(ValueError):
        vet_path("relative/folder")
    with pytest.raises(ValueError):
        vet_path(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="credential"):
        vet_path(str(home / ".ssh"))
    with pytest.raises(ValueError, match="contains"):
        vet_path(str(home))
    assert vet_path(str(home / "Documents")) == home / "Documents"


def test_vet_host_strips_scheme_port_and_case():
    assert vet_host("HTTPS://Models.Example.com:443/x") == "models.example.com"
    assert vet_host("cdn.example.com.") == "cdn.example.com"
    with pytest.raises(ValueError):
        vet_host("")


def test_allow_host_records(tmp_path: Path):
    root = tmp_path / "data"
    h, added = allow_host(root, "https://models.example.com/a", "a VAE")
    assert (h, added) == ("models.example.com", True)
    assert json.loads((root / "egress-allow.json").read_text())["hosts"][0]["host"] == "models.example.com"


def test_cli_allow_and_permissions_round_trip(tmp_path: Path):
    root = tmp_path / "data"
    folder = tmp_path / "refs"
    folder.mkdir()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["allow", "--path", str(folder), "--host", "models.example.com", "--reason", "test", "--data-dir", str(root)],
    )
    assert res.exit_code == 0, res.stdout
    env = _envelope(res)
    assert env["ok"] is True
    assert env["data"]["path"]["added"] is True
    assert env["data"]["host"]["host"] == "models.example.com"

    _json_renderer()
    res = runner.invoke(app, ["permissions", "--data-dir", str(root)])
    assert res.exit_code == 0, res.stdout
    env = _envelope(res)
    assert env["data"]["agent"]["running"] is False
    assert [e["path"] for e in env["data"]["paths"]] == [str(folder)]
    assert [e["host"] for e in env["data"]["hosts"]] == ["models.example.com"]


def test_cli_allow_refuses_a_missing_folder(tmp_path: Path):
    runner = CliRunner()
    res = runner.invoke(app, ["allow", "--path", str(tmp_path / "nope"), "--data-dir", str(tmp_path / "data")])
    assert res.exit_code == 1
    assert _envelope(res)["error"]["code"] == "refused"


def test_cli_allow_needs_something(tmp_path: Path):
    res = CliRunner().invoke(app, ["allow", "--data-dir", str(tmp_path)])
    assert res.exit_code == 2
