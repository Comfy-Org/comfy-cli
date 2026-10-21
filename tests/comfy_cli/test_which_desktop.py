"""`comfy which` / `comfy env` on a machine where the workspace is missing or is Comfy Desktop's.

The default workspace comfy-cli falls back to (``~/Documents/comfy/ComfyUI``)
does not exist on a Desktop-only machine. These commands used to report it with
``ok: true``; now a missing path is an error (`which`) or a flagged warning
(`env`), and a Desktop 2 install is reported with its real split layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli import utils
from comfy_cli.cmdline import app

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "comfy_cli" / "schemas"


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / name).read_text())


@pytest.fixture
def fallback_workspace(tmp_path, monkeypatch) -> Path:
    """comfy-cli's own default workspace, pointed into tmp and not created.

    The cwd is moved to an empty dir so the cwd auto-detect step finds nothing.
    """
    fallback = tmp_path / "Documents" / "comfy" / "ComfyUI"
    monkeypatch.setattr(utils, "get_not_user_set_default_workspace", lambda: str(fallback))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return fallback


def _invoke(*args: str):
    result = CliRunner().invoke(app, list(args))
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    envelope = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else None
    return result, envelope


def test_which_reports_the_desktop_install_when_nothing_else_is_selected(comfy_desktop, fallback_workspace):
    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 0, result.output
    assert envelope["ok"] is True
    data = envelope["data"]
    jsonschema.validate(data, _schema("which.json"))
    assert data["workspace_type"] == "desktop"
    assert data["workspace_path"] == str(comfy_desktop["code"])
    assert data["exists"] is True
    desktop = data["desktop"]
    assert desktop["code_dir"] == str(comfy_desktop["code"])
    assert desktop["data_dir"] == str(comfy_desktop["data"])
    assert desktop["python"] == str(comfy_desktop["python"])
    assert desktop["comfyui_version"] == "v0.34.2"
    assert [root["path"] for root in desktop["model_roots"]] == [
        str(comfy_desktop["data"] / "models"),
        str(comfy_desktop["shared"]),
    ]
    assert desktop["selected_reason"]
    assert [row["install_id"] for row in desktop["installs"]] == ["inst-adopted"]


def test_which_lists_every_desktop_install_and_says_which_it_picked(comfy_desktop, fallback_workspace, tmp_path):
    other = tmp_path / "ComfyUI-Installs" / "Test"
    (other / "ComfyUI").mkdir(parents=True)
    (other / "ComfyUI" / "main.py").write_text("")
    records = [
        {"id": "inst-test", "name": "Test", "installPath": str(other), "lastLaunchedAt": 1_700_000_000_000},
        comfy_desktop["record"],
    ]
    (comfy_desktop["app"] / "installations.json").write_text(json.dumps(records))

    _, envelope = _invoke("--json", "which")

    desktop = envelope["data"]["desktop"]
    assert desktop["install_id"] == "inst-adopted"
    assert "most recently launched of 2" in desktop["selected_reason"]
    assert {row["install_id"]: row["selected"] for row in desktop["installs"]} == {
        "inst-test": False,
        "inst-adopted": True,
    }


def test_which_errors_when_the_default_workspace_does_not_exist(fallback_workspace):
    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "workspace_not_found"
    assert envelope["error"]["details"]["path"] == str(fallback_workspace)
    assert envelope["error"]["details"]["workspace_type"] == "default"
    assert envelope["error"]["hint"]


def test_which_errors_for_a_missing_specified_workspace_even_with_desktop_installed(
    comfy_desktop, fallback_workspace, tmp_path
):
    missing = tmp_path / "typo" / "ComfyUI"

    result, envelope = _invoke("--json", "--workspace", str(missing), "which")

    assert result.exit_code == 1
    assert envelope["error"]["code"] == "workspace_not_found"
    assert envelope["error"]["details"]["path"] == str(missing)
    assert envelope["error"]["details"]["workspace_type"] == "specified"


def test_which_keeps_an_installed_default_workspace_over_desktop(comfy_desktop, fallback_workspace):
    fallback_workspace.mkdir(parents=True)

    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 0, result.output
    assert envelope["data"]["workspace_path"] == str(fallback_workspace)
    assert envelope["data"]["workspace_type"] == "default"
    assert envelope["data"]["exists"] is True
    assert "desktop" not in envelope["data"]


def test_which_names_unreadable_desktop_records_in_the_error(comfy_desktop, fallback_workspace):
    (comfy_desktop["app"] / "installations.json").write_text("{not json")

    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 1
    assert envelope["error"]["code"] == "workspace_not_found"
    problems = envelope["error"]["details"]["desktop"]["problems"]
    assert len(problems) == 1
    assert "installations.json" in problems[0]


def test_which_warns_when_desktop_settings_are_unreadable(comfy_desktop, fallback_workspace):
    (comfy_desktop["app"] / "settings.json").write_text("[broken")

    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 0, result.output
    assert envelope["data"]["workspace_type"] == "desktop"
    assert [w["code"] for w in envelope["data"]["warnings"]] == ["desktop_config_unreadable"]


def test_pretty_which_shows_the_desktop_data_dir(comfy_desktop, fallback_workspace):
    result = CliRunner().invoke(app, ["which"], env={"COMFY_OUTPUT": "pretty", "COLUMNS": "400"})

    assert result.exit_code == 0, result.output
    assert str(comfy_desktop["data"]) in result.output
    assert "desktop" in result.output


def test_env_reports_the_desktop_install(comfy_desktop, fallback_workspace):
    result, envelope = _invoke("--json", "env")

    assert result.exit_code == 0, result.output
    data = envelope["data"]
    jsonschema.validate(data, _schema("env.json"))
    workspace = data["workspace"]
    assert workspace["type"] == "desktop"
    assert workspace["path"] == str(comfy_desktop["code"])
    assert workspace["exists"] is True
    assert workspace["desktop"]["data_dir"] == str(comfy_desktop["data"])
    assert "warnings" not in data


def test_env_flags_a_missing_workspace_instead_of_a_silent_default(fallback_workspace):
    result, envelope = _invoke("--json", "env")

    assert result.exit_code == 0, result.output
    data = envelope["data"]
    jsonschema.validate(data, _schema("env.json"))
    assert data["workspace"]["path"] == str(fallback_workspace)
    assert data["workspace"]["exists"] is False
    assert [w["code"] for w in data["warnings"]] == ["workspace_not_found"]
    assert str(fallback_workspace) in data["warnings"][0]["message"]


@pytest.mark.parametrize("part", ["code", "data"])
def test_which_enriches_an_explicit_desktop_workspace(comfy_desktop, fallback_workspace, part):
    result, envelope = _invoke("--json", "--workspace", str(comfy_desktop[part]), "which")

    assert result.exit_code == 0, result.output
    assert envelope["data"]["workspace_type"] == "specified"
    assert envelope["data"]["workspace_path"] == str(comfy_desktop[part])
    assert envelope["data"]["desktop"]["data_dir"] == str(comfy_desktop["data"])
    assert envelope["data"]["desktop"]["code_dir"] == str(comfy_desktop["code"])


def test_env_warns_when_desktop_settings_are_unreadable(comfy_desktop, fallback_workspace):
    (comfy_desktop["app"] / "settings.json").write_text("[broken")

    result, envelope = _invoke("--json", "env")

    assert result.exit_code == 0, result.output
    assert [w["code"] for w in envelope["data"]["warnings"]] == ["desktop_config_unreadable"]
