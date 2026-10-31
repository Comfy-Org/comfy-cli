"""`comfy which` / `comfy env` on a machine where the workspace is missing or is Comfy Desktop's.

The default workspace comfy-cli falls back to (``~/Documents/comfy/ComfyUI``)
does not exist on a Desktop-only machine. These commands used to report it with
``ok: true``; now a missing path is an error (`which`) or a flagged warning
(`env`), and a Desktop 2 install is reported with its real split layout.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli import desktop_install, utils
from comfy_cli.cmdline import app, workspace_manager
from comfy_cli.env_checker import EnvChecker

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


def test_which_keeps_missing_selected_workspace_separate_from_desktop_candidate(comfy_desktop, fallback_workspace):
    result, envelope = _invoke("--json", "which")

    assert result.exit_code == 1, result.output
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "workspace_not_found"
    details = envelope["error"]["details"]
    assert details["workspace_type"] == workspace_manager.workspace_type.value == "default"
    assert details["path"] == workspace_manager.workspace_path == str(fallback_workspace)
    assert not Path(workspace_manager.workspace_path).is_dir()
    desktop = details["desktop"]
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

    desktop = envelope["error"]["details"]["desktop"]
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

    assert result.exit_code == 1, result.output
    assert envelope["error"]["code"] == "workspace_not_found"
    desktop = envelope["error"]["details"]["desktop"]
    assert any("settings.json" in problem for problem in desktop["problems"])


def test_pretty_which_shows_the_desktop_data_dir(comfy_desktop, fallback_workspace):
    result = CliRunner().invoke(
        app,
        ["--workspace", str(comfy_desktop["code"]), "which"],
        env={"COMFY_OUTPUT": "pretty", "COLUMNS": "400"},
    )

    assert result.exit_code == 0, result.output
    assert str(comfy_desktop["data"]) in result.output
    assert "Desktop data" in result.output
    assert "specified" in result.output


def test_env_reports_the_desktop_install(comfy_desktop, fallback_workspace):
    result, envelope = _invoke("--json", "env")

    assert result.exit_code == 0, result.output
    data = envelope["data"]
    jsonschema.validate(data, _schema("env.json"))
    workspace = data["workspace"]
    assert workspace["type"] == workspace_manager.workspace_type.value == "default"
    assert workspace["path"] == workspace_manager.workspace_path == str(fallback_workspace)
    assert workspace["exists"] is False
    assert workspace["desktop"]["data_dir"] == str(comfy_desktop["data"])
    assert [w["code"] for w in data["warnings"]] == ["workspace_not_found"]


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
    jsonschema.validate(envelope["data"], _schema("which.json"))
    assert envelope["data"]["workspace_type"] == "specified"
    assert envelope["data"]["workspace_path"] == str(comfy_desktop[part])
    assert envelope["data"]["desktop"]["data_dir"] == str(comfy_desktop["data"])
    assert envelope["data"]["desktop"]["code_dir"] == str(comfy_desktop["code"])


def test_env_warns_when_desktop_settings_are_unreadable(comfy_desktop, fallback_workspace):
    (comfy_desktop["app"] / "settings.json").write_text("[broken")

    result, envelope = _invoke("--json", "env")

    assert result.exit_code == 0, result.output
    assert [w["code"] for w in envelope["data"]["warnings"]] == ["workspace_not_found", "desktop_config_unreadable"]


def test_env_renders_hostile_discovery_values_literally_without_changing_json(
    comfy_desktop, fallback_workspace, monkeypatch
):
    raw = "[/]literal[red]x[/red]\x1b[2J\x1b]0;osc-title\x07\x9b2J\r\u202e世界"
    visible = "[/]literal[red]x[/red]\u202e世界"
    selected_path = str(fallback_workspace.parent / raw)
    monkeypatch.setattr(utils, "get_not_user_set_default_workspace", lambda: selected_path)
    found = desktop_install.detect()
    candidate = replace(
        found.selected,
        code_dir=f"code:{raw}",
        data_dir=f"data:{raw}",
        model_roots=(desktop_install.ModelRoot(f"model:{raw}", "data", True),),
    )
    found = replace(found, installs=(candidate,), selected=candidate, problems=(f"warning:{raw}",))
    monkeypatch.setattr(desktop_install, "detect", lambda: found)
    monkeypatch.setattr(EnvChecker(), "fill_print_table", lambda: [])

    json_result, envelope = _invoke("--json", "env")
    assert json_result.exit_code == 0, json_result.output
    workspace = envelope["data"]["workspace"]
    assert workspace["path"] == selected_path
    assert workspace["desktop"]["code_dir"] == f"code:{raw}"
    assert workspace["desktop"]["data_dir"] == f"data:{raw}"
    assert workspace["desktop"]["model_roots"][0]["path"] == f"model:{raw}"
    assert any(warning["message"] == f"warning:{raw}" for warning in envelope["data"]["warnings"])

    result = CliRunner().invoke(app, ["env"], env={"COMFY_OUTPUT": "pretty", "COLUMNS": "600", "NO_COLOR": "1"})

    assert result.exit_code == 0, result.output
    assert str(fallback_workspace.parent / visible) in result.output
    assert f"code:{visible}" in result.output
    assert f"data:{visible}" in result.output
    assert f"model:{visible}" in result.output
    assert f"warning:{visible}" in result.output
    assert "\x1b" not in result.output and "\x9b" not in result.output
    assert "\x07" not in result.output and "\r" not in result.output
    assert "osc-title" not in result.output
    assert "\u202e世界" in result.output


def test_which_renders_hostile_desktop_fields_literally_without_changing_json(
    comfy_desktop, fallback_workspace, monkeypatch
):
    raw = "[/]literal[red]x[/red]\x1b[2J\x1b]0;osc-title\x07\x9b2J\r\u202e世界"
    visible = "[/]literal[red]x[/red]\u202e世界"
    code = fallback_workspace.parent / "selected-comfy"
    desktop_code = fallback_workspace.parent / f"code:{raw}"
    code.mkdir(parents=True)
    (code / "main.py").write_text("")
    found = desktop_install.detect()
    candidate = replace(
        found.selected,
        code_dir=str(desktop_code),
        data_dir=f"data:{raw}",
        python=f"python:{raw}",
        model_roots=(desktop_install.ModelRoot(f"model:{raw}", "data", True),),
    )
    found = replace(
        found, installs=(candidate,), selected=candidate, selected_reason=f"reason:{raw}", problems=(f"warning:{raw}",)
    )
    monkeypatch.setattr(desktop_install, "detect", lambda: found)
    monkeypatch.setattr(desktop_install, "install_for_dir", lambda path, detected: {str(code): candidate}.get(path))
    monkeypatch.setattr("comfy_cli.cmdline.env_checker.check_comfy_server_running", lambda **kwargs: False)

    json_result, envelope = _invoke("--json", "--workspace", str(code), "which")
    assert json_result.exit_code == 0, json_result.output
    data = envelope["data"]
    assert data["workspace_path"] == str(code)
    assert data["desktop"]["code_dir"] == str(desktop_code)
    assert data["desktop"]["data_dir"] == f"data:{raw}"
    assert data["desktop"]["python"] == f"python:{raw}"
    assert data["desktop"]["model_roots"][0]["path"] == f"model:{raw}"
    assert data["desktop"]["selected_reason"] == f"reason:{raw}"
    assert data["warnings"][0]["message"] == f"warning:{raw}"

    result = CliRunner().invoke(
        app, ["--workspace", str(code), "which"], env={"COMFY_OUTPUT": "pretty", "COLUMNS": "600", "NO_COLOR": "1"}
    )

    assert result.exit_code == 0, result.output
    assert str(fallback_workspace.parent / f"code:{visible}") in result.output
    assert f"data:{visible}" in result.output
    assert f"python:{visible}" in result.output
    assert f"model:{visible}" in result.output
    assert f"reason:{visible}" in result.output
    assert f"warning:{visible}" in result.output
    assert "\x1b" not in result.output and "\x9b" not in result.output
    assert "\x07" not in result.output and "\r" not in result.output
    assert "osc-title" not in result.output
    assert "\u202e世界" in result.output
