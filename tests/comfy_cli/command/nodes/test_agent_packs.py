import json
import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.agent_packs import find_agent_packs
from comfy_cli.command.custom_nodes.command import app

runner = CliRunner()


def _record(pack: str, **overrides) -> dict:
    record = {
        "v": 1,
        "pack": pack,
        "route": "install_custom_node",
        "thread_id": "t-1",
        "tool_call_id": "call-1",
        "created_at": "2026-11-01T10:00:00Z",
        "mac": "ab" * 32,
    }
    record.update(overrides)
    return record


def _make_pack(parent, folder: str, record=None, raw: str | None = None):
    pack = parent / folder
    pack.mkdir(parents=True)
    (pack / "__init__.py").write_text("")
    if raw is not None:
        (pack / ".comfy-agent-install").write_text(raw)
    elif record is not None:
        (pack / ".comfy-agent-install").write_text(json.dumps(record))
    return pack


def test_lists_enabled_and_both_kinds_of_disabled_agent_packs(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "restyler", _record("restyler"))
    _make_pack(custom_nodes, "turned_off.disabled", _record("turned_off", route="write"))
    _make_pack(custom_nodes / ".disabled", "moved@1.0.0", _record("moved"))
    _make_pack(custom_nodes, "ComfyUI-Plain")

    packs = find_agent_packs(custom_nodes)

    assert [(p.pack, p.folder, p.state) for p in packs] == [
        ("moved", ".disabled/moved@1.0.0", "disabled-by-manager"),
        ("restyler", "restyler", "enabled"),
        ("turned_off", "turned_off.disabled", "disabled"),
    ]
    assert packs[1].route == "install_custom_node"
    assert packs[1].created_at == "2026-11-01T10:00:00Z"
    assert packs[1].thread_id == "t-1"


@pytest.mark.parametrize(
    ("folder", "raw"),
    [
        ("unknown_version", json.dumps(_record("unknown_version", v=2))),
        ("not_json", "{"),
        ("not_an_object", "null"),
        ("missing_route", json.dumps(_record("missing_route", route=""))),
        ("missing_created", json.dumps(_record("missing_created", created_at=""))),
        ("renamed", json.dumps(_record("someone_else"))),
        ("oversized", json.dumps(_record("oversized", thread_id="x" * 9000))),
    ],
)
def test_ignores_records_that_do_not_name_this_pack_or_are_malformed(tmp_path, folder, raw):
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, folder, raw=raw)

    assert find_agent_packs(custom_nodes) == []


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_ignores_a_record_that_is_a_symlink(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    pack = _make_pack(custom_nodes, "linked")
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_record("linked")))
    (pack / ".comfy-agent-install").symlink_to(target)

    assert find_agent_packs(custom_nodes) == []


def test_missing_custom_nodes_folder_lists_nothing(tmp_path):
    assert find_agent_packs(tmp_path / "custom_nodes") == []


def test_show_agent_created_reads_the_workspace_without_cm_cli(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "restyler", _record("restyler"))
    with (
        patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute,
        patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws,
    ):
        mock_ws.workspace_path = str(tmp_path)
        result = runner.invoke(app, ["show", "agent-created"])

    assert result.exit_code == 0, result.output
    mock_execute.assert_not_called()
    assert "restyler" in result.output
    assert "enabled" in result.output
    assert "not verified" in result.output


def test_show_agent_created_says_when_there_are_none(tmp_path):
    (tmp_path / "custom_nodes").mkdir()
    with patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws:
        mock_ws.workspace_path = str(tmp_path)
        result = runner.invoke(app, ["show", "agent-created"])

    assert result.exit_code == 0, result.output
    assert "No agent-created packs" in result.output


@pytest.fixture
def _reset_renderer():
    yield
    from comfy_cli.output.renderer import reset_renderer_for_testing

    reset_renderer_for_testing()


def test_show_agent_created_json_envelope_lists_packs_as_unverified(tmp_path, _reset_renderer):
    from comfy_cli import cmdline

    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes / ".disabled", "moved", _record("moved"))
    with patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws:
        mock_ws.workspace_path = str(tmp_path)
        result = CliRunner().invoke(cmdline.app, ["--json", "node", "show", "agent-created"])

    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["data"]["verified"] is False
    assert envelope["data"]["packs"] == [
        {
            "pack": "moved",
            "folder": ".disabled/moved",
            "state": "disabled-by-manager",
            "route": "install_custom_node",
            "created_at": "2026-11-01T10:00:00Z",
            "thread_id": "t-1",
            "tool_call_id": "call-1",
        }
    ]
