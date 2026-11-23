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
        ("deeply_nested", "[" * 4000 + "]" * 4000),
        ("bool_version", json.dumps(_record("bool_version", v=True))),
        ("float_version", json.dumps(_record("float_version", v=1.0))),
        ("unsigned", json.dumps({k: v for k, v in _record("unsigned").items() if k != "mac"})),
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


def test_non_string_ids_are_left_empty(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "odd_ids", _record("odd_ids", thread_id=["a"], tool_call_id={"k": 1}))

    [pack] = find_agent_packs(custom_nodes)

    assert (pack.thread_id, pack.tool_call_id) == ("", "")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_ignores_a_manager_disabled_folder_that_is_a_symlink(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    elsewhere = tmp_path / "elsewhere"
    _make_pack(elsewhere, "moved", _record("moved"))
    (custom_nodes / ".disabled").symlink_to(elsewhere)

    assert find_agent_packs(custom_nodes) == []


def test_show_agent_created_escapes_markup_from_the_record(tmp_path):
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "restyler", _record("restyler", route="[/]", created_at="[link=http://x]y[/link]"))
    with patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws:
        mock_ws.workspace_path = str(tmp_path)
        result = runner.invoke(app, ["show", "agent-created"])

    assert result.exit_code == 0, result.output
    assert "[/]" in result.output
    assert "[link=http://x]y[/link]" in result.output


def test_show_agent_created_json_errors_when_custom_nodes_is_missing(tmp_path, _reset_renderer):
    from comfy_cli import cmdline

    with patch("comfy_cli.command.custom_nodes.command.workspace_manager") as mock_ws:
        mock_ws.workspace_path = str(tmp_path)
        result = CliRunner().invoke(cmdline.app, ["--json", "node", "show", "agent-created"])

    assert result.exit_code == 1
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "not_in_workspace"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_symlinked_custom_nodes_root_still_lists_enabled_packs(tmp_path):
    real = tmp_path / "real_custom_nodes"
    _make_pack(real, "restyler", _record("restyler"))
    _make_pack(real / ".disabled", "moved", _record("moved"))
    (tmp_path / "custom_nodes").symlink_to(real)

    packs = find_agent_packs(tmp_path / "custom_nodes")

    assert [(p.pack, p.state) for p in packs] == [("moved", "disabled-by-manager"), ("restyler", "enabled")]


def test_rejects_a_symlinked_record_even_without_o_nofollow(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("symlinks need privileges on Windows")
    custom_nodes = tmp_path / "custom_nodes"
    pack = _make_pack(custom_nodes, "linked")
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_record("linked")))
    (pack / ".comfy-agent-install").symlink_to(target)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    assert find_agent_packs(custom_nodes) == []


def _swap_record_before_open(monkeypatch, swap):
    """Run `swap(path)` after the reader's lstat and before its open, without O_NOFOLLOW."""
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    real_open = os.open

    def swapping_open(path, flags, *args, **kwargs):
        if str(path).endswith(".comfy-agent-install"):
            swap(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)


def test_rejects_a_record_replaced_between_check_and_open(tmp_path, monkeypatch):
    # A regular file swapped in under the same name: a different file than the
    # one checked. Runs on every platform, Windows included.
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "raced", _record("raced"))
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_record("raced")))
    _swap_record_before_open(monkeypatch, lambda path: os.replace(other, path))

    assert find_agent_packs(custom_nodes) == []


def test_rejects_a_record_swapped_for_a_symlink_between_check_and_open(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("creating a symlink needs privileges on Windows; the test above covers the swap there")
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "raced", _record("raced"))
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_record("raced")))

    def to_symlink(path):
        os.remove(path)
        os.symlink(target, path)

    _swap_record_before_open(monkeypatch, to_symlink)

    assert find_agent_packs(custom_nodes) == []


def test_rejects_a_symlink_to_the_checked_record_swapped_in_before_open(tmp_path, monkeypatch):
    # Rename the checked record away and leave a link to it under the old name:
    # device and inode still match, so only the name check after the read catches it.
    if os.name == "nt":
        pytest.skip("creating a symlink needs privileges on Windows")
    custom_nodes = tmp_path / "custom_nodes"
    _make_pack(custom_nodes, "raced", _record("raced"))
    moved = tmp_path / "moved.json"

    def rename_and_link(path):
        os.replace(path, moved)
        os.symlink(moved, path)

    _swap_record_before_open(monkeypatch, rename_and_link)

    assert find_agent_packs(custom_nodes) == []
