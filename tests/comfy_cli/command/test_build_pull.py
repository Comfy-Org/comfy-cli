from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest
from build_pull_support import PullBuilder, invoke_pull, serve
from build_push_support import envelope, invoke_push, local_node, make_workspace, reloaded, write_spec

from comfy_cli.caller import Caller
from comfy_cli.command import build
from comfy_cli.command.build_spec import read_build_spec
from comfy_cli.discovery import COMMAND_SCHEMAS

SCHEMAS_DIR = Path(__file__).parents[3] / "comfy_cli" / "schemas"


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def install_client(monkeypatch: pytest.MonkeyPatch, client: PullBuilder) -> None:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)


def remote_definition() -> dict:
    return {"schema": "distribution-definition/0", "models": [], "customNodes": []}


def test_agentic_pull_needs_an_id_when_the_spec_has_none(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_spec(workspace, models=[], nodes=[])
    client = PullBuilder()
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_id_unknown"
    assert client.calls == []


def test_agentic_pull_without_yes_fetches_then_refuses_without_writing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_spec(workspace, build_id="build-a", models=[], nodes=[])
    before = path.read_bytes()
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)

    result = invoke_pull(workspace)

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_pull_needs_confirm"
    assert [call["method"] for call in client.calls] == ["get_build"]
    assert path.read_bytes() == before


def _vendor_symlink(workspace: Path, tmp_path: Path) -> None:
    vendored = tmp_path / "shared"
    vendored.mkdir()
    (vendored / "lib.py").write_bytes(b"LIB")
    (workspace / "custom_nodes" / "local-node" / "vendor").symlink_to(vendored)


def test_pull_points_the_skip_report_at_the_definition_it_actually_emits(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`location` is numbered against the local spec, but `pull` ships the
    server's node order — so it must be renumbered rather than passed through.
    The remote here deliberately puts another node ahead of `local-node`, so a
    pass-through index would name the wrong entry instead of dangling."""
    # Given
    write_spec(workspace, build_id="build-a")
    _vendor_symlink(workspace, tmp_path)
    client = PullBuilder()
    serve(
        client,
        "build-a",
        {
            "schema": "distribution-definition/0",
            "models": [],
            "customNodes": [
                {"name": "public-pack", "repository": "https://example.test/pack", "gitRef": "abc"},
                {"name": "local-node", "localPath": "local-node"},
            ],
        },
    )
    install_client(monkeypatch, client)

    # When
    result = invoke_pull(workspace, "-y")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    (row,) = data["skipped_symlinks"]
    assert row["localPath"] == "local-node"
    assert row["member"] == "vendor"
    shipped = data["definition"]["customNodes"]
    index = int(row["location"].removeprefix("definition.customNodes[").removesuffix("]"))
    assert shipped[index]["localPath"] == "local-node", "location must resolve inside this payload's definition"
    assert index == 1, "the remote puts public-pack first, so a pass-through local index would be wrong"
    schema = json.loads((SCHEMAS_DIR / "build_pull.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def test_pull_renumbers_every_row_independently(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two skipped nodes must each land on their own entry; a single shared
    index, or one row's index reused for both, would pass a one-row test."""
    # Given
    second = workspace / "custom_nodes" / "second-node"
    second.mkdir()
    (second / "nodes.py").write_bytes(b"SECOND")
    write_spec(
        workspace,
        build_id="build-a",
        nodes=[local_node(), {"name": "second-node", "localPath": "second-node", "source": "local"}],
    )
    _vendor_symlink(workspace, tmp_path)
    (second / "vendor").symlink_to(tmp_path / "shared")
    client = PullBuilder()
    serve(
        client,
        "build-a",
        {
            "schema": "distribution-definition/0",
            "models": [],
            "customNodes": [
                {"name": "second-node", "localPath": "second-node"},
                {"name": "local-node", "localPath": "local-node"},
            ],
        },
    )
    install_client(monkeypatch, client)

    # When
    result = invoke_pull(workspace, "-y")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    shipped = data["definition"]["customNodes"]
    resolved = {
        row["localPath"]: shipped[int(row["location"].removeprefix("definition.customNodes[").removesuffix("]"))]
        for row in data["skipped_symlinks"]
    }
    assert len(data["skipped_symlinks"]) == 2
    assert {path: entry["localPath"] for path, entry in resolved.items()} == {
        "local-node": "local-node",
        "second-node": "second-node",
    }


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
    reason="needs POSIX mode bits that root ignores",
)
def test_an_unreadable_node_file_is_one_envelope_not_a_traceback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pull` repackages every local node before merging, so it hits the same
    failure `push` does. Without its own handler the error is a `ValueError`
    that sails past `except BuildSpecInvalidError` and out as a traceback with
    no envelope at all."""
    # Given
    write_spec(workspace, build_id="build-a")
    secret = workspace / "custom_nodes" / "local-node" / "secret.py"
    secret.write_bytes(b"SECRET")
    os.chmod(secret, 0o000)
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)

    # When
    try:
        result = invoke_pull(workspace, "-y")
    finally:
        os.chmod(secret, 0o644)

    # Then
    error = envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "build_spec_invalid"
    assert "secret.py could not be read" in error["message"]
    assert error["details"]["path"] == str(workspace / "custom_nodes" / "local-node")
    assert "Traceback" not in result.stderr


def test_pull_never_relocates_a_row_onto_a_server_owned_node(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server entry here collides on `localPath` but carries no `source`, so
    it is not a local node and must not absorb the row. Matching on `localPath`
    alone would point the report at somebody else's node."""
    # Given
    write_spec(workspace, build_id="build-a")
    _vendor_symlink(workspace, tmp_path)
    client = PullBuilder()
    serve(
        client,
        "build-a",
        {
            "schema": "distribution-definition/0",
            "models": [],
            "customNodes": [{"name": "someone-elses-pack", "localPath": "local-node"}],
        },
    )
    install_client(monkeypatch, client)

    # When
    result = invoke_pull(workspace, "-y")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    shipped = data["definition"]["customNodes"]
    assert [entry["name"] for entry in shipped] == ["someone-elses-pack"]
    assert shipped[0].get("source") != "local"
    assert "skipped_symlinks" not in data


def test_two_local_nodes_sharing_a_directory_keep_separate_rows(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spec may legitimately point two entries at one directory. Keying the
    relocation on `localPath` alone collapses both rows onto the last index and
    silently stops naming one of the nodes."""
    # Given
    write_spec(
        workspace,
        build_id="build-a",
        nodes=[local_node(name="first"), local_node(name="second")],
    )
    _vendor_symlink(workspace, tmp_path)
    client = PullBuilder()
    serve(
        client,
        "build-a",
        {
            "schema": "distribution-definition/0",
            "models": [],
            "customNodes": [
                {"name": "first", "localPath": "local-node"},
                {"name": "second", "localPath": "local-node"},
            ],
        },
    )
    install_client(monkeypatch, client)

    # When
    result = invoke_pull(workspace, "-y")

    # Then
    assert result.exit_code == 0, result.stderr
    rows = envelope(result)["data"]["skipped_symlinks"]
    assert len(rows) == 2
    assert len({row["location"] for row in rows}) == 2, "both rows collapsed onto one entry"


def test_pull_says_nothing_about_a_skipped_node_it_is_deleting_from_the_spec(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node the remote does not carry is dropped by the pull, so its packaging
    is no longer this spec's problem and warning about it would be noise."""
    # Given
    write_spec(workspace, build_id="build-a")
    _vendor_symlink(workspace, tmp_path)
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)

    # When
    result = invoke_pull(workspace, "-y")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data["definition"]["customNodes"] == []
    assert "skipped_symlinks" not in data
    assert "excluded" not in result.stderr


def test_tty_decline_leaves_the_spec_byte_identical(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_spec(workspace, build_id="build-a", models=[], nodes=[])
    before = path.read_bytes()
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: False)

    result = invoke_pull(workspace, agentic=False)

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before


def test_tty_without_an_id_uses_the_build_picker(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_spec(workspace, models=[], nodes=[])
    client = PullBuilder()
    serve(client, "build-picked", remote_definition())
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.ui.prompt_select", lambda *args, **kwargs: "build-picked")
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: True)

    result = invoke_pull(workspace, agentic=False)

    assert result.exit_code == 0, result.output
    assert reloaded(workspace)["id"] == "build-picked"
    assert [call["method"] for call in client.calls] == ["list_builds", "get_build"]


def test_cross_id_pull_rebinds_and_the_next_plain_push_targets_the_new_build(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_spec(
        workspace,
        build_id="build-a",
        revision="revision-a",
        models=[],
        nodes=[],
        definition_extra={"localOnly": {"kept": True}, "conflict": "local"},
    )
    client = PullBuilder()
    serve(client, "build-b", {**remote_definition(), "serverOnly": {"kept": True}, "conflict": "server"})
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: pytest.fail("-y prompted"))

    pulled = invoke_pull(workspace, "--id", "build-b", "-y")
    after_pull = read_build_spec(workspace / "comfy-build.yaml")
    pushed = invoke_push(workspace)

    assert pulled.exit_code == 0, pulled.stderr
    assert pushed.exit_code == 0, pushed.stderr
    assert (after_pull["id"], after_pull["syncedRevision"]) == ("build-b", "revision-build-b")
    assert envelope(pulled)["data"]["syncedRevision"] == "revision-build-b"
    spec = read_build_spec(workspace / "comfy-build.yaml")
    assert (spec["id"], spec["syncedRevision"], spec["name"], spec["description"]) == (
        "build-b",
        "revision-1",
        "Remote build-b",
        "Description for build-b",
    )
    assert spec["definition"]["localOnly"] == {"kept": True}
    assert spec["definition"]["serverOnly"] == {"kept": True}
    assert spec["definition"]["conflict"] == "server"
    assert [call["id"] for call in client.calls if call["method"] == "update_build"] == ["build-b"]
    assert COMMAND_SCHEMAS["comfy build pull"] == "build_pull"
    schema = json.loads((SCHEMAS_DIR / "build_pull.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(envelope(pulled)["data"])
