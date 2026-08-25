from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest
from build_push_support import (
    RecordingBuilder,
    envelope,
    invoke_push,
    local_node,
    make_workspace,
    reloaded,
    write_spec,
)

from comfy_cli.command import build
from comfy_cli.command.build_package import package_node
from comfy_cli.command.build_spec import JsonObject


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def _install_client(monkeypatch: pytest.MonkeyPatch, client: RecordingBuilder) -> None:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)


def _calls(client: RecordingBuilder, method: str) -> list[JsonObject]:
    return [call for call in client.calls if call["method"] == method]


def _schema(name: str) -> JsonObject:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_is_signed_out_and_zero_http(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    path = write_spec(workspace)
    before = path.read_bytes()
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))
    node_size = package_node(workspace / "custom_nodes" / "local-node").size_bytes

    # When
    result = invoke_push(workspace, "--dry-run")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data["dry_run"] is True
    assert data["upload_count"] == 2
    assert data["upload_bytes"] == len(b"MODEL") + node_size
    assert path.read_bytes() == before


def test_first_push_creates_instead_of_selecting_an_unknown_id(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[], nodes=[])
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    spec = reloaded(workspace)
    assert spec["id"] == "build-created"
    assert spec["syncedRevision"] == "revision-1"
    assert len(_calls(client, "create_build")) == 1
    assert len(_calls(client, "get_build")) == 1


def test_explicit_different_id_refuses_without_force(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace, build_id="build-a", revision="revision-a", models=[], nodes=[])
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace, "--id", "build-b")

    # Then
    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_spec_stale"
    assert client.calls == []
    assert reloaded(workspace)["id"] == "build-a"


def test_forced_rebind_updates_id_and_revision_before_the_next_plain_push(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, build_id="build-a", revision="revision-a", models=[], nodes=[])
    client = RecordingBuilder()
    client.remote_revisions["build-b"] = "revision-b"
    _install_client(monkeypatch, client)

    # When
    forced = invoke_push(workspace, "--id", "build-b", "--force")
    plain = invoke_push(workspace)

    # Then
    assert forced.exit_code == 0, forced.stderr
    assert plain.exit_code == 0, plain.stderr
    spec = reloaded(workspace)
    assert spec["id"] == "build-b"
    updates = _calls(client, "update_build")
    assert [call["id"] for call in updates] == ["build-b", "build-b"]
    assert updates[1]["expectedUpdatedAt"] == "revision-2"


def test_force_exhausts_exactly_three_get_patch_attempts(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="old", models=[], nodes=[])
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "remote"
    client.always_stale = True
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace, "--force")

    # Then
    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_spec_stale"
    assert len(_calls(client, "get_build")) == 3
    assert len(_calls(client, "update_build")) == 3


def test_plain_stale_response_names_pull_and_preserves_spec(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    path = write_spec(workspace, build_id="build-1", revision="old", models=[], nodes=[])
    before = path.read_bytes()
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "remote"
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_stale"
    assert "comfy build pull" in error["hint"]
    assert path.read_bytes() == before


def test_local_node_stale_repository_metadata_never_reaches_the_pin_importer(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(
        workspace,
        models=[],
        nodes=[local_node(repository="https://github.com/wrong/stale", gitRef="old")],
    )
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.snapshots == []
    assert len(client.uploaded) == 1
    wire_node = _calls(client, "create_build")[0]["definition"]["customNodes"][0]
    assert wire_node == {"name": "local-node", "blobId": "blob-1"}


def test_pin_gate_compares_registry_identity_not_display_name(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    nodes = [
        {"name": "Same", "id": "valid-node", "registryVersion": "1.0.0", "source": "registry"},
        {"name": "Same", "id": "missing-node", "registryVersion": "2.0.0", "source": "registry"},
    ]
    write_spec(workspace, models=[], nodes=nodes)
    client = RecordingBuilder()
    client.checked_nodes = [nodes[0]]
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_registry_pin_missing"
    sent_nodes = client.snapshots[0]["snapshots"][0]["customNodes"]
    assert {(node["id"], node["version"]) for node in sent_nodes} == {
        ("valid-node", "1.0.0"),
        ("missing-node", "2.0.0"),
    }
    assert _calls(client, "create_build") == []


def test_pin_gate_normalizes_equivalent_repository_identities(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(
        workspace,
        models=[],
        nodes=[
            {
                "name": "repo-node",
                "source": "git",
                "repository": "git@Example.COM:Owner/Repo.git",
                "gitRef": "main",
            }
        ],
    )
    client = RecordingBuilder()
    client.checked_nodes = [{"name": "repo-node", "repository": "https://example.com/Owner/Repo/"}]
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(client.snapshots) == 1


def test_a_skipped_symlink_is_named_on_stderr_and_carried_in_the_payload(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace)
    node = workspace / "custom_nodes" / "local-node"
    (workspace / "shared").mkdir()
    (workspace / "shared" / "lib.py").write_bytes(b"LIB")
    os.symlink(workspace / "shared", node / "vendor")
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))

    # When
    result = invoke_push(workspace, "--dry-run")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data["skipped_symlinks"] == [
        {"location": "definition.customNodes[0]", "localPath": "local-node", "member": "vendor"}
    ]
    assert "excluded 1 symlink" in result.stderr
    jsonschema.Draft202012Validator(_schema("build_push.json")).validate(data)


def test_a_real_push_keeps_the_skip_report_alongside_its_upload_results(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry-run path returns before `payload.update(...)` adds the release
    keys, so it cannot show that a real push still carries the rows."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    node = workspace / "custom_nodes" / "local-node"
    (workspace / "shared").mkdir()
    (workspace / "shared" / "lib.py").write_bytes(b"LIB")
    os.symlink(workspace / "shared", node / "vendor")
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data["dry_run"] is False
    assert data["uploaded"] == 2
    assert data["skipped_symlinks"] == [
        {"location": "definition.customNodes[0]", "localPath": "local-node", "member": "vendor"}
    ]
    jsonschema.Draft202012Validator(_schema("build_push.json")).validate(data)


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
    reason="needs POSIX mode bits that root ignores",
)
def test_an_unreadable_node_file_is_a_spec_error_rather_than_a_traceback(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace)
    secret = workspace / "custom_nodes" / "local-node" / "secret.py"
    secret.write_bytes(b"SECRET")
    os.chmod(secret, 0o000)
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))

    # When
    try:
        result = invoke_push(workspace, "--dry-run")
    finally:
        os.chmod(secret, 0o644)

    # Then
    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert "secret.py could not be read" in error["message"]
    # The node directory to fix, never the spec YAML: routing this failure
    # through `BuildSpecInvalidError` used to relabel it with the spec's path.
    assert error["details"]["path"] == str(workspace / "custom_nodes" / "local-node")


def test_update_synchronizes_name_and_description(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(
        workspace,
        build_id="build-1",
        revision="revision-remote",
        name="Renamed",
        description="New description",
        models=[],
        nodes=[],
    )
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "revision-remote"
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    (update,) = _calls(client, "update_build")
    assert update["name"] == "Renamed"
    assert update["description"] == "New description"
