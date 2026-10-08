from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
from build_push_support import (
    RecordingBuilder,
    envelope,
    invoke_push,
    local_model,
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


def _definition(spec: JsonObject) -> JsonObject:
    definition = spec["definition"]
    assert isinstance(definition, dict)
    return definition


def _entries(definition: JsonObject, collection: str) -> list[JsonObject]:
    values = definition[collection]
    assert isinstance(values, list)
    assert all(isinstance(value, dict) for value in values)
    return values


def _without_local_cache(before: JsonObject, after: JsonObject) -> tuple[JsonObject, JsonObject]:
    projected_before = deepcopy(before)
    projected_after = deepcopy(after)
    removal = {
        "models": ("blobId", "sha256", "sizeBytes", "sourceUri"),
        "customNodes": ("blobId", "localDigest", "localSizeBytes"),
    }
    for collection, keys in removal.items():
        before_entries = _entries(projected_before, collection)
        after_entries = _entries(projected_after, collection)
        for index, original in enumerate(_entries(before, collection)):
            if original.get("source") != "local":
                continue
            for key in keys:
                before_entries[index].pop(key, None)
                after_entries[index].pop(key, None)
    return projected_before, projected_after


def test_push_changes_only_local_cache_fields_and_derives_the_wire(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    models = [
        local_model(future={"nested": "model"}),
        {
            "type": "vae",
            "filename": "remote.safetensors",
            "source": "url",
            "sourceUri": "https://models.example/remote",
            "future": {"untouched": True},
        },
    ]
    nodes = [
        local_node(future={"nested": "node"}),
        {
            "name": "remote-node",
            "source": "git",
            "repository": "https://github.com/example/remote-node",
            "gitRef": "main",
            "future": {"untouched": True},
        },
    ]
    write_spec(workspace, models=models, nodes=nodes, definition_extra={"futureTop": {"nested": True}})
    before = _definition(reloaded(workspace))
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    after = _definition(reloaded(workspace))
    assert _without_local_cache(before, after)[0] == _without_local_cache(before, after)[1]
    assert _entries(after, "models")[1] == _entries(before, "models")[1]
    assert _entries(after, "customNodes")[1] == _entries(before, "customNodes")[1]
    create = next(call for call in client.calls if call["method"] == "create_build")
    wire = create["definition"]
    assert isinstance(wire, dict)
    local_wire_model = _entries(wire, "models")[0]
    local_wire_node = _entries(wire, "customNodes")[0]
    assert not {"source", "localPath", "localDigest", "localSizeBytes"} & local_wire_model.keys()
    assert not {"source", "localPath", "localDigest", "localSizeBytes"} & local_wire_node.keys()


def test_resolver_match_persists_source_uri_without_a_blob(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    digest = hashlib.sha256(b"MODEL").hexdigest()
    write_spec(workspace, models=[local_model()], nodes=[])
    client = RecordingBuilder()
    client.model_candidates["base.safetensors"] = [{"sourceUri": "https://models.example/new", "sha256": digest}]
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    (model,) = _entries(_definition(reloaded(workspace)), "models")
    assert model["sourceUri"] == "https://models.example/new"
    assert "blobId" not in model
    assert client.blobs == []
    assert envelope(result)["data"]["upload_count"] == 0


def _archive_bytes(workspace: Path, name: str) -> bytes:
    destination = workspace.parent / name
    package_node(workspace / "custom_nodes" / "local-node", destination)
    return destination.read_bytes()


def test_unchanged_node_reuses_blob_then_changed_node_reuploads(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[], nodes=[local_node()])
    client = RecordingBuilder()
    _install_client(monkeypatch, client)
    first_package = _archive_bytes(workspace, "first.zip")

    # When
    first = invoke_push(workspace)
    second = invoke_push(workspace)
    (workspace / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"CHANGED NODE")
    changed_package = _archive_bytes(workspace, "changed.zip")
    third = invoke_push(workspace)

    # Then
    assert [first.exit_code, second.exit_code, third.exit_code] == [0, 0, 0]
    assert client.uploaded == [first_package, changed_package]
    assert len(client.blobs) == 2
    node = _entries(_definition(reloaded(workspace)), "customNodes")[0]
    package = package_node(workspace / "custom_nodes" / "local-node")
    assert (node["localDigest"], node["localSizeBytes"]) == (package.sha256, package.size_bytes)


def test_a_no_op_re_push_does_not_re_read_the_model(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drift detection re-hashes every local model on every push, which is what
    keeps a stale blobId from being published. The memo is what stops that from
    meaning every push re-reads every byte a workspace holds."""
    # Given
    write_spec(workspace, models=[local_model()], nodes=[])
    reads: list[Path] = []
    hash_file = build._sha256_file

    def counting_digest(path: Path) -> str:
        reads.append(path)
        return hash_file(path)

    monkeypatch.setattr(build, "_sha256_file", counting_digest)
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    first = invoke_push(workspace)
    second = invoke_push(workspace)

    # Then
    assert [first.exit_code, second.exit_code] == [0, 0], second.stderr
    assert reads == [workspace / "models" / "checkpoints" / "base.safetensors"]
    assert len(client.blobs) == 1
    model = _entries(_definition(reloaded(workspace)), "models")[0]
    assert model["sha256"] == hashlib.sha256(b"MODEL").hexdigest()


def test_model_change_without_update_refreshes_hash_size_and_only_uploads_once(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[local_model()], nodes=[])
    client = RecordingBuilder()
    _install_client(monkeypatch, client)
    assert invoke_push(workspace).exit_code == 0
    changed = b"MODEL-CONTENT-IS-LONGER"
    (workspace / "models" / "checkpoints" / "base.safetensors").write_bytes(changed)

    # When
    changed_push = invoke_push(workspace)
    unchanged_push = invoke_push(workspace)

    # Then
    assert changed_push.exit_code == 0, changed_push.stderr
    assert unchanged_push.exit_code == 0, unchanged_push.stderr
    assert len(client.blobs) == 2
    assert client.blobs[-1]["sizeBytes"] == len(changed)
    model = _entries(_definition(reloaded(workspace)), "models")[0]
    assert model["sha256"] == hashlib.sha256(changed).hexdigest()
    assert model["sizeBytes"] == len(changed)


def test_stale_source_uri_deletion_survives_409_and_forces_second_resolution(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    old_digest = hashlib.sha256(b"MODEL").hexdigest()
    write_spec(
        workspace,
        build_id="build-1",
        revision="revision-remote",
        models=[local_model(sha256=old_digest, sizeBytes=len(b"MODEL"), sourceUri="https://models.example/old")],
        nodes=[],
    )
    changed = b"CHANGED-MODEL"
    (workspace / "models" / "checkpoints" / "base.safetensors").write_bytes(changed)
    changed_digest = hashlib.sha256(changed).hexdigest()
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "revision-remote"
    client.model_candidates["base.safetensors"] = [
        {"sourceUri": "https://models.example/new", "sha256": changed_digest}
    ]
    client.stale_updates = 1
    _install_client(monkeypatch, client)

    # When
    stale = invoke_push(workspace)
    after_failure = reloaded(workspace)
    retry = invoke_push(workspace)

    # Then
    assert stale.exit_code == 1
    assert envelope(stale)["error"]["code"] == "build_spec_stale"
    failed_model = _entries(_definition(after_failure), "models")[0]
    assert failed_model == local_model(sha256=old_digest, sizeBytes=len(b"MODEL"))
    assert retry.exit_code == 0, retry.stderr
    assert len([call for call in client.calls if call["method"] == "resolve_models"]) == 2
    saved_model = _entries(_definition(reloaded(workspace)), "models")[0]
    assert saved_model["sourceUri"] == "https://models.example/new"
    assert saved_model["sha256"] == changed_digest
