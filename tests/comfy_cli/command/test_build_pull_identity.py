from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from build_pull_support import PullBuilder, invoke_pull, serve
from build_push_support import envelope, make_workspace, reloaded, write_spec

from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject


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


def remote_definition(*, models: list[dict] | None = None, nodes: list[dict] | None = None) -> dict:
    return {"schema": "distribution-definition/0", "models": models or [], "customNodes": nodes or []}


def test_first_pull_without_blob_cache_matches_nodes_at_id_tier(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_spec(
        workspace,
        build_id="build-a",
        models=[],
        nodes=[{"name": "local-name", "id": "node-id", "source": "registry", "localPath": "kept"}],
    )
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=[{"name": "server-name", "id": "node-id", "mark": "id"}]))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    assert reloaded(workspace)["definition"]["customNodes"] == [
        {"name": "server-name", "id": "node-id", "source": "registry", "localPath": "kept", "mark": "id"}
    ]


def test_node_tiers_are_blob_then_id_then_normalized_repository_then_name(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local: list[JsonObject] = [
        {
            "name": "local-blob",
            "blobId": "blob-1",
            "source": "cached",
            "localPath": "p-blob",
            "localDigest": "digest",
            "localSizeBytes": 10,
        },
        {
            "name": "local-id",
            "id": "node-id",
            "source": "registry",
            "localPath": "p-id",
            "localUnknown": "kept",
            "conflict": "local",
        },
        {"name": "local-repo", "repository": "git@Example.COM:Org/Repo.git", "source": "git", "localPath": "p-repo"},
        {"name": "same-name", "source": "named", "localPath": "p-name"},
    ]
    remote = [
        {"name": "same-name", "mark": "name"},
        {"name": "remote-repo", "repository": "https://example.com/Org/Repo/", "mark": "repository"},
        {"name": "remote-id", "id": "node-id", "mark": "id", "serverUnknown": "kept", "conflict": "server"},
        {"name": "remote-blob", "blobId": "blob-1", "mark": "blob"},
        {"name": "server-only", "blobId": "server-blob", "mark": "unmatched"},
    ]
    write_spec(workspace, build_id="build-a", models=[], nodes=local)
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=remote))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    nodes = reloaded(workspace)["definition"]["customNodes"]
    assert {node["mark"]: node.get("localPath") for node in nodes} == {
        "blob": "p-blob",
        "id": "p-id",
        "repository": "p-repo",
        "name": "p-name",
        "unmatched": None,
    }
    by_mark = {node["mark"]: node for node in nodes}
    assert (by_mark["blob"]["blobId"], by_mark["blob"]["localDigest"], by_mark["blob"]["localSizeBytes"]) == (
        "blob-1",
        "digest",
        10,
    )
    assert (by_mark["id"]["localUnknown"], by_mark["id"]["serverUnknown"], by_mark["id"]["conflict"]) == (
        "kept",
        "kept",
        "server",
    )


def test_id_matches_are_removed_before_a_shared_name_can_collide(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local: list[JsonObject] = [
        {"name": "Shared", "id": "one", "source": "registry", "localPath": "one"},
        {"name": "Shared", "id": "two", "source": "registry", "localPath": "two"},
    ]
    remote = [{"name": "Shared", "id": "two", "mark": "two"}, {"name": "Shared", "id": "one", "mark": "one"}]
    write_spec(workspace, build_id="build-a", models=[], nodes=local)
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=remote))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    assert {node["mark"]: node["localPath"] for node in reloaded(workspace)["definition"]["customNodes"]} == {
        "one": "one",
        "two": "two",
    }


def test_empty_tier_keys_fall_through_instead_of_colliding(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local: list[JsonObject] = [
        {"name": "One", "blobId": "", "id": "one"},
        {"name": "Two", "blobId": "", "id": "two"},
    ]
    remote = [
        {"name": "Two", "blobId": "", "id": "two", "mark": "two"},
        {"name": "One", "blobId": "", "id": "one", "mark": "one"},
    ]
    write_spec(workspace, build_id="build-a", models=[], nodes=local)
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=remote))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    assert {node["id"]: node["mark"] for node in reloaded(workspace)["definition"]["customNodes"]} == {
        "one": "one",
        "two": "two",
    }


def test_active_tier_ambiguity_is_build_spec_invalid_and_names_candidates(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_spec(
        workspace,
        build_id="build-a",
        models=[],
        nodes=[{"name": "Shared", "id": "one"}, {"name": "Shared", "id": "two"}],
    )
    before = path.read_bytes()
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=[{"name": "Shared"}]))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert "customNodes[0]" in error["message"] and "customNodes[1]" in error["message"]
    assert path.read_bytes() == before


def test_model_digest_tuple_and_collided_digest_group_match_greedily(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_root = workspace / "models"
    for relative, content in (
        ("a/one.safetensors", b"ONE"),
        ("b/two.safetensors", b"TWO"),
        ("x/same.safetensors", b"SAME"),
        ("y/same.safetensors", b"SAME"),
    ):
        path = models_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    local = [
        {
            "type": "a",
            "filename": "one.safetensors",
            "localPath": "a/one.safetensors",
            "source": "local",
            "sha256": digest(b"ONE"),
            "sizeBytes": 3,
            "blobId": "blob-one",
        },
        {
            "type": "b",
            "filename": "two.safetensors",
            "localPath": "b/two.safetensors",
            "source": "local",
            "sha256": digest(b"TWO"),
            "sizeBytes": 3,
            "blobId": "blob-two",
        },
        {"type": "x", "filename": "same.safetensors", "localPath": "x/same.safetensors", "source": "local"},
        {"type": "y", "filename": "same.safetensors", "localPath": "y/same.safetensors", "source": "local"},
    ]
    remote = [
        {"type": "moved", "filename": "renamed.safetensors", "sha256": digest(b"ONE"), "mark": "digest"},
        {"type": "b", "filename": "two.safetensors", "sha256": "f" * 64, "mark": "tuple"},
        {"type": "y", "filename": "same.safetensors", "sha256": digest(b"SAME"), "mark": "same-y"},
        {"type": "x", "filename": "same.safetensors", "sha256": digest(b"SAME"), "mark": "same-x"},
    ]
    write_spec(workspace, build_id="build-a", models=local, nodes=[])
    client = PullBuilder()
    serve(client, "build-a", remote_definition(models=remote))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    models = reloaded(workspace)["definition"]["models"]
    assert {model["mark"]: model["localPath"] for model in models} == {
        "digest": "a/one.safetensors",
        "tuple": "b/two.safetensors",
        "same-x": "x/same.safetensors",
        "same-y": "y/same.safetensors",
    }
    by_mark = {model["mark"]: model for model in models}
    assert (by_mark["digest"]["blobId"], by_mark["digest"]["sha256"], by_mark["digest"]["sizeBytes"]) == (
        "blob-one",
        digest(b"ONE"),
        3,
    )
    assert by_mark["tuple"]["blobId"] == "blob-two"
