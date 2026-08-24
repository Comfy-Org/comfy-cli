from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from build_pull_support import PullBuilder, invoke_pull, serve
from build_push_support import envelope, invoke_push, make_workspace, reloaded, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_package import node_content_identity


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


@pytest.mark.parametrize(("stored_digest", "keeps_blob"), [(None, False), ("current", True)])
def test_pull_recomputes_node_identity_before_deciding_whether_its_blob_is_valid(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, stored_digest: str | None, keeps_blob: bool
) -> None:
    digest, size = node_content_identity(workspace / "custom_nodes" / "local-node")
    node: dict = {
        "name": "local-node",
        "localPath": "local-node",
        "source": "local",
        "blobId": "blob-local",
        "localDigest": digest if stored_digest == "current" else "stale",
        "localSizeBytes": size,
    }
    write_spec(workspace, build_id="build-a", models=[], nodes=[node])
    client = PullBuilder()
    serve(client, "build-a", remote_definition(nodes=[{"name": "local-node", "blobId": "blob-local"}]))
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    pulled = reloaded(workspace)["definition"]["customNodes"][0]
    assert (pulled.get("blobId") == "blob-local") is keeps_blob
    assert (pulled["localDigest"], pulled["localSizeBytes"]) == (digest, size)


@pytest.mark.parametrize(
    ("local_uri", "stored_sha", "expected_uri"),
    [
        (None, None, None),
        ("https://local/kept", hashlib.sha256(b"MODEL").hexdigest(), "https://local/kept"),
        ("https://local/stale", "0" * 64, None),
    ],
    ids=["server-only-is-ignored", "local-unchanged-is-retained", "local-changed-is-deleted"],
)
def test_source_uri_follows_its_three_case_rule(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_uri: str | None,
    stored_sha: str | None,
    expected_uri: str | None,
) -> None:
    local: dict = {
        "type": "checkpoints",
        "filename": "base.safetensors",
        "localPath": "checkpoints/base.safetensors",
        "source": "local",
    }
    if local_uri is not None:
        local["sourceUri"] = local_uri
    if stored_sha is not None:
        local["sha256"] = stored_sha
    write_spec(workspace, build_id="build-a", models=[local], nodes=[])
    client = PullBuilder()
    serve(
        client,
        "build-a",
        remote_definition(
            models=[
                {
                    "type": "checkpoints",
                    "filename": "base.safetensors",
                    "sha256": stored_sha or "1" * 64,
                    "sourceUri": "https://server/ignored",
                }
            ]
        ),
    )
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 0, result.stderr
    assert reloaded(workspace)["definition"]["models"][0].get("sourceUri") == expected_uri


def test_push_pull_update_push_cycle_has_no_diff_and_an_identical_wire_definition(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment: dict = {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None}
    write_spec(
        workspace,
        models=[
            {
                "type": "checkpoints",
                "filename": "base.safetensors",
                "localPath": "checkpoints/base.safetensors",
                "source": "local",
            }
        ],
        nodes=[
            {"name": "local-node", "repository": None, "gitRef": None, "localPath": "local-node", "source": "local"}
        ],
        definition_extra={
            "baseComfyVersion": "v0.3.0",
            "pipDependencies": "example==1.0.0\n",
            "environment": environment,
        },
    )
    monkeypatch.setattr(
        build,
        "capture_pip_provenance",
        lambda python: {"pipDependencies": "example==1.0.0\n", "environment": environment},
    )
    client = PullBuilder()
    install_client(monkeypatch, client)

    first_push = invoke_push(workspace)
    first_wire = next(call["definition"] for call in client.calls if call["method"] == "create_build")
    assert isinstance(first_wire, dict)
    serve(client, "build-created", first_wire)
    pulled = invoke_pull(workspace, "-y")
    before_update = (workspace / "comfy-build.yaml").read_bytes()
    updated = CliRunner(mix_stderr=False).invoke(
        cli_app,
        ["build", "update", "-y", "--python", sys.executable, "--comfy-version", "0.3.0", str(workspace)],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )
    after_update = (workspace / "comfy-build.yaml").read_bytes()
    second_push = invoke_push(workspace)

    assert first_push.exit_code == 0, first_push.stderr
    assert pulled.exit_code == 0, pulled.stderr
    assert updated.exit_code == 0, updated.stderr
    assert second_push.exit_code == 0, second_push.stderr
    assert envelope(updated)["data"]["summary"] == "no changes"
    assert after_update == before_update
    assert [call["definition"] for call in client.calls if call["method"] == "update_build"][-1] == first_wire
