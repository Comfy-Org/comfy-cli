"""Authoritative model identity must come from bytes, never matching metadata."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from build_pull_support import PullBuilder, invoke_pull, serve
from build_push_support import (
    RecordingBuilder,
    envelope,
    invoke_push,
    local_model,
    make_workspace,
    reloaded,
    write_spec,
)

from comfy_cli.command import build
from comfy_cli.command.build_validation import project_wire_definition


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def model_entry(workspace: Path) -> dict:
    (model,) = reloaded(workspace)["definition"]["models"]
    return model


def rewrite_preserving_metadata(path: Path, data: bytes, *, replace: bool = False) -> None:
    before = path.stat()
    assert len(data) == before.st_size
    if replace:
        staged = path.with_suffix(".replacement")
        staged.write_bytes(data)
        staged.replace(path)
    else:
        path.write_bytes(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)


@pytest.mark.parametrize("reference", ["blobId", "sourceUri"])
@pytest.mark.parametrize("mutation", ["rewrite", "replacement", "middle_chunk"])
def test_push_drops_stale_references_after_metadata_preserving_edits(workspace, monkeypatch, reference, mutation):
    path = workspace / "models/checkpoints/base.safetensors"
    original = b"MODEL" if mutation != "middle_chunk" else b"x" * (2 * 1024 * 1024 + 32)
    changed = (
        b"OTHER" if mutation != "middle_chunk" else original[: 1024 * 1024 + 7] + b"y" + original[1024 * 1024 + 8 :]
    )
    path.write_bytes(original)
    write_spec(workspace, models=[local_model()], nodes=[])
    client = RecordingBuilder()
    if reference == "sourceUri":
        client.model_candidates[path.name] = [
            {"sourceUri": "https://models.example/old", "sha256": hashlib.sha256(original).hexdigest()}
        ]
    monkeypatch.setattr(build, "_builder_client", lambda *args: client)
    assert invoke_push(workspace).exit_code == 0
    old_reference = model_entry(workspace)[reference]
    before_uploads = len(client.uploaded)
    rewrite_preserving_metadata(path, changed, replace=mutation == "replacement")

    result = invoke_push(workspace)

    assert result.exit_code == 0, result.stderr
    model = model_entry(workspace)
    assert model["sha256"] == hashlib.sha256(changed).hexdigest()
    assert model["sizeBytes"] == len(changed)
    assert model.get(reference) != old_reference
    assert "sourceUri" not in model
    assert client.uploaded[before_uploads:] == [changed]
    saved_wire = [call["definition"] for call in client.calls if call["method"] == "update_build"][-1]
    assert saved_wire["models"][0]["sha256"] == model["sha256"]
    assert saved_wire["models"][0]["blobId"] == model["blobId"]
    assert invoke_push(workspace).exit_code == 0
    assert len(client.uploaded) == before_uploads + 1


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("reference", ["blobId", "sourceUri"])
def test_pull_reconciles_fresh_local_identity_before_merging(workspace, monkeypatch, dry_run, reference):
    write_spec(workspace, models=[local_model()], nodes=[])
    client = PullBuilder()
    if reference == "sourceUri":
        client.model_candidates["base.safetensors"] = [
            {"sourceUri": "https://models.example/old", "sha256": hashlib.sha256(b"MODEL").hexdigest()}
        ]
    monkeypatch.setattr(build, "_builder_client", lambda *args: client)
    assert invoke_push(workspace).exit_code == 0
    spec = reloaded(workspace)
    serve(client, spec["id"], project_wire_definition(deepcopy(spec["definition"])))
    path = workspace / "models/checkpoints/base.safetensors"
    rewrite_preserving_metadata(path, b"OTHER")
    before = (workspace / "comfy-build.yaml").read_bytes()
    digests = []
    hash_file = build._sha256_file

    def record_digest(path):
        digest = hash_file(path)
        digests.append(digest)
        return digest

    monkeypatch.setattr(build, "_sha256_file", record_digest)
    calls = len(client.calls)
    result = invoke_pull(workspace, "--dry-run" if dry_run else "-y")

    assert result.exit_code == 0, result.stderr
    assert digests == [hashlib.sha256(b"OTHER").hexdigest()]
    assert [call["method"] for call in client.calls[calls:]] == ["get_build"]
    if dry_run:
        assert (workspace / "comfy-build.yaml").read_bytes() == before
        assert envelope(result)["changed"] is False
    else:
        model = model_entry(workspace)
        assert model["sha256"] == hashlib.sha256(b"OTHER").hexdigest()
        assert "blobId" not in model and "sourceUri" not in model


def test_forged_legacy_digest_memo_cannot_authorize_blob_reuse(workspace, monkeypatch):
    path = workspace / "models/checkpoints/base.safetensors"
    forged = "0" * 64
    write_spec(workspace, models=[local_model(sha256=forged, blobId="stale-blob")], nodes=[])
    memo = Path(os.environ["XDG_CACHE_HOME"]) / "comfy-cli/build-model-digests.json"
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(
        json.dumps(
            {
                str(path.resolve()): {
                    "mtimeNs": path.stat().st_mtime_ns,
                    "sizeBytes": path.stat().st_size,
                    "sha256": forged,
                }
            }
        )
    )
    original_memo = memo.read_bytes()
    client = RecordingBuilder()
    monkeypatch.setattr(build, "_builder_client", lambda *args: client)

    result = invoke_push(workspace)

    assert result.exit_code == 0, result.stderr
    assert model_entry(workspace)["sha256"] == hashlib.sha256(b"MODEL").hexdigest()
    assert model_entry(workspace)["blobId"] != "stale-blob"
    assert client.uploaded == [b"MODEL"]
    assert memo.read_bytes() == original_memo


@pytest.mark.parametrize("reference", ["blobId", "sourceUri"])
def test_push_dry_run_detects_drift_without_saving_or_sending(workspace, monkeypatch, reference):
    write_spec(workspace, models=[local_model()], nodes=[])
    client = RecordingBuilder()
    if reference == "sourceUri":
        client.model_candidates["base.safetensors"] = [
            {"sourceUri": "https://models.example/old", "sha256": hashlib.sha256(b"MODEL").hexdigest()}
        ]
    monkeypatch.setattr(build, "_builder_client", lambda *args: client)
    assert invoke_push(workspace).exit_code == 0
    rewrite_preserving_metadata(workspace / "models/checkpoints/base.safetensors", b"OTHER")
    before = (workspace / "comfy-build.yaml").read_bytes()
    calls, uploaded = len(client.calls), len(client.uploaded)

    result = invoke_push(workspace, "--dry-run")

    assert result.exit_code == 0, result.stderr
    assert envelope(result)["data"]["upload_count"] == 1
    assert envelope(result)["data"]["upload_bytes"] == 5
    assert (workspace / "comfy-build.yaml").read_bytes() == before
    assert len(client.calls) == calls
    assert len(client.uploaded) == uploaded


def test_warm_metadata_cannot_hide_a_model_read_failure(workspace, monkeypatch):
    write_spec(workspace, models=[local_model()], nodes=[])
    client = RecordingBuilder()
    monkeypatch.setattr(build, "_builder_client", lambda *args: client)
    assert invoke_push(workspace).exit_code == 0
    before = (workspace / "comfy-build.yaml").read_bytes()
    calls = len(client.calls)

    def unreadable(path):
        raise PermissionError("model cannot be read")

    monkeypatch.setattr(build, "_sha256_file", unreadable)
    result = invoke_push(workspace)

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_spec_invalid"
    assert (workspace / "comfy-build.yaml").read_bytes() == before
    assert not any(call["method"] in {"create_blob", "update_build", "create_release"} for call in client.calls[calls:])
