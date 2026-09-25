from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import sys
import urllib.error
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest
import requests
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

from comfy_cli.builder_api import BuilderClient
from comfy_cli.command import build
from comfy_cli.command.build_package import package_node
from comfy_cli.command.build_paths import resolve_build_paths
from comfy_cli.command.build_push import pending_uploads, prepare_push
from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer


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


def test_a_dry_run_tells_a_person_what_it_would_upload(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace)
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))

    # When
    result = invoke_push(workspace, "--dry-run", agentic=False)

    # Then
    assert result.exit_code == 0, result.output
    assert "2 files" in result.stdout
    assert "to upload, 0 already held" in result.stdout
    assert "nothing was sent" in result.stdout


def test_a_dry_run_with_nothing_to_upload_says_so(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace, models=[], nodes=[])
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))

    # When
    result = invoke_push(workspace, "--dry-run", agentic=False)

    # Then
    assert result.exit_code == 0, result.output
    assert "0 files" in result.stdout
    assert "nothing was sent" in result.stdout


def test_a_json_dry_run_prints_only_its_envelope(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace)
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))

    # When
    result = invoke_push(workspace, "--dry-run")

    # Then
    assert result.exit_code == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, result.stdout
    assert json.loads(lines[0])["data"]["dry_run"] is True
    assert result.stderr == ""


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
        os.chmod(secret, 0o600)

    # Then
    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert "secret.py could not be read" in error["message"]
    # The node directory to fix, never the spec YAML: routing this failure
    # through `BuildSpecInvalidError` used to relabel it with the spec's path.
    assert error["details"]["path"] == str(workspace / "custom_nodes" / "local-node")


class _InterruptedBuilder(RecordingBuilder):
    """Drops the connection once ``survive`` uploads have landed."""

    def __init__(self, survive: int) -> None:
        super().__init__()
        self.survive = survive

    def upload_blob(self, upload_url: str, path: Path, progress: Callable[[int], None] | None = None) -> None:
        if len(self.uploaded) >= self.survive:
            raise requests.ConnectionError("connection reset mid-upload")
        super().upload_blob(upload_url, path, progress)


def test_an_interrupted_push_keeps_the_blobs_it_already_uploaded(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec file is the resume store, so a blob id that never reaches it is
    lost: the retry re-sends bytes the builder already holds and orphans them,
    since it mints a fresh id per create_blob and deduplicates nothing."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    interrupted = _InterruptedBuilder(survive=1)
    _install_client(monkeypatch, interrupted)

    # When
    failed = invoke_push(workspace)
    checkpointed = reloaded(workspace)
    resumed = prepare_push(checkpointed, resolve_build_paths(str(workspace)), build._sha256_file)

    # Then
    assert failed.exit_code == 1
    assert envelope(failed)["error"]["code"] == "build_builder_error"
    definition = checkpointed["definition"]
    assert isinstance(definition, dict)
    assert definition["models"][0]["blobId"] == "blob-1"
    assert "blobId" not in definition["customNodes"][0]
    assert [(upload.kind, upload.index) for upload in pending_uploads(resumed)] == [("node_zip", 0)]


#: A presigned GCS PUT URL: everything after the `?` IS the credential, and it
#: stays valid for as long as `X-Goog-Expires` says.
SIGNED_URL = (
    "https://storage.googleapis.com/comfy-blobs/blob-1"
    "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
    "&X-Goog-Credential=builder%40comfy.iam.gserviceaccount.com%2F20260906%2Fauto%2Fstorage%2Fgoog4_request"
    "&X-Goog-Date=20260906T090000Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=6d1f3c9ab0e5147a2d8f"
)


class _SignedUrlBuilder(RecordingBuilder):
    """Fails the first upload with an exception quoting the presigned URL."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def upload_blob(self, upload_url: str, path: Path, progress: Callable[[int], None] | None = None) -> None:
        raise self.error


@pytest.mark.parametrize(
    "error, kept",
    [
        pytest.param(
            requests.HTTPError(f"403 Client Error: Forbidden for url: {SIGNED_URL}"),
            "403 Client Error: Forbidden for url: https://storage.googleapis.com/comfy-blobs/blob-1",
            id="raise_for_status-quotes-the-whole-url",
        ),
        pytest.param(
            requests.ConnectionError(
                "HTTPSConnectionPool(host='storage.googleapis.com', port=443): Max retries exceeded with url: "
                "/comfy-blobs/blob-1?X-Goog-Signature=6d1f3c9ab0e5147a2d8f (Caused by NewConnectionError('boom'))"
            ),
            "Max retries exceeded with url: /comfy-blobs/blob-1 (Caused by NewConnectionError('boom'))",
            id="connection-error-quotes-the-path",
        ),
    ],
)
def test_a_failed_upload_keeps_the_signature_out_of_the_envelope(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, kept: str
) -> None:
    """`upload_assets` runs inside `_builder_call`, so an upload that fails hands
    its exception to the transport branch of `_report_builder_error`, which
    interpolates it whole. Both shapes `requests` produces quote what they were
    talking to -- and for a presigned PUT the query string IS the credential, so
    an ordinary failed upload writes a still-valid `X-Goog-Signature` to stdout,
    into the JSON envelope and into any CI log that captured it. The host and the
    path are what make the failure diagnosable, so only the query comes off."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    failing = _SignedUrlBuilder(error)
    failing.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, failing)

    # When
    result = invoke_push(workspace)

    # Then
    message = envelope(result)["error"]["message"]
    assert (
        "X-Goog-Signature" in result.output,
        "X-Goog-Credential" in result.output,
        "?" in message,
        kept in message,
    ) == (False, False, False, True)


@pytest.mark.parametrize(
    "error, code, kept",
    [
        pytest.param(
            requests.exceptions.SSLError(
                f"HTTPSConnectionPool(host='storage.googleapis.com', port=443): Max retries exceeded with url: "
                f"{SIGNED_URL} (Caused by SSLError(SSLCertVerificationError(1, "
                "'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate')))"
            ),
            "tls_verify_failed",
            "Max retries exceeded with url: https://storage.googleapis.com/comfy-blobs/blob-1 (Caused by",
            id="tls-branch-redacts-too",
        ),
        pytest.param(
            requests.exceptions.InvalidURL(f"Invalid URL {SIGNED_URL!r}: No host supplied."),
            "build_builder_error",
            "Invalid URL 'https://storage.googleapis.com/comfy-blobs/blob-1",
            id="malformed-url-is-the-builders-failure-and-redacted",
        ),
    ],
)
def test_the_other_two_failure_paths_keep_the_signature_out_too(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, code: str, kept: str
) -> None:
    """Two paths skipped the redaction above. A certificate failure exits
    `_report_builder_error` before the transport branch, and used to interpolate
    the exception raw. And `requests.exceptions.InvalidURL` (with `MissingSchema`
    and `InvalidSchema`) subclasses `ValueError` as well as `RequestException`, so
    with the `ValueError` clause first a malformed builder-supplied upload URL was
    relabelled `build_missing_input` -- the caller's fault -- and printed whole."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    failing = _SignedUrlBuilder(error)
    failing.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, failing)

    # When
    result = invoke_push(workspace)

    # Then
    err = envelope(result)["error"]
    assert (
        err["code"],
        "X-Goog-Signature" in result.output,
        "X-Goog-Credential" in result.output,
        "?" in err["message"],
        kept in err["message"],
    ) == (code, False, False, False, True)


def test_resuming_an_interrupted_push_uploads_only_what_is_missing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    _install_client(monkeypatch, _InterruptedBuilder(survive=1))
    assert invoke_push(workspace).exit_code == 1
    retry = RecordingBuilder()
    retry.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, retry)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert [blob["kind"] for blob in retry.blobs] == ["node_zip"]
    assert envelope(result)["data"]["uploaded"] == 1
    definition = reloaded(workspace)["definition"]
    assert isinstance(definition, dict)
    assert definition["models"][0]["blobId"] == "blob-1"


def _upload_events(result) -> list[JsonObject]:
    """The progress lines a `--json` push writes to stderr, among whatever else is there."""
    events = []
    for line in result.stderr.splitlines():
        if line.startswith("{"):
            parsed = json.loads(line)
            if parsed.get("schema") == "event/1":
                events.append(parsed)
    return events


def test_a_push_reports_each_upload_on_stderr_and_keeps_stdout_the_envelope(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent's stdout is a pipe, so it resolves to single-envelope JSON mode:
    progress has to reach it without putting a second document on stdout."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0")
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 1
    events = _upload_events(result)
    validator = jsonschema.Draft202012Validator(_schema("build_push_event.json"))
    for event in events:
        validator.validate(event)
    assert [(event["type"], event.get("file")) for event in events] == [
        ("upload_plan", None),
        ("upload_progress", "base.safetensors"),
        ("upload_complete", "base.safetensors"),
        ("upload_progress", "local-node.zip"),
        ("upload_complete", "local-node.zip"),
    ]
    assert events[0]["files"] == 2
    assert events[0]["already_held"] == 0
    assert events[0]["bytes_total"] == envelope(result)["data"]["upload_bytes"]
    assert events[-1]["overall_bytes_done"] == events[0]["bytes_total"]


def test_a_retry_says_what_it_already_holds_before_it_sends_the_rest(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a push that died after the model landed
    write_spec(workspace, build_id="build-1", revision="revision-0")
    _install_client(monkeypatch, _InterruptedBuilder(survive=1))
    interrupted = invoke_push(workspace)
    assert interrupted.exit_code == 1
    # The file that failed has a start and no completion: the error envelope follows it.
    assert [(event["type"], event.get("file")) for event in _upload_events(interrupted)][-2:] == [
        ("upload_complete", "base.safetensors"),
        ("upload_progress", "local-node.zip"),
    ]
    retry = RecordingBuilder()
    retry.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, retry)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    plan = _upload_events(result)[0]
    assert (plan["type"], plan["files"], plan["already_held"]) == ("upload_plan", 1, 1)


def test_a_push_with_nothing_to_upload_still_prints_the_plan(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a spec whose blobs all landed on an earlier push
    write_spec(workspace, build_id="build-1", revision="revision-0")
    client = RecordingBuilder()
    client.remote_revisions["build-1"] = "revision-0"
    _install_client(monkeypatch, client)
    assert invoke_push(workspace).exit_code == 0

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    events = _upload_events(result)
    assert [event["type"] for event in events] == ["upload_plan"]
    assert (events[0]["files"], events[0]["bytes_total"], events[0]["already_held"]) == (0, 0, 2)
    assert envelope(result)["data"]["uploaded"] == 0


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


def test_push_refuses_model_entries_the_builder_would_refuse_before_any_upload(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder would save these and refuse the release. Push names all four,
    the folder's case included since it reads the builder's list, and sends nothing."""
    # Given
    write_spec(
        workspace,
        models=[
            {"type": "Loras", "sourceUri": "https://h.example/a.safetensors"},
            {"type": "loras", "sourceUri": "https://civitai.com/api/download/models/128713"},
            {
                "type": "loras",
                "filename": "add_detail (v1.1).safetensors",
                "sourceUri": "https://h.example/b.safetensors",
            },
            {"type": "loras", "sha256": "8A0B5F4C2E", "sourceUri": "https://h.example/c.safetensors"},
        ],
        nodes=[],
    )
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert sorted(issue["field"].rsplit(".", 1)[1] for issue in error["details"]["invalid"]) == [
        "filename",
        "filename",
        "sha256",
        "type",
    ]
    assert client.calls == []
    assert reloaded(workspace)["id"] is None


#: sha256 of the fixture model's bytes (``make_workspace``), so push keeps the link.
_BASE_DIGEST = hashlib.sha256(b"MODEL").hexdigest()


@pytest.mark.parametrize(
    ("entry", "field"),
    [
        pytest.param(local_model(sourceUri="http://h.example/base.safetensors"), "sourceUri", id="http"),
        pytest.param(
            {
                k: v
                for k, v in local_model(sourceUri="https://civitai.com/api/download/models/1").items()
                if k != "filename"
            },
            "filename",
            id="no-extension-no-filename",
        ),
        pytest.param(local_model(sourceUri="https://h.example/%zz.safetensors"), "sourceUri", id="go-cannot-parse"),
    ],
)
def test_push_refuses_a_local_models_kept_link_before_any_upload(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, entry: JsonObject, field: str
) -> None:
    """A local model whose file still matches its sha256 keeps its link and is not
    uploaded, so the builder reads that link: push checks it as it would any other."""
    # Given
    write_spec(workspace, models=[{**entry, "sha256": _BASE_DIGEST}], nodes=[])
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 1
    error = envelope(result)["error"]
    assert error["code"] == "build_spec_invalid"
    assert [issue["field"] for issue in error["details"]["invalid"]] == [f"definition.models[0].{field}"]
    assert client.calls == []


def test_push_uploads_a_local_model_whose_changed_file_drops_a_bad_link(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(
        workspace, models=[local_model(sha256="0" * 64, sourceUri="http://h.example/base.safetensors")], nodes=[]
    )
    client = RecordingBuilder()
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stdout
    (create,) = _calls(client, "create_build")
    assert create["definition"]["models"][0]["blobId"] == "blob-1"
    assert "sourceUri" not in create["definition"]["models"][0]


def test_push_does_not_refuse_for_a_folder_list_it_could_not_read(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[{"type": "Loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])
    client = RecordingBuilder()
    reads: list[str] = []

    def unreachable() -> list[str]:
        reads.append("read")
        raise requests.ConnectionError("builder unreachable")

    monkeypatch.setattr(client, "list_model_directories", unreachable)
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stdout
    assert len(_calls(client, "create_build")) == 1
    assert reads == ["read"]
    assert _FOLDER_CASE_UNCHECKED in " ".join(result.stderr.split())


def _a_list_body() -> list[str]:
    """What ``list_model_directories`` does with a 200 whose body is a JSON list."""
    client = BuilderClient("https://builder.test", "token")
    client._send = lambda url, **kwargs: (200, ["loras"])  # type: ignore[method-assign]
    return client.list_model_directories()


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(http.client.IncompleteRead(b"{"), id="cut-off-body"),
        pytest.param(ResponseTooLarge("over the cap"), id="oversized-body"),
        pytest.param(_a_list_body, id="list-body"),
    ],
)
def test_push_does_not_stop_for_any_failure_reading_the_folder_list(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, read
) -> None:
    # Given
    write_spec(workspace, models=[{"type": "loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])
    client = RecordingBuilder()
    reads: list[str] = []

    def failing() -> list[str]:
        reads.append("read")
        if callable(read):
            return read()
        raise read

    monkeypatch.setattr(client, "list_model_directories", failing)
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stdout
    assert len(_calls(client, "create_build")) == 1
    assert reads == ["read"]
    assert _FOLDER_CASE_UNCHECKED in " ".join(result.stderr.split())


#: What a push or validate says when it could not check a folder's case.
_FOLDER_CASE_UNCHECKED = "a folder's case (`Loras` for `loras`) was not checked"


class _FolderList:
    """A client whose folder list answers with *listed*, counting the reads."""

    def __init__(self, listed: object) -> None:
        self.listed = listed
        self.reads = 0

    def list_model_directories(self) -> object:
        self.reads += 1
        return self.listed


@pytest.mark.parametrize(
    "definition",
    [
        pytest.param({}, id="no-models"),
        pytest.param({"models": []}, id="empty-models"),
        pytest.param(["models"], id="not-a-mapping"),
    ],
)
def test_a_spec_without_models_reads_no_folder_list(definition: JsonValue) -> None:
    # Given
    client = _FolderList(["loras"])

    # When
    directories = build._model_directories(get_renderer(), client, {"definition": definition})

    # Then
    assert directories is None
    assert client.reads == 0


@pytest.mark.parametrize(
    ("listed", "directories"),
    [
        pytest.param(["loras", 3], frozenset({"loras"}), id="names-kept"),
        pytest.param("loras", None, id="a-string"),
        pytest.param({"loras": True}, None, id="a-mapping"),
        pytest.param([], None, id="empty"),
        pytest.param([1, None], None, id="no-names"),
    ],
)
def test_a_folder_list_refuses_nothing_unless_it_names_folders(
    listed: object, directories: frozenset[str] | None
) -> None:
    # Given
    client = _FolderList(listed)

    # When
    read = build._model_directories(get_renderer(), client, {"definition": {"models": [{"type": "loras"}]}})

    # Then
    assert read == directories
    assert client.reads == 1


def test_a_dry_run_says_it_did_not_check_a_folders_case(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run reads nothing from the builder, so "Loras" passes it and the real
    push refuses it."""
    # Given
    write_spec(workspace, models=[{"type": "Loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])

    # When
    result = invoke_push(workspace, "--dry-run", agentic=False)

    # Then
    assert result.exit_code == 0, result.output
    assert _FOLDER_CASE_UNCHECKED in " ".join(result.stdout.split())


@pytest.mark.parametrize(
    ("args", "checked"),
    [
        pytest.param(("--dry-run",), False, id="dry-run"),
        pytest.param((), True, id="push"),
    ],
)
def test_the_payload_says_whether_a_folders_case_was_checked(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], checked: bool
) -> None:
    """The line a dry run prints never reaches an agent, so the payload carries it."""
    # Given
    write_spec(workspace, models=[{"type": "loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])
    _install_client(monkeypatch, RecordingBuilder())

    # When
    result = invoke_push(workspace, *args)

    # Then
    assert result.exit_code == 0, result.stdout
    data = envelope(result)["data"]
    assert data["folder_case_checked"] is checked
    jsonschema.Draft202012Validator(_schema("build_push.json")).validate(data)


def test_a_push_whose_folder_list_names_no_folder_says_the_case_was_not_checked(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, models=[{"type": "Loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])
    client = RecordingBuilder()
    monkeypatch.setattr(client, "list_model_directories", lambda: [])
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stdout
    assert envelope(result)["data"]["folder_case_checked"] is False
    assert _FOLDER_CASE_UNCHECKED in " ".join(result.stderr.split())


def test_a_push_without_models_says_nothing_of_a_folders_case(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    write_spec(workspace, models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--dry-run")

    # Then
    assert result.exit_code == 0, result.stdout
    assert "folder_case_checked" not in envelope(result)["data"]


def test_a_refused_release_names_each_model_the_push_saved(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cut counts the models of the definition the push just saved; push has
    that definition, so each line names the model too."""
    # Given
    write_spec(
        workspace,
        models=[
            {"type": "loras", "filename": "a.safetensors", "sourceUri": "https://h.example/a.safetensors"},
            {"type": "loras", "sourceUri": "https://h.example/b.safetensors?token=s3cr3t"},
        ],
        nodes=[],
    )
    client = RecordingBuilder()
    invalid = [
        {"field": "models[0].sourceUri", "reason": "link is not reachable"},
        {"field": "models[7].type", "reason": "no such model"},
        {"field": "targets[0]", "reason": "not buildable"},
    ]

    def refused(build_id: str, targets: list[JsonObject] | None = None) -> tuple[str, str]:
        raise urllib.error.HTTPError(
            "https://builder.test/v1/builds/build-1/releases",
            400,
            "Bad Request",
            {},
            io.BytesIO(json.dumps({"error": "INVALID_DEFINITION", "invalid": invalid}).encode()),
        )

    monkeypatch.setattr(client, "create_release", refused)
    _install_client(monkeypatch, client)

    # When
    result = invoke_push(workspace, "--release", "--target", "linux/nvidia")

    # Then
    error = envelope(result)["error"]
    assert error["code"] == "build_definition_invalid"
    # The spec sorts its models, and the entry with no filename comes first.
    (create,) = _calls(client, "create_build")
    assert "filename" not in create["definition"]["models"][0]
    assert error["message"].splitlines()[1:] == [
        "  models[0].sourceUri (https://h.example/b.safetensors): link is not reachable",
        "  models[7].type: no such model",
        "  targets[0]: not buildable",
    ]
    assert [issue.get("model") for issue in error["details"]["invalid"]] == [
        "https://h.example/b.safetensors",
        None,
        None,
    ]
    assert "s3cr3t" not in result.output
