from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import jsonschema
import pytest
from build_pull_support import PullBuilder, serve
from build_push_support import envelope, invoke_push, make_workspace
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_diff import diff_definitions
from comfy_cli.discovery import COMMAND_SCHEMAS

SCHEMAS_DIR = Path(__file__).parents[3] / "comfy_cli" / "schemas"
STATUS_SCHEMA = json.loads((SCHEMAS_DIR / "build_status.json").read_text(encoding="utf-8"))

# Pinned so the rescan `status` runs is comparable with the one `init` wrote: a
# real `pip freeze` differs between runs, which would read as permanent drift.
FROZEN_PROVENANCE = {
    "pipDependencies": "example==1.0.0\n",
    "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
}


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(build, "capture_pip_provenance", lambda python: dict(FROZEN_PROVENANCE))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def _invoke(root: Path, args: list[str], *, token: str | None, agentic: bool):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        [*args, str(root)],
        env={
            "AI_AGENT": "1" if agentic else None,
            "COMFY_OUTPUT": "json" if agentic else "pretty",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def invoke_init(root: Path):
    args = ["build", "init", "--name", "Status", "--python", sys.executable, "--comfy-version", "0.3"]
    return _invoke(root, args, token="tok_test", agentic=True)


def invoke_status(root: Path, *extra: str, token: str | None = "tok_test", agentic: bool = True):
    # The same `--python` / `--comfy-version` `init` used: `status` rescans, and
    # an unpinned rescan would answer from this machine instead of the fixture.
    args = ["build", "status", *extra, "--python", sys.executable, "--comfy-version", "0.3"]
    return _invoke(root, args, token=token, agentic=agentic)


def install_client(monkeypatch: pytest.MonkeyPatch, client: PullBuilder) -> None:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)


def synced_workspace(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[PullBuilder, str]:
    """An install whose spec was freshly scanned AND freshly pushed.

    Both of `status`'s comparisons are clean here — the spec matches the install
    and its ``syncedRevision`` is the Build's ``updatedAt`` — which is the only
    honest baseline for asserting that either half can move on its own.
    """
    client = PullBuilder()
    install_client(monkeypatch, client)
    initialized = invoke_init(root)
    assert initialized.exit_code == 0, initialized.stderr
    pushed = invoke_push(root)
    assert pushed.exit_code == 0, pushed.stderr
    build_id = envelope(pushed)["data"]["id"]
    assert isinstance(build_id, str)
    return client, build_id


def data(result) -> dict:
    payload = envelope(result)["data"]
    assert isinstance(payload, dict)
    return payload


def refuse(name: str):
    def refusing(*args, **kwargs):
        pytest.fail(f"status reached the mutating helper {name}")

    return refusing


def test_signed_out_status_reports_no_partial_local_block(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build, "_scan_install", refuse("_scan_install"))
    monkeypatch.setattr("comfy_cli.builder_api.request_json", refuse("request_json"))

    result = invoke_status(workspace, token=None)

    assert result.exit_code == 1
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, result.stdout
    body = json.loads(lines[0])
    assert body["ok"] is False
    assert body["error"]["code"] == "build_not_signed_in"
    assert body.get("data") is None
    assert "local" not in json.dumps(body)


def test_clean_spec_and_clean_install_report_both_halves_clean(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _client, build_id = synced_workspace(workspace, monkeypatch)

    result = invoke_status(workspace)

    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["build"]["id"] == build_id
    assert payload["spec"]["path"] == str(workspace / "comfy-build.yaml")
    assert payload["remote"]["behind"] is False
    assert payload["local"]["drift"]["models"] == {"added": 0, "removed": 0, "changed": 0}
    assert payload["local"]["drift"]["customNodes"] == {"added": 0, "removed": 0, "changed": 0}
    assert payload["local"]["drift"]["pipDependencies"] == "unchanged"
    assert payload["local"]["drift"]["baseComfyVersion"] == "unchanged"
    assert payload["hint"] is None


def test_diverged_remote_and_drifted_install_are_reported_together(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, build_id = synced_workspace(workspace, monkeypatch)
    client.remote_revisions[build_id] = "revision-moved"
    (workspace / "models" / "checkpoints" / "extra.safetensors").write_bytes(b"EXTRA")

    result = invoke_status(workspace)

    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["remote"] == {"revision": "revision-moved", "behind": True}
    assert payload["local"]["drift"]["models"]["added"] == 1
    assert "comfy build pull" in payload["hint"]


def test_a_diverged_remote_leaves_local_drift_clean(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, build_id = synced_workspace(workspace, monkeypatch)
    client.remote_revisions[build_id] = "revision-moved"

    result = invoke_status(workspace)

    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["remote"]["behind"] is True
    assert payload["local"]["drift"]["models"] == {"added": 0, "removed": 0, "changed": 0}
    assert payload["local"]["drift"]["customNodes"] == {"added": 0, "removed": 0, "changed": 0}


def test_a_drifted_install_leaves_the_remote_in_sync(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced_workspace(workspace, monkeypatch)
    (workspace / "custom_nodes" / "local-node" / "extra.py").write_bytes(b"MORE")

    result = invoke_status(workspace)

    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["remote"]["behind"] is False
    assert payload["hint"] is None
    assert payload["local"]["drift"]["customNodes"]["changed"] == 1


def test_status_writes_nothing(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced_workspace(workspace, monkeypatch)
    spec_path = workspace / "comfy-build.yaml"
    before = spec_path.read_bytes()
    for mutation in ("write_build_spec", "_write_spec", "prepare_push", "merge_pulled_spec", "upload_assets"):
        monkeypatch.setattr(build, mutation, refuse(mutation))

    result = invoke_status(workspace)

    assert result.exit_code == 0, result.stderr
    assert spec_path.read_bytes() == before
    assert envelope(result)["changed"] is False


def test_agentic_status_needs_an_id_when_the_spec_has_none(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = PullBuilder()
    install_client(monkeypatch, client)
    initialized = invoke_init(workspace)
    assert initialized.exit_code == 0, initialized.stderr

    result = invoke_status(workspace)

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_id_unknown"
    assert client.calls == []


def test_tty_status_without_an_id_uses_the_build_picker(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = PullBuilder()
    install_client(monkeypatch, client)
    serve(client, "build-picked", {"schema": "distribution-definition/0", "models": [], "customNodes": []})
    initialized = invoke_init(workspace)
    assert initialized.exit_code == 0, initialized.stderr
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.ui.prompt_select", lambda *args, **kwargs: "build-picked")

    result = invoke_status(workspace, agentic=False)

    assert result.exit_code == 0, result.stderr
    assert [call["method"] for call in client.calls] == ["list_builds", "get_build"]


def test_status_payload_validates_against_its_registered_schema(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, build_id = synced_workspace(workspace, monkeypatch)
    client.remote_revisions[build_id] = "revision-moved"
    (workspace / "models" / "checkpoints" / "extra.safetensors").write_bytes(b"EXTRA")

    result = invoke_status(workspace)

    assert COMMAND_SCHEMAS["comfy build status"] == "build_status"
    jsonschema.Draft202012Validator(STATUS_SCHEMA).validate(data(result))


def test_status_warns_about_a_skipped_symlink_without_touching_its_payload(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`status` rescans, so it sees the omission and says so on stderr — but it
    persists no digest, so unlike init/update/push/pull it carries no rows in
    the envelope."""
    # Given
    synced_workspace(workspace, monkeypatch)
    vendored = tmp_path / "shared"
    vendored.mkdir()
    (vendored / "lib.py").write_bytes(b"LIB")
    (workspace / "custom_nodes" / "local-node" / "vendor").symlink_to(vendored)

    # When
    result = invoke_status(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert "excluded 1 symlink" in result.stderr
    assert "skipped_symlinks" not in data(result)
    jsonschema.Draft202012Validator(STATUS_SCHEMA).validate(data(result))


def test_drift_reports_the_update_diffs_counts_without_its_entries() -> None:
    stored = {"models": [{"type": "checkpoints", "filename": "a.safetensors"}], "customNodes": []}
    updated = {"models": [], "customNodes": [{"name": "pack"}], "pipDependencies": "x==1\n"}

    drift = diff_definitions(stored, updated).as_drift()

    assert drift == {
        "models": {"added": 0, "removed": 1, "changed": 0},
        "customNodes": {"added": 1, "removed": 0, "changed": 0},
        "pipDependencies": "changed",
    }


# --- the two comparisons are independent --------------------------------------


def test_a_spec_with_no_install_still_reports_spec_vs_remote(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given no install, When status runs, Then the remote comparison still lands.

    `status` promises drift "from the remote Build AND from the install", but a
    missing install refused the whole command with `build_models_dir_missing` —
    so a spec written by hand, which is the workflow the authoring skill
    documents, had no way to check spec-vs-remote at all. That is the exact
    comparison the failure skill says to read before choosing between `pull` and
    `push --force`.
    """
    # Given
    client, build_id = synced_workspace(workspace, monkeypatch)
    client.remote_revisions[build_id] = "revision-moved"
    shutil.rmtree(workspace / "models")

    # When
    result = invoke_status(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["remote"] == {"revision": "revision-moved", "behind": True}
    assert payload["local"]["scanned"] is False
    assert payload["local"]["drift"] is None
    assert str(workspace / "models") in payload["local"]["reason"]
    assert "comfy build pull" in payload["hint"]


def test_no_scan_reports_the_remote_half_without_touching_the_install(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given --no-scan, When status runs, Then the install is never read."""
    # Given
    synced_workspace(workspace, monkeypatch)
    monkeypatch.setattr(build, "_scan_install", refuse("_scan_install"))

    # When
    result = invoke_status(workspace, "--no-scan")

    # Then
    assert result.exit_code == 0, result.stderr
    assert data(result)["local"] == {"scanned": False, "reason": "--no-scan was passed", "drift": None}


def test_an_unreadable_python_degrades_the_local_half_only(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a Python that yields no provenance, When status runs, Then it reports.

    Pointing `status` at directories it could not scan escalated to demanding a
    working ComfyUI Python and failing outright, which is the same defect one
    level down: an unusable install must cost the drift half, not the report.
    """
    # Given
    synced_workspace(workspace, monkeypatch)
    monkeypatch.setattr(build, "capture_pip_provenance", lambda python: None)

    # When
    result = invoke_status(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["local"]["scanned"] is False
    assert payload["local"]["drift"] is None
    assert "provenance" in payload["local"]["reason"]


def test_a_skipped_scan_is_not_readable_as_a_clean_one(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a skipped scan, When compared with a real one, Then the shapes differ.

    `drift: null` beside `scanned: false` is the whole distinction: an absent
    comparison must never be readable as agreement.
    """
    # Given
    synced_workspace(workspace, monkeypatch)

    # When
    scanned = data(invoke_status(workspace))["local"]
    skipped = data(invoke_status(workspace, "--no-scan"))["local"]

    # Then
    assert scanned["scanned"] is True and scanned["drift"] is not None
    assert skipped["scanned"] is False and skipped["drift"] is None


def test_an_unpackageable_node_directory_degrades_the_local_half_only(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a node dir that will not package, When status runs, Then it reports.

    `init` and `push` must fail on this — packaging is all-or-nothing and they
    exist to package. `status` does not: an unpackageable directory is one more
    reason the drift half cannot be computed, and it must not take the
    spec-vs-remote half down with it.
    """
    # Given
    from comfy_cli.command.build_package import NodePackageError

    synced_workspace(workspace, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise NodePackageError(path=workspace / "custom_nodes" / "local-node", reason="a symlink loop")

    monkeypatch.setattr(build, "scan_custom_nodes", refuse)

    # When
    result = invoke_status(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    payload = data(result)
    assert payload["local"]["scanned"] is False
    assert "cannot package" in payload["local"]["reason"]
    assert payload["remote"]["behind"] is False
