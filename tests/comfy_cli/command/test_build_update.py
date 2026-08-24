"""`comfy build update` — rescan, diff, confirm, rewrite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import read_build_spec, write_build_spec
from comfy_cli.discovery import COMMAND_SCHEMAS

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"


@pytest.fixture(autouse=True)
def stable_command_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build,
        "capture_pip_provenance",
        lambda python: {
            "pipDependencies": "example==1.0.0\n",
            "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
        },
    )


@pytest.fixture
def install(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "checkpoints" / "base.safetensors").write_bytes(b"MODEL")
    (root / "custom_nodes" / "local-node").mkdir(parents=True)
    (root / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"NODE")
    return root


def _run(runner: CliRunner, verb: str, root: Path, *args: str):
    return runner.invoke(
        cli_app,
        ["build", verb, *args, "--python", sys.executable, "--comfy-version", "0.3.0", str(root)],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )


def _envelope(result) -> dict:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.output
    return json.loads(lines[-1])


@pytest.fixture
def spec_file(install: Path) -> Path:
    result = _run(CliRunner(), "init", install, "--name", "Foo")
    assert result.exit_code == 0, result.output
    return install / "comfy-build.yaml"


def _add_model(install: Path, filename: str = "extra.safetensors") -> None:
    (install / "models" / "loras").mkdir(exist_ok=True)
    (install / "models" / "loras" / filename).write_bytes(b"EXTRA")


def test_two_updates_of_an_unchanged_install_leave_the_file_byte_identical(install: Path, spec_file: Path) -> None:
    # Given
    runner = CliRunner()
    assert _run(runner, "update", install, "-y").exit_code == 0
    after_first = spec_file.read_bytes()

    # When
    second = _run(runner, "update", install, "-y")

    # Then
    assert second.exit_code == 0, second.output
    assert spec_file.read_bytes() == after_first


def test_an_update_of_an_unchanged_install_reports_an_empty_diff(install: Path, spec_file: Path) -> None:
    # Given
    runner = CliRunner()
    assert _run(runner, "update", install, "-y").exit_code == 0

    # When
    envelope = _envelope(_run(runner, "update", install, "-y"))

    # Then
    diff = envelope["data"]["diff"]
    assert envelope["command"] == "build update"
    assert envelope["changed"] is False
    assert diff["models"] == {"added": 0, "removed": 0, "changed": 0, "entries": []}
    assert diff["customNodes"] == {"added": 0, "removed": 0, "changed": 0, "entries": []}
    assert diff["pipDependencies"] == "unchanged"
    assert diff["baseComfyVersion"] == "unchanged"
    assert envelope["data"]["summary"] == "no changes"


def test_a_drifted_install_reports_the_new_model_and_writes_it(install: Path, spec_file: Path) -> None:
    # Given
    _add_model(install)

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"]["written"] is True
    models = envelope["data"]["diff"]["models"]
    assert (models["added"], models["removed"], models["changed"]) == (1, 0, 0)
    assert models["entries"] == [{"change": "added", "name": "loras/extra.safetensors", "fields": []}]
    definition = read_build_spec(spec_file)["definition"]
    assert isinstance(definition, dict)
    assert {entry["filename"] for entry in definition["models"]} == {"base.safetensors", "extra.safetensors"}


def test_a_removed_model_is_reported_and_dropped(install: Path, spec_file: Path) -> None:
    # Given
    (install / "models" / "checkpoints" / "base.safetensors").unlink()

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
    models = _envelope(result)["data"]["diff"]["models"]
    assert (models["added"], models["removed"], models["changed"]) == (0, 1, 0)
    assert models["entries"] == [{"change": "removed", "name": "checkpoints/base.safetensors", "fields": []}]
    definition = read_build_spec(spec_file)["definition"]
    assert isinstance(definition, dict)
    assert definition["models"] == []


def test_dry_run_reports_the_diff_and_writes_nothing_at_all(install: Path, spec_file: Path) -> None:
    # Given
    _add_model(install)
    before_bytes = spec_file.read_bytes()
    before_mtime = spec_file.stat().st_mtime_ns

    # When
    result = _run(CliRunner(), "update", install, "--dry-run")

    # Then
    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["data"]["dry_run"] is True
    assert envelope["data"]["written"] is False
    assert envelope["changed"] is False
    assert envelope["data"]["diff"]["models"]["added"] == 1
    assert spec_file.read_bytes() == before_bytes
    assert spec_file.stat().st_mtime_ns == before_mtime


def test_dry_run_never_asks_for_confirmation(install: Path, spec_file: Path, monkeypatch) -> None:
    # Given: --dry-run stops before the prompt, so an agentic caller is fine.
    _add_model(install)
    monkeypatch.setattr(build, "confirm", lambda *args, **kwargs: pytest.fail("confirmation was requested"))

    # When
    result = _run(CliRunner(), "update", install, "--dry-run")

    # Then
    assert result.exit_code == 0, result.output


def test_a_non_interactive_caller_without_yes_refuses_and_writes_nothing(install: Path, spec_file: Path) -> None:
    # Given
    _add_model(install)
    before = spec_file.read_bytes()

    # When
    result = _run(CliRunner(), "update", install)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_update_needs_confirm"
    assert error["details"]["missing"] == ["--yes"]
    assert spec_file.read_bytes() == before


def test_the_spec_metadata_survives_the_rescan(install: Path, spec_file: Path) -> None:
    # Given
    spec = read_build_spec(spec_file)
    spec.update({"id": "bld_1", "name": "Named", "description": "why", "syncedRevision": "2026-08-01T12:00:00Z"})
    write_build_spec(spec_file, spec)
    _add_model(install)

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
    rewritten = read_build_spec(spec_file)
    assert {key: rewritten[key] for key in ("id", "name", "description", "syncedRevision")} == {
        "id": "bld_1",
        "name": "Named",
        "description": "why",
        "syncedRevision": "2026-08-01T12:00:00Z",
    }
    assert rewritten["definition"] != spec["definition"]


def test_a_rescan_keeps_a_cached_blob_whose_bytes_never_moved(install: Path, spec_file: Path) -> None:
    # Given: the shape `push` leaves behind once an entry has been uploaded.
    spec = read_build_spec(spec_file)
    definition = spec["definition"]
    assert isinstance(definition, dict)
    definition["models"][0]["blobId"] = "blob-1"
    definition["customNodes"][0]["blobId"] = "blob-2"
    write_build_spec(spec_file, spec)
    before = spec_file.read_bytes()

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
    assert _envelope(result)["data"]["summary"] == "no changes"
    assert spec_file.read_bytes() == before


def test_a_rescan_drops_a_cached_blob_whose_bytes_moved(install: Path, spec_file: Path) -> None:
    # Given
    spec = read_build_spec(spec_file)
    definition = spec["definition"]
    assert isinstance(definition, dict)
    definition["models"][0]["blobId"] = "blob-1"
    write_build_spec(spec_file, spec)
    (install / "models" / "checkpoints" / "base.safetensors").write_bytes(b"REPLACED BYTES")

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
    rewritten = read_build_spec(spec_file)["definition"]
    assert isinstance(rewritten, dict)
    assert "blobId" not in rewritten["models"][0]


def test_a_missing_spec_is_build_spec_not_found(install: Path) -> None:
    # Given / When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_spec_not_found"
    assert error["details"]["path"] == str(install / "comfy-build.yaml")


def test_the_emitted_payload_validates_against_its_registered_schema(install: Path, spec_file: Path) -> None:
    # Given
    _add_model(install)

    # When
    envelope = _envelope(_run(CliRunner(), "update", install, "-y"))

    # Then
    assert COMMAND_SCHEMAS["comfy build update"] == "build_update"
    schema = json.loads((SCHEMAS_DIR / "build_update.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(envelope["data"])


def test_update_never_constructs_a_builder_client(install: Path, spec_file: Path, monkeypatch) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))
    _add_model(install)

    # When
    result = _run(CliRunner(), "update", install, "-y")

    # Then
    assert result.exit_code == 0, result.output
