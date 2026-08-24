"""`comfy build init/update --from-snapshot` — authoring through the builder's importer.

Every test drives the real ``BuilderClient`` and stubs only the shared
``request_json`` seam, so the URL, the HTTP verb and the request body asserted
here are the ones the command would really put on the wire.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import read_build_spec, write_build_spec

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"

_RESOLVE_URL_SUFFIX = "/v1/snapshots/resolve"

#: A bare ComfyUI Desktop snapshot — the shape actually written under
#: `.launcher/snapshots/`, which `as_snapshot_envelope` has to wrap.
_SNAPSHOT = {
    "comfyui": {"baseTag": "0.3.40"},
    "customNodes": [{"type": "cnr", "id": "comfyui-essentials", "dirName": "comfyui-essentials", "version": "1.1.0"}],
    "pipPackages": {"numpy": "1.26.4"},
    "pythonVersion": "3.12.7",
}

_IMPORTED_DEFINITION = {
    "baseComfyVersion": "v0.3.40",
    "models": [],
    "customNodes": [
        {"name": "comfyui-essentials", "id": "comfyui-essentials", "registryVersion": "1.1.0"},
        {"name": "hand-rolled", "repository": "https://github.com/org/hand-rolled", "gitRef": "a" * 40},
    ],
    "pipDependencies": "numpy==1.26.4\n",
}

_IMPORT_REPORT = {
    "pythonSatisfied": False,
    "notInRegistry": ["ghost-pack"],
    "skippedPins": ["torch"],
}

_SCAN_ONLY_FLAGS = ("--models-dir", "--custom-nodes-dir", "--python", "--comfy-url")


class _Recorder:
    """Stands in for the shared ``request_json`` seam, recording each request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        self.calls.append({"method": method, "url": url, "body": body})
        return 200, {"definition": dict(_IMPORTED_DEFINITION), "report": dict(_IMPORT_REPORT)}

    @property
    def resolve_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["url"].endswith(_RESOLVE_URL_SUFFIX)]


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
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", rec)
    return rec


@pytest.fixture
def signed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def install(tmp_path: Path) -> Path:
    root = tmp_path / "install"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "checkpoints" / "base.safetensors").write_bytes(b"MODEL")
    (root / "custom_nodes" / "local-node").mkdir(parents=True)
    (root / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"NODE")
    return root


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "desktop-snapshot.json"
    path.write_text(json.dumps(_SNAPSHOT), encoding="utf-8")
    return path


def _run(verb: str, root: Path, *args: str, token: str | None = "tok_test"):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        ["build", verb, *args, str(root)],
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": "json",
            "NO_COLOR": "1",
            "COLUMNS": "400",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def _scan_init(install: Path) -> Path:
    """A spec written by the ordinary local `init`, for the `update` tests."""
    result = CliRunner(mix_stderr=False).invoke(
        cli_app,
        ["build", "init", "--name", "Foo", "--python", sys.executable, "--comfy-version", "0.3.0", str(install)],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )
    assert result.exit_code == 0, result.output
    return install / "comfy-build.yaml"


@pytest.fixture
def spec_file(install: Path) -> Path:
    return _scan_init(install)


def _envelope(result) -> dict:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stdout
    return json.loads(lines[-1])


def _definition(path: Path) -> dict:
    definition = read_build_spec(path)["definition"]
    assert isinstance(definition, dict)
    return definition


def _flat(text: str) -> str:
    """Stderr with Rich's line wrapping collapsed, so a substring survives it."""
    return " ".join(text.split())


# --- init --------------------------------------------------------------------


def test_init_issues_exactly_one_resolve_carrying_the_snapshot_envelope(
    install: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(recorder.calls) == 1
    call = recorder.resolve_calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"https://builder.test{_RESOLVE_URL_SUFFIX}"
    assert call["body"] == {"snapshot": build.as_snapshot_envelope(_SNAPSHOT)}


def test_init_writes_the_importers_definition_with_an_unsynced_spec(
    install: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    spec = read_build_spec(install / "comfy-build.yaml")
    assert spec["id"] is None
    assert spec["syncedRevision"] is None
    assert spec["name"] == "Foo"
    definition = _definition(install / "comfy-build.yaml")
    assert definition["baseComfyVersion"] == "v0.3.40"
    assert definition["pipDependencies"] == "numpy==1.26.4\n"
    assert definition["models"] == []
    assert isinstance(definition["customNodes"], list)
    assert {node["name"] for node in definition["customNodes"]} == {"comfyui-essentials", "hand-rolled"}


def test_init_never_scans_the_install(
    install: Path, snapshot: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_scan_install", lambda *args, **kwargs: pytest.fail("the install was scanned"))

    # When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr


def test_init_renders_every_advisory_on_stderr_and_one_envelope_on_stdout(
    install: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given
    expected = build.report_advisories(_IMPORT_REPORT)
    assert expected, "the fixture report must produce advisories or this passes vacuously"

    # When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    stderr = _flat(result.stderr)
    for line in expected:
        assert _flat(line) in stderr
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 1
    assert _envelope(result)["data"]["advisories"] == expected


def test_init_signed_out_refuses_before_the_snapshot_file_is_touched(
    install: Path, tmp_path: Path, recorder: _Recorder, signed_out: None
) -> None:
    # Given: a path that does not exist, so reading first would be a different code.
    missing = tmp_path / "never-read.json"

    # When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(missing), token=None)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_not_signed_in"
    assert recorder.calls == []
    assert not (install / "comfy-build.yaml").exists()


def test_init_reports_an_unreadable_snapshot_once_signed_in(install: Path, tmp_path: Path, recorder: _Recorder) -> None:
    # Given
    missing = tmp_path / "absent.json"

    # When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(missing))

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_spec_invalid"
    assert recorder.calls == []


@pytest.mark.parametrize("flag", _SCAN_ONLY_FLAGS)
def test_init_refuses_a_scan_only_flag_beside_the_snapshot(
    install: Path, snapshot: Path, recorder: _Recorder, flag: str
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-snapshot", str(snapshot), flag, "whatever")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["conflict"] == ["--from-snapshot", flag]
    assert flag in error["message"] and "--from-snapshot" in error["message"]
    assert recorder.calls == []
    assert not (install / "comfy-build.yaml").exists()


def test_init_names_every_conflicting_flag_at_once(install: Path, snapshot: Path, recorder: _Recorder) -> None:
    # Given / When
    result = _run(
        "init",
        install,
        "--name",
        "Foo",
        "--from-snapshot",
        str(snapshot),
        "--models-dir",
        str(install / "models"),
        "--python",
        sys.executable,
    )

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["details"]["conflict"] == ["--from-snapshot", "--models-dir", "--python"]


def test_init_leaves_the_scan_flags_usable_without_a_snapshot(install: Path, recorder: _Recorder) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--python", sys.executable, "--comfy-version", "0.3.0")

    # Then
    assert result.exit_code == 0, result.stderr
    assert recorder.calls == []
    assert _envelope(result)["data"]["source"] == "scan"


@pytest.mark.parametrize("source", ["scan", "snapshot"])
def test_the_init_payload_validates_against_its_registered_schema(
    install: Path, snapshot: Path, recorder: _Recorder, source: str
) -> None:
    # Given
    args = (
        ["--from-snapshot", str(snapshot)]
        if source == "snapshot"
        else ["--python", sys.executable, "--comfy-version", "0.3.0"]
    )

    # When
    envelope = _envelope(_run("init", install, "--name", "Foo", *args))

    # Then
    schema = json.loads((SCHEMAS_DIR / "build_init.json").read_text(encoding="utf-8"))
    assert envelope["data"]["source"] == source
    jsonschema.Draft202012Validator(schema).validate(envelope["data"])


# --- update ------------------------------------------------------------------


def test_update_issues_exactly_one_resolve_carrying_the_snapshot_envelope(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("update", install, "-y", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(recorder.calls) == 1
    call = recorder.resolve_calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"https://builder.test{_RESOLVE_URL_SUFFIX}"
    assert call["body"] == {"snapshot": build.as_snapshot_envelope(_SNAPSHOT)}


def test_update_preserves_the_spec_metadata_across_an_import(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given
    spec = read_build_spec(spec_file)
    spec.update({"id": "bld_1", "name": "Named", "description": "why", "syncedRevision": "2026-08-01T12:00:00Z"})
    write_build_spec(spec_file, spec)

    # When
    result = _run("update", install, "-y", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    rewritten = read_build_spec(spec_file)
    assert {key: rewritten[key] for key in ("id", "name", "description", "syncedRevision")} == {
        "id": "bld_1",
        "name": "Named",
        "description": "why",
        "syncedRevision": "2026-08-01T12:00:00Z",
    }
    assert rewritten["definition"] != spec["definition"]
    assert _definition(spec_file)["baseComfyVersion"] == "v0.3.40"


def test_update_reports_the_import_through_the_ordinary_diff(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given / When
    envelope = _envelope(_run("update", install, "-y", "--from-snapshot", str(snapshot)))

    # Then
    diff = envelope["data"]["diff"]
    assert envelope["data"]["source"] == "snapshot"
    assert envelope["data"]["snapshot"] == str(snapshot)
    assert envelope["data"]["advisories"] == build.report_advisories(_IMPORT_REPORT)
    assert envelope["changed"] is True
    assert envelope["data"]["written"] is True
    # A snapshot describes no models, so the scanned one is dropped and both
    # imported packs replace the single local one.
    assert (diff["models"]["added"], diff["models"]["removed"]) == (0, 1)
    assert (diff["customNodes"]["added"], diff["customNodes"]["removed"]) == (2, 1)


def test_update_never_rescans_the_install(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_scan_install", lambda *args, **kwargs: pytest.fail("the install was rescanned"))

    # When
    result = _run("update", install, "-y", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr


def test_update_still_requires_confirmation_for_an_agentic_caller(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given
    before = spec_file.read_bytes()

    # When
    result = _run("update", install, "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_update_needs_confirm"
    assert spec_file.read_bytes() == before


def test_update_dry_run_imports_but_writes_nothing(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given
    before = spec_file.read_bytes()

    # When
    result = _run("update", install, "--dry-run", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["written"] is False
    assert len(recorder.resolve_calls) == 1
    assert spec_file.read_bytes() == before


def test_update_renders_every_advisory_on_stderr_and_one_envelope_on_stdout(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given
    expected = build.report_advisories(_IMPORT_REPORT)

    # When
    result = _run("update", install, "-y", "--from-snapshot", str(snapshot))

    # Then
    assert result.exit_code == 0, result.stderr
    stderr = _flat(result.stderr)
    for line in expected:
        assert _flat(line) in stderr
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 1


def test_update_signed_out_refuses_before_the_snapshot_file_is_touched(
    install: Path, spec_file: Path, tmp_path: Path, recorder: _Recorder, signed_out: None
) -> None:
    # Given
    missing = tmp_path / "never-read.json"
    before = spec_file.read_bytes()

    # When
    result = _run("update", install, "-y", "--from-snapshot", str(missing), token=None)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_not_signed_in"
    assert recorder.calls == []
    assert spec_file.read_bytes() == before


@pytest.mark.parametrize("flag", _SCAN_ONLY_FLAGS)
def test_update_refuses_a_scan_only_flag_beside_the_snapshot(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder, flag: str
) -> None:
    # Given
    before = spec_file.read_bytes()

    # When
    result = _run("update", install, "-y", "--from-snapshot", str(snapshot), flag, "whatever")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["conflict"] == ["--from-snapshot", flag]
    assert recorder.calls == []
    assert spec_file.read_bytes() == before


def test_the_update_payload_validates_against_its_registered_schema(
    install: Path, spec_file: Path, snapshot: Path, recorder: _Recorder
) -> None:
    # Given / When
    envelope = _envelope(_run("update", install, "-y", "--from-snapshot", str(snapshot)))

    # Then
    schema = json.loads((SCHEMAS_DIR / "build_update.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(envelope["data"])
