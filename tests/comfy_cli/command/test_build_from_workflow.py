"""`comfy build init/update --from-workflow` — authoring from a graph file.

The sibling of ``test_build_from_snapshot``. Every test drives the real
``BuilderClient`` and stubs only the shared ``request_json`` seam, so the URL,
the HTTP verb and the request body asserted here are the ones the command would
really put on the wire.

The single most important assertion in this module is that the request goes to
``/v1/workflows/resolve``. The builder also serves ``/v1/builds/from-workflow``,
which reads the same graph but *creates a build row* — and a spec `init` writes
to a local file, or an `update` that amends a build which already exists, has
nowhere to put a build the server minted on its own.
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
from comfy_cli.command.build_spec import read_build_spec

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"

_RESOLVE_URL_SUFFIX = "/v1/workflows/resolve"

#: The editing ("UI") dialect. The API export is a flat map of node ids; both
#: are accepted, because the builder is what reads them.
_UI_WORKFLOW = {"nodes": [{"type": "KSampler"}, {"type": "WAS_Image_Blend"}], "links": []}
_API_WORKFLOW = {"3": {"class_type": "KSampler", "inputs": {}}}

#: A workflow names no model sources, so an imported definition carries none.
_IMPORTED_DEFINITION = {
    "baseComfyVersion": "v0.3.40",
    "models": [],
    "customNodes": [{"name": "comfyui-essentials", "id": "comfyui-essentials", "registryVersion": "1.1.0"}],
}

_IMPORT_REPORT = {
    "pinnedToLatest": True,
    "unresolvedClasses": ["WAS_Image_Blend"],
    "models": [{"filename": "sd15.safetensors", "status": "missing", "usedBy": ["CheckpointLoaderSimple"]}],
    "partnerClasses": {"LumaImageNode": "Luma"},
}

_SCAN_ONLY_FLAGS = ("--models-dir", "--custom-nodes-dir", "--python", "--comfy-url")


class _Recorder:
    """Stands in for the shared ``request_json`` seam, recording each request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        self.calls.append({"method": method, "url": url, "body": body, "timeout": timeout})
        return 200, {"definition": dict(_IMPORTED_DEFINITION), "report": dict(_IMPORT_REPORT), "format": "ui"}


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
def workflow(tmp_path: Path) -> Path:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(_UI_WORKFLOW), encoding="utf-8")
    return path


def _run(verb: str, root: Path, *args: str, token: str | None = "tok_test"):
    return CliRunner().invoke(
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
    result = CliRunner().invoke(
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


def _flat(text: str) -> str:
    """Stderr with Rich's line wrapping collapsed, so a substring survives it."""
    return " ".join(text.split())


# --- init --------------------------------------------------------------------


def test_init_resolves_the_workflow_rather_than_creating_a_build(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"https://builder.test{_RESOLVE_URL_SUFFIX}"
    assert call["body"] == {"workflow": _UI_WORKFLOW}


def test_init_gives_the_registry_sweep_longer_than_a_plain_post(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    """The builder looks every distinct node class up in the registry behind a
    budget of its own, so the shared 30s POST default truncates a large graph."""
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    assert recorder.calls[0]["timeout"] == 90.0


def test_init_takes_the_api_export_dialect_unchanged(install: Path, tmp_path: Path, recorder: _Recorder) -> None:
    """The API export has no `nodes` and no `links`, and the builder reads it. A
    frontend-shaped check here would refuse a file the server accepts."""
    # Given
    path = tmp_path / "api.json"
    path.write_text(json.dumps(_API_WORKFLOW), encoding="utf-8")

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(path))

    # Then
    assert result.exit_code == 0, result.stderr
    assert recorder.calls[0]["body"] == {"workflow": _API_WORKFLOW}


def test_init_writes_the_importers_definition_with_an_unsynced_spec(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    spec = read_build_spec(install / "comfy-build.yaml")
    assert spec["id"] is None
    assert spec["syncedRevision"] is None
    assert spec["name"] == "Foo"
    definition = spec["definition"]
    assert isinstance(definition, dict)
    assert definition["baseComfyVersion"] == "v0.3.40"
    # A workflow names no model sources, so the spec starts with none.
    assert definition["models"] == []


def test_init_never_scans_the_install(
    install: Path, workflow: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_scan_install", lambda *args, **kwargs: pytest.fail("the install was scanned"))

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr


def test_init_names_the_workflow_as_the_source_in_its_payload(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["source"] == "workflow"
    assert data["workflow"] == str(workflow)
    assert "snapshot" not in data


def test_init_renders_every_advisory_on_stderr_and_one_envelope_on_stdout(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    # Given
    expected = build.report_advisories(_IMPORT_REPORT)
    assert expected, "the fixture report must produce advisories or this passes vacuously"

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    stderr = _flat(result.stderr)
    for line in expected:
        assert _flat(line) in stderr
    assert len([line for line in result.stdout.splitlines() if line.strip()]) == 1
    assert _envelope(result)["data"]["advisories"] == expected


def test_init_signed_out_refuses_before_the_workflow_file_is_touched(
    install: Path, tmp_path: Path, recorder: _Recorder, signed_out: None
) -> None:
    # Given: a path that does not exist, so reading first would be a different code.
    missing = tmp_path / "never-read.json"

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(missing), token=None)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_not_signed_in"
    assert recorder.calls == []
    assert not (install / "comfy-build.yaml").exists()


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        pytest.param(None, "absent", id="no-such-file"),
        pytest.param("{not json", "unparseable", id="not-json"),
        pytest.param("[1, 2, 3]", "not an object", id="json-but-not-an-object"),
    ],
)
def test_init_reports_an_unreadable_workflow_under_its_own_code(
    install: Path, tmp_path: Path, recorder: _Recorder, contents: str | None, reason: str
) -> None:
    """A workflow that cannot be read is `build_workflow_invalid`, not the spec
    code: the file the user named is a graph, and the spec does not exist yet."""
    # Given
    path = tmp_path / "graph.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(path))

    # Then
    assert result.exit_code == 1, reason
    error = _envelope(result)["error"]
    assert error["code"] == "build_workflow_invalid"
    assert error["details"]["path"] == str(path)
    assert recorder.calls == []


@pytest.mark.parametrize("flag", _SCAN_ONLY_FLAGS)
def test_init_refuses_a_scan_only_flag_beside_the_workflow(
    install: Path, workflow: Path, recorder: _Recorder, flag: str
) -> None:
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow), flag, "whatever")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["conflict"] == ["--from-workflow", flag]
    assert recorder.calls == []
    assert not (install / "comfy-build.yaml").exists()


def test_init_refuses_both_import_flags_at_once(install: Path, workflow: Path, recorder: _Recorder) -> None:
    """Two sources for one definition, and they disagree by construction: a
    snapshot records what was installed, a workflow only what a graph refers to."""
    # Given / When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow), "--from-snapshot", str(workflow))

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["conflict"] == ["--from-snapshot", "--from-workflow"]
    assert recorder.calls == []
    assert not (install / "comfy-build.yaml").exists()


def test_the_init_payload_validates_against_its_registered_schema(
    install: Path, workflow: Path, recorder: _Recorder
) -> None:
    # Given
    schema = json.loads((SCHEMAS_DIR / "build_init.json").read_text(encoding="utf-8"))

    # When
    result = _run("init", install, "--name", "Foo", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    jsonschema.validate(_envelope(result)["data"], schema)


# --- update ------------------------------------------------------------------


def test_update_resolves_the_workflow_rather_than_creating_a_build(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder
) -> None:
    """`update` amends a build that already exists, so the one-call creator is
    not merely redundant here — it has no meaning at all."""
    # Given / When
    result = _run("update", install, "--yes", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"https://builder.test{_RESOLVE_URL_SUFFIX}"
    assert call["body"] == {"workflow": _UI_WORKFLOW}


def test_update_reports_the_import_through_the_ordinary_diff(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder
) -> None:
    # Given / When
    result = _run("update", install, "--yes", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["source"] == "workflow"
    assert data["workflow"] == str(workflow)
    assert data["written"] is True
    assert data["diff"]


def test_update_never_rescans_the_install(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_scan_install", lambda *args, **kwargs: pytest.fail("the install was scanned"))

    # When
    result = _run("update", install, "--yes", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr


def test_update_still_requires_confirmation_for_an_agentic_caller(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder
) -> None:
    """Under --json nothing prompts, so the confirmation comes back as a refusal
    envelope rather than opening a TUI on the stream the envelope is written to."""
    # Given / When
    result = _run("update", install, "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "build_update_needs_confirm"


def test_update_dry_run_imports_but_writes_nothing(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder
) -> None:
    # Given
    before = spec_file.read_text(encoding="utf-8")

    # When
    result = _run("update", install, "--dry-run", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["written"] is False
    assert spec_file.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("flag", _SCAN_ONLY_FLAGS)
def test_update_refuses_a_scan_only_flag_beside_the_workflow(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder, flag: str
) -> None:
    # Given / When
    result = _run("update", install, "--yes", "--from-workflow", str(workflow), flag, "whatever")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"]["conflict"] == ["--from-workflow", flag]
    assert recorder.calls == []


def test_the_update_payload_validates_against_its_registered_schema(
    install: Path, workflow: Path, spec_file: Path, recorder: _Recorder
) -> None:
    # Given
    schema = json.loads((SCHEMAS_DIR / "build_update.json").read_text(encoding="utf-8"))

    # When
    result = _run("update", install, "--yes", "--from-workflow", str(workflow))

    # Then
    assert result.exit_code == 0, result.stderr
    jsonschema.validate(_envelope(result)["data"], schema)
