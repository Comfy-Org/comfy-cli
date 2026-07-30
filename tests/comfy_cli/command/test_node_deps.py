"""Tests for ``comfy node deps`` — per-pack declared-vs-installed dependencies.

Builds a fixture workspace with fake ``custom_nodes/<pack>/requirements.txt`` +
``pyproject.toml`` under tmp_path and mocks the single ``pip list --format=json``
subprocess to a fixed payload, so nothing touches a real venv or the network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from comfy_cli.caller import Caller
from comfy_cli.command import node_deps as node_deps_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

# What the mocked workspace venv has installed. ``Pillow`` is deliberately
# capitalized to exercise canonicalization against a lowercase declaration.
INSTALLED = [
    {"name": "opencv-python", "version": "4.9.0.80"},
    {"name": "Pillow", "version": "10.2.0"},
    {"name": "numpy", "version": "1.26.4"},
]


@pytest.fixture(autouse=True)
def _renderer_isolation():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


class _FakePipList:
    """Stand-in for ``subprocess.run``; records every invocation."""

    def __init__(self, payload=INSTALLED, returncode: int = 0, stdout: str | None = None):
        self.calls: list[list[str]] = []
        self._payload = payload
        self._returncode = returncode
        self._stdout = stdout

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        stdout = self._stdout if self._stdout is not None else json.dumps(self._payload)
        if self._returncode != 0:
            raise subprocess.CalledProcessError(self._returncode, cmd, output=stdout, stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


@pytest.fixture
def fake_pip(monkeypatch) -> _FakePipList:
    fake = _FakePipList()
    monkeypatch.setattr(node_deps_cmd.subprocess, "run", fake)
    return fake


def _make_pack(workspace: Path, name: str, *, requirements: str | None = None, pyproject: str | None = None) -> Path:
    pack = workspace / "custom_nodes" / name
    pack.mkdir(parents=True)
    if requirements is not None:
        (pack / "requirements.txt").write_text(requirements)
    if pyproject is not None:
        (pack / "pyproject.toml").write_text(pyproject)
    return pack


@pytest.fixture
def workspace(tmp_path) -> Path:
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)

    _make_pack(
        ws,
        "comfyui-impact-pack",
        requirements=(
            "# a comment line\n"
            "\n"
            "opencv-python>=4.7  # inline comment\n"
            "ultralytics\n"
            "numpy<1.20\n"
            "pillow\n"
            "--extra-index-url https://example.invalid/simple\n"
            "-r other-requirements.txt\n"
            "git+https://example.invalid/pkg.git\n"
        ),
    )
    _make_pack(
        ws,
        "Registry-Pack",
        pyproject=(
            '[project]\nname = "registry-pack"\nversion = "1.0.0"\nlicense = {text = "MIT"}\n'
            'dependencies = ["numpy>=1.20", "requests"]\n'
        ),
    )
    return ws


def _packs_by_name(report: dict) -> dict[str, dict]:
    return {p["pack"]: p for p in report["packs"]}


def _reqs_by_raw(pack: dict) -> dict[str, dict]:
    return {r["raw"]: r for r in pack["requirements"]}


# ---------------------------------------------------------------------------
# statuses
# ---------------------------------------------------------------------------


def test_statuses_satisfied_mismatch_missing_unparseable(workspace, fake_pip):
    report, warnings = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert not warnings
    pack = _packs_by_name(report)["comfyui-impact-pack"]
    assert pack["status"] == "installed"
    assert pack["requirement_files"] == ["requirements.txt"]
    reqs = _reqs_by_raw(pack)

    assert reqs["opencv-python>=4.7"]["status"] == "satisfied"
    assert reqs["opencv-python>=4.7"]["installed"] == "4.9.0.80"
    assert reqs["opencv-python>=4.7"]["specifier"] == ">=4.7"

    # installed but rejected by the specifier → both versions reported
    assert reqs["numpy<1.20"]["status"] == "mismatch"
    assert reqs["numpy<1.20"]["installed"] == "1.26.4"

    # declared, not in the venv
    assert reqs["ultralytics"]["status"] == "missing"
    assert reqs["ultralytics"]["installed"] is None
    # a bare name is satisfied by whatever is installed
    assert reqs["pillow"]["specifier"] == ""

    for raw in (
        "--extra-index-url https://example.invalid/simple",
        "-r other-requirements.txt",
        "git+https://example.invalid/pkg.git",
    ):
        assert reqs[raw]["status"] == "unparseable", raw
        assert reqs[raw]["name"] is None

    assert pack["summary"] == {
        "satisfied": 2,  # opencv-python, pillow
        "mismatch": 1,  # numpy
        "missing": 1,  # ultralytics
        "unparseable": 3,
        "unknown": 0,
    }
    # comments and blank lines are dropped, everything else is kept
    assert len(pack["requirements"]) == 7


def test_name_canonicalization_matches_case_insensitively(workspace, fake_pip):
    """A lowercase ``pillow`` declaration resolves against installed ``Pillow``."""
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    pillow = _reqs_by_raw(_packs_by_name(report)["comfyui-impact-pack"])["pillow"]
    assert pillow["installed"] == "10.2.0"
    assert pillow["status"] == "satisfied"


def test_pyproject_dependencies_are_reported(workspace, fake_pip):
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    pack = _packs_by_name(report)["Registry-Pack"]
    assert pack["requirement_files"] == ["pyproject.toml"]
    reqs = _reqs_by_raw(pack)
    assert reqs["numpy>=1.20"]["source"] == "pyproject.toml"
    assert reqs["numpy>=1.20"]["status"] == "satisfied"
    assert reqs["requests"]["status"] == "missing"


def test_both_sources_are_unioned_with_per_line_source(tmp_path, fake_pip):
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _make_pack(
        ws,
        "dual-pack",
        requirements="numpy>=1.20\nrequests\n",
        pyproject=(
            '[project]\nname = "dual-pack"\nversion = "1.0.0"\nlicense = {text = "MIT"}\n'
            # ``numpy>=1.20`` is declared in BOTH files (deduped, requirements.txt
            # wins the source), ``pillow`` only here.
            'dependencies = ["numpy>=1.20", "pillow"]\n'
        ),
    )

    report, _ = node_deps_cmd.build_report(str(ws), python="/fake/python")
    pack = _packs_by_name(report)["dual-pack"]

    assert pack["requirement_files"] == ["requirements.txt", "pyproject.toml"]
    reqs = _reqs_by_raw(pack)
    assert len(pack["requirements"]) == 3
    assert reqs["numpy>=1.20"]["source"] == "requirements.txt"
    assert reqs["requests"]["source"] == "requirements.txt"
    assert reqs["pillow"]["source"] == "pyproject.toml"


def test_environment_marker_is_surfaced_not_evaluated(tmp_path, fake_pip):
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _make_pack(ws, "marker-pack", requirements='pywin32; sys_platform == "win32"\n')

    report, _ = node_deps_cmd.build_report(str(ws), python="/fake/python")
    req = _packs_by_name(report)["marker-pack"]["requirements"][0]

    assert req["name"] == "pywin32"
    assert req["marker"] == 'sys_platform == "win32"'
    assert req["status"] == "missing"


# ---------------------------------------------------------------------------
# pip list behavior
# ---------------------------------------------------------------------------


def test_non_pep440_installed_version_is_unknown_not_a_crash(tmp_path, monkeypatch):
    """A broken `.dist-info` version must not abort the whole report."""
    monkeypatch.setattr(
        node_deps_cmd.subprocess,
        "run",
        _FakePipList(payload=[{"name": "weirdpkg", "version": "not-a-version"}]),
    )
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _make_pack(ws, "weird-pack", requirements="weirdpkg>=1.0\nweirdpkg\n")

    report, _ = node_deps_cmd.build_report(str(ws), python="/fake/python")
    reqs = _reqs_by_raw(_packs_by_name(report)["weird-pack"])

    assert reqs["weirdpkg>=1.0"]["status"] == "unknown"
    assert reqs["weirdpkg>=1.0"]["installed"] == "not-a-version"
    # a bare name never needs a version comparison, so it stays satisfied
    assert reqs["weirdpkg"]["status"] == "satisfied"


def test_pip_list_runs_once_for_the_whole_report(workspace, fake_pip):
    node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert len(fake_pip.calls) == 1, f"pip list ran {len(fake_pip.calls)}x — must be once per report"
    assert fake_pip.calls[0] == ["/fake/python", "-m", "pip", "list", "--format=json"]


def test_pip_list_failure_degrades_to_unknown_with_a_warning(workspace, monkeypatch):
    monkeypatch.setattr(node_deps_cmd.subprocess, "run", _FakePipList(returncode=2))

    report, warnings = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert [w["code"] for w in warnings] == ["installed_versions_unavailable"]
    pack = _packs_by_name(report)["comfyui-impact-pack"]
    reqs = _reqs_by_raw(pack)
    assert reqs["opencv-python>=4.7"]["status"] == "unknown"
    assert reqs["opencv-python>=4.7"]["installed"] is None
    # unparseable lines stay unparseable — the pip probe can't change that
    assert reqs["-r other-requirements.txt"]["status"] == "unparseable"
    assert pack["summary"]["unknown"] == 4
    assert pack["summary"]["unparseable"] == 3


def test_pip_list_unparseable_output_degrades_to_unknown(workspace, monkeypatch):
    monkeypatch.setattr(node_deps_cmd.subprocess, "run", _FakePipList(stdout="not json at all"))

    report, warnings = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert [w["code"] for w in warnings] == ["installed_versions_unavailable"]
    assert _packs_by_name(report)["Registry-Pack"]["requirements"][0]["status"] == "unknown"


def test_pip_list_missing_interpreter_does_not_crash(workspace, monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("no such python")

    monkeypatch.setattr(node_deps_cmd.subprocess, "run", _boom)

    report, warnings = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert report is not None
    assert [w["code"] for w in warnings] == ["installed_versions_unavailable"]


# ---------------------------------------------------------------------------
# pack filtering
# ---------------------------------------------------------------------------


def test_pack_name_filter_is_case_insensitive(workspace, fake_pip):
    report, _ = node_deps_cmd.build_report(str(workspace), ["REGISTRY-pack"], python="/fake/python")

    assert [p["pack"] for p in report["packs"]] == ["Registry-Pack"]
    assert report["packs"][0]["status"] == "installed"


def test_unknown_pack_name_reports_not_installed(workspace, fake_pip):
    report, _ = node_deps_cmd.build_report(str(workspace), ["nope-pack", "comfyui-impact-pack"], python="/fake/python")

    rows = _packs_by_name(report)
    assert rows["nope-pack"]["status"] == "not_installed"
    assert rows["nope-pack"]["path"] is None
    assert rows["nope-pack"]["requirements"] == []
    assert rows["comfyui-impact-pack"]["status"] == "installed"


def test_no_names_reports_every_pack(workspace, fake_pip):
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")

    assert sorted(p["pack"] for p in report["packs"]) == ["Registry-Pack", "comfyui-impact-pack"]


def test_unreadable_requirements_file_warns_instead_of_crashing(tmp_path, fake_pip, monkeypatch):
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _make_pack(ws, "locked-pack", requirements="numpy\n")

    real_read_text = Path.read_text

    def _deny(self, *a, **k):
        if self.name == "requirements.txt":
            raise PermissionError("denied")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _deny)

    report, warnings = node_deps_cmd.build_report(str(ws), python="/fake/python")

    assert [w["code"] for w in warnings] == ["pack_read_error"]
    assert _packs_by_name(report)["locked-pack"]["requirements"] == []


def test_pack_without_declarations_reports_empty_requirements(tmp_path, fake_pip):
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _make_pack(ws, "bare-pack")

    report, _ = node_deps_cmd.build_report(str(ws), python="/fake/python")
    pack = _packs_by_name(report)["bare-pack"]

    assert pack["requirement_files"] == []
    assert pack["requirements"] == []
    assert pack["summary"]["satisfied"] == 0


# ---------------------------------------------------------------------------
# workspace-level fields
# ---------------------------------------------------------------------------


def test_compiled_lock_presence_and_path(workspace, fake_pip):
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")
    assert report["compiled_lock"] == {"present": False, "path": None}

    (workspace / "requirements.compiled").write_text("numpy==1.26.4\n")
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")
    assert report["compiled_lock"]["present"] is True
    assert report["compiled_lock"]["path"] == str(workspace / "requirements.compiled")


def test_missing_custom_nodes_dir_is_not_an_error(tmp_path, fake_pip):
    ws = tmp_path / "ComfyUI"
    ws.mkdir()

    report, warnings = node_deps_cmd.build_report(str(ws), python="/fake/python")

    assert report["packs"] == []
    assert not warnings


def test_no_workspace_returns_none(tmp_path):
    report, warnings = node_deps_cmd.build_report(str(tmp_path / "nope"), python="/fake/python")
    assert report is None
    assert warnings == []

    report, _ = node_deps_cmd.build_report(None, python="/fake/python")
    assert report is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _force_json_renderer() -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def test_cli_json_envelope_shape(workspace, fake_pip, capsys):
    r = _force_json_renderer()
    node_deps_cmd.execute(r, str(workspace))

    out = capsys.readouterr().out.strip()
    # Envelope contract: stdout must be exactly ONE line. The registry parser's
    # own warnings must land on stderr, not corrupt the envelope.
    assert len(out.splitlines()) == 1, f"stdout not a single envelope line:\n{out}"
    envelope = json.loads(out)

    assert envelope["ok"] is True
    assert envelope["command"] == "node deps"
    assert envelope["data"]["workspace"] == str(workspace)
    assert isinstance(envelope["data"]["packs"], list)
    assert envelope["data"]["warnings"] == []
    assert r.exit_code == 0  # a report, not a check: missing deps stay exit 0


def test_cli_no_workspace_emits_error_envelope(tmp_path, capsys):
    import typer

    r = _force_json_renderer()
    # typer.Exit is what makes the *process* exit non-zero; renderer.error alone
    # only records the code (see `comfy which`).
    with pytest.raises(typer.Exit) as exc:
        node_deps_cmd.execute(r, str(tmp_path / "nope"))
    assert exc.value.exit_code == 1

    envelope = json.loads(capsys.readouterr().out.strip())
    assert envelope["ok"] is False
    assert envelope["command"] == "node deps"
    assert envelope["error"]["code"] == "not_in_workspace"
    assert envelope["error"]["hint"]
    assert r.exit_code == 1


def test_cli_pretty_mode_renders_without_crashing(workspace, fake_pip, capsys):
    reset_renderer_for_testing()
    r = Renderer(mode=OutputMode.PRETTY)
    set_renderer(r)

    node_deps_cmd.execute(r, str(workspace))

    out = capsys.readouterr().out
    assert "comfyui-impact-pack" in out
    assert "satisfied" in out


def test_report_validates_against_shipped_schema(workspace, fake_pip):
    import jsonschema

    from comfy_cli import discovery

    schema = discovery._read_schema("node_deps")
    report, _ = node_deps_cmd.build_report(str(workspace), python="/fake/python")
    report["warnings"] = []
    jsonschema.validate(report, schema)

    # a not-installed row and an unknown-status report must validate too
    report_filtered, _ = node_deps_cmd.build_report(str(workspace), ["nope-pack"], python="/fake/python")
    report_filtered["warnings"] = []
    jsonschema.validate(report_filtered, schema)


def test_command_registers_a_schema():
    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy node deps"] == "node_deps"


def test_deps_command_is_registered_on_the_node_app():
    from comfy_cli.command.custom_nodes.command import app

    names = {c.name for c in app.registered_commands}
    assert "deps" in names
    assert {"deps-in-workflow", "install-deps"} <= names, "must not shadow the existing deps-* commands"
