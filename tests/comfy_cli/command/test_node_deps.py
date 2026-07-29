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


@pytest.fixture(autouse=True)
def _isolated_registry_cache(tmp_path, monkeypatch):
    """Point the shared ``outdated.json`` registry cache at a throwaway dir.

    ``--registry`` reuses ``comfy outdated``'s cache file, which is resolved
    from ``XDG_CACHE_HOME`` at call time — without this, a test run would read
    and rewrite the developer's real cache.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


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


# ---------------------------------------------------------------------------
# --registry (not-yet-installed candidates)
# ---------------------------------------------------------------------------


def _node(node_id: str, version: str | None = "1.2.3", dependencies=None):
    """A registry ``Node`` as ``RegistryAPI.get_node`` would return one."""
    from comfy_cli.registry.types import Node, NodeVersion

    latest = None
    if version is not None:
        latest = NodeVersion(
            changelog="",
            # ``map_node_version`` defaults a missing key to [], so [] is what
            # "the registry published no dependency metadata" looks like here.
            dependencies=[] if dependencies is None else dependencies,
            deprecated=False,
            id=f"{node_id}@{version}",
            version=version,
            download_url="https://example.invalid/pack.zip",
        )
    return Node(id=node_id, name=node_id, description="", latest_version=latest)


class _FakeRegistry:
    """Records every ``get_node`` call; ``install_node`` must never be reached."""

    def __init__(self, node=None, error: Exception | None = None):
        self.calls: list[str] = []
        self._node = node
        self._error = error

    def get_node(self, node_id):
        self.calls.append(node_id)
        if self._error is not None:
            raise self._error
        return self._node

    def install_node(self, node_id, version=None):  # pragma: no cover - a failure, not a path
        raise AssertionError("install_node must never be called: it records an install + analytics event")


def _registry_row(report: dict) -> dict:
    rows = [p for p in report["packs"] if p.get("registry")]
    assert len(rows) == 1, f"expected exactly one registry row, got {[p['pack'] for p in report['packs']]}"
    return rows[0]


def test_registry_candidate_is_diffed_against_the_same_pip_list(workspace, fake_pip):
    api = _FakeRegistry(
        _node(
            "comfyui-impact-pack",
            "8.28.3",
            # satisfied / mismatch / missing / unparseable, in that order.
            ["numpy>=1.20", "opencv-python<4.0", "segment-anything", "-r extra.txt"],
        )
    )
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["comfyui-impact-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    assert api.calls == ["comfyui-impact-pack"]
    assert row["pack"] == "comfyui-impact-pack"
    assert row["status"] == "registry"
    assert row["path"] is None
    assert row["version"] == "8.28.3"
    assert row["declared"] == ["numpy>=1.20", "opencv-python<4.0", "segment-anything", "-r extra.txt"]
    assert row["requirement_files"] == ["registry"]
    assert "warning" not in row
    assert warnings == []

    reqs = _reqs_by_raw(row)
    assert reqs["numpy>=1.20"]["status"] == "satisfied"
    assert reqs["numpy>=1.20"]["installed"] == "1.26.4"
    assert reqs["opencv-python<4.0"]["status"] == "mismatch"
    assert reqs["opencv-python<4.0"]["installed"] == "4.9.0.80"
    assert reqs["segment-anything"]["status"] == "missing"
    assert reqs["-r extra.txt"]["status"] == "unparseable"
    assert all(r["source"] == "registry" for r in row["requirements"])
    assert row["summary"] == {"satisfied": 1, "mismatch": 1, "missing": 1, "unparseable": 1, "unknown": 0}

    # A single `pip list`, shared with the installed packs — not one per entry.
    assert len(fake_pip.calls) == 1


def test_registry_without_dependency_metadata_warns_honestly(workspace, fake_pip):
    api = _FakeRegistry(_node("bare-pack", "1.0.0", []))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["bare-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    # `declared: null`, NOT `[]`: the API cannot distinguish "declares nothing"
    # from "field absent", so we must not claim the former.
    assert row["declared"] is None
    assert row["registry"] is True
    assert row["requirements"] == []
    assert row["requirement_files"] == []
    assert row["warning"] == "registry did not return dependency metadata for this pack"
    assert [w["code"] for w in warnings] == ["registry_no_dependency_metadata"]


def test_registry_node_without_latest_version_warns_honestly(workspace, fake_pip):
    api = _FakeRegistry(_node("unpublished-pack", version=None))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["unpublished-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    assert row["version"] is None
    assert row["declared"] is None
    assert row["warning"] == "registry did not return dependency metadata for this pack"
    assert [w["code"] for w in warnings] == ["registry_no_dependency_metadata"]


def test_registry_non_list_dependencies_do_not_become_one_row_per_character(workspace, fake_pip):
    """A malformed payload must degrade to the honest warning, not to garbage."""
    api = _FakeRegistry(_node("odd-pack", "1.0.0", "numpy"))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["odd-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    assert row["declared"] is None
    assert row["requirements"] == []
    assert row["warning"] == "registry did not return dependency metadata for this pack"
    assert [w["code"] for w in warnings] == ["registry_no_dependency_metadata"]


def test_registry_blank_only_dependencies_degrade_to_the_warning(workspace, fake_pip):
    api = _FakeRegistry(_node("blank-pack", "1.0.0", ["", "   "]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["blank-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    assert row["declared"] is None
    assert [w["code"] for w in warnings] == ["registry_no_dependency_metadata"]


def test_registry_network_failure_is_a_per_entry_warning_not_a_failed_command(workspace, fake_pip):
    import requests

    api = _FakeRegistry(error=requests.exceptions.RequestException("connection timed out"))
    report, warnings = node_deps_cmd.build_report(
        str(workspace),
        ["comfyui-impact-pack"],
        registry_ids=["some-pack"],
        python="/fake/python",
        registry_api=api,
    )

    row = _registry_row(report)
    assert row["declared"] is None
    assert row["version"] is None
    assert "connection timed out" in row["warning"]
    assert [w["code"] for w in warnings] == ["registry_unavailable"]

    # The installed pack's section is untouched by the registry failure.
    installed = _packs_by_name(report)["comfyui-impact-pack"]
    assert installed["status"] == "installed"
    assert _reqs_by_raw(installed)["numpy<1.20"]["status"] == "mismatch"


def test_registry_never_calls_install_node(workspace, fake_pip, monkeypatch):
    """The install endpoint records an install + analytics event server-side."""
    from unittest import mock

    from comfy_cli.registry import RegistryAPI

    with (
        mock.patch.object(RegistryAPI, "install_node", autospec=True) as install,
        mock.patch.object(RegistryAPI, "get_node", autospec=True) as get_node,
    ):
        get_node.return_value = _node("some-pack", "2.0.0", ["numpy"])
        # No injected api: build_report must construct the real RegistryAPI and
        # still only reach for the read-only endpoint.
        report, warnings = node_deps_cmd.build_report(str(workspace), registry_ids=["some-pack"], python="/fake/python")

    install.assert_not_called()
    assert get_node.call_count == 1
    assert _registry_row(report)["declared"] == ["numpy"]
    assert warnings == []


def test_registry_lookup_is_cached_and_refresh_bypasses_it(workspace, fake_pip):
    api = _FakeRegistry(_node("cached-pack", "3.0.0", ["numpy"]))

    for _ in range(2):
        report, _ = node_deps_cmd.build_report(
            str(workspace), registry_ids=["cached-pack"], python="/fake/python", registry_api=api
        )
        assert _registry_row(report)["version"] == "3.0.0"
    assert api.calls == ["cached-pack"], "second lookup must be served from the 1h cache"

    node_deps_cmd.build_report(
        str(workspace), registry_ids=["cached-pack"], python="/fake/python", registry_api=api, refresh=True
    )
    assert api.calls == ["cached-pack", "cached-pack"], "--refresh must bypass the cache"


def test_registry_failure_is_not_cached(workspace, fake_pip):
    """A transient outage must not poison the next hour of lookups."""
    import requests

    api = _FakeRegistry(error=requests.exceptions.RequestException("boom"))
    for _ in range(2):
        node_deps_cmd.build_report(str(workspace), registry_ids=["flaky-pack"], python="/fake/python", registry_api=api)
    assert api.calls == ["flaky-pack", "flaky-pack"]


def test_registry_only_invocation_does_not_dump_every_installed_pack(workspace, fake_pip):
    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    report, _ = node_deps_cmd.build_report(
        str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api
    )

    assert [p["pack"] for p in report["packs"]] == ["some-pack"]


def test_registry_is_additive_with_positional_pack_names(workspace, fake_pip):
    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    report, _ = node_deps_cmd.build_report(
        str(workspace),
        ["Registry-Pack"],
        registry_ids=["some-pack"],
        python="/fake/python",
        registry_api=api,
    )

    assert [p["pack"] for p in report["packs"]] == ["Registry-Pack", "some-pack"]
    assert report["packs"][0]["status"] == "installed"


def test_registry_ids_are_deduped_and_blanks_dropped(workspace, fake_pip):
    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace),
        registry_ids=["some-pack", "  some-pack ", "", "   "],
        python="/fake/python",
        registry_api=api,
    )

    assert [p["pack"] for p in report["packs"]] == ["some-pack"]
    assert api.calls == ["some-pack"]
    # The blanks are reported, not silently swallowed.
    assert [w["code"] for w in warnings] == ["registry_invalid_node_id"] * 2


def test_registry_ids_dedupe_case_insensitively(workspace, fake_pip):
    """The registry resolves ids case-insensitively (`GET /nodes/comfyui-lcm`
    302s to `/nodes/ComfyUI-LCM`), so two spellings are one pack — not two
    network calls, two cache entries and two rows.
    """
    api = _FakeRegistry(_node("Some-Pack", "1.0.0", ["numpy"]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace),
        registry_ids=["Some-Pack", "some-pack", "SOME-PACK"],
        python="/fake/python",
        registry_api=api,
    )

    assert [p["pack"] for p in report["packs"]] == ["Some-Pack"], "first spelling seen wins"
    assert api.calls == ["Some-Pack"]
    assert warnings == []


@pytest.mark.parametrize(
    "bad_id",
    [
        "some-pack/install",  # the side-effecting install endpoint's own URL
        "../nodes/other",
        "some-pack?version=1",
        "some-pack#frag",
        "some pack",
        "https://api.comfy.org/nodes/some-pack",
    ],
)
def test_registry_id_outside_the_safe_charset_is_rejected_without_a_request(workspace, fake_pip, bad_id):
    """`get_node` interpolates the id straight into `GET {base}/nodes/{id}`, so a
    `/` retargets the request — `<pack>/install` builds exactly `install_node`'s
    URL, the endpoint this feature promises never to touch. Reject before the call.
    """
    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=[bad_id], python="/fake/python", registry_api=api
    )

    assert api.calls == [], "an invalid id must never reach the network"
    assert report["packs"] == []
    assert [w["code"] for w in warnings] == ["registry_invalid_node_id"]


def test_all_registry_ids_rejected_does_not_dump_every_installed_pack(workspace, fake_pip):
    """A targeted pre-install question whose ids were all rejected must answer
    with the warning, not silently fall back to the whole-workspace report.
    """
    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["", "bad/id"], python="/fake/python", registry_api=api
    )

    assert report["packs"] == []
    assert [w["code"] for w in warnings] == ["registry_invalid_node_id"] * 2


def test_registry_404_is_distinct_from_a_transient_outage(workspace, fake_pip):
    """`registry_unavailable`'s hint says "check the network, retry with
    --refresh" — an agent following that against a 404 retries forever.
    """
    from comfy_cli.registry import NodeFetchError

    api = _FakeRegistry(error=NodeFetchError("Failed to retrieve node: 404 - Node not found", status_code=404))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["no-such-pack"], python="/fake/python", registry_api=api
    )

    assert [w["code"] for w in warnings] == ["registry_node_not_found"]
    assert "404" in _registry_row(report)["warning"]


def test_registry_5xx_stays_registry_unavailable(workspace, fake_pip):
    from comfy_cli.registry import NodeFetchError

    api = _FakeRegistry(error=NodeFetchError("Failed to retrieve node: 503 - upstream down", status_code=503))
    _, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api
    )

    assert [w["code"] for w in warnings] == ["registry_unavailable"]


def test_registry_error_text_is_truncated_before_entering_the_envelope(workspace, fake_pip):
    """A captive portal answers with a whole HTML page; it must not be copied
    verbatim into the single-line JSON envelope consumers parse.
    """
    api = _FakeRegistry(error=Exception("Failed to retrieve node: 502 - " + ("<html>padding</html>" * 500)))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api
    )

    message = _registry_row(report)["warning"]
    assert len(message) < node_deps_cmd.MAX_REGISTRY_ERROR_CHARS + 120
    assert message.endswith("… (truncated)")
    assert warnings[0]["message"] == message


def test_registry_partial_dependency_list_is_flagged_not_silently_shortened(workspace, fake_pip):
    """`["numpy", null]` must not render as complete metadata containing only
    numpy — the dropped entry could have been the conflicting one.
    """
    api = _FakeRegistry(_node("mixed-pack", "1.0.0", ["numpy>=1.20", None, 123]))
    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["mixed-pack"], python="/fake/python", registry_api=api
    )

    row = _registry_row(report)
    assert row["declared"] == ["numpy>=1.20"]
    assert [w["code"] for w in warnings] == ["registry_partial_dependency_metadata"]
    assert "incomplete" in row["warning"]


def test_malformed_cache_entry_does_not_abort_the_report(workspace, fake_pip):
    """A cache entry written by another comfy-cli version (or hand-edited) can
    hold `[null]`; reaching `_classify` it would raise AttributeError and kill
    the whole report — the opposite of this module's degrade-to-a-warning design.
    """
    from comfy_cli.command import outdated as outdated_cmd

    api = _FakeRegistry(_node("cached-pack", "1.0.0", ["numpy"]))
    cache = outdated_cmd._load_cache()
    key = node_deps_cmd._registry_cache_key(api, "cached-pack")
    outdated_cmd._cache_set(cache, key, {"version": "1.0.0", "dependencies": [None, 123, "numpy>=1.20"]})
    outdated_cmd._save_cache(cache)

    report, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["cached-pack"], python="/fake/python", registry_api=api
    )

    assert api.calls == [], "served from cache"
    assert _registry_row(report)["declared"] == ["numpy>=1.20"]
    assert warnings == []


def test_registry_cache_key_is_scoped_to_the_registry_base_url(workspace, fake_pip):
    """Staging metadata must not be served to a later production run, where it
    could hide a real dependency conflict.
    """

    class _Scoped(_FakeRegistry):
        def __init__(self, base_url, node):
            super().__init__(node)
            self.base_url = base_url

    staging = _Scoped("https://stagingapi.comfy.org", _node("some-pack", "9.9.9-staging", ["numpy"]))
    production = _Scoped("https://api.comfy.org", _node("some-pack", "1.0.0", ["numpy"]))

    node_deps_cmd.build_report(str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=staging)
    report, _ = node_deps_cmd.build_report(
        str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=production
    )

    assert production.calls == ["some-pack"], "the staging entry must not satisfy a production lookup"
    assert _registry_row(report)["version"] == "1.0.0"


def test_saving_the_cache_preserves_a_concurrent_writers_keys(workspace, fake_pip):
    """`node deps` is a second writer of the file `comfy outdated` owns: it must
    merge into a freshly re-read dict, not blind-write the one it loaded.
    """
    from comfy_cli.command import outdated as outdated_cmd

    class _RacingRegistry(_FakeRegistry):
        """Writes a rival key *after* build_report has loaded the cache."""

        def get_node(self, node_id):
            rival = outdated_cmd._load_cache()
            outdated_cmd._cache_set(rival, "pack:written-by-outdated", "2.0.0")
            outdated_cmd._save_cache(rival)
            return super().get_node(node_id)

    api = _RacingRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    node_deps_cmd.build_report(str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api)

    saved = outdated_cmd._load_cache()
    assert outdated_cmd._cache_get(saved, "pack:written-by-outdated") == "2.0.0"
    assert node_deps_cmd._registry_cache_key(api, "some-pack") in saved


def test_expired_registry_cache_entries_are_pruned_on_save(workspace, fake_pip):
    """The key space is arbitrary caller-supplied ids holding whole dependency
    lists, and `_save_cache` rewrites every key — without pruning it grows without bound.
    """
    import time

    from comfy_cli.command import outdated as outdated_cmd

    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy"]))
    cache = outdated_cmd._load_cache()
    stale_key = f"{node_deps_cmd.REGISTRY_CACHE_PREFIX}https://api.comfy.org:long-gone-pack"
    cache[stale_key] = {"value": {"version": "1", "dependencies": ["numpy"]}, "ts": time.time() - 999_999}
    # A *fresh* registry entry and `outdated`'s own expired keys are both left alone.
    fresh_key = f"{node_deps_cmd.REGISTRY_CACHE_PREFIX}https://api.comfy.org:still-fresh"
    outdated_cmd._cache_set(cache, fresh_key, {"version": "1", "dependencies": ["numpy"]})
    cache["pack:someone-elses-expired"] = {"value": "1.0.0", "ts": time.time() - 999_999}
    outdated_cmd._save_cache(cache)

    node_deps_cmd.build_report(str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api)

    saved = outdated_cmd._load_cache()
    assert stale_key not in saved
    assert fresh_key in saved
    assert "pack:someone-elses-expired" in saved, "pruning is scoped to this command's own key space"


def test_registry_rows_validate_against_the_shipped_schema(workspace, fake_pip):
    import jsonschema

    from comfy_cli import discovery

    schema = discovery._read_schema("node_deps")

    api = _FakeRegistry(_node("some-pack", "1.0.0", ["numpy>=1.20", "-r extra.txt"]))
    report, _ = node_deps_cmd.build_report(
        str(workspace), ["Registry-Pack"], registry_ids=["some-pack"], python="/fake/python", registry_api=api
    )
    report["warnings"] = []
    jsonschema.validate(report, schema)

    # the no-metadata / unreachable shapes must validate too
    bare, _ = node_deps_cmd.build_report(
        str(workspace), registry_ids=["bare"], python="/fake/python", registry_api=_FakeRegistry(_node("bare", "1.0"))
    )
    bare["warnings"] = []
    jsonschema.validate(bare, schema)


def test_registry_pretty_mode_renders_version_and_status(workspace, fake_pip, capsys):
    from unittest import mock

    from comfy_cli.registry import RegistryAPI

    reset_renderer_for_testing()
    r = Renderer(mode=OutputMode.PRETTY)
    set_renderer(r)

    with mock.patch.object(RegistryAPI, "get_node", autospec=True) as get_node:
        get_node.return_value = _node("some-pack", "9.9.9", ["numpy>=1.20"])
        node_deps_cmd.execute(r, str(workspace), registry_ids=["some-pack"])

    out = capsys.readouterr().out
    assert "some-pack" in out
    assert "9.9.9" in out
    assert "satisfied" in out


def test_registry_pretty_mode_renders_the_warning_row(workspace, fake_pip, capsys):
    from unittest import mock

    from comfy_cli.registry import RegistryAPI

    reset_renderer_for_testing()
    r = Renderer(mode=OutputMode.PRETTY)
    set_renderer(r)

    with mock.patch.object(RegistryAPI, "get_node", autospec=True) as get_node:
        get_node.return_value = _node("bare-pack", "1.0.0", [])
        node_deps_cmd.execute(r, str(workspace), registry_ids=["bare-pack"])

    out = capsys.readouterr().out
    assert "bare-pack" in out
    assert "did not return dependency metadata" in out
    assert r.exit_code == 0  # a report, not a check


def test_registry_flags_are_registered_on_the_deps_command():
    import click
    import typer

    from comfy_cli.command.custom_nodes.command import app

    command = next(c for c in app.registered_commands if c.name == "deps")
    click_command = typer.main.get_command_from_info(command, pretty_exceptions_short=False, rich_markup_mode="rich")
    opts = {opt for p in click_command.params if isinstance(p, click.Option) for opt in p.opts}
    assert {"--registry", "--refresh"} <= opts


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


def test_a_failed_refresh_does_not_resurrect_an_expired_cache_entry(workspace, fake_pip):
    """The failed lookup writes nothing, so the stale entry it could not replace
    must still be pruned rather than carried forward as a fresh write.
    """
    import time

    import requests

    from comfy_cli.command import outdated as outdated_cmd

    api = _FakeRegistry(error=requests.exceptions.RequestException("boom"))
    key = node_deps_cmd._registry_cache_key(api, "some-pack")
    cache = outdated_cmd._load_cache()
    cache[key] = {"value": {"version": "1", "dependencies": ["numpy"]}, "ts": time.time() - 999_999}
    outdated_cmd._save_cache(cache)

    _, warnings = node_deps_cmd.build_report(
        str(workspace), registry_ids=["some-pack"], python="/fake/python", registry_api=api
    )

    assert [w["code"] for w in warnings] == ["registry_unavailable"]
    assert key not in outdated_cmd._load_cache()
