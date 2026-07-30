"""Tests for ``comfy outdated`` — installed-vs-latest reporting.

Builds a real fixture workspace (a git-checkout core + one registry pack +
one git pack) under tmp_path and mocks the GitHub/registry HTTP so nothing
touches the network. Covers the outdated verdicts, the JSON envelope shape,
the 1h cache (warm-cache serves latest even when the network is down), and the
network-failure path (``latest: null`` + exit 0).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from comfy_cli.caller import Caller
from comfy_cli.command import outdated as outdated_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(cwd: Path, filename: str, content: str, message: str) -> None:
    (cwd / filename).write_text(content)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-m", message)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Stand-in for RegistryAPI: get_node(id) -> object with .latest_version.version."""

    def __init__(self, versions: dict[str, str], raises: bool = False):
        self._versions = versions
        self._raises = raises

    def get_node(self, node_id: str):
        if self._raises:
            raise RuntimeError("registry unreachable")

        class _NV:
            pass

        class _Node:
            pass

        nv = _NV()
        nv.version = self._versions.get(node_id, "0.0.0")
        node = _Node()
        node.latest_version = nv
        return node


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Pin the outdated cache under tmp so tests don't share disk state."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def workspace(tmp_path) -> Path:
    """A fixture ComfyUI workspace: git core + registry pack + git pack."""
    ws = tmp_path / "ComfyUI"
    ws.mkdir()

    # --- core: git checkout on tag v0.3.40 ---
    _git(ws, "init")
    _commit(ws, "main.py", "# comfy\n", "initial")
    _git(ws, "tag", "v0.3.40")

    custom_nodes = ws / "custom_nodes"
    custom_nodes.mkdir()

    # --- registry pack: pyproject with a node id + version ---
    reg = custom_nodes / "comfy-registry-pack"
    reg.mkdir()
    (reg / "pyproject.toml").write_text('[project]\nname = "comfy-registry-pack"\nversion = "1.0.0"\n')

    # --- git pack: local HEAD behind its remote HEAD ---
    # Pin the bare repo's default branch to ``main`` so its HEAD symref resolves
    # to the branch the seed pushes below. Without this, a runner whose
    # ``init.defaultBranch`` is ``master`` (the git default) leaves the bare
    # repo's HEAD dangling, so ``git ls-remote <remote> HEAD`` returns nothing
    # and the git-pack latest lookup silently fails.
    remote = tmp_path / "gitpack-remote.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "gitpack-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _commit(seed, "node.py", "v1\n", "pack v1")
    _git(seed, "push", "origin", "HEAD:refs/heads/main")

    gitpack = custom_nodes / "git-pack"
    _git(tmp_path, "clone", str(remote), str(gitpack))
    # advance the remote so the pack's local HEAD is now stale
    _commit(seed, "node.py", "v2\n", "pack v2")
    _git(seed, "push", "origin", "HEAD:refs/heads/main")

    return ws


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


def _packs_by_name(report: dict) -> dict[str, dict]:
    return {p["name"]: p for p in report["packs"]}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_stale_install_flags_core_and_packs(workspace, monkeypatch):
    """Acceptance: a stale install reports core.outdated true + pack rows."""

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    registry = _FakeRegistry({"comfy-registry-pack": "1.2.0"})

    report, warnings = outdated_cmd.build_report(str(workspace), registry_api=registry)

    assert report["core"]["installed"] == "v0.3.40"
    assert report["core"]["latest"] == "v0.3.41"
    assert report["core"]["outdated"] is True
    assert report["core"]["commit"]  # short HEAD present for a git checkout

    packs = _packs_by_name(report)
    assert packs["comfy-registry-pack"]["source"] == "registry"
    assert packs["comfy-registry-pack"]["installed"] == "1.0.0"
    assert packs["comfy-registry-pack"]["latest"] == "1.2.0"
    assert packs["comfy-registry-pack"]["outdated"] is True

    assert packs["git-pack"]["source"] == "git"
    assert packs["git-pack"]["outdated"] is True
    assert packs["git-pack"]["installed"] != packs["git-pack"]["latest"]

    assert "checked_at" in report
    assert not warnings


def test_up_to_date_install_not_flagged(workspace, monkeypatch):
    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.40"},
    )
    registry = _FakeRegistry({"comfy-registry-pack": "1.0.0"})

    report, _ = outdated_cmd.build_report(str(workspace), registry_api=registry)

    assert report["core"]["outdated"] is False
    assert _packs_by_name(report)["comfy-registry-pack"]["outdated"] is False


def test_network_failure_yields_null_latest_and_no_warnings_crash(workspace, monkeypatch):
    """GitHub + registry down → latest: null, outdated: false, warnings, no raise."""

    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr("comfy_cli.command.install.get_latest_release", _boom)
    registry = _FakeRegistry({}, raises=True)

    report, warnings = outdated_cmd.build_report(str(workspace), registry_api=registry)

    assert report["core"]["latest"] is None
    assert report["core"]["outdated"] is False
    packs = _packs_by_name(report)
    assert packs["comfy-registry-pack"]["latest"] is None
    assert packs["comfy-registry-pack"]["outdated"] is False
    assert warnings  # each failed lookup surfaced a warning


def test_warm_cache_serves_latest_without_network(workspace, monkeypatch):
    """First call populates the 1h cache; second call serves it even offline."""

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    registry = _FakeRegistry({"comfy-registry-pack": "1.2.0"})

    outdated_cmd.build_report(str(workspace), registry_api=registry)  # warms cache

    # Now sever the network; the warm cache must still yield the same verdicts.
    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr("comfy_cli.command.install.get_latest_release", _boom)
    report, warnings = outdated_cmd.build_report(str(workspace), registry_api=_FakeRegistry({}, raises=True))

    assert report["core"]["latest"] == "v0.3.41"
    assert report["core"]["outdated"] is True
    assert _packs_by_name(report)["comfy-registry-pack"]["latest"] == "1.2.0"
    assert not warnings


def test_refresh_bypasses_cache(workspace, monkeypatch):
    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    outdated_cmd.build_report(str(workspace), registry_api=_FakeRegistry({"comfy-registry-pack": "1.2.0"}))

    # --refresh must re-query even with a warm cache; a down network → null.
    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr("comfy_cli.command.install.get_latest_release", _boom)
    report, warnings = outdated_cmd.build_report(
        str(workspace), refresh=True, registry_api=_FakeRegistry({}, raises=True)
    )

    assert report["core"]["latest"] is None
    assert warnings


def test_is_outdated_incomparable_sha_vs_version_not_flagged():
    """A bare SHA (shallow/no-tag checkout) vs a version tag is incomparable."""
    # both versions → semantic
    assert outdated_cmd._is_outdated("v0.3.40", "v0.3.41") is True
    assert outdated_cmd._is_outdated("v0.3.41", "v0.3.41") is False
    # checkout ahead of the latest tag → not flagged
    assert outdated_cmd._is_outdated("v0.3.41-5-gabc1234", "v0.3.41") is False
    # both opaque (git SHAs) → any difference is "behind"
    assert outdated_cmd._is_outdated("abc1234", "def5678") is True
    assert outdated_cmd._is_outdated("abc1234", "abc1234") is False
    # exactly one parses → incomparable, must NOT false-positive
    assert outdated_cmd._is_outdated("abc1234def", "v0.3.41") is False
    assert outdated_cmd._is_outdated("v0.3.41", "abc1234def") is False


def test_shallow_core_checkout_not_falsely_flagged(tmp_path, monkeypatch):
    """A no-tag core checkout resolves to a SHA; don't flag it vs a version."""
    ws = tmp_path / "ComfyUI"
    ws.mkdir()
    _git(ws, "init")
    _commit(ws, "main.py", "# comfy\n", "initial")  # no tag → describe --always = SHA
    (ws / "custom_nodes").mkdir()

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    report, _ = outdated_cmd.build_report(str(ws), registry_api=_FakeRegistry({}))
    assert report["core"]["latest"] == "v0.3.41"
    assert report["core"]["installed"]  # a SHA
    assert report["core"]["outdated"] is False  # SHA vs version is incomparable


def test_missing_workspace_warns_not_crashes(tmp_path):
    report, warnings = outdated_cmd.build_report(str(tmp_path / "nope"))
    assert report["core"]["installed"] is None
    assert report["packs"] == []
    assert warnings


def test_git_pack_with_pyproject_falls_back_to_git_when_unregistered(tmp_path, monkeypatch):
    """A git-installed pack shipping a pyproject that the registry doesn't know
    must fall back to the git HEAD comparison, not report unknown/not-outdated."""
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _git(ws, "init")
    _commit(ws, "main.py", "# comfy\n", "initial")

    # git pack whose local HEAD is behind its remote HEAD, but which also ships
    # a pyproject (so the registry branch is entered first).
    remote = tmp_path / "gp-remote.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "gp-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _commit(seed, "node.py", "v1\n", "pack v1")
    _git(seed, "push", "origin", "HEAD:refs/heads/main")

    pack = ws / "custom_nodes" / "hybrid-pack"
    _git(tmp_path, "clone", str(remote), str(pack))
    (pack / "pyproject.toml").write_text('[project]\nname = "hybrid-pack"\nversion = "1.0.0"\n')
    _commit(seed, "node.py", "v2\n", "pack v2")
    _git(seed, "push", "origin", "HEAD:refs/heads/main")

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.40"},
    )
    # Registry doesn't know this pack (404 → get_node raises) → latest None
    # → must fall back to git.
    report, _ = outdated_cmd.build_report(str(ws), registry_api=_FakeRegistry({}, raises=True))

    pack_row = _packs_by_name(report)["hybrid-pack"]
    assert pack_row["source"] == "git"
    assert pack_row["installed"] != pack_row["latest"]
    assert pack_row["outdated"] is True


def test_git_pack_without_origin_warns(tmp_path, monkeypatch):
    """A git pack with no ``origin`` remote surfaces an explicit warning."""
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _git(ws, "init")
    _commit(ws, "main.py", "# comfy\n", "initial")

    pack = ws / "custom_nodes" / "orphan-pack"
    pack.mkdir()
    _git(pack, "init")
    _commit(pack, "node.py", "v1\n", "pack v1")  # no origin remote configured

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.40"},
    )
    report, warnings = outdated_cmd.build_report(str(ws), registry_api=_FakeRegistry({}))

    assert _packs_by_name(report)["orphan-pack"]["latest"] is None
    assert any("no origin remote" in w for w in warnings)


def test_git_pack_rejects_option_injecting_remote(tmp_path, monkeypatch):
    """A malicious ``origin`` URL (``ext::``/``-``-prefixed) must not execute:
    the transport allowlist + ``--`` separator degrade it to latest: null."""
    ws = tmp_path / "ComfyUI"
    (ws / "custom_nodes").mkdir(parents=True)
    _git(ws, "init")
    _commit(ws, "main.py", "# comfy\n", "initial")

    pack = ws / "custom_nodes" / "evil-pack"
    pack.mkdir()
    _git(pack, "init")
    _commit(pack, "node.py", "v1\n", "pack v1")
    canary = tmp_path / "pwned"
    _git(pack, "remote", "add", "origin", f"ext::sh -c 'touch {canary}; true'")

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.40"},
    )
    report, warnings = outdated_cmd.build_report(str(ws), registry_api=_FakeRegistry({}))

    assert not canary.exists(), "ext:: transport executed — RCE not blocked"
    assert _packs_by_name(report)["evil-pack"]["latest"] is None
    assert any("could not reach git remote" in w for w in warnings)


def test_report_validates_against_shipped_schema(workspace, monkeypatch):
    """The report payload must satisfy comfy_cli/schemas/outdated.json."""
    import jsonschema

    from comfy_cli import discovery

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    schema = discovery._read_schema("outdated")

    # both a live report and a network-down report must validate
    report_ok, _ = outdated_cmd.build_report(
        str(workspace), registry_api=_FakeRegistry({"comfy-registry-pack": "1.2.0"})
    )
    jsonschema.validate(report_ok, schema)

    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    report_down, _ = outdated_cmd.build_report(
        str(workspace), refresh=True, registry_api=_FakeRegistry({}, raises=True)
    )
    jsonschema.validate(report_down, schema)


def test_cli_json_envelope_shape(workspace, monkeypatch, capsys):
    monkeypatch.setattr(
        "comfy_cli.command.install.get_latest_release",
        lambda *a, **k: {"tag": "v0.3.41"},
    )
    monkeypatch.setattr(
        outdated_cmd,
        "RegistryAPI",
        lambda: _FakeRegistry({"comfy-registry-pack": "1.2.0"}),
    )

    r = _force_json_renderer()
    outdated_cmd.execute(r, str(workspace))

    out = capsys.readouterr().out.strip()
    # Envelope contract: stdout must be exactly ONE line (the JSON envelope).
    # The registry pack's pyproject has no license, which makes the shared
    # parser warn — that side-message must land on stderr, not stdout.
    assert len(out.splitlines()) == 1, f"stdout not a single envelope line:\n{out}"
    envelope = json.loads(out)
    assert envelope["ok"] is True
    assert envelope["command"] == "outdated"
    assert envelope["data"]["core"]["outdated"] is True
    assert isinstance(envelope["data"]["packs"], list)
