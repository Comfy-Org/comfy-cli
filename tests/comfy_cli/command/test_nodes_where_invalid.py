"""``comfy nodes`` renders a ``where_invalid`` envelope for a bad routing value.

``_resolved_where`` is written to survive a *corrupt persisted config* by
dropping ``where_default`` and re-resolving. That fallback cannot help when the
**flag/env/project** value is the bad one: the retry re-parses the same value and
``where._parse`` raises again. Uncaught, that escaped as a raw Python traceback
with **nothing on stdout** — the worst possible shape for the machine consumers
these verbs exist for, since every other routed command reaches
``where.resolve_default_or_exit`` and gets an envelope.

These tests pin both halves: the unrecoverable sources now emit the shared
envelope, and the config-recovery branch still recovers.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import where as where_module
from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

FIXTURE = "tests/comfy_cli/fixtures/nodes_path_object_info.json"

# Every verb that loads a graph, i.e. every verb routed through
# ``_get_graph`` -> ``_resolved_where``. ``refresh`` is deliberately absent: it
# accepts ``--where`` as a legacy no-op and never resolves routing at all.
GRAPH_VERBS: list[list[str]] = [
    ["ls"],
    ["show", "KSampler"],
    ["search", "KSampler"],
    ["upstream", "KSampler"],
    ["downstream", "KSampler"],
    ["path", "MODEL", "IMAGE"],
    ["types"],
    ["categories"],
    ["widget-catalog"],
]


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture(autouse=True)
def isolated_routing_sources(monkeypatch: pytest.MonkeyPatch):
    """Pin every routing source the tests don't set themselves.

    Without this the assertions would read the developer's real ``COMFY_WHERE``,
    a ``comfy.yaml`` above the checkout, their persisted ``where_default``, and
    their cloud credentials.
    """
    monkeypatch.delenv(where_module.ENV_DEFAULT, raising=False)
    monkeypatch.setattr("comfy_cli.project.find_project", lambda *a, **kw: None)
    monkeypatch.setattr(where_module, "_has_cloud_credentials", lambda: False)
    _set_persisted_where_default(monkeypatch, None)


def _set_persisted_where_default(monkeypatch: pytest.MonkeyPatch, value: str | None):
    """Make ``ConfigManager().get("where_default")`` answer *value*.

    ``ConfigManager`` is ``@singleton``-wrapped, so the module-level name is a
    factory, not the class — patch the real class off the instance it hands back.
    """
    from comfy_cli.config_manager import ConfigManager

    cls = type(ConfigManager())
    real_get = cls.get

    def fake_get(self, key, *a, **kw):
        if key == where_module.CONFIG_KEY_WHERE_DEFAULT:
            return value
        return real_get(self, key, *a, **kw)

    monkeypatch.setattr(cls, "get", fake_get)


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _invoke(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run a ``nodes`` verb and return ``(exit_code, stdout, stderr)``.

    ``standalone_mode`` is left ON — unlike the passthrough tests in
    ``test_nodes_cli.py``, the whole claim here is about the process exit code,
    and only click's own standalone handling turns ``typer.Exit`` into one.
    """
    _force_json_renderer()
    result = CliRunner().invoke(nodes_cmd.app, args)
    captured = capsys.readouterr()
    out = captured.out or result.stdout or ""
    return result.exit_code, out, captured.err


def _sole_envelope(out: str) -> dict[str, Any]:
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one envelope on stdout, got {len(lines)} lines: {lines!r}"
    return json.loads(lines[0])


def _assert_where_invalid(code: int, out: str, err: str, bad_value: str):
    assert code == 1, f"expected exit 1, got {code} (stdout={out!r})"
    env = _sole_envelope(out)
    assert env["ok"] is False
    assert env["error"]["code"] == "where_invalid"
    assert bad_value in env["error"]["message"]
    # The whole point: a machine consumer reads `error.code`, not a stack trace
    # on stderr with an empty stdout.
    for stream in (out, err):
        assert "Traceback" not in stream
        assert "ValueError" not in stream


class TestBadFlag:
    @pytest.mark.parametrize("verb", GRAPH_VERBS, ids=lambda v: "-".join(v))
    def test_bad_where_flag_renders_envelope(self, verb, capsys):
        """Every graph-loading verb, not just ``show`` — they share one resolver."""
        code, out, err = _invoke([*verb, "--where", "bogus", "--input", FIXTURE], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_bad_where_flag_wins_over_a_valid_config(self, monkeypatch, capsys):
        """A *valid* persisted default must not paper over an explicit bad flag."""
        _set_persisted_where_default(monkeypatch, "local")
        code, out, err = _invoke(["show", "KSampler", "--where", "bogus", "--input", FIXTURE], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_hint_is_the_shared_constant(self, capsys):
        """Pin the drift the module constant exists to prevent: this envelope and
        ``resolve_default_or_exit``'s must stay byte-identical."""
        _, out, _err = _invoke(["show", "KSampler", "--where", "bogus", "--input", FIXTURE], capsys)
        assert _sole_envelope(out)["error"]["hint"] == where_module.WHERE_INVALID_HINT


class TestBadEnvAndProject:
    def test_bad_comfy_where_env_renders_envelope(self, monkeypatch, capsys):
        monkeypatch.setenv(where_module.ENV_DEFAULT, "bogus")
        code, out, err = _invoke(["show", "KSampler", "--input", FIXTURE], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_bad_project_defaults_where_renders_envelope(self, monkeypatch, capsys):
        """``defaults.where`` in the governing ``comfy.yaml`` is re-read by the
        fallback too, so it is just as unrecoverable as the flag."""
        from pathlib import Path

        from comfy_cli.project import Project

        project = Project(root=Path("/nonexistent"), config={"schema": "project/1", "defaults": {"where": "bogus"}})
        monkeypatch.setattr("comfy_cli.project.find_project", lambda *a, **kw: project)
        code, out, err = _invoke(["show", "KSampler", "--input", FIXTURE], capsys)
        _assert_where_invalid(code, out, err, "bogus")


class TestConfigRecoveryStillWorks:
    """Regression control for the branch this change must NOT break."""

    def test_corrupt_config_with_valid_flag_still_succeeds(self, monkeypatch, capsys):
        _set_persisted_where_default(monkeypatch, "garbage")
        code, out, err = _invoke(["show", "KSampler", "--where", "local", "--input", FIXTURE], capsys)
        assert code == 0, f"corrupt config recovery regressed (stdout={out!r})"
        env = _sole_envelope(out)
        assert env["ok"] is True
        assert env["where"] == "local"

    def test_corrupt_config_with_no_flag_still_falls_back(self, monkeypatch, capsys):
        """No flag, no env, no project: dropping the bad config leaves the
        auto-detect default, which must still resolve rather than exit."""
        _set_persisted_where_default(monkeypatch, "garbage")
        code, out, err = _invoke(["show", "KSampler", "--input", FIXTURE], capsys)
        assert code == 0, f"corrupt config recovery regressed (stdout={out!r})"
        env = _sole_envelope(out)
        assert env["ok"] is True
        assert env["where"] == "local"


class TestResolvedWhereUnit:
    """The resolver itself, without the CLI layer in the way."""

    def test_returns_the_label_for_a_valid_flag(self):
        assert nodes_cmd._resolved_where("cloud") == "cloud"
        assert nodes_cmd._resolved_where("local") == "local"

    def test_raises_typer_exit_for_a_bad_flag(self):
        import typer

        _force_json_renderer()
        with pytest.raises(typer.Exit) as excinfo:
            nodes_cmd._resolved_where("bogus")
        assert excinfo.value.exit_code == 1
