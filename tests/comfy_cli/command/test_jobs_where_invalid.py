"""``comfy jobs`` renders a ``where_invalid`` envelope for a bad routing value.

The sibling of :mod:`tests.comfy_cli.command.test_nodes_where_invalid`, for the
other recover-first command. ``jobs._is_cloud`` used to swallow the ``ValueError``
and ``return False`` on the assumption that ``cmdline.py``'s top-level ``--where``
had already rejected a bad value. That only holds for ``comfy --where X jobs ls``:
a **per-command** ``comfy jobs ls --where bogus``, an exported ``COMFY_WHERE``, or a
stale ``defaults.where`` reaches no such validator, so the verb silently routed to
local and exited **0 with ``ok: true``** — a machine consumer got a successful
*local* answer to a question about a target it never named, which is worse than the
traceback the ``nodes`` half of this change fixed.

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
from comfy_cli.command import jobs as jobs_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

# Every verb that stamps routing at entry, i.e. every verb routed through
# ``_stamp_where`` -> ``_is_cloud``. Each calls it as its first statement, so a
# bad value exits before any host/port resolution or network call — that is what
# makes these safe to run offline.
ROUTED_VERBS: list[list[str]] = [
    ["ls"],
    ["status", "abc123"],
    ["wait", "abc123"],
    ["cancel", "abc123"],
    ["watch", "abc123"],
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
    """Run a ``jobs`` verb and return ``(exit_code, stdout, stderr)``.

    ``standalone_mode`` is left ON: the whole claim here is about the process
    exit code, and only click's own standalone handling turns ``typer.Exit``
    into one.
    """
    _force_json_renderer()
    result = CliRunner().invoke(jobs_cmd.app, args)
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
    # The regression this file exists for: the old code exited 0 with a *local*
    # success envelope, so a consumer keying on `ok` never learned it was
    # answered about the wrong target.
    for stream in (out, err):
        assert "Traceback" not in stream
        assert "ValueError" not in stream


class TestBadFlag:
    @pytest.mark.parametrize("verb", ROUTED_VERBS, ids=lambda v: "-".join(v))
    def test_bad_where_flag_renders_envelope(self, verb, capsys):
        """Every routed verb, not just ``ls`` — they share one resolver."""
        code, out, err = _invoke([*verb, "--where", "bogus"], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_bad_where_flag_wins_over_a_valid_config(self, monkeypatch, capsys):
        """A *valid* persisted default must not paper over an explicit bad flag."""
        _set_persisted_where_default(monkeypatch, "local")
        code, out, err = _invoke(["ls", "--where", "bogus"], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_hint_is_the_shared_constant(self, capsys):
        """Pin the drift the module constant exists to prevent: this envelope,
        ``nodes``', and ``resolve_default_or_exit``'s must stay byte-identical."""
        _, out, _err = _invoke(["ls", "--where", "bogus"], capsys)
        assert _sole_envelope(out)["error"]["hint"] == where_module.WHERE_INVALID_HINT


class TestBadEnvAndProject:
    def test_bad_comfy_where_env_renders_envelope(self, monkeypatch, capsys):
        """``COMFY_WHERE`` is also how the top-level ``comfy --where`` reaches a
        subcommand, so this is the path an exported bad value takes."""
        monkeypatch.setenv(where_module.ENV_DEFAULT, "bogus")
        code, out, err = _invoke(["ls"], capsys)
        _assert_where_invalid(code, out, err, "bogus")

    def test_bad_project_defaults_where_renders_envelope(self, monkeypatch, capsys):
        """``defaults.where`` in the governing ``comfy.yaml`` is re-read by the
        fallback too, so it is just as unrecoverable as the flag."""
        from pathlib import Path

        from comfy_cli.project import Project

        project = Project(root=Path("/nonexistent"), config={"schema": "project/1", "defaults": {"where": "bogus"}})
        monkeypatch.setattr("comfy_cli.project.find_project", lambda *a, **kw: project)
        code, out, err = _invoke(["ls"], capsys)
        _assert_where_invalid(code, out, err, "bogus")


class TestConfigRecoveryStillWorks:
    """Regression control for the branch this change must NOT break: a corrupt
    persisted ``where_default`` is still non-fatal, exactly as before."""

    def test_corrupt_config_with_valid_flag_still_succeeds(self, monkeypatch, capsys):
        _set_persisted_where_default(monkeypatch, "garbage")
        code, out, _err = _invoke(["ls", "--where", "local", "--local-only"], capsys)
        assert code == 0, f"corrupt config recovery regressed (stdout={out!r})"
        env = _sole_envelope(out)
        assert env["ok"] is True
        assert env["where"] == "local"

    def test_corrupt_config_with_no_flag_still_falls_back(self, monkeypatch, capsys):
        """No flag, no env, no project: dropping the bad config leaves the
        auto-detect default, which must still resolve rather than exit."""
        _set_persisted_where_default(monkeypatch, "garbage")
        code, out, _err = _invoke(["ls", "--local-only"], capsys)
        assert code == 0, f"corrupt config recovery regressed (stdout={out!r})"
        env = _sole_envelope(out)
        assert env["ok"] is True
        assert env["where"] == "local"


class TestIsCloudUnit:
    """The resolver itself, without the CLI layer in the way."""

    def test_returns_the_target_for_a_valid_flag(self):
        assert jobs_cmd._is_cloud("cloud") is True
        assert jobs_cmd._is_cloud("local") is False

    def test_raises_typer_exit_for_a_bad_flag(self):
        import typer

        _force_json_renderer()
        with pytest.raises(typer.Exit) as excinfo:
            jobs_cmd._is_cloud("bogus")
        assert excinfo.value.exit_code == 1
