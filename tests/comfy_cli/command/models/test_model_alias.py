"""`models` (plural) → `model` (singular) merge + deprecation alias.

Asserts the discovery leaves resolve under the singular `model` noun, that the
plural `models` spelling still works as a HIDDEN alias, and that invoking the
alias prints a single yellow deprecation warning to stderr while the canonical
`model` spelling stays warning-free.
"""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app
from comfy_cli.deprecation import add_deprecated_alias, deprecated_help
from comfy_cli.help_json import build_help_json, iter_command_paths
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

_DISCOVERY_LEAVES = ("search", "show", "list-folders", "list-folder")


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


# ---------------------------------------------------------------------------
# Resolution: the discovery leaves live under `model`, and `models` still works.
# ---------------------------------------------------------------------------


class TestResolution:
    def test_discovery_leaves_resolve_under_model(self):
        paths = set(iter_command_paths(app))
        for leaf in _DISCOVERY_LEAVES:
            assert f"comfy model {leaf}" in paths, (leaf, sorted(p for p in paths if "model" in p))

    def test_local_ops_still_under_model(self):
        # The merge must not displace the local-filesystem ops.
        paths = set(iter_command_paths(app))
        for leaf in ("download", "remove", "list"):
            assert f"comfy model {leaf}" in paths, leaf

    def test_plural_alias_still_resolves(self):
        paths = set(iter_command_paths(app))
        for leaf in _DISCOVERY_LEAVES:
            assert f"comfy models {leaf}" in paths, leaf

    def test_models_group_is_hidden_model_is_visible(self):
        commands = build_help_json(app)["commands"]
        assert commands["models"]["hidden"] is True
        assert commands["model"]["hidden"] is False
        # The deprecation banner rides on the hidden group's help.
        assert "DEPRECATED" in (commands["models"]["help"] or "")


# ---------------------------------------------------------------------------
# Helper unit tests — isolated from the real command tree.
# ---------------------------------------------------------------------------


class TestHelper:
    def test_deprecated_help_prefixes_banner(self):
        assert deprecated_help("model", "Discover things.") == "[DEPRECATED — use `comfy model`] Discover things."
        # Empty original help degrades to the bare banner.
        assert deprecated_help("model") == "[DEPRECATED — use `comfy model`]"

    def _build_parent_with_alias(self):
        source = typer.Typer(help="Do a thing.")

        @source.command("go")
        def go():
            typer.echo("ran")

        parent = typer.Typer()
        parent.add_typer(source, name="thing")
        add_deprecated_alias(parent, source, old_name="things", new_name="thing")
        return parent

    def test_alias_invocation_warns_and_runs(self, capsys):
        parent = self._build_parent_with_alias()
        result = CliRunner().invoke(parent, ["things", "go"])
        assert result.exit_code == 0, result.output
        # The wrapped command still executes (reused implementation).
        assert "ran" in result.output
        # And the yellow warning fires — rprint in pretty mode goes to stdout.
        combined = result.output + capsys.readouterr().err
        assert "deprecated" in combined.lower()
        assert "'comfy things …'" in combined

    def test_canonical_invocation_does_not_warn(self, capsys):
        parent = self._build_parent_with_alias()
        result = CliRunner().invoke(parent, ["thing", "go"])
        assert result.exit_code == 0, result.output
        assert "ran" in result.output
        combined = result.output + capsys.readouterr().err
        assert "deprecated" not in combined.lower()

    def test_alias_carries_nested_sub_groups(self):
        # A source app with a nested sub-group (add_typer) must have that group
        # carried onto the alias too — not silently dropped — and the warning
        # still fires for leaves reached through the nested group.
        source = typer.Typer(help="Root.")
        sub = typer.Typer()

        @sub.command("leaf")
        def leaf():
            typer.echo("leaf-ran")

        source.add_typer(sub, name="sub")

        parent = typer.Typer()
        parent.add_typer(source, name="thing")
        add_deprecated_alias(parent, source, old_name="things", new_name="thing")

        result = CliRunner().invoke(parent, ["things", "sub", "leaf"])
        assert result.exit_code == 0, result.output
        assert "leaf-ran" in result.output
        assert "deprecated" in result.output.lower()

    def test_alias_composes_source_callback(self, capsys):
        # A source app that declares its own group callback (with an option) must
        # have both the deprecation warning AND its own callback run, with the
        # callback's option preserved on the alias.
        seen: list[bool] = []
        source = typer.Typer()

        @source.callback()
        def _cb(flag: bool = typer.Option(False, "--flag")):
            seen.append(flag)

        @source.command("go")
        def go():
            typer.echo("ran")

        parent = typer.Typer()
        parent.add_typer(source, name="thing")
        add_deprecated_alias(parent, source, old_name="things", new_name="thing")

        result = CliRunner().invoke(parent, ["things", "--flag", "go"])
        assert result.exit_code == 0, result.output
        assert "ran" in result.output
        combined = result.output + capsys.readouterr().err
        assert "deprecated" in combined.lower()
        # The source callback ran, and its --flag option resolved on the alias.
        assert seen == [True], seen


# ---------------------------------------------------------------------------
# End-to-end through the real root app: `comfy models list-folders` warns,
# `comfy model list-folders` does not.
# ---------------------------------------------------------------------------


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch):
    from comfy_cli.target import Target

    fake = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-api-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    return fake


def _fake_resp(body: bytes):
    class _Resp:
        def __init__(self):
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n: int | None = None):
            return body if n is None else body[:n]

    return _Resp()


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, payload):
    def _fake(req, timeout=None):
        return _fake_resp(json.dumps(payload).encode())

    # `_http_get_json` routes every request through `comfy_cli.http.request_json`,
    # which opens via the module's `_AUTHED_OPENER` (built with
    # NoRedirectHandler) rather than the global `urllib.request.urlopen` — see
    # the same pattern in tests/comfy_cli/command/models/test_search.py.
    import comfy_cli.http as http_mod

    monkeypatch.setattr(http_mod._AUTHED_OPENER, "open", _fake)


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)


_CLOUD_FOLDERS = [{"name": "checkpoints", "folders": ["checkpoints"]}]


class TestEndToEnd:
    def _invoke(self, monkeypatch, noun):
        _force_json_renderer()
        _patch_urlopen(monkeypatch, _CLOUD_FOLDERS)
        return CliRunner().invoke(app, ["--json", noun, "list-folders", "--where", "cloud"], standalone_mode=False)

    def test_plural_invocation_warns_on_stderr(self, cloud_target, monkeypatch):
        result = self._invoke(monkeypatch, "models")
        # Command still works: an ok envelope lands on stdout.
        stdout = (result.stdout or "").strip()
        assert stdout, f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
        env = json.loads(stdout.splitlines()[-1])
        assert env["ok"] is True, env
        # Deprecation warning fired to stderr (JSON mode keeps stdout clean).
        assert "deprecated" in result.stderr.lower(), result.stderr

    def test_singular_invocation_is_silent(self, cloud_target, monkeypatch):
        result = self._invoke(monkeypatch, "model")
        stdout = (result.stdout or "").strip()
        assert stdout, f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
        env = json.loads(stdout.splitlines()[-1])
        assert env["ok"] is True, env
        assert "deprecated" not in result.stderr.lower(), result.stderr
