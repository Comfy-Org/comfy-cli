"""Layer 2: CLI envelope tests for ``comfy nodes`` — --where passthrough,
--cloud-disabled note, and --query removal.

Follows the same fixture patterns as test_nodes_introspect.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    """Pin the renderer to JSON so tests can read envelopes off stdout."""
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _fake_object_info() -> dict[str, Any]:
    """A small object_info dict covering the cases the tests assert on."""
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {}},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "category": "loaders",
            "display_name": "Load Checkpoint",
            "description": "Loads a diffusion model checkpoint.",
            "output_node": False,
            "python_module": "nodes",
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "sampler_name": [["euler", "heun", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras", "simple"], {"default": "normal"}],
                },
            },
            "input_order": {"required": ["model", "positive", "steps", "sampler_name", "scheduler"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise the latent via the provided model.",
            "output_node": False,
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "clip": ["CLIP"],
                    "text": ["STRING", {"multiline": True}],
                },
            },
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode (Prompt)",
            "description": "Encode prompt text to conditioning.",
            "output_node": False,
            "python_module": "nodes",
        },
        "SaveImage": {
            "input": {"required": {}},
            "output": [],
            "category": "image",
            "display_name": "Save Image",
            "description": "Save image to disk.",
            "output_node": True,
            "python_module": "nodes",
        },
    }


def _fake_graph():
    """Build a Graph from the fake object_info."""
    from comfy_cli.cql.engine import Graph

    return Graph.from_object_info(_fake_object_info())


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    """Bypass network/file loading; serve the fake graph straight to the command."""
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _fake_graph())


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(nodes_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code})"
    return json.loads(captured.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# --where passthrough tests
# ---------------------------------------------------------------------------


class _WhereSpy:
    """Records what ``where=`` value ``_get_graph`` received."""

    def __init__(self):
        self.captured_where: Any = "NOT_CALLED"

    def __call__(self, *a, **kw):
        self.captured_where = kw.get("where")
        return _fake_graph()


class TestLsWhereFlag:
    def test_ls_passes_where_to_get_graph(self, monkeypatch, capsys):
        spy = _WhereSpy()
        monkeypatch.setattr(nodes_cmd, "_get_graph", spy)
        _run(["ls", "--where", "cloud"], capsys)
        assert spy.captured_where == "cloud"

    def test_ls_default_where_is_none(self, monkeypatch, capsys):
        spy = _WhereSpy()
        monkeypatch.setattr(nodes_cmd, "_get_graph", spy)
        _run(["ls"], capsys)
        assert spy.captured_where is None


class TestSearchWhereFlag:
    def test_search_passes_where(self, monkeypatch, capsys):
        spy = _WhereSpy()
        monkeypatch.setattr(nodes_cmd, "_get_graph", spy)
        _run(["search", "KSampler", "--where", "cloud"], capsys)
        assert spy.captured_where == "cloud"


class TestUpstreamWhereFlag:
    def test_upstream_passes_where(self, monkeypatch, capsys):
        spy = _WhereSpy()
        monkeypatch.setattr(nodes_cmd, "_get_graph", spy)
        _run(["upstream", "KSampler", "--where", "cloud"], capsys)
        assert spy.captured_where == "cloud"


class TestDownstreamWhereFlag:
    def test_downstream_passes_where(self, monkeypatch, capsys):
        spy = _WhereSpy()
        monkeypatch.setattr(nodes_cmd, "_get_graph", spy)
        _run(["downstream", "CheckpointLoaderSimple", "--where", "cloud"], capsys)
        assert spy.captured_where == "cloud"


# ---------------------------------------------------------------------------
# --cloud-disabled note tests
# ---------------------------------------------------------------------------


class TestCloudDisabledNote:
    def test_cloud_disabled_on_cloud_shows_note(self, patched_loader, monkeypatch, capsys):
        monkeypatch.setattr(nodes_cmd, "_resolved_where", lambda where: "cloud")
        env = _run(["ls", "--cloud-disabled"], capsys)
        assert env["data"]["count"] == 0
        assert "cloud_note" in env["data"]
        assert "local server" in env["data"]["cloud_note"].lower() or "local" in env["data"]["cloud_note"].lower()

    def test_cloud_disabled_on_local_no_note(self, patched_loader, monkeypatch, capsys):
        monkeypatch.setattr(nodes_cmd, "_resolved_where", lambda where: "local")
        env = _run(["ls", "--cloud-disabled"], capsys)
        assert env["data"]["count"] == 0
        assert "cloud_note" not in env["data"]


# ---------------------------------------------------------------------------
# --query flag removed
# ---------------------------------------------------------------------------


class TestQueryFlagRemoved:
    def test_query_flag_rejected(self):
        runner = CliRunner()
        result = runner.invoke(nodes_cmd.app, ["ls", "--query", "produces IMAGE"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# nodes refresh — re-fetch annotation data from comfy-complete
# ---------------------------------------------------------------------------


class TestNodesRefresh:
    def test_refresh_reports_remote_success(self, monkeypatch, capsys):
        from comfy_cli.cql import annotations_source

        fake = [
            {"name": "supported_nodes.yaml", "source": "remote", "bytes": 100, "path": "/c/supported_nodes.yaml"},
            {
                "name": "cloud_disable_config.yaml",
                "source": "remote",
                "bytes": 50,
                "path": "/c/cloud_disable_config.yaml",
            },
        ]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh"], capsys)
        assert env["data"]["refreshed"] is True
        assert env["data"]["files"] == fake

    def test_refresh_reports_bundled_fallback(self, monkeypatch, capsys):
        from comfy_cli.cql import annotations_source

        fake = [
            {"name": "supported_nodes.yaml", "source": "bundled", "bytes": 100, "path": None, "error": "offline"},
            {"name": "cloud_disable_config.yaml", "source": "bundled", "bytes": 50, "path": None, "error": "offline"},
        ]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh"], capsys)
        assert env["data"]["refreshed"] is False

    def test_refresh_still_accepts_the_legacy_where_flag(self, monkeypatch, capsys):
        """``--where`` steers nothing now, but the CLI's own error hints and two
        shipped SKILL.md files told people to type it. Rejecting it would turn
        "you followed the hint" into ``No such option`` (exit 2)."""
        from comfy_cli.cql import annotations_source

        fake = [
            {"name": "supported_nodes.yaml", "source": "remote", "bytes": 100, "path": "/c/supported_nodes.yaml"},
            {
                "name": "cloud_disable_config.yaml",
                "source": "remote",
                "bytes": 50,
                "path": "/c/cloud_disable_config.yaml",
            },
        ]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh", "--where", "cloud"], capsys)
        assert env["data"]["refreshed"] is True

    def test_refresh_reports_unavailable_when_no_source_at_all(self, monkeypatch, capsys):
        """Package data missing *and* the fetch failed — neither remote nor bundled."""
        from comfy_cli.cql import annotations_source

        fake = [
            {"name": n, "source": "unavailable", "bytes": 0, "path": None, "error": "dns failure"}
            for n in ("supported_nodes.yaml", "cloud_disable_config.yaml")
        ]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh"], capsys)
        assert env["data"]["refreshed"] is False
        assert all(f["source"] == "unavailable" for f in env["data"]["files"])

    def test_refresh_reports_remote_with_cache_error(self, monkeypatch, capsys):
        """Downloaded fine, couldn't save it — still a successful refresh."""
        from comfy_cli.cql import annotations_source

        fake = [
            {"name": n, "source": "remote", "bytes": 100, "path": None, "cache_error": "disk full"}
            for n in ("supported_nodes.yaml", "cloud_disable_config.yaml")
        ]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh"], capsys)
        assert env["data"]["refreshed"] is True
        assert all(f["cache_error"] == "disk full" for f in env["data"]["files"])

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            # `_run` pins the JSON renderer, so the pretty branch — where the
            # unavailable/cache_error wording actually lives — needs its own pass.
            ({"source": "unavailable", "bytes": 0, "path": None, "error": "dns failure"}, "dns failure"),
            ({"source": "remote", "bytes": 100, "path": None, "cache_error": "disk full"}, "disk full"),
            ({"source": "bundled", "bytes": 100, "path": None}, "remote unavailable"),
            ({"source": "unavailable", "bytes": 0, "path": None}, "no source"),
        ],
    )
    def test_refresh_pretty_output_explains_every_source(self, monkeypatch, entry, expected):
        """Each source renders a reason; a missing `error` never prints as blank."""
        from comfy_cli.cql import annotations_source

        fake = [{"name": "supported_nodes.yaml", **entry}]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        reset_renderer_for_testing()  # pretty is the default renderer
        result = CliRunner().invoke(nodes_cmd.app, ["refresh"])
        assert result.exit_code == 0, result.output
        assert expected in result.output


class TestNodesRefreshPublishedContract:
    """`nodes refresh`'s payload is a published contract (`comfy --json discover`).

    Nothing validated the `nodes` payloads against `schemas/nodes.json`, so this
    command's new `refreshed`/`files` fields shipped undocumented — an agent
    reading `discover` would not have known they existed. Pin both directions:
    the schema describes the fields, and the command emits what it describes.
    """

    @staticmethod
    def _schema():
        from comfy_cli import discovery

        # `load_all_schemas` returns {name, title, schema} — the JSON Schema is
        # the inner value. Validating the wrapper instead would assert nothing:
        # `name`/`schema` are not validation keywords, so it accepts any input.
        return discovery.load_all_schemas()["nodes"]["schema"]

    def test_schema_documents_the_refresh_fields(self):
        props = self._schema()["properties"]
        assert "refreshed" in props, "schemas/nodes.json does not describe `refreshed`"
        assert "files" in props, "schemas/nodes.json does not describe `files`"
        assert set(props["files"]["items"]["properties"]["source"]["enum"]) == {
            "remote",
            "bundled",
            "unavailable",
        }

    @pytest.mark.parametrize(
        "entry",
        [
            {"source": "remote", "bytes": 100, "path": "/c/annotations.json"},
            {"source": "remote", "bytes": 100, "path": None, "cache_error": "disk full"},
            {"source": "bundled", "bytes": 100, "path": None, "error": "offline"},
            {"source": "unavailable", "bytes": 0, "path": None, "error": "dns failure"},
        ],
    )
    def test_emitted_payload_validates_against_the_schema(self, monkeypatch, capsys, entry):
        import jsonschema

        from comfy_cli.cql import annotations_source

        fake = [{"name": n, **entry} for n in ("supported_nodes.yaml", "cloud_disable_config.yaml")]
        monkeypatch.setattr(annotations_source, "refresh_annotations", lambda: fake)
        env = _run(["refresh"], capsys)
        jsonschema.Draft202012Validator(self._schema()).validate(env["data"])

    def test_real_refresh_output_validates_offline(self, monkeypatch, capsys):
        """Not just the fakes — the genuine offline code path emits valid shape."""
        import jsonschema

        monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "1")
        env = _run(["refresh"], capsys)
        jsonschema.Draft202012Validator(self._schema()).validate(env["data"])
        assert env["data"]["refreshed"] is False
        assert {f["source"] for f in env["data"]["files"]} == {"bundled"}


# local target resolution — config.background parity with `comfy run`
# ---------------------------------------------------------------------------
#
# `comfy nodes` is the agent's node-discovery surface, so it must read the same
# server `comfy run` submits to. It used to resolve `--host`/`--port` >
# COMFY_LOCAL_URL > 127.0.0.1:8188, skipping the persisted `config.background`
# step that `run`/`jobs` honor via `host_port.resolve_host_port` — so with
# ComfyUI launched in the background on a non-default port, discovery listed a
# different server's nodes than the one the workflow would execute on.


BACKGROUND_PORT = 8388


def _set_background(monkeypatch, background):
    """Point `host_port.resolve_host_port` at a synthetic `config.background`.

    ConfigManager already drops a record whose pid is dead, so a tuple here
    stands for a LIVE background server.
    """

    class _FakeConfigManager:
        def __init__(self):
            self.background = background

    monkeypatch.setattr("comfy_cli.host_port.ConfigManager", _FakeConfigManager)
    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)


@pytest.fixture
def captured_target(monkeypatch):
    """Stub the resilient live loader, recording the host/port it was handed."""
    seen: dict[str, Any] = {}

    def _fake_resilient_load(*, mode="local", host=None, port=None, input_path=None, on_stale=None):
        seen.update(mode=mode, host=host, port=port)
        return _fake_object_info()

    monkeypatch.setattr("comfy_cli.cql.loader.resilient_load_object_info", _fake_resilient_load)
    monkeypatch.setattr(nodes_cmd, "_resolved_where", lambda where: "local")
    return seen


class TestLocalTargetResolution:
    def test_ls_honors_background_server(self, monkeypatch, capsys, captured_target):
        """No flags + a live background server on a non-default port → discovery
        queries THAT server, not 127.0.0.1:8188."""
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

        env = _run(["ls", "--limit", "1"], capsys)

        assert env["ok"] is True
        assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", BACKGROUND_PORT)

    def test_ls_explicit_flags_beat_background(self, monkeypatch, capsys, captured_target):
        """Precedence unchanged: explicit `--host`/`--port` still win."""
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

        _run(["ls", "--limit", "1", "--host", "127.0.0.1", "--port", "9000"], capsys)

        assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", 9000)

    def test_ls_without_background_defaults_to_8188(self, monkeypatch, capsys, captured_target):
        """With nothing recorded, the default target is unchanged."""
        _set_background(monkeypatch, None)

        _run(["ls", "--limit", "1"], capsys)

        assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", 8188)

    def test_show_honors_background_server(self, monkeypatch, capsys, captured_target):
        """The same resolution applies to every `comfy nodes` subcommand — they
        all share `_get_graph`."""
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

        _run(["show", "KSampler"], capsys)

        assert captured_target["port"] == BACKGROUND_PORT

    def test_input_path_skips_host_resolution(self, monkeypatch, tmp_path, capsys, captured_target):
        """`--input` is offline mode: no live fetch, and the recorded background
        server is never consulted."""
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))
        dump = tmp_path / "oi.json"
        dump.write_text(json.dumps(_fake_object_info()), encoding="utf-8")

        env = _run(["ls", "--limit", "1", "--input", str(dump)], capsys)

        assert env["ok"] is True
        # The live loader is stubbed by `captured_target`; asserting it stayed
        # untouched is what actually pins the `input_path is None` guard. Without
        # it a regression would fall through to a real request and could still
        # pass off a stale on-disk cache.
        assert captured_target == {}

    def test_wildcard_background_host_is_canonicalized(self, monkeypatch, capsys, captured_target):
        """`comfy launch -- --listen 0.0.0.0` records the wildcard BIND address.
        Used as a destination it trips the object_info loopback guard, and here
        the resulting LoadError is swallowed by `resilient_load_object_info` — so
        `nodes ls/show/search` would serve the last cached dump while claiming to
        read the live server. It is canonicalized to loopback instead."""
        _set_background(monkeypatch, ("0.0.0.0", BACKGROUND_PORT, 4242))

        _run(["ls", "--limit", "1"], capsys)

        assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", BACKGROUND_PORT)

    def test_empty_host_flag_is_rejected(self, monkeypatch, captured_target):
        """`--host ""` must error rather than falling through to the background
        server — the same guard `comfy run` gets from `resolve_host_port`."""
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

        result = CliRunner().invoke(nodes_cmd.app, ["ls", "--limit", "1", "--host", ""])

        assert result.exit_code == 2
        assert captured_target == {}

    @pytest.mark.parametrize("bad_port", ["0", "99999"])
    def test_out_of_range_port_flag_is_rejected(self, monkeypatch, captured_target, bad_port):
        _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

        result = CliRunner().invoke(nodes_cmd.app, ["ls", "--limit", "1", "--port", bad_port])

        assert result.exit_code == 2
        assert captured_target == {}
