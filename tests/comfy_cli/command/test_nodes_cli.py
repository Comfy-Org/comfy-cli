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
