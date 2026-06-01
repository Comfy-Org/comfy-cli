"""Tests for ``comfy nodes`` — agent-facing node-class introspection.

The commands are thin wrappers over the CQL stub's loader, so the heavy
lifting is in the wiring (filter precedence, error codes, envelope shape).
Tests use a hand-rolled graph fixture instead of hitting a live server.
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
    # The renderer writes to its own stream; capsys captures stdout.
    captured = capsys.readouterr().out
    if not captured.strip():
        # Renderer wrote to its bound stream (sys.stdout in JSON mode); the
        # CliRunner may have stolen it. Fall back to result.stdout.
        captured = result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code})"
    return json.loads(captured.strip().splitlines()[-1])


class TestLs:
    def test_no_filter_returns_all(self, patched_loader, capsys):
        env = _run(["ls"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 4

    def test_produces_filter(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "MODEL"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "CheckpointLoaderSimple"

    def test_produces_filter_case_insensitive(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "model"], capsys)
        assert env["data"]["count"] == 1

    def test_accepts_filter(self, patched_loader, capsys):
        env = _run(["ls", "--accepts", "MODEL"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_category_glob(self, patched_loader, capsys):
        env = _run(["ls", "--category", "sampling*"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_category_sql_percent_pattern(self, patched_loader, capsys):
        """Agents from the SQL-CQL grammar might still send `%`."""
        env = _run(["ls", "--category", "samp%"], capsys)
        assert env["data"]["count"] == 1

    def test_limit(self, patched_loader, capsys):
        env = _run(["ls", "--limit", "2"], capsys)
        assert env["data"]["count"] == 2

    def test_filter_block_present_in_envelope(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "LATENT", "--category", "samp*"], capsys)
        f = env["data"]["filter"]
        assert f["produces"] == "LATENT"
        assert f["accepts"] is None
        assert f["category"] == "samp*"


class TestShow:
    def test_basic_envelope(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        assert env["ok"] is True
        d = env["data"]
        assert d["name"] == "KSampler"
        assert d["category"] == "sampling"
        assert d["output_types"] == ["LATENT"]
        # Inputs include section + type + options.
        inputs = {i["name"]: i for i in d["inputs"]}
        assert "model" in inputs and inputs["model"]["type"] == "MODEL"
        assert inputs["steps"]["options"]["default"] == 20

    def test_inputs_sorted_required_first(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        sections = [i["section"] for i in env["data"]["inputs"]]
        # No `optional` in the fixture; just verify all required come before any non-required.
        first_optional = next((i for i, s in enumerate(sections) if s != "required"), len(sections))
        last_required = max((i for i, s in enumerate(sections) if s == "required"), default=-1)
        assert last_required < first_optional

    def test_node_not_found_emits_structured_error(self, patched_loader, capsys):
        env = _run(["show", "Nonexistent"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "node_not_found"
        assert "close_matches" in env["error"]["details"]

    def test_node_not_found_suggests_close_matches(self, patched_loader, capsys):
        # Lowercase typo should still surface KSampler as a close match.
        env = _run(["show", "ksampler"], capsys)
        close = env["error"]["details"]["close_matches"]
        assert "KSampler" in close

    def test_choices_populated_for_local_enum(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        inputs = {i["name"]: i for i in env["data"]["inputs"]}
        assert inputs["sampler_name"]["choices"] == ["euler", "heun", "dpmpp_2m"]

    def test_choices_populated_for_cloud_combo(self, patched_loader, capsys):
        """Cloud-API COMBO inputs nest their choices at options.options.
        `nodes show` should normalize them into the same `choices` array
        as local ENUM inputs — agents shouldn't need to know which shape
        the graph happens to use."""
        env = _run(["show", "KSampler"], capsys)
        inputs = {i["name"]: i for i in env["data"]["inputs"]}
        assert inputs["scheduler"]["choices"] == ["normal", "karras", "simple"]
        # And the raw options block is still passed through for callers
        # that need the default / min / max metadata.
        assert inputs["scheduler"]["options"]["default"] == "normal"


class TestSearch:
    def test_exact_name_wins(self, patched_loader, capsys):
        env = _run(["search", "KSampler"], capsys)
        assert env["data"]["count"] >= 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_substring_match(self, patched_loader, capsys):
        env = _run(["search", "checkpoint"], capsys)
        assert env["data"]["count"] >= 1
        assert env["data"]["rows"][0]["name"] == "CheckpointLoaderSimple"

    def test_description_match(self, patched_loader, capsys):
        env = _run(["search", "Denoise"], capsys)
        # KSampler's description contains "Denoise"; should match.
        names = [r["name"] for r in env["data"]["rows"]]
        assert "KSampler" in names

    def test_no_match_returns_empty(self, patched_loader, capsys):
        env = _run(["search", "xyzzy_nothing"], capsys)
        assert env["data"]["count"] == 0
        assert env["data"]["rows"] == []

    def test_limit_caps_results(self, patched_loader, capsys):
        env = _run(["search", "e", "--limit", "2"], capsys)
        assert env["data"]["count"] <= 2


class TestFlattenCategoryTree:
    """Pin the shape contract for the wasm CategoryTree, since the flattener
    has to know the (capital-cased) field names the Go side emits."""

    def test_walks_nested_children(self):
        tree = {
            "Root": {
                "Name": "",
                "FullPath": "",
                "Children": {
                    "loaders": {
                        "FullPath": "loaders",
                        "Count": 22,
                        "Children": {
                            "advanced": {
                                "FullPath": "loaders/advanced",
                                "Count": 4,
                                "Children": {},
                            },
                        },
                    },
                    "sampling": {
                        "FullPath": "sampling",
                        "Count": 8,
                        "Children": {},
                    },
                },
            },
        }
        from comfy_cli.command.nodes import _flatten_category_tree

        flat = _flatten_category_tree(tree)
        flat_dict = dict(flat)
        assert flat_dict["loaders"] == 22
        assert flat_dict["loaders/advanced"] == 4
        assert flat_dict["sampling"] == 8

    def test_empty_or_malformed_returns_empty(self):
        from comfy_cli.command.nodes import _flatten_category_tree

        assert _flatten_category_tree({}) == []
        assert _flatten_category_tree({"Root": None}) == []
        assert _flatten_category_tree("not a dict") == []  # type: ignore[arg-type]
