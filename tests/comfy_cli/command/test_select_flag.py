"""`--select <expr>` wiring on the four heaviest read commands (V1-011).

ONE selector implementation (``comfy_cli.selector``) projected onto:

  - ``comfy templates ls --select``
  - ``comfy nodes show --select``
  - ``comfy workflow slots --select``
  - ``comfy generate list --select``  (the invocation the cloud agent's
    ``list_generate_models`` tool shells — services/agent/internal/loop/tools.go
    execs ``generate list --json``)

Pinned here per command: the envelope's ``data`` becomes the selected slice,
sibling ``selected_bytes``/``total_bytes`` fields appear, fail-open on a miss
(ok:true, exit 0, key inventory + ``select_no_match`` advisory), pretty-mode
rendering, and — crucially — that WITHOUT ``--select`` the envelope is
byte-identical to the pre-flag output (``test_no_select_output_unchanged``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.command import templates as templates_cmd
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


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


def _force_pretty_renderer():
    r = Renderer.resolve(
        is_stdout_tty=True,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        no_json_flag=True,
    )
    r.mode = OutputMode.PRETTY
    set_renderer(r)
    return r


def _envelope(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope in stdout:\n{stdout}")


def _assert_byte_fields(env: dict) -> None:
    # selected_bytes counts the serialized emitted slice; total_bytes the full
    # payload. On a fail-open miss the inventory slice can serialize LARGER
    # than a small payload, so no ordering is asserted here — match-path tests
    # assert selected < total themselves.
    assert env["selected_bytes"] == len(json.dumps(env["data"], ensure_ascii=False).encode("utf-8"))
    assert isinstance(env["total_bytes"], int) and env["total_bytes"] > 0


def _assert_fail_open(env: dict) -> None:
    assert env["ok"] is True
    assert env["error"] is None
    assert "inventory" in env["data"]
    assert env["data"]["warnings"][0]["code"] == "select_no_match"


# ---------------------------------------------------------------------------
# templates ls
# ---------------------------------------------------------------------------

GALLERY_FIXTURE = [
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Image",
        "type": "image",
        "templates": [
            {
                "name": "image_flux2",
                "title": "Flux 2 Image",
                "description": "Text-to-image using Flux 2 via the BFL API.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["API", "Text to Image"],
                "models": ["Flux 2"],
                "logos": [{"provider": ["Black Forest Labs"]}],
            },
            {
                "name": "image_z_image",
                "title": "Z Image",
                "description": "Local SDXL-style text-to-image.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["Local", "Text to Image"],
                "models": ["Z Image"],
                "logos": [{"provider": "Z"}],
            },
        ],
    },
]


@pytest.fixture
def gallery_file(tmp_path: Path) -> str:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(GALLERY_FIXTURE))
    return str(path)


class TestTemplatesLsSelect:
    def test_select_projects_data(self, gallery_file):
        _force_json_renderer()
        result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--select", "rows.#.name"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["ok"] is True
        assert env["data"] == ["image_flux2", "image_z_image"]
        _assert_byte_fields(env)
        assert env["selected_bytes"] < env["total_bytes"]

    def test_select_miss_fails_open_with_inventory(self, gallery_file):
        _force_json_renderer()
        result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--select", "not.a.key"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        _assert_fail_open(env)
        assert set(env["data"]["inventory"]) >= {"rows", "matched", "filters"}
        _assert_byte_fields(env)

    def test_select_works_in_pretty_mode(self, gallery_file):
        _force_pretty_renderer()
        result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--select", "rows.#.name"])
        assert result.exit_code == 0, result.output
        assert "image_flux2" in result.output
        assert "image_z_image" in result.output
        # The selection replaced the human table.
        assert "Flux 2 Image" not in result.output

    def test_pretty_miss_prints_inventory_and_hint(self, gallery_file):
        _force_pretty_renderer()
        result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file, "--select", "not.a.key"])
        assert result.exit_code == 0, result.output
        assert "matched nothing" in result.output
        assert "rows" in result.output  # inventory names the real keys

    def test_no_select_output_unchanged(self, gallery_file):
        """WITHOUT --select the envelope is byte-identical to the pre-flag
        serialization: exact key set, exact order, no byte-count siblings."""
        _force_json_renderer()
        result = runner.invoke(templates_cmd.app, ["ls", "--gallery", gallery_file])
        assert result.exit_code == 0, result.output
        line = result.output.strip().splitlines()[-1]
        rows = [
            {
                "name": t["name"],
                "title": t["title"],
                "output_type": "image",
                "category_title": "Image",
                "tags": t["tags"],
                "models": t["models"],
                "providers": ["Black Forest Labs"] if t["name"] == "image_flux2" else ["Z"],
                "description": t["description"][:120],
            }
            for t in GALLERY_FIXTURE[0]["templates"]
        ]
        expected = {
            "schema": "envelope/1",
            "type": "envelope",
            "ok": True,
            "command": "templates ls",
            "version": "",
            "where": None,
            "data": {
                "total_in_gallery": 2,
                "matched": 2,
                "shown": 2,
                "filters": {
                    "type": None,
                    "category": None,
                    "tag": None,
                    "model": None,
                    "provider": None,
                    "name": None,
                },
                "rows": rows,
            },
            "error": None,
        }
        assert line == json.dumps(expected, ensure_ascii=False)


# ---------------------------------------------------------------------------
# nodes show
# ---------------------------------------------------------------------------


def _nodes_object_info() -> dict[str, Any]:
    return {
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "sampler_name": [["euler", "heun", "dpmpp_2m"]],
                },
            },
            "input_order": {"required": ["model", "steps", "sampler_name"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise the latent via the provided model.",
            "output_node": False,
            "python_module": "nodes",
        },
    }


@pytest.fixture
def patched_nodes_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: Graph.from_object_info(_nodes_object_info()))


class TestNodesShowSelect:
    def test_select_projects_inputs(self, patched_nodes_graph):
        _force_json_renderer()
        result = runner.invoke(nodes_cmd.app, ["show", "KSampler", "--select", "inputs.#.name"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["ok"] is True
        assert env["data"] == ["model", "steps", "sampler_name"]
        _assert_byte_fields(env)

    def test_select_comma_multi(self, patched_nodes_graph):
        _force_json_renderer()
        result = runner.invoke(nodes_cmd.app, ["show", "KSampler", "--select", "name,category"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"] == {"name": "KSampler", "category": "sampling"}

    def test_select_miss_fails_open(self, patched_nodes_graph):
        _force_json_renderer()
        result = runner.invoke(nodes_cmd.app, ["show", "KSampler", "--select", "a..b"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        _assert_fail_open(env)
        assert "inputs" in env["data"]["inventory"]

    def test_select_scalar_pretty_prints_plain(self, patched_nodes_graph):
        _force_pretty_renderer()
        result = runner.invoke(nodes_cmd.app, ["show", "KSampler", "--select", "category"])
        assert result.exit_code == 0, result.output
        assert "sampling" in result.output
        assert '"sampling"' not in result.output  # bare string, no JSON quotes


# ---------------------------------------------------------------------------
# workflow slots
# ---------------------------------------------------------------------------


def _slots_object_info() -> dict[str, Any]:
    return {
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": ["CLIP"],
                },
            },
            "input_order": {"required": ["clip", "text"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode",
            "python_module": "nodes",
        },
    }


def _slots_workflow() -> dict:
    return {
        "nodes": [
            {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["a cat in space"]},
        ],
        "links": [],
    }


@pytest.fixture
def patched_workflow_graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(workflow_cmd, "_get_graph", lambda *a, **kw: Graph.from_object_info(_slots_object_info()))


class TestWorkflowSlotsSelect:
    def _write(self, tmp_path: Path) -> Path:
        p = tmp_path / "wf.json"
        p.write_text(json.dumps(_slots_workflow()), encoding="utf-8")
        return p

    def test_select_projects_addresses(self, patched_workflow_graph, tmp_path):
        _force_json_renderer()
        path = self._write(tmp_path)
        result = runner.invoke(workflow_cmd.app, ["slots", str(path), "--select", "slots.#.address"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["ok"] is True
        assert env["data"] == ["6.text"]
        _assert_byte_fields(env)

    def test_select_count(self, patched_workflow_graph, tmp_path):
        _force_json_renderer()
        path = self._write(tmp_path)
        result = runner.invoke(workflow_cmd.app, ["slots", str(path), "--select", "count"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"] == 1

    def test_select_miss_fails_open(self, patched_workflow_graph, tmp_path):
        _force_json_renderer()
        path = self._write(tmp_path)
        result = runner.invoke(workflow_cmd.app, ["slots", str(path), "--select", "widgets"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        _assert_fail_open(env)
        assert "slots" in env["data"]["inventory"]


# ---------------------------------------------------------------------------
# generate list  (what the cloud agent's list_generate_models tool shells)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_tracking(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _isolate_spec_caches():
    from comfy_cli.command.generate import spec as _spec

    _spec.load_raw_spec.cache_clear()
    _spec._registry.cache_clear()
    yield
    _spec.load_raw_spec.cache_clear()
    _spec._registry.cache_clear()


class TestGenerateListSelect:
    def test_select_projects_aliases(self):
        from comfy_cli.cmdline import app as cli_app

        _force_json_renderer()
        result = runner.invoke(cli_app, ["generate", "list", "--json", "--select", "models.#.alias"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["ok"] is True
        assert env["command"] == "generate list"
        assert isinstance(env["data"], list) and env["data"]
        assert "flux-pro" in env["data"]
        _assert_byte_fields(env)

    def test_select_eq_form(self):
        from comfy_cli.cmdline import app as cli_app

        _force_json_renderer()
        result = runner.invoke(cli_app, ["generate", "list", "--json", "--select=count"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert isinstance(env["data"], int) and env["data"] >= 1

    def test_select_miss_fails_open(self):
        from comfy_cli.cmdline import app as cli_app

        _force_json_renderer()
        result = runner.invoke(cli_app, ["generate", "list", "--json", "--select", "bogus.path"])
        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        _assert_fail_open(env)
        assert set(env["data"]["inventory"]) >= {"models", "count"}
