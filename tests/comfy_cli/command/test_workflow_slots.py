"""Tests for `comfy workflow slots/set-slot/vary` — slot editing on frontend-format workflows.

Tests both template/subgraph mode and direct mode (regular workflows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


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


def _object_info():
    return {
        "KSampler": {
            "input": {
                "required": {
                    "model": "MODEL",
                    "positive": "CONDITIONING",
                    "negative": "CONDITIONING",
                    "latent_image": "LATENT",
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32, "control_after_generate": True}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "cfg": ["FLOAT", {"default": 8.0}],
                    "sampler_name": [["euler", "euler_ancestral", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras"]],
                    "denoise": ["FLOAT", {"default": 1.0}],
                },
            },
            "input_order": {
                "required": [
                    "model",
                    "positive",
                    "negative",
                    "latent_image",
                    "seed",
                    "steps",
                    "cfg",
                    "sampler_name",
                    "scheduler",
                    "denoise",
                ]
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": "CLIP",
                },
            },
            "input_order": {"required": ["clip", "text"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode",
            "python_module": "nodes",
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                },
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "latent",
            "display_name": "Empty Latent Image",
            "python_module": "nodes",
        },
        "GeminiImage2Node": {
            "input": {
                "required": {
                    "prompt": ["STRING", {"default": ""}],
                    "model": "COMBO",
                    "seed": ["INT", {"default": 42, "control_after_generate": True}],
                    "aspect_ratio": "COMBO",
                    "resolution": "COMBO",
                    "response_modalities": "COMBO",
                },
                "optional": {
                    "images": "IMAGE",
                    "files": "GEMINI_INPUT_FILES",
                    "system_prompt": ["STRING", {"default": ""}],
                },
            },
            "input_order": {
                "required": ["prompt", "model", "seed", "aspect_ratio", "resolution", "response_modalities"],
                "optional": ["images", "files", "system_prompt"],
            },
            "output": ["IMAGE", "STRING"],
            "output_name": ["IMAGE", "STRING"],
            "category": "api node/image/Gemini",
            "display_name": "Nano Banana Pro",
            "python_module": "nodes",
        },
    }


def _fake_graph():
    return Graph.from_object_info(_object_info())


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _subgraph_graph():
    """Graph built from committed object_info covering the fetched-template inner nodes."""
    oi = json.loads((_FIXTURES / "subgraph_object_info.json").read_text(encoding="utf-8"))
    return Graph.from_object_info(oi)


def _subgraph_workflow() -> dict:
    """The committed minimal fetched-gallery template with UUID subgraph instances."""
    return json.loads((_FIXTURES / "subgraph_template_ui.json").read_text(encoding="utf-8"))


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_cmd, "_get_graph", lambda *a, **kw: _fake_graph())


@pytest.fixture
def patched_subgraph_graph(monkeypatch):
    monkeypatch.setattr(workflow_cmd, "_get_graph", lambda *a, **kw: _subgraph_graph())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_workflow(tmp_path: Path, data: dict, name: str = "test.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _direct_workflow():
    return {
        "nodes": [
            {
                "id": 3,
                "type": "KSampler",
                "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
            },
            {
                "id": 6,
                "type": "CLIPTextEncode",
                "widgets_values": ["a cat in space"],
            },
            {
                "id": 7,
                "type": "EmptyLatentImage",
                "widgets_values": [512, 512, 1],
            },
        ],
        "links": [],
    }


def _api_node_workflow():
    return {
        "nodes": [
            {
                "id": 263,
                "type": "GeminiImage2Node",
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": None},
                    {"name": "files", "type": "GEMINI_INPUT_FILES", "link": None},
                ],
                "widgets_values": [
                    "fill the masked region",
                    "gemini-3-pro-image-preview",
                    41439150623705,
                    "randomize",
                    "auto",
                    "4K",
                    "IMAGE+TEXT",
                    "always produce an image",
                ],
            }
        ],
        "links": [],
    }


def _template_workflow():
    return {
        "nodes": [
            {
                "id": 1,
                "type": "MyTemplate",
                "properties": {
                    "proxyWidgets": [
                        [10, "text"],
                        [11, "seed"],
                    ],
                },
            },
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "name": "MyTemplate",
                    "inputs": [
                        {"name": "text", "type": "STRING"},
                        {"name": "seed", "type": "INT"},
                    ],
                    "nodes": [
                        {
                            "id": 10,
                            "type": "CLIPTextEncode",
                            "widgets_values": ["hello world"],
                        },
                        {
                            "id": 11,
                            "type": "KSampler",
                            "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
                        },
                    ],
                },
            ],
        },
    }


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    lines = [ln for ln in captured.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:500]})")


# ---------------------------------------------------------------------------
# slots — direct mode
# ---------------------------------------------------------------------------


class TestSlotsDirectMode:
    def test_slots_shows_widget_inputs(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] > 0, "expected at least one slot"

    def test_slots_includes_address_and_current_value(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["slots", str(path)], capsys)
        slots = env["data"]["slots"]
        by_addr = {s["address"]: s for s in slots}
        assert "6.text" in by_addr
        assert by_addr["6.text"]["current_value"] == "a cat in space"

    def test_slots_rejects_api_format(self, patched_graph, tmp_path, capsys):
        api_wf = {"3": {"class_type": "KSampler", "inputs": {}}}
        path = _write_workflow(tmp_path, api_wf)
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_frontend_format"

    def test_slots_rejects_missing_file(self, patched_graph, capsys):
        env = _run(["slots", "/nonexistent/path.json"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_slots_keeps_api_node_dynamic_combo_values_aligned(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _api_node_workflow())
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        slots = {s["address"]: s["current_value"] for s in env["data"]["slots"]}
        assert slots["263.model"] == "gemini-3-pro-image-preview"
        assert slots["263.seed"] == 41439150623705
        assert slots["263.aspect_ratio"] == "auto"
        assert slots["263.resolution"] == "4K"
        assert slots["263.response_modalities"] == "IMAGE+TEXT"
        assert slots["263.system_prompt"] == "always produce an image"


# ---------------------------------------------------------------------------
# set-slot — direct mode
# ---------------------------------------------------------------------------


class TestSetSlotDirectMode:
    def test_set_slot_modifies_in_place(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["set-slot", str(path), '6.text="a dog"'], capsys)
        assert env["ok"] is True
        on_disk = json.loads(path.read_text())
        clip_node = next(n for n in on_disk["nodes"] if n["id"] == 6)
        assert clip_node["widgets_values"][0] == "a dog"

    def test_set_slot_stdout_mode(self, patched_graph, tmp_path, capsys):
        wf = _direct_workflow()
        path = _write_workflow(tmp_path, wf)
        original_text = path.read_text()
        # --stdout prints to stdout instead of modifying file
        _force_json_renderer()
        runner = CliRunner()
        runner.invoke(
            workflow_cmd.app,
            ["set-slot", str(path), '6.text="a dog"', "--stdout"],
            standalone_mode=False,
        )
        # File should be unchanged
        assert path.read_text() == original_text

    def test_set_slot_invalid_format(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["set-slot", str(path), "bad_no_equals"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_slot_invalid"

    def test_set_slot_unknown_node(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["set-slot", str(path), "99.text=foo"], capsys)
        assert env["ok"] is False
        assert "not found" in env["error"]["message"].lower()


# ---------------------------------------------------------------------------
# vary — direct mode
# ---------------------------------------------------------------------------


class TestVaryDirectMode:
    def test_vary_produces_files(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        out_dir = tmp_path / "out"
        env = _run(
            [
                "vary",
                str(path),
                "--slot",
                "3.seed=[1,2,3]",
                "--out-dir",
                str(out_dir),
            ],
            capsys,
        )
        assert env["ok"] is True
        assert env["data"]["count"] == 3
        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 3

    def test_vary_mismatched_lengths_rejected(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(
            [
                "vary",
                str(path),
                "--slot",
                "3.seed=[1,2,3]",
                "--slot",
                "3.steps=[10,20]",
                "--out-dir",
                str(tmp_path / "out"),
            ],
            capsys,
        )
        assert env["ok"] is False
        assert "same length" in env["error"]["message"].lower()

    def test_vary_non_list_rejected(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(
            [
                "vary",
                str(path),
                "--slot",
                "3.seed=42",
                "--out-dir",
                str(tmp_path / "out"),
            ],
            capsys,
        )
        assert env["ok"] is False
        assert "array" in env["error"]["message"].lower()


# ---------------------------------------------------------------------------
# slots — template mode
# ---------------------------------------------------------------------------


class TestSlotsTemplateMode:
    def test_slots_template_shows_declared_inputs_only(self, patched_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _template_workflow())
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        slots = env["data"]["slots"]
        names = {s["name"] for s in slots}
        assert names == {"text", "seed"}, f"expected only template-declared inputs, got {names}"
        assert env["data"]["count"] == 2


# ---------------------------------------------------------------------------
# Agent scenario: seed sweep end-to-end
# ---------------------------------------------------------------------------


class TestAgentSeedSweepScenario:
    def test_seed_sweep_e2e(self, patched_graph, tmp_path, capsys):
        """Simulates what an agent actually does:
        1. Write workflow
        2. Run `slots` to discover addresses
        3. Run `vary` with seed list
        4. Read back variants and verify each has the right seed
        """
        # Step 1: write workflow
        path = _write_workflow(tmp_path, _direct_workflow())

        # Step 2: discover slots
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        slots = env["data"]["slots"]
        seed_slot = next((s for s in slots if s["name"] == "seed"), None)
        assert seed_slot is not None, "agent must find a 'seed' slot"
        seed_addr = seed_slot["address"]
        assert seed_addr == "3.seed"

        # Step 3: produce variants
        out_dir = tmp_path / "variants"
        env = _run(
            [
                "vary",
                str(path),
                "--slot",
                f"{seed_addr}=[10,20,30]",
                "--out-dir",
                str(out_dir),
            ],
            capsys,
        )
        assert env["ok"] is True
        assert env["data"]["count"] == 3

        # Step 4: verify each variant
        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 3
        seeds = []
        for f in files:
            wf = json.loads(f.read_text())
            ks = next(n for n in wf["nodes"] if n["type"] == "KSampler")
            # seed is the first widget in input_order
            seeds.append(ks["widgets_values"][0])
        assert seeds == [10, 20, 30]

    def test_prompt_and_seed_sweep_e2e(self, patched_graph, tmp_path, capsys):
        """Multi-slot sweep: vary both prompt and seed."""
        path = _write_workflow(tmp_path, _direct_workflow())
        out_dir = tmp_path / "variants"
        env = _run(
            [
                "vary",
                str(path),
                "--slot",
                '6.text=["cat","dog","fox"]',
                "--slot",
                "3.seed=[1,2,3]",
                "--out-dir",
                str(out_dir),
            ],
            capsys,
        )
        assert env["ok"] is True
        assert env["data"]["count"] == 3

        files = sorted(out_dir.glob("*.json"))
        prompts = []
        seeds = []
        for f in files:
            wf = json.loads(f.read_text())
            clip = next(n for n in wf["nodes"] if n["type"] == "CLIPTextEncode")
            ks = next(n for n in wf["nodes"] if n["type"] == "KSampler")
            prompts.append(clip["widgets_values"][0])
            seeds.append(ks["widgets_values"][0])
        assert prompts == ["cat", "dog", "fox"]
        assert seeds == [1, 2, 3]


# ---------------------------------------------------------------------------
# slots — nested subgraph (fetched gallery templates)
# ---------------------------------------------------------------------------
#
# Fetched templates wrap their logic in subgraph instances whose class_type is
# a UUID. Multiple defs collide on the cosmetic name "New Subgraph", and the
# curated proxyWidgets reference deleted interior ids (e.g. "-1"), so the real
# editable inner inputs (prompt / seed / image inside GeminiImage2Node) stay
# hidden. `slots` must recurse INTO subgraphs and surface them with stable
# nested addresses: ``<instanceId>/<innerNodeId>.<input>`` (and deeper for
# nested subgraphs). set-slot / vary must accept those addresses.


def _interior_node(workflow: dict, sg_id: str, inner_id) -> dict:
    sg = next(s for s in workflow["definitions"]["subgraphs"] if s["id"] == sg_id)
    return next(n for n in sg["nodes"] if str(n.get("id")) == str(inner_id))


D33 = "d33c1791-dfd2-4102-8540-aa63e4434cd2"  # instance 10's subgraph def
F22 = "f2228dc9-64e8-43b1-a4ca-b8a57eed8f64"  # instance 19's subgraph def (same name, diff def)
BATCH = "da09b826-d678-40e0-a4e4-5f2178043ab6"  # nested subgraph (Batch Prompt Iterator)


class TestSlotsNestedSubgraph:
    def test_slots_recurses_into_subgraph_inner_inputs(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        by_addr = {s["address"]: s for s in env["data"]["slots"]}
        # The editable inner prompt + seed of the Gemini node inside instance 10.
        assert "10/9.prompt" in by_addr, f"missing nested prompt slot; got {sorted(by_addr)[:20]}"
        assert "10/9.seed" in by_addr
        assert by_addr["10/9.seed"]["current_value"] == 1

    def test_nested_addresses_are_instance_scoped(self, patched_subgraph_graph, tmp_path, capsys):
        """Two instances of same-named ('New Subgraph') but DIFFERENT defs must
        not collide — each surfaces its own inner values under its own id."""
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["slots", str(path)], capsys)
        by_addr = {s["address"]: s for s in env["data"]["slots"]}
        assert by_addr["10/9.seed"]["current_value"] == 1
        assert by_addr["19/9.seed"]["current_value"] == 2

    def test_slots_recurses_into_doubly_nested_subgraph(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["slots", str(path)], capsys)
        addrs = {s["address"] for s in env["data"]["slots"]}
        # instance 10 -> inner node 3 (Batch Prompt Iterator subgraph) -> inner
        # node 7 (PrimitiveStringMultiline) widget "value".
        assert "10/3/7.value" in addrs, f"missing doubly-nested slot; got {sorted(addrs)[:30]}"


class TestSetSlotNestedSubgraph:
    def test_set_slot_writes_into_subgraph_inner_node(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["set-slot", str(path), '10/9.prompt="a red fox in snow"'], capsys)
        assert env["ok"] is True, env
        wf = json.loads(path.read_text())
        gemini = _interior_node(wf, D33, 9)
        # prompt is widget index 0 for GeminiImage2Node.
        assert gemini["widgets_values"][0] == "a red fox in snow"

    def test_set_slot_nested_does_not_touch_sibling_instance(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        _run(["set-slot", str(path), "10/9.seed=777"], capsys)
        wf = json.loads(path.read_text())
        # seed is widget index 2 (index 3 is control_after_generate).
        assert _interior_node(wf, D33, 9)["widgets_values"][2] == 777
        # The other instance's def must be untouched.
        assert _interior_node(wf, F22, 9)["widgets_values"][2] == 2

    def test_set_slot_doubly_nested(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["set-slot", str(path), '10/3/7.value="iterated prompt"'], capsys)
        assert env["ok"] is True, env
        wf = json.loads(path.read_text())
        prim = _interior_node(wf, BATCH, 7)
        assert prim["widgets_values"][0] == "iterated prompt"

    def test_set_slot_unknown_nested_inner_node(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        env = _run(["set-slot", str(path), "10/999.prompt=foo"], capsys)
        assert env["ok"] is False
        assert "not found" in env["error"]["message"].lower()


def test_cli_hints_use_id_based_slot_addresses():
    """Slot addresses resolve by node id; hints must not teach title-style addresses."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[3] / "comfy_cli/command/workflow.py").read_text(encoding="utf-8")
    assert "positive_prompt.text" not in src


class TestVaryNestedSubgraph:
    def test_vary_over_nested_prompt(self, patched_subgraph_graph, tmp_path, capsys):
        path = _write_workflow(tmp_path, _subgraph_workflow())
        out_dir = tmp_path / "out"
        env = _run(
            ["vary", str(path), "--slot", '10/9.prompt=["cat","dog","fox"]', "--out-dir", str(out_dir)],
            capsys,
        )
        assert env["ok"] is True
        assert env["data"]["count"] == 3
        prompts = []
        for f in sorted(out_dir.glob("*.json")):
            wf = json.loads(f.read_text())
            prompts.append(_interior_node(wf, D33, 9)["widgets_values"][0])
        assert prompts == ["cat", "dog", "fox"]


# ---------------------------------------------------------------------------
# stale cache envelope — workflow slots/set-slot/vary surface stale flag
# ---------------------------------------------------------------------------
#
# When resilient_load_object_info falls back to a cached copy it fires
# on_stale(host_key, reason). The three slot commands must inject
# stale=True and warnings=[{code:"object_info_stale", ...}] into the
# emitted payload so agents can detect cache staleness.


def _stale_get_graph(on_stale_key: str, on_stale_reason: str):
    """Return a _get_graph replacement that fires on_stale and returns a usable Graph."""

    def _patched(input_path, host, port, on_stale=None):
        if on_stale is not None:
            on_stale(on_stale_key, on_stale_reason)
        return _fake_graph()

    return _patched


class TestStaleEnvelope:
    def test_slots_surfaces_stale_in_envelope(self, monkeypatch, tmp_path, capsys):
        """workflow slots must inject stale=True + warnings when graph came from stale cache."""
        monkeypatch.setattr(workflow_cmd, "_get_graph", _stale_get_graph("cloud:127.0.0.1:8188", "connection refused"))
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["slots", str(path)], capsys)
        assert env["ok"] is True
        assert env["data"].get("stale") is True, "stale flag must be set in data payload"
        warnings = env["data"].get("warnings") or []
        assert any(w.get("code") == "object_info_stale" for w in warnings), (
            f"expected a warning with code='object_info_stale', got {warnings}"
        )

    def test_set_slot_surfaces_stale_in_envelope(self, monkeypatch, tmp_path, capsys):
        """workflow set-slot must inject stale=True + warnings when graph came from stale cache."""
        monkeypatch.setattr(workflow_cmd, "_get_graph", _stale_get_graph("local:127.0.0.1:8188", "timeout"))
        path = _write_workflow(tmp_path, _direct_workflow())
        env = _run(["set-slot", str(path), '6.text="new prompt"'], capsys)
        assert env["ok"] is True
        assert env["data"].get("stale") is True, "stale flag must be set in data payload"
        warnings = env["data"].get("warnings") or []
        assert any(w.get("code") == "object_info_stale" for w in warnings), (
            f"expected a warning with code='object_info_stale', got {warnings}"
        )

    def test_vary_surfaces_stale_in_envelope(self, monkeypatch, tmp_path, capsys):
        """workflow vary must inject stale=True + warnings when graph came from stale cache."""
        monkeypatch.setattr(workflow_cmd, "_get_graph", _stale_get_graph("local:127.0.0.1:8188", "boom"))
        path = _write_workflow(tmp_path, _direct_workflow())
        out_dir = tmp_path / "out"
        env = _run(
            ["vary", str(path), "--slot", "3.seed=[1,2,3]", "--out-dir", str(out_dir)],
            capsys,
        )
        assert env["ok"] is True
        assert env["data"].get("stale") is True, "stale flag must be set in data payload"
        warnings = env["data"].get("warnings") or []
        assert any(w.get("code") == "object_info_stale" for w in warnings), (
            f"expected a warning with code='object_info_stale', got {warnings}"
        )
