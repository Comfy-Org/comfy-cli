"""Scenarios for `comfy workflow add-node/connect/set-widget/delete` — the
CRDT-ready structured-edit primitives.

These are the observation layer for the red→green loop. Each command must:
  * mutate a frontend-format workflow file (or --stdout), and
  * emit a structured, CRDT-mergeable op in the envelope's `data.op`.

The op-model correctness properties (fidelity / idempotency / convergence /
conflict-detection / name-safety / api-validity) are exercised directly against
`comfy_cli.workflow_ops`, which the commands wrap.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.command import workflow_edit
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


def _object_info() -> dict[str, Any]:
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "category": "loaders",
            "display_name": "Load Checkpoint",
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": "CLIP"}},
            "input_order": {"required": ["clip", "text"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode",
            "python_module": "nodes",
        },
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
                    "sampler_name": [["euler", "euler_ancestral"]],
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
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                }
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "latent",
            "display_name": "Empty Latent Image",
            "python_module": "nodes",
        },
        "VAEDecode": {
            "input": {"required": {"samples": "LATENT", "vae": "VAE"}},
            "input_order": {"required": ["samples", "vae"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "VAE Decode",
            "python_module": "nodes",
        },
        "KlingFLFTest": {
            "input": {
                "required": {
                    "first_frame": "IMAGE",
                    "last_frame": "IMAGE",
                    "prompt": ["STRING", {"default": ""}],
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {
                                    "key": "kling-v3",
                                    "inputs": {
                                        "required": {
                                            "resolution": ["COMBO", {"default": "1080p", "options": ["4k", "1080p", "720p"]}]
                                        }
                                    },
                                }
                            ]
                        },
                    ],
                }
            },
            "input_order": {"required": ["first_frame", "last_frame", "prompt", "model"]},
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
            "category": "video",
            "display_name": "Kling FLF (test)",
            "python_module": "nodes",
        },
        "PrimitiveFloat": {
            "input": {"required": {"value": ["FLOAT", {"default": 1.0}]}},
            "input_order": {"required": ["value"]},
            "output": ["FLOAT"],
            "output_name": ["FLOAT"],
            "category": "primitive",
            "display_name": "Float",
            "python_module": "nodes",
        },
        "BatchImagesNode": {
            "input": {"required": {"images": "COMFY_AUTOGROW_V3"}},
            "input_order": {"required": ["images"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "image/batch",
            "display_name": "Batch Images",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _base_workflow() -> dict:
    """A minimal but wired frontend-format graph: EmptyLatentImage -> KSampler."""
    return {
        "last_node_id": 7,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 3,
                "type": "KSampler",
                "pos": [100, 100],
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "positive", "type": "CONDITIONING", "link": None},
                    {"name": "negative", "type": "CONDITIONING", "link": None},
                    {"name": "latent_image", "type": "LATENT", "link": 1},
                ],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
            },
            {
                "id": 7,
                "type": "EmptyLatentImage",
                "pos": [0, 0],
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}],
                "widgets_values": [512, 512, 1],
            },
        ],
        "links": [[1, 7, 0, 3, 3, "LATENT"]],
    }


def _autogrow_workflow() -> dict:
    """A BatchImagesNode (autogrow `images` input) plus two IMAGE sources, so
    two connects can race onto the same autogrow base."""
    return {
        "last_node_id": 21,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 10,
                "type": "BatchImagesNode",
                "pos": [200, 0],
                "inputs": [{"name": "images", "type": "COMFY_AUTOGROW_V3", "link": None}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
            {
                "id": 20,
                "type": "VAEDecode",
                "pos": [0, 0],
                "inputs": [{"name": "samples", "type": "LATENT", "link": None}, {"name": "vae", "type": "VAE", "link": None}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
            {
                "id": 21,
                "type": "VAEDecode",
                "pos": [0, 100],
                "inputs": [{"name": "samples", "type": "LATENT", "link": None}, {"name": "vae", "type": "VAE", "link": None}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
        ],
        "links": [],
    }


def _two_instance_subgraph_workflow() -> dict:
    """Two top-level instances (57, 58) of ONE shared subgraph definition, so an
    interior write must fork the shared def before mutating it."""
    wf = _subgraph_workflow()
    inst57 = next(n for n in wf["nodes"] if n["id"] == 57)
    inst58 = copy.deepcopy(inst57)
    inst58["id"] = 58
    inst58["pos"] = [400, 0]
    wf["nodes"].append(inst58)
    wf["last_node_id"] = 58
    return wf


def _write(tmp_path: Path, data: dict, name: str = "wf.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


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
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


# ---------------------------------------------------------------------------
# add-node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_adds_node_and_emits_op(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "VAEDecode"], capsys)
        assert env["ok"] is True
        op = env["data"]["op"]
        assert op["op"] == "add_node"
        assert op["class_type"] == "VAEDecode"
        assert isinstance(op["op_id"], str) and op["op_id"]
        nid = op["node_id"]
        on_disk = json.loads(path.read_text())
        node = next(n for n in on_disk["nodes"] if n["id"] == nid)
        assert node["type"] == "VAEDecode"
        # inputs/outputs materialized from object_info so it is connectable
        assert {i["name"] for i in node["inputs"]} == {"samples", "vae"}
        assert [o["name"] for o in node["outputs"]] == ["IMAGE"]

    def test_combo_widgets_default_to_first_choice(self, patched_graph, tmp_path, capsys):
        """A fresh node must not leave COMBO widgets null (would fail at runtime)."""
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "KSampler"], capsys)
        nid = env["data"]["op"]["node_id"]
        node = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == nid)
        assert None not in node["widgets_values"], node["widgets_values"]
        assert "euler" in node["widgets_values"]  # sampler_name first choice
        assert "normal" in node["widgets_values"]  # scheduler first choice

    def test_where_flag_threads_to_catalog_resolver(self, monkeypatch, tmp_path, capsys):
        captured: dict = {}

        def fake_get_graph(input_path, host, port, where=None):
            captured["where"] = where
            return _graph()

        monkeypatch.setattr(workflow_edit, "_get_graph", fake_get_graph)
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "VAEDecode", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert captured["where"] == "cloud"

    def test_ids_are_large_ints_and_collision_free(self, patched_graph, tmp_path, capsys):
        """CRDT identity: leaderless, collision-free, int-typed (converter-safe)."""
        path = _write(tmp_path, _base_workflow())
        env1 = _run(["add-node", str(path), "VAEDecode"], capsys)
        env2 = _run(["add-node", str(path), "VAEDecode"], capsys)
        id1, id2 = env1["data"]["op"]["node_id"], env2["data"]["op"]["node_id"]
        assert isinstance(id1, int) and isinstance(id2, int)
        assert id1 != id2
        # Not a small sequential counter value — minted from a large space.
        assert id1 > 10_000 and id2 > 10_000


# ---------------------------------------------------------------------------
# set-widget (name-addressed)
# ---------------------------------------------------------------------------


class TestSetWidget:
    def test_sets_widget_by_name_and_records_old_new(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "3.steps", "35"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_widget"
        assert op["node_id"] == 3
        assert op["widget"] == "steps"
        assert op["value"] == 35
        assert op["old"] == 20
        on_disk = json.loads(path.read_text())
        ks = next(n for n in on_disk["nodes"] if n["id"] == 3)
        # widget_order: seed, control_after_generate, steps -> index 2
        assert ks["widgets_values"][2] == 35

    def test_unknown_widget_errors(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "3.nope", "1"], capsys)
        assert env["ok"] is False

    def test_shape_mismatch_rejected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "3.steps", "notanumber"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"

    def test_error_envelope_command_is_qualified(self, patched_graph, tmp_path, capsys):
        """The error envelope's `command` must match the success one (not bare `workflow`)."""
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "3.nope", "1"], capsys)
        assert env["command"] == "workflow set-widget"


# ---------------------------------------------------------------------------
# set-widget on SUBGRAPH-based templates (the modern gallery templates)
#
# Modern ComfyUI templates wrap their real nodes inside a subgraph *instance*
# (a top-level node whose `type` is a subgraph UUID). `slots` advertises the
# instance's promoted inputs as flat `<instance>.<input>` addresses (e.g.
# `57.text`). set-widget MUST accept the SAME address slots emits — descending
# into the subgraph definition and writing the interior node's widget — plus the
# nested `<instance>/<inner>.<input>` form the skill documents.
# ---------------------------------------------------------------------------


_SG_UUID = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"


def _subgraph_workflow() -> dict:
    """A minimal curated subgraph template (derived from a fetched gallery
    template): a top-level subgraph instance `57` whose promoted `text`/`seed`/
    `steps` route through `proxyWidgets` to interior CLIPTextEncode `27` and
    KSampler `3`, plus a plain top-level node `9` so we can prove top-level edits
    still work alongside subgraph edits."""
    return {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 57,
                "type": _SG_UUID,
                "pos": [0, 0],
                "inputs": [],
                "outputs": [],
                "properties": {"proxyWidgets": [["27", "text"], ["3", "seed"], ["3", "steps"]]},
            },
            {
                "id": 9,
                "type": "EmptyLatentImage",
                "pos": [10, 10],
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                "widgets_values": [512, 512, 1],
            },
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": _SG_UUID,
                    "name": "Text to Image",
                    "inputs": [
                        {"name": "text", "type": "STRING"},
                        {"name": "seed", "type": "INT"},
                        {"name": "steps", "type": "INT"},
                    ],
                    "nodes": [
                        {"id": 27, "type": "CLIPTextEncode", "widgets_values": ["old prompt"]},
                        {"id": 3, "type": "KSampler", "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0]},
                    ],
                }
            ]
        },
    }


def _interior(wf: dict, inner_id) -> dict:
    sg = wf["definitions"]["subgraphs"][0]
    return next(n for n in sg["nodes"] if str(n["id"]) == str(inner_id))


class TestSetWidgetSubgraph:
    def test_flat_promoted_address_writes_interior_node(self, patched_graph, tmp_path, capsys):
        """`57.text` — the exact address `slots` advertises — writes CLIPTextEncode 27."""
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57.text", "a cat on a bicycle"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_widget"
        assert op["node_id"] == 57
        assert op["value"] == "a cat on a bicycle"
        assert op["old"] == "old prompt"
        # op is self-describing + replayable: resolved interior path + widget.
        assert op["path"] == ["57", "27"]
        assert op["inner_widget"] == "text"
        # CRDT stamping preserved.
        assert isinstance(op["op_id"], str) and op["op_id"]
        assert op["stamp"] == [0, "cli"]
        # value landed on the interior node, in the definition (persists on disk).
        wf = json.loads(path.read_text())
        assert _interior(wf, 27)["widgets_values"][0] == "a cat on a bicycle"

    def test_flat_promoted_int_input_writes_ksampler(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57.seed", "12345"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["op"]["path"] == ["57", "3"]
        wf = json.loads(path.read_text())
        assert _interior(wf, 3)["widgets_values"][0] == 12345  # seed is index 0

    def test_nested_interior_address_writes_interior_node(self, patched_graph, tmp_path, capsys):
        """`57/27.text` (the nested form the skill documents) hits the same widget."""
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57/27.text", "a nested cat"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["node_id"] == "57/27"
        assert op["path"] == ["57", "27"]
        assert op["inner_widget"] == "text"
        wf = json.loads(path.read_text())
        assert _interior(wf, 27)["widgets_values"][0] == "a nested cat"

    def test_flat_and_nested_share_a_conflict_target(self):
        """Flat `57.text` and nested `57/27.text` land on the same interior
        widget, so their ops must resolve to the SAME CRDT write target (they
        converge — one does not silently clobber the other undetected)."""
        wf = _subgraph_workflow()
        _, flat = workflow_ops.set_widget(copy.deepcopy(wf), _graph(), 57, "text", "A")
        _, nested = workflow_ops.set_widget(copy.deepcopy(wf), _graph(), "57/27", "text", "B")
        assert workflow_ops._write_target(flat) == workflow_ops._write_target(nested)
        assert workflow_ops.detect_conflict(flat, nested) is True  # different values, same target

    def test_slots_and_set_widget_agree(self, patched_graph, monkeypatch, tmp_path, capsys):
        """The self-consistency the bug broke: every flat address `slots` emits
        for the subgraph instance is accepted by set-widget."""
        # slots resolves its graph via workflow.py's _get_graph; set-widget via
        # workflow_edit.py's. patched_graph covers the latter; patch the former too.
        monkeypatch.setattr(workflow_cmd, "_get_graph", lambda *a, **kw: _graph())
        path = _write(tmp_path, _subgraph_workflow())
        slots_env = _run(["slots", str(path)], capsys)
        addrs = [s["address"] for s in slots_env["data"]["slots"] if str(s["address"]).startswith("57.")]
        assert addrs, slots_env  # the instance's promoted inputs are advertised flat
        for addr in ("57.text", "57.seed", "57.steps"):
            assert addr in addrs
            env = _run(["set-widget", str(path), addr, "3" if addr != "57.text" else "x"], capsys)
            assert env["ok"] is True, (addr, env)

    def test_unknown_promoted_input_errors_cleanly(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57.nope", "1"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert "not found on subgraph node 57" in env["error"]["message"]

    def test_type_mismatch_on_promoted_int_rejected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57.seed", '"notanumber"'], capsys)
        assert env["ok"] is False
        # unchanged on disk (edit rejected before write).
        wf = json.loads(path.read_text())
        assert _interior(wf, 3)["widgets_values"][0] == 42

    def test_top_level_edit_still_works_with_subgraphs_present(self, patched_graph, tmp_path, capsys):
        """A plain top-level node in a workflow that also contains subgraphs is
        still edited directly (no regression)."""
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "9.width", "768"], capsys)
        assert env["ok"] is True, env
        assert "path" not in env["data"]["op"]  # direct top-level op, not a subgraph op
        wf = json.loads(path.read_text())
        node9 = next(n for n in wf["nodes"] if n["id"] == 9)
        assert node9["widgets_values"][0] == 768


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connects_and_wires_slots(self, patched_graph, tmp_path, capsys):
        wf = _base_workflow()
        path = _write(tmp_path, wf)
        # add a VAEDecode then connect KSampler.LATENT -> VAEDecode.samples
        add = _run(["add-node", str(path), "VAEDecode"], capsys)
        vae_id = add["data"]["op"]["node_id"]
        env = _run(["connect", str(path), "3.LATENT", f"{vae_id}.samples"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "connect"
        link_id = op["link_id"]
        on_disk = json.loads(path.read_text())
        link = next(ln for ln in on_disk["links"] if ln[0] == link_id)
        assert link[1] == 3 and link[3] == vae_id  # from KSampler -> to VAEDecode
        vae = next(n for n in on_disk["nodes"] if n["id"] == vae_id)
        samples = next(i for i in vae["inputs"] if i["name"] == "samples")
        assert samples["link"] == link_id

    def test_autogrow_input_grows_a_slot_per_connection(self, patched_graph, tmp_path, capsys):
        """COMFY_AUTOGROW inputs (BatchImagesNode.images) grow images.image0/1… — the
        assembly wiring the CRDT/apply path needs for video."""
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        a = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        b = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        batch = _run(["add-node", str(path), "BatchImagesNode"], capsys)["data"]["op"]["node_id"]
        e1 = _run(["connect", str(path), f"{a}.IMAGE", f"{batch}.images"], capsys)
        e2 = _run(["connect", str(path), f"{b}.IMAGE", f"{batch}.images"], capsys)
        assert e1["ok"] and e2["ok"], (e1, e2)
        assert e1["data"]["op"]["grow"]["name"] == "images.image0"
        assert e2["data"]["op"]["grow"]["name"] == "images.image1"
        bn = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == batch)
        grown = [i for i in bn["inputs"] if i["name"].startswith("images.image")]
        assert {i["name"] for i in grown} == {"images.image0", "images.image1"}
        assert all(i["link"] is not None and i["type"] == "IMAGE" for i in grown)

    def test_connect_converts_widget_to_input(self, patched_graph, tmp_path, capsys):
        """connect onto a widget-backed input (KSampler.cfg) converts it to a link;
        widgets_values stays intact and the converter uses the link (fps-style wiring)."""
        path = _write(tmp_path, _base_workflow())  # KSampler id 3, widgets len 7
        src = _run(["add-node", str(path), "PrimitiveFloat"], capsys)["data"]["op"]["node_id"]
        env = _run(["connect", str(path), f"{src}.FLOAT", "3.cfg"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["op"]["grow"] == {"name": "cfg", "type": "FLOAT", "widget": "cfg"}
        wf = json.loads(path.read_text())
        ks = next(n for n in wf["nodes"] if n["id"] == 3)
        cfg_in = next(i for i in ks["inputs"] if i["name"] == "cfg")
        assert cfg_in["link"] is not None and cfg_in["widget"] == {"name": "cfg"}
        assert len(ks["widgets_values"]) == 7  # value kept → positional alignment holds

        from comfy_cli.workflow_to_api import convert_ui_to_api

        api = convert_ui_to_api(wf, _object_info())
        assert api["3"]["inputs"]["cfg"] == [str(src), 0]  # cfg is a link now
        assert api["3"]["inputs"]["steps"] == 20  # other widgets still aligned

    def test_type_mismatch_rejected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        add = _run(["add-node", str(path), "VAEDecode"], capsys)
        vae_id = add["data"]["op"]["node_id"]
        # KSampler LATENT output cannot feed a VAE-typed input.
        env = _run(["connect", str(path), "3.LATENT", f"{vae_id}.vae"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"

    def test_replacing_input_link_scrubs_the_old_one(self, patched_graph, tmp_path, capsys):
        """Re-wiring an occupied input must retire the previous link, not orphan it."""
        path = _write(tmp_path, _base_workflow())
        # KSampler.latent_image already holds link 1 (from EmptyLatentImage 7).
        add = _run(["add-node", str(path), "EmptyLatentImage"], capsys)
        new_src = add["data"]["op"]["node_id"]
        env = _run(["connect", str(path), f"{new_src}.LATENT", "3.latent_image"], capsys)
        assert env["ok"] is True, env
        new_link = env["data"]["op"]["link_id"]
        wf = json.loads(path.read_text())
        # old link 1 is gone entirely; only the new link references latent_image
        assert all(ln[0] != 1 for ln in wf["links"])
        ks = next(n for n in wf["nodes"] if n["id"] == 3)
        assert next(i for i in ks["inputs"] if i["name"] == "latent_image")["link"] == new_link
        # old source's out-links no longer carry the retired link
        old_src = next(n for n in wf["nodes"] if n["id"] == 7)
        assert 1 not in (old_src["outputs"][0]["links"] or [])


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_deletes_node_and_incident_links(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["delete-node", str(path), "7"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "delete_node"
        assert op["node_id"] == 7
        on_disk = json.loads(path.read_text())
        assert all(n["id"] != 7 for n in on_disk["nodes"])
        # link 1 (7 -> 3) must be gone, and KSampler.latent_image cleared
        assert all(ln[1] != 7 and ln[3] != 7 for ln in on_disk["links"])
        ks = next(n for n in on_disk["nodes"] if n["id"] == 3)
        latent = next(i for i in ks["inputs"] if i["name"] == "latent_image")
        assert latent["link"] is None


    def test_nested_subgraph_address_missing_interior_node_errors(self, patched_graph, tmp_path, capsys):
        # A nested address into a graph with no such subgraph/interior node fails
        # cleanly (the top-level workflow here has no subgraph instance 10).
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "10/9.prompt", "x"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"


# ---------------------------------------------------------------------------
# invariant: the edit surface operates on UI (frontend) format ONLY
# (API format is a throwaway produced only at `run`)
# ---------------------------------------------------------------------------


class TestUiFormatOnly:
    _API = {"3": {"class_type": "KSampler", "inputs": {}}}  # API format: dict keyed by ids

    def test_add_node_rejects_api_format(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, self._API)
        env = _run(["add-node", str(path), "VAEDecode"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_frontend_format"

    def test_apply_rejects_api_format(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, self._API)
        ops = tmp_path / "ops.json"
        ops.write_text("[]", encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(ops)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_frontend_format"

    def test_set_widget_rejects_api_format(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, self._API)
        env = _run(["set-widget", str(path), "3.steps", "20"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_frontend_format"


# ---------------------------------------------------------------------------
# ls-nodes
# ---------------------------------------------------------------------------


class TestLsNodes:
    def test_lists_nodes(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["ls-nodes", str(path)], capsys)
        assert env["ok"] is True
        ids = {n["id"] for n in env["data"]["nodes"]}
        assert ids == {3, 7}
        types = {n["type"] for n in env["data"]["nodes"]}
        assert types == {"KSampler", "EmptyLatentImage"}


# ---------------------------------------------------------------------------
# apply — batch with aliases
# ---------------------------------------------------------------------------


class TestApplyBatch:
    def _empty(self, tmp_path):
        return _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})

    def test_builds_graph_in_one_pass_with_aliases(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        specs = [
            {"op": "add_node", "class_type": "CheckpointLoaderSimple", "as": "ckpt"},
            {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},
            {"op": "add_node", "class_type": "EmptyLatentImage", "as": "lat"},
            {"op": "connect", "from": "ckpt.MODEL", "to": "ks.model"},
            {"op": "connect", "from": "ckpt.CLIP", "to": "pos.clip"},
            {"op": "connect", "from": "pos.CONDITIONING", "to": "ks.positive"},
            {"op": "connect", "from": "lat.LATENT", "to": "ks.latent_image"},
            {"op": "set_widget", "node": "pos", "widget": "text", "value": "a cat"},
            {"op": "set_widget", "node": "ks", "widget": "steps", "value": 30},
        ]
        ops_path = tmp_path / "ops.json"
        ops_path.write_text(json.dumps(specs), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(ops_path)], capsys)
        assert env["ok"] is True, env
        assert env["data"]["count"] == 10
        assert set(env["data"]["aliases"]) == {"ckpt", "pos", "ks", "lat"}
        # the aliased KSampler really got wired
        from comfy_cli.workflow_to_api import convert_ui_to_api

        api = convert_ui_to_api(json.loads(path.read_text()), _object_info())
        ks_id = str(env["data"]["aliases"]["ks"])
        assert ks_id in api
        assert api[ks_id]["inputs"]["model"][0] == str(env["data"]["aliases"]["ckpt"])

    def test_batch_is_atomic_on_failure(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        before = path.read_text()
        specs = [
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},
            {"op": "add_node", "class_type": "NoSuchNode"},  # fails
        ]
        ops_path = tmp_path / "ops.json"
        ops_path.write_text(json.dumps(specs), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(ops_path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert path.read_text() == before, "failed batch must not write a partial graph"


# ---------------------------------------------------------------------------
# dynamic combo (COMFY_DYNAMICCOMBO_V3) — set_widget on model + model.resolution
# ---------------------------------------------------------------------------


class TestDynamicCombo:
    def test_add_node_fills_dynamiccombo_defaults(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        nid = _run(["add-node", str(path), "KlingFLFTest"], capsys)["data"]["op"]["node_id"]
        g = _graph()
        node = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == nid)
        order = g.widget_order("KlingFLFTest")
        assert "model" in order and "model.resolution" in order
        wv = node["widgets_values"]
        assert wv[order.index("model")] == "kling-v3"  # first key
        assert wv[order.index("model.resolution")] == "1080p"  # sub default

    def test_set_widget_dynamiccombo_selector_and_sub(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        nid = _run(["add-node", str(path), "KlingFLFTest"], capsys)["data"]["op"]["node_id"]
        e1 = _run(["set-widget", str(path), f"{nid}.model", "kling-v3"], capsys)
        e2 = _run(["set-widget", str(path), f"{nid}.model.resolution", "720p"], capsys)
        assert e1["ok"] and e2["ok"], (e1, e2)
        g = _graph()
        order = g.widget_order("KlingFLFTest")
        wv = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == nid)["widgets_values"]
        assert wv[order.index("model.resolution")] == "720p"

        from comfy_cli.workflow_to_api import convert_ui_to_api

        api = convert_ui_to_api(json.loads(path.read_text()), _object_info())
        assert api[str(nid)]["inputs"]["model"] == "kling-v3"
        assert api[str(nid)]["inputs"]["model.resolution"] == "720p"


# ---------------------------------------------------------------------------
# recipes — parameterized op-batches (apply --param)
# ---------------------------------------------------------------------------


class TestRecipes:
    def _recipe(self):
        return {
            "recipe": "t2i",
            "params": {"positive": {"type": "string"}, "steps": {"type": "int", "default": 20}},
            "ops": [
                {"op": "add_node", "class_type": "KSampler", "as": "ks"},
                {"op": "set_widget", "node": "ks", "widget": "steps", "value": "${steps}"},
                {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
                {"op": "set_widget", "node": "pos", "widget": "text", "value": "a ${positive} scene"},
            ],
        }

    def _empty(self, tmp_path):
        return _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})

    def test_param_substitution_is_typed(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(self._recipe()), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(rp), "--param", "positive=quiet forest", "--param", "steps=35"], capsys)
        assert env["ok"] is True, env
        wf = json.loads(path.read_text())
        g = _graph()
        ks = next(n for n in wf["nodes"] if n["type"] == "KSampler")
        pos = next(n for n in wf["nodes"] if n["type"] == "CLIPTextEncode")
        assert ks["widgets_values"][g.widget_order("KSampler").index("steps")] == 35  # int, not "35"
        assert pos["widgets_values"][g.widget_order("CLIPTextEncode").index("text")] == "a quiet forest scene"

    def test_default_used_when_param_omitted(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(self._recipe()), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(rp), "--param", "positive=x"], capsys)
        assert env["ok"] is True
        wf = json.loads(path.read_text())
        g = _graph()
        ks = next(n for n in wf["nodes"] if n["type"] == "KSampler")
        assert ks["widgets_values"][g.widget_order("KSampler").index("steps")] == 20  # declared default

    def test_missing_required_param_errors(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(self._recipe()), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(rp)], capsys)  # positive omitted, no default
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert "positive" in env["error"]["message"]

    def test_unknown_param_errors(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(self._recipe()), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(rp), "--param", "positive=x", "--param", "nope=1"], capsys)
        assert env["ok"] is False
        assert "nope" in env["error"]["message"]

    def test_bad_type_errors(self, patched_graph, tmp_path, capsys):
        path = self._empty(tmp_path)
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(self._recipe()), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(rp), "--param", "positive=x", "--param", "steps=notanint"], capsys)
        assert env["ok"] is False
        assert "int" in env["error"]["message"]


# ---------------------------------------------------------------------------
# foreach — bulk-instantiate a recipe over N param-sets
# ---------------------------------------------------------------------------


class TestForeach:
    def _recipe(self, tmp_path):
        rp = tmp_path / "r.json"
        rp.write_text(
            json.dumps(
                {
                    "recipe": "t2i",
                    "params": {"positive": {"type": "string"}, "steps": {"type": "int", "default": 20}},
                    "ops": [
                        {"op": "add_node", "class_type": "KSampler", "as": "ks"},
                        {"op": "set_widget", "node": "ks", "widget": "steps", "value": "${steps}"},
                        {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
                        {"op": "set_widget", "node": "pos", "widget": "text", "value": "${positive}"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return rp

    def test_foreach_materializes_one_workflow_per_param_set(self, patched_graph, tmp_path, capsys):
        rp = self._recipe(tmp_path)
        params = tmp_path / "sets.jsonl"
        params.write_text('{"positive":"a cat","steps":10}\n{"positive":"a dog","steps":30}\n', encoding="utf-8")
        out = tmp_path / "out"
        env = _run(["foreach", str(rp), "--params", str(params), "--out-dir", str(out)], capsys)
        assert env["ok"] is True, env
        assert env["data"]["count"] == 2
        files = sorted(out.glob("*.json"))
        assert len(files) == 2
        g = _graph()
        seen = []
        for f in files:
            wf = json.loads(f.read_text())
            pos = next(n for n in wf["nodes"] if n["type"] == "CLIPTextEncode")
            ks = next(n for n in wf["nodes"] if n["type"] == "KSampler")
            seen.append(
                (
                    pos["widgets_values"][g.widget_order("CLIPTextEncode").index("text")],
                    ks["widgets_values"][g.widget_order("KSampler").index("steps")],
                )
            )
        assert seen == [("a cat", 10), ("a dog", 30)]  # each param-set → its own workflow

    def test_foreach_accepts_json_array(self, patched_graph, tmp_path, capsys):
        rp = self._recipe(tmp_path)
        params = tmp_path / "sets.json"
        params.write_text(json.dumps([{"positive": "x"}, {"positive": "y"}, {"positive": "z"}]), encoding="utf-8")
        out = tmp_path / "out"
        env = _run(["foreach", str(rp), "--params", str(params), "--out-dir", str(out)], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 3  # steps uses the default

    def test_foreach_bad_param_set_fails(self, patched_graph, tmp_path, capsys):
        rp = self._recipe(tmp_path)
        params = tmp_path / "sets.jsonl"
        params.write_text('{"steps":10}\n', encoding="utf-8")  # missing required positive
        out = tmp_path / "out"
        env = _run(["foreach", str(rp), "--params", str(params), "--out-dir", str(out)], capsys)
        assert env["ok"] is False
        assert "positive" in env["error"]["message"]


# ---------------------------------------------------------------------------
# capture — project a graph into a recipe; round-trips through apply
# ---------------------------------------------------------------------------


class TestCapture:
    def test_capture_roundtrips_through_apply(self, patched_graph, tmp_path, capsys):
        src = _write(tmp_path, _base_workflow())
        cap = _run(["capture", str(src), "--name", "base"], capsys)
        assert cap["ok"] is True, cap
        recipe = cap["data"]["recipe_doc"]
        # a non-default widget (KSampler seed=42) is captured; defaults are not
        assert any(o["op"] == "set_widget" and o["widget"] == "seed" and o["value"] == 42 for o in recipe["ops"])
        assert not any(o.get("widget") == "steps" for o in recipe["ops"])  # steps=20 is the default

        empty = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}, "empty.json")
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(recipe), encoding="utf-8")
        applied = _run(["apply", str(empty), "--ops", str(rp)], capsys)
        assert applied["ok"] is True, applied

        rebuilt = json.loads(empty.read_text())
        orig = _base_workflow()
        assert sorted(n["type"] for n in rebuilt["nodes"]) == sorted(n["type"] for n in orig["nodes"])
        assert len(rebuilt["links"]) == len(orig["links"])
        g = _graph()
        ks = next(n for n in rebuilt["nodes"] if n["type"] == "KSampler")
        assert ks["widgets_values"][g.widget_order("KSampler").index("seed")] == 42  # preserved

        from comfy_cli.workflow_to_api import convert_ui_to_api

        api = convert_ui_to_api(rebuilt, _object_info())
        ks_api = next(v for v in api.values() if v["class_type"] == "KSampler")
        assert isinstance(ks_api["inputs"].get("latent_image"), list)  # wiring preserved

    def test_capture_lifts_widget_to_param_even_at_default(self, patched_graph, tmp_path, capsys):
        """`--param` promotes a widget to a ${param} hole even when its value is the
        node default (the footgun: capture would otherwise drop it)."""
        # EmptyLatentImage.width=512 IS the default → normally not captured.
        src = _write(tmp_path, _base_workflow())
        lat_id = next(n["id"] for n in _base_workflow()["nodes"] if n["type"] == "EmptyLatentImage")
        cap = _run(["capture", str(src), "--param", f"{lat_id}.width=w"], capsys)
        assert cap["ok"] is True, cap
        recipe = cap["data"]["recipe_doc"]
        assert "w" in recipe["params"] and recipe["params"]["w"]["type"] == "int"
        assert any(o["op"] == "set_widget" and o.get("value") == "${w}" for o in recipe["ops"])
        # and it applies with an override
        empty = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}, "e.json")
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(recipe), encoding="utf-8")
        env = _run(["apply", str(empty), "--ops", str(rp), "--param", "w=768"], capsys)
        assert env["ok"] is True, env
        g = _graph()
        lat = next(n for n in json.loads(empty.read_text())["nodes"] if n["type"] == "EmptyLatentImage")
        assert lat["widgets_values"][g.widget_order("EmptyLatentImage").index("width")] == 768

    def test_capture_param_rejects_unknown_target(self, patched_graph, tmp_path, capsys):
        src = _write(tmp_path, _base_workflow())
        env = _run(["capture", str(src), "--param", "3.nope=x"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"

    def test_capture_rejects_subgraphs(self, patched_graph, tmp_path, capsys):
        wf = _base_workflow()
        wf["definitions"] = {"subgraphs": [{"id": "sg", "name": "x", "nodes": []}]}
        path = _write(tmp_path, wf)
        env = _run(["capture", str(path)], capsys)
        assert env["ok"] is False
        assert "subgraph" in env["error"]["message"].lower()


# ---------------------------------------------------------------------------
# op-model correctness — direct against workflow_ops (P1..P7)
# ---------------------------------------------------------------------------


class TestOpModel:
    def _ops(self):
        from comfy_cli import workflow_ops

        return workflow_ops

    def test_p1_fidelity_apply_equals_primitive(self):
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        for make in (
            lambda w: ops.add_node(w, g, "VAEDecode"),
            lambda w: ops.set_widget(w, g, 3, "steps", 33),
            lambda w: ops.delete_node(w, g, 7),
        ):
            direct, op = make(copy.deepcopy(base))
            replayed = ops.apply_op(copy.deepcopy(base), op, g)
            assert ops.canonical(replayed) == ops.canonical(direct)

    def test_p2_idempotency(self):
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, op = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 33)
        once = ops.apply_op(copy.deepcopy(base), op, g)
        twice = ops.apply_op(ops.apply_op(copy.deepcopy(base), op, g), op, g)
        assert ops.canonical(once) == ops.canonical(twice)

    def test_p3_convergence_nonoverlapping(self):
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, op_add = ops.add_node(copy.deepcopy(base), g, "VAEDecode", actor="agent")
        _, op_set = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 50, actor="human")
        ab = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_add, g), op_set, g)
        ba = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_set, g), op_add, g)
        assert ops.canonical(ab) == ops.canonical(ba)

    def test_p4_conflict_detection(self):
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, a = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 10, actor="human")
        _, b = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 20, actor="agent")
        _, c = ops.set_widget(copy.deepcopy(base), g, 3, "cfg", 7.0, actor="agent")
        assert ops.detect_conflict(a, b) is True
        assert ops.detect_conflict(a, c) is False

    def test_p5_widget_name_safe_under_layout_shift(self):
        """Name-addressed op survives a concurrent widget-layout shift."""
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, op = ops.set_widget(copy.deepcopy(base), g, 3, "denoise", 0.5, actor="agent")
        # Simulate another peer prepending a widget slot on the same node
        # (indices all shift by one); a name-keyed op must still hit denoise.
        shifted = copy.deepcopy(base)
        ks = next(n for n in shifted["nodes"] if n["id"] == 3)
        ks["widgets_values"] = ["INJECTED", *ks["widgets_values"]]
        # apply must re-resolve by name, not blindly by the original index
        out = ops.apply_op(shifted, op, g)
        ks_out = next(n for n in out["nodes"] if n["id"] == 3)
        order = g.widget_order("KSampler")
        assert ks_out["widgets_values"][order.index("denoise")] == 0.5
        assert ks_out["widgets_values"][0] == "INJECTED"

    def test_p7_api_convert_valid(self):
        ops = self._ops()
        from comfy_cli.workflow_to_api import convert_ui_to_api

        g = _graph()
        wf = _base_workflow()
        wf, _ = ops.add_node(wf, g, "VAEDecode")
        vae_id = next(n["id"] for n in wf["nodes"] if n["type"] == "VAEDecode")
        wf, _ = ops.connect(wf, g, 3, "LATENT", vae_id, "samples")
        api = convert_ui_to_api(wf, _object_info())
        assert isinstance(api, dict)
        # the added VAEDecode survives conversion with an int-keyed id
        assert str(vae_id) in api

    # -- convergence properties (P8..P11): a merge consumer replays ops in any
    #    order; apply must be TOTAL (never crash) and ORDER-INDEPENDENT (both
    #    orders reach the same canonical graph). One property per convergence
    #    bug the deep review flagged.

    def test_p8_totality_delete_wins_over_concurrent_edit(self):
        """A write to a concurrently-deleted node is a no-op (delete wins),
        never a crash. {delete(3), set_widget(3)} converges in either order."""
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, op_del = ops.delete_node(copy.deepcopy(base), g, 3, actor="human")
        _, op_set = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 99, actor="agent")
        # neither order may raise; both must converge to "node 3 gone".
        del_then_set = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_del, g), op_set, g)
        set_then_del = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_set, g), op_del, g)
        assert ops.canonical(del_then_set) == ops.canonical(set_then_del)
        assert all(n["id"] != 3 for n in del_then_set["nodes"])

    def test_p8_totality_connect_to_deleted_node_is_noop(self):
        """A connect whose endpoint was concurrently deleted no-ops without
        crashing or leaving a dangling link."""
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        # wire EmptyLatentImage(7).LATENT -> KSampler(3).latent_image, then race a
        # delete of the source node 7.
        _, op_conn = ops.connect(copy.deepcopy(base), g, 7, "LATENT", 3, "latent_image", actor="agent")
        _, op_del = ops.delete_node(copy.deepcopy(base), g, 7, actor="human")
        out = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_del, g), op_conn, g)
        assert all(n["id"] != 7 for n in out["nodes"])
        # no link may reference the deleted node 7 (as source or target).
        assert all(ln[1] != 7 and ln[3] != 7 for ln in out.get("links") or [])

    def test_p9_autogrow_connects_are_commutative(self):
        """Two concurrent autogrow connects to the same base must both survive
        (no clobber) and converge regardless of apply order."""
        ops = self._ops()
        g = _graph()
        base = _autogrow_workflow()
        _, op1 = ops.connect(copy.deepcopy(base), g, 20, "IMAGE", 10, "images", actor="a")
        _, op2 = ops.connect(copy.deepcopy(base), g, 21, "IMAGE", 10, "images", actor="b")
        ab = ops.apply_op(ops.apply_op(copy.deepcopy(base), op1, g), op2, g)
        ba = ops.apply_op(ops.apply_op(copy.deepcopy(base), op2, g), op1, g)
        # both source links survive in either order (no silent connection loss).
        for out in (ab, ba):
            link_srcs = {ln[1] for ln in out.get("links") or []}
            assert {20, 21} <= link_srcs, out.get("links")
        # ...and the two orders converge.
        assert ops.canonical(ab) == ops.canonical(ba)

    def test_p9_autogrow_grow_id_survives_api_conversion(self):
        """The ``grow_id`` bookkeeping (persisted on grown slots as their
        convergence identity) must not break API conversion — both wired sources
        reach the flat API prompt."""
        ops = self._ops()
        from comfy_cli.workflow_to_api import convert_ui_to_api

        g = _graph()
        base = _autogrow_workflow()
        _, op1 = ops.connect(copy.deepcopy(base), g, 20, "IMAGE", 10, "images", actor="a")
        _, op2 = ops.connect(copy.deepcopy(base), g, 21, "IMAGE", 10, "images", actor="b")
        wf = ops.apply_op(ops.apply_op(copy.deepcopy(base), op1, g), op2, g)
        assert any(i.get("grow_id") for i in next(n for n in wf["nodes"] if n["id"] == 10)["inputs"])
        api = convert_ui_to_api(wf, _object_info())
        wired = list(api["10"]["inputs"].values())
        assert ["20", 0] in wired and ["21", 0] in wired, wired

    def test_p10_subgraph_fork_is_deterministic(self):
        """Forking a shared subgraph definition must mint a deterministic id, so
        two replicas replaying the same ops reach byte-identical graphs."""
        ops = self._ops()
        g = _graph()
        base = _two_instance_subgraph_workflow()
        _, op_a = ops.set_widget(copy.deepcopy(base), g, "57/27", "text", "A", actor="a")
        _, op_b = ops.set_widget(copy.deepcopy(base), g, "58/27", "text", "B", actor="b")
        replica1 = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_a, g), op_b, g)
        replica2 = ops.apply_op(ops.apply_op(copy.deepcopy(base), op_a, g), op_b, g)
        # deterministic fork ids => two independent replays are identical.
        assert ops.canonical(replica1) == ops.canonical(replica2)

    def test_p11_concurrent_widget_writes_resolve_by_stamp(self):
        """Two concurrent writes to the same widget converge on the higher-stamp
        value regardless of apply order (last-writer-wins by causal stamp)."""
        ops = self._ops()
        g = _graph()
        base = _base_workflow()
        _, lo = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 10, actor="human", base_version=5)
        _, hi = ops.set_widget(copy.deepcopy(base), g, 3, "steps", 20, actor="agent", base_version=7)
        lo_hi = ops.apply_op(ops.apply_op(copy.deepcopy(base), lo, g), hi, g)
        hi_lo = ops.apply_op(ops.apply_op(copy.deepcopy(base), hi, g), lo, g)
        assert ops.canonical(lo_hi) == ops.canonical(hi_lo)
        order = g.widget_order("KSampler")
        ks = next(n for n in lo_hi["nodes"] if n["id"] == 3)
        assert ks["widgets_values"][order.index("steps")] == 20  # higher base_version wins
