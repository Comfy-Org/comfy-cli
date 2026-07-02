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


    def test_nested_subgraph_address_routed_to_set_slot(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "10/9.prompt", "x"], capsys)
        assert env["ok"] is False
        assert "set-slot" in (env["error"]["hint"] or "")


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
        from comfy_cli.command import workflow_edit  # graph for widget order

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
