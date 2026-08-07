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
                                            "resolution": [
                                                "COMBO",
                                                {"default": "1080p", "options": ["4k", "1080p", "720p"]},
                                            ]
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


def _object_info_with_autogrow_template(template: dict) -> dict[str, Any]:
    """``_object_info()`` with ``BatchImagesNode.images`` carrying a V3 autogrow
    element-naming template — the shape checked into
    ``tests/comfy_cli/fixtures/subgraph_object_info.json`` for the live cloud
    BatchImagesNode: ``["COMFY_AUTOGROW_V3", {"template": {"prefix": ..., ...}}]``.
    ``template`` here is just the ``names``/``prefix`` pair the engine consumes."""
    info = copy.deepcopy(_object_info())
    info["BatchImagesNode"]["input"]["required"]["images"] = ["COMFY_AUTOGROW_V3", {"template": template}]
    return info


def _graph_with_autogrow_template(template: dict) -> Graph:
    return Graph.from_object_info(_object_info_with_autogrow_template(template))


def _object_info_with_inputcount() -> dict[str, Any]:
    """``_object_info()`` plus ``ImageBatchMulti`` — the kijai KJNodes family
    (also MaskBatchMulti, ConditioningMultiCombine, JoinStringMulti, …) that
    is NOT autogrow-typed: the schema declares fixed ``image_1``/``image_2``
    inputs plus a required INT ``inputcount`` widget the node reads at
    runtime to decide how many ``image_N`` slots to look at. Shape pinned
    against the production catalog snapshot
    (``services/ingest/data/object_info.json``, key ``ImageBatchMulti``):

        "required": {
          "inputcount": ["INT", {"default": 2, "min": 2, "max": 1000, "step": 1}],
          "image_1": ["IMAGE"]
        },
        "optional": {"image_2": ["IMAGE"]}

    Detection signal for the connect path (see ``_inputcount_family_match`` in
    ``workflow_ops.py``): a required INT widget literally named ``inputcount``
    PLUS a ``{elem}_1`` sibling input for the requested element — both must be
    present so an unrelated node with a coincidental ``foo_1`` input is never
    misclassified. Bare 1-based keys (``image_3``) are the CORRECT wire
    address for this family — unlike autogrow's dotted ``base.elemN`` — which
    is exactly what prod agents were sending and the CLI wrongly refused.
    """
    info = copy.deepcopy(_object_info())
    info["ImageBatchMulti"] = {
        "input": {
            "required": {
                "inputcount": ["INT", {"default": 2, "min": 2, "max": 1000, "step": 1}],
                "image_1": ["IMAGE"],
            },
            "optional": {"image_2": ["IMAGE"]},
        },
        "input_order": {"required": ["inputcount", "image_1"], "optional": ["image_2"]},
        "output": ["IMAGE"],
        "output_name": ["images"],
        "category": "KJNodes/image",
        "display_name": "Image Batch Multi",
        "python_module": "custom_nodes.ComfyUI-KJNodes",
    }
    return info


def _graph_with_inputcount() -> Graph:
    return Graph.from_object_info(_object_info_with_inputcount())


def _inputcount_workflow(existing: int = 2) -> dict:
    """An ``ImageBatchMulti`` node (id 20) with ``existing`` numbered
    ``image_N`` inputs already wired (from dummy VAEDecode sources 21, 22,
    …), plus one more unwired VAEDecode source (id 20 + existing + 1) to
    connect the next slot from."""
    nodes = []
    links = []
    link_id = 0
    for i in range(1, existing + 1):
        src_id = 20 + i
        nodes.append(
            {
                "id": src_id,
                "type": "VAEDecode",
                "pos": [0, i * 100],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [link_id]}],
                "widgets_values": [],
            }
        )
        links.append([link_id, src_id, 0, 20, i - 1, "IMAGE"])
        link_id += 1
    batch_inputs = [{"name": f"image_{i}", "type": "IMAGE", "link": i - 1} for i in range(1, existing + 1)]
    nodes.append(
        {
            "id": 20,
            "type": "ImageBatchMulti",
            "pos": [400, 0],
            "inputs": batch_inputs,
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": [existing],
        }
    )
    extra_src_id = 20 + existing + 1
    nodes.append(
        {
            "id": extra_src_id,
            "type": "VAEDecode",
            "pos": [0, (existing + 1) * 100],
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": None},
                {"name": "vae", "type": "VAE", "link": None},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": [],
        }
    )
    return {"last_node_id": extra_src_id, "last_link_id": link_id, "nodes": nodes, "links": links}


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
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
            {
                "id": 21,
                "type": "VAEDecode",
                "pos": [0, 100],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            },
        ],
        "links": [],
    }


def _convergence_base() -> dict:
    """A graph rich enough to exercise every op kind concurrently: two widget
    nodes (KSampler 3, EmptyLatentImage 7), an autogrow sink (BatchImagesNode 10),
    and two IMAGE sources (20, 21)."""
    wf = _base_workflow()
    wf["nodes"].append(
        {
            "id": 10,
            "type": "BatchImagesNode",
            "pos": [300, 0],
            "inputs": [{"name": "images", "type": "COMFY_AUTOGROW_V3", "link": None}],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "widgets_values": [],
        }
    )
    for nid, y in ((20, 0), (21, 120)):
        wf["nodes"].append(
            {
                "id": nid,
                "type": "VAEDecode",
                "pos": [150, y],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            }
        )
    wf["last_node_id"] = 21
    return wf


# Each spec mints one well-formed op against a fresh _convergence_base(); all are
# causally independent (they reference only base nodes), so any subset is a valid
# concurrent batch off the same base_version.
_CONVERGENCE_OP_SPECS = [
    "set_steps",
    "set_cfg",
    "set_width",
    "set_steps2",  # a second write to steps => LWW race with set_steps
    "del_ksampler",
    "del_latent",
    "connect_latent",
    "connect_latent2",  # a second link into the same concrete slot => a conflict
    "autogrow_20",
    "autogrow_21",  # a second autogrow onto the same base
    "add_vae",
]


def _make_convergence_op(spec: str, rng, g) -> dict:
    b = _convergence_base()
    a = rng.choice("abc")
    v = rng.randint(0, 3)
    if spec == "set_steps":
        _, op = workflow_ops.set_widget(b, g, 3, "steps", rng.randint(1, 40), actor=a, base_version=v)
    elif spec == "set_cfg":
        _, op = workflow_ops.set_widget(b, g, 3, "cfg", float(rng.randint(1, 15)), actor=a, base_version=v)
    elif spec == "set_width":
        _, op = workflow_ops.set_widget(b, g, 7, "width", rng.choice([256, 512, 768, 1024]), actor=a, base_version=v)
    elif spec == "set_steps2":
        _, op = workflow_ops.set_widget(b, g, 3, "steps", rng.randint(41, 80), actor=a, base_version=v)
    elif spec == "del_ksampler":
        _, op = workflow_ops.delete_node(b, g, 3, actor=a)
    elif spec == "del_latent":
        _, op = workflow_ops.delete_node(b, g, 7, actor=a)
    elif spec == "connect_latent":
        _, op = workflow_ops.connect(b, g, 7, "LATENT", 3, "latent_image", actor=a, base_version=v)
    elif spec == "connect_latent2":
        _, op = workflow_ops.connect(b, g, 7, "LATENT", 3, "latent_image", actor=a, base_version=v)
    elif spec == "autogrow_20":
        _, op = workflow_ops.connect(b, g, 20, "IMAGE", 10, "images", actor=a, base_version=v)
    elif spec == "autogrow_21":
        _, op = workflow_ops.connect(b, g, 21, "IMAGE", 10, "images", actor=a, base_version=v)
    elif spec == "add_vae":
        _, op = workflow_ops.add_node(b, g, "VAEDecode", actor=a)
    else:  # pragma: no cover - guard against a typo in the spec list
        raise AssertionError(f"unknown convergence op spec {spec!r}")
    return op


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

    def test_add_node_without_at_does_not_stack(self):
        """No explicit `pos` → the layout-aware cascade default, not the old
        blind [0, 0] every node used to land on (they used to stack exactly)."""
        g = _graph()
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, op1 = workflow_ops.add_node(wf, g, "KSampler")
        wf, op2 = workflow_ops.add_node(wf, g, "KSampler")
        n1, n2 = wf["nodes"][-2], wf["nodes"][-1]
        assert n1["pos"] != [0, 0] and n2["pos"] != [0, 0]
        assert n1["pos"] != n2["pos"]
        # position is frozen into the op for convergent replay
        assert op1["pos"] == n1["pos"]
        assert op2["pos"] == n2["pos"]

    def test_add_node_explicit_pos_is_honored(self):
        """`pos=` still passes straight through unchanged — Task 3's pre-pass
        depends on this."""
        g = _graph()
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, op = workflow_ops.add_node(wf, g, "KSampler", pos=[400, 200])
        assert wf["nodes"][-1]["pos"] == [400, 200]
        assert op["pos"] == [400, 200]

    def test_add_node_size_reflects_widget_count(self):
        """Size is estimated from the node's real inputs/outputs/widgets, not
        the old blind [210, 100] default."""
        g = _graph()
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, _ = workflow_ops.add_node(wf, g, "KSampler")
        node = wf["nodes"][-1]
        assert node["size"] != [210, 100]  # no longer the blind default
        assert node["size"][1] >= 60

    def test_add_node_cmd_rejects_single_coordinate(self, patched_graph, tmp_path, capsys):
        """`--at 400` (no comma, so `--at=400`) used to silently become a
        length-1 `pos` list instead of erroring — arity must be enforced."""
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "KSampler", "--at", "400"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "workflow_edit_invalid"


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

    def test_flattened_colon_address_writes_interior_node(self, patched_graph, tmp_path, capsys):
        """`57:27.text` — the FLATTENED id namespace `validate` and server
        node_errors report after UI→API lowering (workflow_to_api composes inner
        ids as `<outer>:<inner>`) — is accepted as an alias for `57/27.text`.
        Without this, an agent that copies a node id out of a validate error gets
        `node 57:27 not found in workflow` from the very tool that should fix it."""
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57:27.text", "a flattened cat"], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["path"] == ["57", "27"]
        assert op["inner_widget"] == "text"
        wf = json.loads(path.read_text())
        assert _interior(wf, 27)["widgets_values"][0] == "a flattened cat"

    def test_flattened_colon_converges_with_nested_form(self):
        """`57:27.text` and `57/27.text` resolve to the SAME CRDT write target."""
        wf = _subgraph_workflow()
        _, colon = workflow_ops.set_widget(copy.deepcopy(wf), _graph(), "57:27", "text", "A")
        _, nested = workflow_ops.set_widget(copy.deepcopy(wf), _graph(), "57/27", "text", "B")
        assert workflow_ops._write_target(colon) == workflow_ops._write_target(nested)

    def test_flattened_colon_address_unknown_interior_errors(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _subgraph_workflow())
        env = _run(["set-widget", str(path), "57:99.text", "x"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert "interior node 99 not found" in env["error"]["message"]

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

    def test_autogrow_rejects_malformed_slot_targets(self, patched_graph, tmp_path, capsys):
        """A dotted autogrow target that is not the next sequential slot — an index gap
        (images.image2), a doubled prefix (images.images.image0), a stray element
        (images.foo), or a trailing dot — is rejected with the fix, not silently grown
        into a key the server cannot map. Regression: prod connects mis-addressed
        autogrow slots and the CLI grew bogus inputs that failed only at submit time."""
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        src = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        batch = _run(["add-node", str(path), "BatchImagesNode"], capsys)["data"]["op"]["node_id"]

        for bad in ("images.image2", "images.images.image0", "images.foo", "images."):
            env = _run(["connect", str(path), f"{src}.IMAGE", f"{batch}.{bad}"], capsys)
            assert env["ok"] is False, (bad, env)
            assert env["error"]["code"] == "workflow_edit_invalid", (bad, env)
            # Actionable: names the base and the exact next free key to use instead.
            assert "images.image0" in env["error"]["message"], (bad, env)
            # And the bogus target is never minted onto the node.
            bn = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == batch)
            assert not any(i["name"] == bad for i in bn["inputs"]), (bad, bn["inputs"])

    def test_autogrow_accepts_the_exact_next_slot_key(self, patched_graph, tmp_path, capsys):
        """The bare base auto-appends, and the EXACT next sequential key is also accepted,
        so an agent addressing images.image0 then images.image1 still wires cleanly."""
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        a = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        b = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        batch = _run(["add-node", str(path), "BatchImagesNode"], capsys)["data"]["op"]["node_id"]
        e1 = _run(["connect", str(path), f"{a}.IMAGE", f"{batch}.images.image0"], capsys)  # exact next
        assert e1["ok"] is True, e1
        assert e1["data"]["op"]["grow"]["name"] == "images.image0"
        e2 = _run(["connect", str(path), f"{b}.IMAGE", f"{batch}.images.image1"], capsys)  # exact next
        assert e2["ok"] is True, e2
        assert e2["data"]["op"]["grow"]["name"] == "images.image1"

    def test_autogrow_accepts_bare_element_names(self, patched_graph, tmp_path, capsys):
        """A bare element name (`image0`, `image1` — no `images.` prefix) is the guess
        agents make on classic batch nodes, and was the top alpha workflow-edit
        failure. The canonical next element grows; a non-next element is rejected
        with an error that names the base and the exact next free key."""
        path = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})
        a = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        b = _run(["add-node", str(path), "VAEDecode"], capsys)["data"]["op"]["node_id"]
        batch = _run(["add-node", str(path), "BatchImagesNode"], capsys)["data"]["op"]["node_id"]

        # 1-indexed first guess (`image1` on an empty base) is rejected, and the
        # error teaches both recovery paths.
        env = _run(["connect", str(path), f"{a}.IMAGE", f"{batch}.image1"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "workflow_edit_invalid", env
        assert "images.image0" in env["error"]["message"], env
        assert "'images'" in env["error"]["message"], env

        # 0-indexed sequential guesses just work.
        e0 = _run(["connect", str(path), f"{a}.IMAGE", f"{batch}.image0"], capsys)
        assert e0["ok"] is True, e0
        assert e0["data"]["op"]["grow"]["name"] == "images.image0"
        e1 = _run(["connect", str(path), f"{b}.IMAGE", f"{batch}.image1"], capsys)
        assert e1["ok"] is True, e1
        assert e1["data"]["op"]["grow"]["name"] == "images.image1"

        # A name unrelated to any autogrow element still gets the generic error.
        env = _run(["connect", str(path), f"{a}.IMAGE", f"{batch}.frames3"], capsys)
        assert env["ok"] is False and "not found" in env["error"]["message"], env

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

    def test_inputcount_family_bare_key_grows_and_bumps_count(self):
        """kijai ``inputcount`` family (ImageBatchMulti et al.): bare 1-based
        ``image_3`` IS the correct wire address (unlike autogrow's dotted
        ``base.elemN``) — prod agents sent exactly this and the CLI wrongly
        refused it. Growing the slot must ALSO bump the ``inputcount`` widget
        to N, or the node never reads the new slot at runtime. Detection
        signal pinned against ImageBatchMulti's real object_info entry (see
        ``_object_info_with_inputcount``): a required INT ``inputcount``
        widget plus a ``{elem}_1`` sibling input."""
        g = _graph_with_inputcount()
        wf = _inputcount_workflow(existing=2)  # image_1, image_2 already wired
        wf, op = workflow_ops.connect(wf, g, 23, "IMAGE", 20, "image_3", actor="a")
        assert op["grow"]["name"] == "image_3"
        node = next(n for n in wf["nodes"] if n["id"] == 20)
        assert any(i["name"] == "image_3" and i["link"] is not None for i in node["inputs"])
        idx = g.widget_order("ImageBatchMulti").index("inputcount")
        assert node["widgets_values"][idx] == 3

    def test_inputcount_family_out_of_sequence_rejected_with_next_key(self):
        """Skipping ahead (``image_5`` when only 2 slots exist) is rejected
        with the guided next-free-key error, mirroring autogrow's
        out-of-sequence guidance — never a silent/bogus grow."""
        g = _graph_with_inputcount()
        wf = _inputcount_workflow(existing=2)
        with pytest.raises(ValueError, match="image_3"):
            workflow_ops.connect(wf, g, 23, "IMAGE", 20, "image_5", actor="a")
        node = next(n for n in wf["nodes"] if n["id"] == 20)
        assert not any(i["name"] == "image_5" for i in node["inputs"])

    def test_inputcount_family_concurrent_connects_stay_bare_and_converge(self):
        """Two concurrent connects both minted against the same next slot
        (``image_3``) must both survive (no clobber, mirroring autogrow's P9
        commutativity) — and the loser's collision-resolved name MUST stay a
        bare inputcount key (``image_4``), never autogrow's dotted
        ``base.elemN`` fallback, which the server can't map. The inputcount
        widget bump uses each op's mint-time-planned value (not one re-derived
        from its post-collision slot number) specifically so the two apply
        orders converge to the SAME final value — a value derived from the
        renamed slot would make the winning stamp carry different values in
        each order and break convergence (P9)."""
        g = _graph_with_inputcount()
        base = _inputcount_workflow(existing=2)
        extra_src_id = base["last_node_id"]  # the unwired VAEDecode _inputcount_workflow adds
        base["nodes"].append(
            {
                "id": extra_src_id + 1,
                "type": "VAEDecode",
                "pos": [0, 999],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            }
        )
        _, op1 = workflow_ops.connect(copy.deepcopy(base), g, extra_src_id, "IMAGE", 20, "image_3", actor="a")
        _, op2 = workflow_ops.connect(copy.deepcopy(base), g, extra_src_id + 1, "IMAGE", 20, "image_3", actor="b")
        ab = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op1, g), op2, g)
        ba = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op2, g), op1, g)
        idx = g.widget_order("ImageBatchMulti").index("inputcount")
        for out in (ab, ba):
            node = next(n for n in out["nodes"] if n["id"] == 20)
            names = {i["name"] for i in node["inputs"]}
            assert names == {"image_1", "image_2", "image_3", "image_4"}, names  # both survive, bare
            assert not any("." in n for n in names)  # never the dotted autogrow fallback
            assert node["widgets_values"][idx] == 3  # each op planned 3 at mint time
        assert workflow_ops.canonical(ab) == workflow_ops.canonical(ba)

    def test_non_family_unknown_input_error_unchanged(self):
        """A non-family node (BatchImagesNode, autogrow-typed but NOT in the
        inputcount family) keeps the exact generic not-found error text for a
        key that matches neither autogrow nor inputcount shapes."""
        g = _graph()
        wf = _autogrow_workflow()
        with pytest.raises(ValueError, match="not found on node"):
            workflow_ops.connect(wf, g, 20, "IMAGE", 10, "bogus_7", actor="a")


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
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_graph_but_preserves_id_counters(self):
        g = _graph()
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, _ = workflow_ops.add_node(wf, g, "KSampler")
        wf, _ = workflow_ops.add_node(wf, g, "KSampler")
        wf["groups"] = [{"title": "g"}]
        last_node = wf["last_node_id"]
        wf, op = workflow_ops.clear(wf)
        assert wf["nodes"] == [] and wf["links"] == [] and wf["groups"] == []
        assert wf["last_node_id"] == last_node  # ids stay monotonic across a clear
        assert op["op"] == "clear"

    def test_clear_op_replays_idempotently(self):
        g = _graph()
        wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, _ = workflow_ops.add_node(wf, g, "KSampler")
        cleared, op = workflow_ops.clear(copy.deepcopy(wf))
        replay = workflow_ops.apply_op(copy.deepcopy(wf), op, g)
        replay = workflow_ops.apply_op(replay, op, g)  # second apply is a no-op
        assert replay["nodes"] == cleared["nodes"] == []

    def test_clear_cmd_empties_workflow_and_emits_op(self, patched_graph, tmp_path, capsys):
        wf = _base_workflow()
        wf["groups"] = [{"title": "g"}]
        path = _write(tmp_path, wf)
        env = _run(["clear", str(path)], capsys)
        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "clear"
        on_disk = json.loads(path.read_text())
        assert on_disk["nodes"] == [] and on_disk["links"] == [] and on_disk["groups"] == []


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

    def test_duplicate_alias_is_rejected(self, patched_graph, tmp_path, capsys):
        """A repeated `as` name would silently clobber the earlier node — reject it."""
        path = self._empty(tmp_path)
        before = path.read_text()
        specs = [
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},  # duplicate alias
        ]
        ops_path = tmp_path / "ops.json"
        ops_path.write_text(json.dumps(specs), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(ops_path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert "ks" in env["error"]["message"] and "already defined" in env["error"]["message"]
        assert path.read_text() == before

    def test_missing_field_names_the_spec(self, patched_graph, tmp_path, capsys):
        """A bare KeyError becomes an actionable `spec #i (...) is missing ...`."""
        path = self._empty(tmp_path)
        specs = [{"op": "add_node", "class_type": "KSampler", "as": "ks"}, {"op": "set_widget", "node": "ks"}]
        ops_path = tmp_path / "ops.json"
        ops_path.write_text(json.dumps(specs), encoding="utf-8")
        env = _run(["apply", str(path), "--ops", str(ops_path)], capsys)
        assert env["ok"] is False
        assert "spec #1" in env["error"]["message"] and "missing required field" in env["error"]["message"]

    def test_apply_specs_batch_has_no_stacked_nodes(self):
        """apply_specs must layout-assign positions for a batch of add_nodes
        that don't specify `at`, so they don't all land stacked at the origin."""
        wf: dict = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        specs = [
            {"op": "add_node", "class_type": "KSampler", "as": "a"},
            {"op": "add_node", "class_type": "KSampler", "as": "b"},
            {"op": "add_node", "class_type": "KSampler", "as": "c"},
        ]
        wf, ops, _ = workflow_ops.apply_specs(wf, _graph(), specs)
        positions = [tuple(n["pos"]) for n in wf["nodes"]]
        assert len(set(positions)) == len(positions)
        assert (0, 0) not in positions


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
        env = _run(
            ["apply", str(path), "--ops", str(rp), "--param", "positive=quiet forest", "--param", "steps=35"], capsys
        )
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

    def test_foreach_surfaces_partial_writes_on_mid_batch_failure(self, patched_graph, tmp_path, capsys):
        """foreach writes per param-set; a mid-batch failure leaves earlier files
        on disk, so the error must surface them (not leave the caller blind)."""
        rp = self._recipe(tmp_path)
        params = tmp_path / "sets.jsonl"
        # #0 valid → written; #1 missing required `positive` → fails.
        params.write_text('{"positive":"a cat"}\n{"steps":10}\n', encoding="utf-8")
        out = tmp_path / "out"
        env = _run(["foreach", str(rp), "--params", str(params), "--out-dir", str(out)], capsys)
        assert env["ok"] is False
        written = env["error"]["details"]["written"]
        assert len(written) == 1 and written[0].endswith("_000.json")
        assert list(out.glob("*.json"))  # the partial file really is on disk
        assert "before the failure" in env["error"]["hint"]


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

    def test_autogrow_uses_schema_prefix_zero_based(self):
        """A ``{"prefix": "frame"}`` template names grown slots verbatim from
        the schema, 0-based (images.frame0, images.frame1) — a prefix that
        deliberately differs from the ``{base[:-1]}`` pluralization guess
        ("image"), so this only passes when the name truly comes from the
        schema template, not the heuristic."""
        ops = self._ops()
        g = _graph_with_autogrow_template({"prefix": "frame"})
        wf = _autogrow_workflow()
        wf, op1 = ops.connect(wf, g, 20, "IMAGE", 10, "images", actor="a")
        wf, op2 = ops.connect(wf, g, 21, "IMAGE", 10, "images", actor="a")
        assert op1["grow"]["name"] == "images.frame0"
        assert op2["grow"]["name"] == "images.frame1"
        grown = [
            i["name"]
            for i in next(n for n in wf["nodes"] if n["id"] == 10)["inputs"]
            if str(i["name"]).startswith("images.")
        ]
        assert grown == ["images.frame0", "images.frame1"]

    def test_autogrow_uses_schema_names_verbatim(self):
        """A ``{"names": [...]}`` template uses the literal element names from
        the schema, not a pluralization guess — e.g. a node whose V3 definition
        calls its slots "first"/"second" rather than "image0"/"image1"."""
        ops = self._ops()
        g = _graph_with_autogrow_template({"names": ["first", "second"]})
        wf = _autogrow_workflow()
        wf, op1 = ops.connect(wf, g, 20, "IMAGE", 10, "images", actor="a")
        wf, op2 = ops.connect(wf, g, 21, "IMAGE", 10, "images", actor="a")
        assert op1["grow"]["name"] == "images.first"
        assert op2["grow"]["name"] == "images.second"
        grown = [
            i["name"]
            for i in next(n for n in wf["nodes"] if n["id"] == 10)["inputs"]
            if str(i["name"]).startswith("images.")
        ]
        assert grown == ["images.first", "images.second"]

    def test_autogrow_without_template_keeps_heuristic(self):
        """No schema template (offline edit, or a catalog entry — like this
        file's ``_object_info()`` — that never populated one) keeps the
        historical ``{base}.{base[:-1]}{N}`` guess. Regression: the existing
        fixture's contract must not change just because the feature shipped."""
        ops = self._ops()
        g = _graph()  # BatchImagesNode.images is bare "COMFY_AUTOGROW_V3": no template
        wf = _autogrow_workflow()
        wf, op = ops.connect(wf, g, 20, "IMAGE", 10, "images", actor="a")
        assert op["grow"]["name"] == "images.image0"
        grown = [
            i["name"]
            for i in next(n for n in wf["nodes"] if n["id"] == 10)["inputs"]
            if str(i["name"]).startswith("images.")
        ]
        assert grown == ["images.image0"]

    def test_autogrow_fills_a_gap_instead_of_colliding(self):
        """A gapped legacy run (``image0`` + ``image2``) must not mint a SECOND
        ``image2``. Both namers used to seed N from the count of ``images.``-
        prefixed inputs, so a gap made the count land on an occupied slot and
        the grow clobbered a wired input."""
        from comfy_cli.workflow_ops import _next_autogrow_name, _plan_autogrow

        ins = [{"name": "images.image0"}, {"name": "images.image2"}]
        assert _plan_autogrow(ins, "images", "IMAGE")["name"] == "images.image1"
        assert _next_autogrow_name(ins, "images.image0") == "images.image1"

    def test_autogrow_ignores_non_conforming_siblings(self):
        """A sibling that isn't a mintable slot name (``images.foo``) must not
        advance the counter — counting prefix matches skipped ``image0``."""
        from comfy_cli.workflow_ops import _next_autogrow_name, _plan_autogrow

        ins = [{"name": "images.foo"}]
        assert _plan_autogrow(ins, "images", "IMAGE")["name"] == "images.image0"
        assert _next_autogrow_name(ins, "images.foo") == "images.image0"

    def test_autogrow_gap_fill_respects_a_names_template(self):
        """Gap-filling is computed from the names we'd actually mint, so a
        ``names`` template fills its own vocabulary rather than a guessed stem."""
        from comfy_cli.workflow_ops import _plan_autogrow

        tpl = {"names": ["first", "second", "third"]}
        ins = [{"name": "images.first"}, {"name": "images.third"}]
        assert _plan_autogrow(ins, "images", "IMAGE", tpl)["name"] == "images.second"

    def test_autogrow_still_appends_on_a_clean_run(self):
        """The common case is unchanged: a gapless run grows at the end."""
        from comfy_cli.workflow_ops import _plan_autogrow

        ins = [{"name": "images.image0"}, {"name": "images.image1"}]
        assert _plan_autogrow(ins, "images", "IMAGE")["name"] == "images.image2"
        assert _plan_autogrow([], "images", "IMAGE")["name"] == "images.image0"

    def test_p9_autogrow_names_template_converges(self):
        """Two concurrent autogrow connects onto a ``names``-templated base still
        converge to the schema's two literal element names in either apply
        order — the conflict-resolution path (``_next_autogrow_name``) must
        derive from the schema too, not just the first-planned request."""
        ops = self._ops()
        g = _graph_with_autogrow_template({"names": ["first", "second"]})
        base = _autogrow_workflow()
        _, op1 = ops.connect(copy.deepcopy(base), g, 20, "IMAGE", 10, "images", actor="a")
        _, op2 = ops.connect(copy.deepcopy(base), g, 21, "IMAGE", 10, "images", actor="b")
        ab = ops.apply_op(ops.apply_op(copy.deepcopy(base), op1, g), op2, g)
        ba = ops.apply_op(ops.apply_op(copy.deepcopy(base), op2, g), op1, g)
        for out in (ab, ba):
            ins = next(n for n in out["nodes"] if n["id"] == 10)["inputs"]
            names = {i["name"] for i in ins if str(i["name"]).startswith("images.")}
            assert names == {"images.first", "images.second"}, names
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

    # -- sufficiency (P12..P13): P8..P11 prove specific bugs are fixed; these
    #    prove the op model's CONTRACT holds across randomized inputs — every op
    #    pair either converges or is flagged (never silently diverges), and
    #    canonical() is a sound equality oracle (folds only immaterial detail).

    def test_p12_every_op_pair_converges_or_is_flagged(self):
        """The load-bearing invariant: any two concurrent ops EITHER converge
        under replay OR are reported by ``detect_conflict``. A silent divergence
        (order matters, but nothing flagged it) is the failure this rules out.
        Proved over randomized pairs with a fixed seed (reproducible)."""
        import itertools
        import random

        ops = self._ops()
        g = _graph()
        rng = random.Random(20260707)
        checked = 0
        for _ in range(400):
            specs = rng.sample(_CONVERGENCE_OP_SPECS, k=rng.randint(2, 4))
            pool = [_make_convergence_op(s, rng, g) for s in specs]
            for a, b in itertools.combinations(pool, 2):
                base = _convergence_base()
                ab = ops.apply_op(ops.apply_op(copy.deepcopy(base), a, g), b, g)
                ba = ops.apply_op(ops.apply_op(copy.deepcopy(base), b, g), a, g)
                if ops.canonical(ab) != ops.canonical(ba):
                    assert ops.detect_conflict(a, b), (
                        "SILENT DIVERGENCE",
                        (a["op"], a.get("widget") or a.get("to_node")),
                        (b["op"], b.get("widget") or b.get("to_node")),
                    )
                checked += 1
        assert checked > 1000  # the harness genuinely exercised many pairs

    def test_p12b_conflict_free_sets_fully_converge(self):
        """Higher-order: a set of pairwise-non-conflicting ops converges across
        ALL apply orders (not just pairs) — catches 3-way interactions."""
        import itertools
        import random

        ops = self._ops()
        g = _graph()
        rng = random.Random(4242)
        trials = 0
        for _ in range(300):
            specs = rng.sample(_CONVERGENCE_OP_SPECS, k=rng.randint(2, 4))
            pool = [_make_convergence_op(s, rng, g) for s in specs]
            free: list[dict] = []
            for op in pool:  # greedily keep a maximal conflict-free subset
                if all(not ops.detect_conflict(op, kept) for kept in free):
                    free.append(op)
            if len(free) < 2:
                continue
            perms = list(itertools.permutations(free))
            if len(perms) > 24:
                perms = rng.sample(perms, 24)
            cans = []
            for perm in perms:
                wf = _convergence_base()
                for op in perm:
                    wf = ops.apply_op(wf, op, g)
                cans.append(ops.canonical(wf))
            for c in cans[1:]:
                assert c == cans[0], "conflict-free set diverged across apply orders"
            trials += 1
        assert trials > 50

    def test_p13_canonical_is_a_sound_equality_oracle(self):
        """``canonical`` must fold away ONLY immaterial detail (apply
        bookkeeping, node/link/def ordering, a grown slot's display name) and
        must PRESERVE every material difference — otherwise a real divergence
        could hide behind a false ``canonical`` match."""
        ops = self._ops()
        g = _graph()
        base = _convergence_base()
        c0 = ops.canonical(base)
        assert c0 == ops.canonical(copy.deepcopy(base))  # stable / reflexive

        # Immaterial differences must NOT change canonical.
        immaterial = copy.deepcopy(base)
        immaterial["nodes"] = list(reversed(immaterial["nodes"]))
        immaterial["links"] = list(reversed(immaterial["links"]))
        immaterial["_applied_ops"] = ["deadbeef"]
        immaterial["_widget_stamps"] = {"('widget', 3, 'steps')": [9, "z", "op"]}
        assert ops.canonical(immaterial) == c0

        # Material differences MUST change canonical.
        widget = copy.deepcopy(base)
        next(n for n in widget["nodes"] if n["id"] == 3)["widgets_values"][2] = 999
        assert ops.canonical(widget) != c0  # a changed widget value
        removed_node = copy.deepcopy(base)
        removed_node["nodes"] = [n for n in removed_node["nodes"] if n["id"] != 7]
        assert ops.canonical(removed_node) != c0  # a removed node
        removed_link = copy.deepcopy(base)
        removed_link["links"] = []
        assert ops.canonical(removed_link) != c0  # a removed link

        # Autogrow: the grown slot's DISPLAY NAME is immaterial (order-dependent),
        # but WHICH SOURCE it wires is material. Prove canonical draws that line.
        _, o20 = ops.connect(_convergence_base(), g, 20, "IMAGE", 10, "images", actor="a")
        _, o21 = ops.connect(_convergence_base(), g, 21, "IMAGE", 10, "images", actor="b")
        grown = ops.apply_op(ops.apply_op(_convergence_base(), o20, g), o21, g)
        renamed = copy.deepcopy(grown)
        for inp in next(n for n in renamed["nodes"] if n["id"] == 10)["inputs"]:
            if inp.get("grow_id") is not None:
                inp["name"] = f"images.renamed{inp['grow_id']}"  # cosmetic only
        assert ops.canonical(renamed) == ops.canonical(grown)  # name is immaterial
        rewired = copy.deepcopy(grown)
        for ln in rewired["links"]:
            if ln[1] == 21:
                ln[1] = 20  # a grown slot now sourced from a different node
        assert ops.canonical(rewired) != ops.canonical(grown)  # source is material


class TestOpResolutionSuggestions:
    """The edit ops enrich a *not-found* error with the real id/address, so an
    agent that rebuilt an identifier from memory (hitting a wrong node, a real
    sibling, or a nonexistent id) self-corrects in one step instead of looping.
    Covers the whole edit surface: set_widget, connect, delete_node."""

    def test_set_widget_wrong_node_suggests_the_widgets_real_address(self):
        # 'steps' lives on KSampler (3), not EmptyLatentImage (7).
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError, match=r"Did you mean:.*3\.steps \(KSampler\)"):
            workflow_ops.set_widget(wf, g, 7, "steps", 20)

    def test_set_widget_missing_node_suggests_the_widgets_real_address(self):
        # Node 999 doesn't exist (mirrors a wrong id/separator); 'steps' is on 3.
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError, match=r"Did you mean:.*3\.steps \(KSampler\)"):
            workflow_ops.set_widget(wf, g, 999, "steps", 20)

    def test_set_widget_unknown_widget_no_false_suggestion(self):
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, g, 3, "no_such_widget", 1)
        assert "Did you mean" not in str(ei.value)

    def test_set_widget_shape_error_is_not_enriched(self):
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, g, 3, "steps", "not_an_int")
        assert "Did you mean" not in str(ei.value)

    def test_connect_missing_node_lists_available_nodes(self):
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError, match=r"Nodes in this workflow:.*KSampler"):
            workflow_ops.connect(wf, g, 999, "LATENT", 3, "latent_image")

    def test_delete_missing_node_lists_available_nodes(self):
        g, wf = _graph(), _base_workflow()
        with pytest.raises(ValueError, match=r"Nodes in this workflow:.*KSampler"):
            workflow_ops.delete_node(wf, g, 999)


class TestSetWidgetModelNormalization:
    """set_widget auto-corrects a mangled COMBO/model value to the real option so
    the model actually loads even when the agent rebuilds the name from memory
    (e.g. adds a directory prefix) — the reliable fix for 'model not found'."""

    def test_prefixed_combo_value_is_normalized_in_the_op(self):
        g, wf = _graph(), _base_workflow()
        _, op = workflow_ops.set_widget(wf, g, 3, "sampler_name", "samplers/euler")
        assert op["value"] == "euler"  # the real option, prefix stripped
        assert any(w.get("code") == "normalized_value" for w in op.get("warnings", []))

    def test_exact_value_is_untouched_and_unwarned(self):
        g, wf = _graph(), _base_workflow()
        _, op = workflow_ops.set_widget(wf, g, 3, "sampler_name", "euler")
        assert op["value"] == "euler"
        assert not any(w.get("code") == "normalized_value" for w in op.get("warnings", []))

    def test_unknown_value_is_left_for_validate_to_flag(self):
        g, wf = _graph(), _base_workflow()
        _, op = workflow_ops.set_widget(wf, g, 3, "sampler_name", "totally_made_up")
        assert op["value"] == "totally_made_up"  # not silently changed
        assert any(w.get("code") == "unknown_enum_value" for w in op.get("warnings", []))


class TestWhereInvalid:
    """A bad --where surfaces the agent-first error envelope, not a raw traceback.

    Regression: _get_graph only caught LoadError, so resolve_default's ValueError
    on an invalid --where escaped uncaught out of every edit command.
    """

    def test_set_widget_bad_where_emits_envelope(self, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["set-widget", str(path), "3.seed", "5", "--where", "clowd"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "where_invalid"
