"""`set-widget` applies a model Cloud loads from its asset library even though
the catalog does not list it, and still refuses one nothing backs."""

from __future__ import annotations

import json

from test_workflow_edit import (  # type: ignore[import-not-found]
    _run,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import workflow_ops
from comfy_cli.command import workflow_edit
from comfy_cli.cql import model_assets
from comfy_cli.cql.engine import Graph

INT8 = "minimax_h3_video_vae_int8_convrot.safetensors"
FP16 = "minimax_h3_video_vae_fp16.safetensors"


def _setup(tmp_path, monkeypatch):
    graph = Graph.from_object_info(
        {
            "VAELoader": {
                "input": {"required": {"vae_name": [[FP16, "ae.safetensors"], {}]}},
                "input_order": {"required": ["vae_name"]},
                "output": ["VAE"],
                "output_name": ["VAE"],
                "category": "loaders",
                "display_name": "Load VAE",
                "description": "",
                "output_node": False,
                "python_module": "nodes",
            }
        }
    )
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "VAELoader")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: graph)
    return path, op["node_id"]


def test_set_widget_applies_an_asset_backed_model(tmp_path, capsys, monkeypatch):
    path, nid = _setup(tmp_path, monkeypatch)
    model_assets.set_lookup(lambda name: name == INT8)
    env = _run(["set-widget", str(path), f"{nid}.vae_name", INT8], capsys)
    assert env["ok"] is True, env
    node = json.loads(path.read_text())["nodes"][0]
    assert node["widgets_values"][0] == INT8


def test_set_widget_still_refuses_a_model_nothing_backs(tmp_path, capsys, monkeypatch):
    path, nid = _setup(tmp_path, monkeypatch)
    before = path.read_text()
    model_assets.set_lookup(lambda name: False)
    env = _run(["set-widget", str(path), f"{nid}.vae_name", INT8], capsys)
    assert env["ok"] is False
    assert env["error"]["code"] == "unknown_enum_value"
    assert path.read_text() == before
