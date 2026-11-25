"""`set-widget --where cloud` accepts a model the workspace uploaded to its Comfy Cloud library.

Cloud's ``object_info`` lists Cloud's own models only, so the edit refused every
upload with ``unknown_enum_value``. The refusal now asks the library first (see
``comfy_cli.cloud_own_models``); a value nobody holds is still refused, and the
hint says the library was checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _run,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import cloud_own_models, workflow_ops
from comfy_cli.cloud_own_models import OwnModels
from comfy_cli.command import workflow_edit
from comfy_cli.cql.engine import Graph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

UPLOAD = {"name": "my_finetune_v1.safetensors", "tags": ["models", "checkpoints"]}

CHECKPOINT_LOADER = {
    "input": {"required": {"ckpt_name": [["sd_xl_base_1.0.safetensors"], {}]}},
    "input_order": {"required": ["ckpt_name"]},
    "output": ["MODEL", "CLIP", "VAE"],
    "output_name": ["MODEL", "CLIP", "VAE"],
    "category": "loaders",
    "display_name": "Load Checkpoint",
    "description": "",
    "output_node": False,
    "python_module": "nodes",
}


class Library:
    def __init__(self, *rows: dict):
        self.rows = list(rows)
        self.asked: list[str] = []

    def __call__(self, name: str) -> list[dict]:
        self.asked.append(name)
        return [r for r in self.rows if name in r["name"]]


@pytest.fixture
def library(monkeypatch) -> Library:
    lib = Library(UPLOAD)
    monkeypatch.setattr(cloud_own_models, "for_cloud", lambda: OwnModels(lib))
    return lib


def _checkpoint_workflow(tmp_path, monkeypatch) -> tuple[Path, int]:
    graph = Graph.from_object_info({"CheckpointLoaderSimple": CHECKPOINT_LOADER})
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: graph)
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "CheckpointLoaderSimple")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    return path, op["node_id"]


def _widgets(path: Path) -> list:
    return json.loads(path.read_text())["nodes"][0]["widgets_values"]


def test_cloud_set_widget_writes_an_upload_the_library_holds(tmp_path, capsys, monkeypatch, library):
    path, node_id = _checkpoint_workflow(tmp_path, monkeypatch)
    env = _run(["set-widget", str(path), f"{node_id}.ckpt_name", UPLOAD["name"], "--where", "cloud"], capsys)
    assert env["ok"] is True, env
    assert env["data"]["op"]["value"] == UPLOAD["name"]
    assert _widgets(path) == [UPLOAD["name"]]
    assert library.asked == [UPLOAD["name"]]


def test_cloud_set_widget_refuses_a_model_nobody_holds(tmp_path, capsys, monkeypatch, library):
    path, node_id = _checkpoint_workflow(tmp_path, monkeypatch)
    before = path.read_text()
    env = _run(["set-widget", str(path), f"{node_id}.ckpt_name", "nobody.safetensors", "--where", "cloud"], capsys)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "unknown_enum_value"
    assert err["hint"].startswith("no model in your Comfy Cloud model library is named 'nobody.safetensors'"), err
    assert "comfy model upload" in err["details"]["library"]
    assert path.read_text() == before


def test_cloud_set_widget_never_asks_about_a_non_model_value(tmp_path, capsys, monkeypatch, library):
    path, node_id = _checkpoint_workflow(tmp_path, monkeypatch)
    env = _run(["set-widget", str(path), f"{node_id}.ckpt_name", "not a file", "--where", "cloud"], capsys)
    assert env["error"]["code"] == "unknown_enum_value"
    assert "library" not in env["error"]["details"]
    assert library.asked == []


def test_cloud_set_widget_never_writes_an_upload_into_an_input_that_picks_no_model(
    tmp_path, capsys, monkeypatch, library
):
    # The library holds the file, but the sampler's options are samplers: an
    # upload named *.safetensors is refused there, and nothing is written.
    sampler = json.loads(json.dumps(CHECKPOINT_LOADER))
    sampler["input"]["required"] = {"sampler_name": [["euler", "dpmpp_2m"], {}]}
    sampler["input_order"]["required"] = ["sampler_name"]
    graph = Graph.from_object_info({"SamplerPick": sampler})
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: graph)
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "SamplerPick")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    before = path.read_text()
    env = _run(["set-widget", str(path), f"{op['node_id']}.sampler_name", UPLOAD["name"], "--where", "cloud"], capsys)
    assert env["error"]["code"] == "unknown_enum_value"
    assert "library" not in env["error"]["details"]
    assert library.asked == []
    assert path.read_text() == before


@pytest.mark.parametrize("catalog", [["--where", "local"], ["--input", "object_info.json"]])
def test_set_widget_off_clouds_live_catalog_reads_no_library(tmp_path, capsys, monkeypatch, library, catalog):
    path, node_id = _checkpoint_workflow(tmp_path, monkeypatch)
    env = _run(["set-widget", str(path), f"{node_id}.ckpt_name", UPLOAD["name"], *catalog], capsys)
    assert env["error"]["code"] == "unknown_enum_value"
    assert library.asked == []


def test_cloud_set_widget_admits_an_upload_into_a_promoted_subgraph_widget(tmp_path, capsys, monkeypatch, library):
    graph = Graph.from_object_info(json.loads((_FIXTURES / "object_info_subgraph_promoted.json").read_text()))
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: graph)
    library.rows = [{"name": "my_unet.safetensors"}]
    path = tmp_path / "wf.json"
    path.write_text((_FIXTURES / "gallery" / "image_z_image_turbo.json").read_text(encoding="utf-8"))
    env = _run(["set-widget", str(path), "57.unet_name", "my_unet.safetensors", "--where", "cloud"], capsys)
    assert env["ok"] is True, env
    host = next(n for n in json.loads(path.read_text())["nodes"] if n["id"] == 57)
    assert "my_unet.safetensors" in host["widgets_values"]
