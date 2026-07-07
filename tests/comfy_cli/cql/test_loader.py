"""Loader tests: object_info, API workflow, and pre-shaped graph inputs."""

from __future__ import annotations

import json

import pytest

from comfy_cli.cql import loader
from comfy_cli.cql.errors import CQLRuntimeError
from comfy_cli.cql.loader import _load_from_server, load_graph, normalize

OBJECT_INFO = {
    "KSampler": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0}],
                "model": ["MODEL"],
                "scheduler": [["normal", "karras"]],
            },
            "optional": {
                "denoise": ["FLOAT", {"default": 1.0}],
            },
        },
        "output": ["LATENT"],
        "category": "sampling",
        "display_name": "K Sampler",
        "description": "samples",
    },
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": ["STRING"]}},
        "output": ["MODEL", "CLIP", "VAE"],
        "category": "loaders",
    },
}


API_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "model": ["4", 0]},
        "_meta": {"title": "Sampler"},
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
    },
}


def test_normalize_object_info_extracts_nodes_and_inputs():
    g = normalize(OBJECT_INFO)
    names = {n["name"] for n in g["nodes"]}
    assert names == {"KSampler", "CheckpointLoaderSimple"}
    ks = next(n for n in g["nodes"] if n["name"] == "KSampler")
    assert ks["category"] == "sampling"
    assert ks["display_name"] == "K Sampler"
    assert ks["output_types"] == ["LATENT"]
    # Inputs were flattened with section labels.
    seed = next(i for i in g["inputs"] if i["node"] == "KSampler" and i["name"] == "seed")
    assert seed["type"] == "INT"
    assert seed["section"] == "required"
    assert seed["options"]["default"] == 0
    # Choices captured for combo inputs.
    sch = next(i for i in g["inputs"] if i["name"] == "scheduler")
    assert sch["type"] == "ENUM"
    assert sch["choices"] == ["normal", "karras"]


def test_normalize_object_info_aggregates_categories():
    g = normalize(OBJECT_INFO)
    by_name = {c["name"]: c["node_count"] for c in g["categories"]}
    assert by_name == {"sampling": 1, "loaders": 1}


def test_normalize_api_workflow_marks_references():
    g = normalize(API_WORKFLOW)
    nodes_by_id = {n["id"]: n for n in g["nodes"]}
    assert nodes_by_id["3"]["class_type"] == "KSampler"
    assert nodes_by_id["3"]["title"] == "Sampler"
    seed = next(i for i in g["inputs"] if i["node_id"] == "3" and i["name"] == "seed")
    assert seed["is_reference"] is False
    assert seed["value"] == 42
    model = next(i for i in g["inputs"] if i["node_id"] == "3" and i["name"] == "model")
    assert model["is_reference"] is True
    assert model["ref_node"] == "4"
    assert model["ref_slot"] == 0


def test_normalize_preshaped_graph_pass_through():
    pre = {
        "nodes": [{"name": "Foo"}],
        "inputs": [],
        "categories": [{"name": "x", "node_count": 1}],
    }
    g = normalize(pre)
    assert g["nodes"][0]["name"] == "Foo"


def test_load_graph_from_file(tmp_path):
    p = tmp_path / "object_info.json"
    p.write_text(json.dumps(OBJECT_INFO))
    g = load_graph(input_path=str(p))
    assert {n["name"] for n in g["nodes"]} == {"KSampler", "CheckpointLoaderSimple"}


def test_load_graph_missing_source_raises():
    with pytest.raises(CQLRuntimeError):
        load_graph()


def test_load_graph_bad_json(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    with pytest.raises(CQLRuntimeError):
        load_graph(input_path=str(p))


def test_normalize_rejects_garbage():
    with pytest.raises(CQLRuntimeError):
        normalize({"foo": 1, "bar": "baz"})


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _n=None):
        return self._payload


def test_load_from_server_refuses_non_loopback_host():
    # SSRF guard: a public host must never be fetched by the local loader.
    with pytest.raises(CQLRuntimeError, match="non-loopback"):
        _load_from_server("example.com", 8188, timeout=0.1)


def test_load_from_server_accepts_loopback(monkeypatch):
    # 127.0.0.1 passes the guard and proceeds to the fetch (mocked here).
    payload = json.dumps(OBJECT_INFO).encode("utf-8")
    monkeypatch.setattr(loader._LOADER_OPENER, "open", lambda *a, **k: _FakeResp(payload))
    g = _load_from_server("127.0.0.1", 8188, timeout=0.1)
    assert {n["name"] for n in g["nodes"]} == {"KSampler", "CheckpointLoaderSimple"}
