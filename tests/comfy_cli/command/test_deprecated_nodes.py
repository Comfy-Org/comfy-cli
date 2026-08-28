"""Deprecated node classes: hidden from discovery, refused by add_node.

ComfyUI marks a retired class with ``deprecated: true`` in object_info and
suffixes its display name with "(DEPRECATED)" / "(Legacy)"; the successor is
registered under the bare display name (``ImageBatch`` -> ``BatchImagesNode``).
The frontend hides such classes from its node library by default. The agent
kept building on ``ImageBatch`` (BE-7684) because ``nodes search`` ranked it
like any live class and ``add-node`` accepted it, so:

  * ``nodes search`` / ``nodes ls`` drop deprecated rows unless
    ``--include-deprecated`` is passed.
  * ``workflow add-node`` and an ``add_node`` op in ``workflow apply`` refuse
    a deprecated class with ``code=node_deprecated`` and name the live
    replacement; ``--allow-deprecated`` / ``"allow_deprecated": true`` adds it
    anyway for a user who asked for that exact node.

Editing a deprecated node that is ALREADY on the graph is untouched: connect
and set_widget never go through add_node.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _force_json_renderer,
    _object_info,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)
from typer.testing import CliRunner

from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.command import workflow_edit
from comfy_cli.cql.engine import Graph


def _object_info_with_deprecated() -> dict[str, Any]:
    info = copy.deepcopy(_object_info())
    info["ImageBatch"] = {
        "input": {"required": {"image1": "IMAGE", "image2": "IMAGE"}},
        "input_order": {"required": ["image1", "image2"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "image/batch",
        "display_name": "Batch Images (DEPRECATED)",
        "deprecated": True,
        "python_module": "nodes",
    }
    # Deprecated free class whose only same-name live twin is a paid partner
    # node: no replacement may be offered.
    info["OldFreeUpscale"] = {
        "input": {"required": {"image": "IMAGE"}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "image/upscale",
        "display_name": "Upscale (Legacy)",
        "deprecated": True,
        "python_module": "nodes",
    }
    info["PaidUpscale"] = {
        "input": {"required": {"image": "IMAGE"}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "partner/image",
        "display_name": "Upscale",
        "api_node": True,
        "python_module": "comfy_api_nodes",
    }
    # Deprecated loader that is the ONLY producer of its type.
    info["OldLoader"] = {
        "input": {"required": {"name": [["a"]]}},
        "input_order": {"required": ["name"]},
        "output": ["OLDTHING"],
        "output_name": ["OLDTHING"],
        "category": "loaders",
        "display_name": "Old Loader",
        "deprecated": True,
        "python_module": "nodes",
    }
    info["OldSampler"] = {
        "input": {"required": {"model": "MODEL"}},
        "input_order": {"required": ["model"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
        "display_name": "Old Sampler",
        "deprecated": True,
        "python_module": "nodes",
    }
    return info


def _graph() -> Graph:
    return Graph.from_object_info(_object_info_with_deprecated())


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph())


def _run(app, args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    result = CliRunner().invoke(app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


def _names(env: dict) -> list[str]:
    return [r["name"] for r in env["data"]["rows"]]


class TestAddNode:
    def test_refused_with_replacement(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(workflow_cmd.app, ["add-node", str(path), "ImageBatch"], capsys)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "node_deprecated", err
        assert err["details"] == {"requested": "ImageBatch", "replacement": "BatchImagesNode"}
        assert "BatchImagesNode" in err["hint"]
        assert "allow_deprecated" in err["hint"]
        # Atomic: the file is untouched.
        assert json.loads(path.read_text())["last_node_id"] == _base_workflow()["last_node_id"]

    def test_refused_without_replacement_points_at_search(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(workflow_cmd.app, ["add-node", str(path), "OldSampler"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "node_deprecated"
        assert env["error"]["details"]["replacement"] is None
        assert "nodes search" in env["error"]["hint"]

    def test_no_replacement_across_billing_class(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(workflow_cmd.app, ["add-node", str(path), "OldFreeUpscale"], capsys)
        assert env["error"]["code"] == "node_deprecated"
        assert env["error"]["details"]["replacement"] is None
        assert "PaidUpscale" not in env["error"]["hint"]

    def test_allow_deprecated_adds(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(workflow_cmd.app, ["add-node", str(path), "ImageBatch", "--allow-deprecated"], capsys)
        assert env["ok"] is True, env
        assert any(n["type"] == "ImageBatch" for n in json.loads(path.read_text())["nodes"])

    def test_live_class_unaffected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(workflow_cmd.app, ["add-node", str(path), "BatchImagesNode"], capsys)
        assert env["ok"] is True, env


class TestApplyBatch:
    def test_batch_refused(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        ops = _write(tmp_path, [{"op": "add_node", "class_type": "ImageBatch", "as": "b"}], "ops.json")
        env = _run(workflow_cmd.app, ["apply", str(path), "--ops", str(ops)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "node_deprecated", env
        assert env["error"]["details"]["replacement"] == "BatchImagesNode"

    def test_batch_allow_deprecated_adds(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        ops = _write(
            tmp_path, [{"op": "add_node", "class_type": "ImageBatch", "as": "b", "allow_deprecated": True}], "ops.json"
        )
        env = _run(workflow_cmd.app, ["apply", str(path), "--ops", str(ops)], capsys)
        assert env["ok"] is True, env
        assert any(n["type"] == "ImageBatch" for n in json.loads(path.read_text())["nodes"])


class TestDiscovery:
    def test_search_hides_deprecated_by_default(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["search", "batch images"], capsys)
        assert env["ok"] is True
        assert _names(env) == ["BatchImagesNode"]

    def test_search_include_deprecated(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["search", "batch images", "--include-deprecated"], capsys)
        assert set(_names(env)) == {"BatchImagesNode", "ImageBatch"}
        flags = {r["name"]: r.get("deprecated") for r in env["data"]["rows"]}
        assert flags == {"BatchImagesNode": None, "ImageBatch": True}

    def test_downstream_and_path_skip_deprecated(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["downstream", "VAEDecode"], capsys)
        assert env["ok"] is True
        assert "ImageBatch" not in _names(env)
        env = _run(nodes_cmd.app, ["path", "IMAGE", "IMAGE", "--emit-ops"], capsys)
        assert env["ok"] is True
        assert "ImageBatch" not in json.dumps(env["data"])

    def test_search_close_match_fallback_skips_deprecated(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["search", "OldSamplr"], capsys)
        assert env["ok"] is True
        assert "OldSampler" not in _names(env)

    def test_deprecated_only_producer_is_not_free(self):
        g = _graph()
        assert "OLDTHING" not in g.free_types()
        assert g._free_producer("OLDTHING", g.free_types()) is None

    def test_ls_hides_deprecated_by_default(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["ls", "--category", "image/batch"], capsys)
        assert _names(env) == ["BatchImagesNode"]
        assert env["data"]["filter"]["include_deprecated"] is None

    def test_ls_include_deprecated(self, patched_graph, capsys):
        env = _run(nodes_cmd.app, ["ls", "--category", "image/batch", "--include-deprecated"], capsys)
        assert set(_names(env)) == {"BatchImagesNode", "ImageBatch"}
        assert env["data"]["filter"]["include_deprecated"] is True


class TestExistingDeprecatedNodeReplays:
    """A deprecated class already on a graph is not the caller's choice, so the
    op batches that reproduce that graph must still apply."""

    def _workflow_with_image_batch(self) -> dict:
        wf = _base_workflow()
        wf["nodes"].append(
            {
                "id": 9,
                "type": "ImageBatch",
                "pos": [300, 300],
                "inputs": [
                    {"name": "image1", "type": "IMAGE", "link": None},
                    {"name": "image2", "type": "IMAGE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
                "widgets_values": [],
            }
        )
        wf["last_node_id"] = 9
        return wf

    def test_capture_then_apply(self, patched_graph, tmp_path, capsys):
        src = _write(tmp_path, self._workflow_with_image_batch(), "src.json")
        recipe = tmp_path / "recipe.json"
        env = _run(workflow_cmd.app, ["capture", str(src), "--out", str(recipe)], capsys)
        assert env["ok"] is True, env
        empty = _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}, "empty.json")
        env = _run(workflow_cmd.app, ["apply", str(empty), "--ops", str(recipe)], capsys)
        assert env["ok"] is True, env
        assert any(n["type"] == "ImageBatch" for n in json.loads(empty.read_text())["nodes"])

    def test_replace_ops_apply(self, patched_graph, tmp_path, capsys):
        from comfy_cli import workflow_ops

        empty = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        ops = workflow_ops.replace_ops(empty, self._workflow_with_image_batch())
        wf, applied, _ = workflow_ops.apply_specs(copy.deepcopy(empty), _graph(), ops)
        assert any(n["type"] == "ImageBatch" for n in wf["nodes"])
