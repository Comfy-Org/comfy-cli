"""The catalog graph parses a node class on first use, not on load.

A catalog holds thousands of classes; a command that reads one workflow touches
a handful. These tests pin both halves of that contract: lookups parse only the
classes asked for, and everything that needs the whole catalog (type indexes,
listings, path search, annotations) answers exactly as a fully parsed graph.
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
from comfy_cli.cql import engine as cql_engine
from comfy_cli.cql.engine import Graph
from comfy_cli.cql.widget_catalog import build_types
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

FIXTURES = Path(__file__).parent.parent / "fixtures"
SD15_UI = FIXTURES / "sd15_ui_workflow.json"
SD15_OI = FIXTURES / "sd15_object_info.json"
PATH_OI = FIXTURES / "nodes_path_object_info.json"

SD15_CLASSES = {
    "CLIPTextEncode",
    "CheckpointLoaderSimple",
    "EmptyLatentImage",
    "KSampler",
    "SaveImage",
    "VAEDecode",
}


def _filler(i: int) -> dict[str, Any]:
    return {
        "input": {"required": {"image": ["IMAGE"], "amount": ["FLOAT", {"default": 1.0, "min": 0.0}]}},
        "input_order": {"required": ["image", "amount"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": f"filler/{i % 3}",
        "display_name": f"Filler {i}",
        "python_module": "custom_nodes.filler_pack",
    }


def _padded_sd15_object_info(n: int = 200) -> dict[str, Any]:
    info = json.loads(SD15_OI.read_text())
    for i in range(n):
        info[f"Filler{i:03d}"] = _filler(i)
    return info


@pytest.fixture
def parse_calls(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = cql_engine._parse_morphism

    def counting(node_id: str, raw: dict) -> cql_engine.Morphism:
        calls.append(node_id)
        return real(node_id, raw)

    monkeypatch.setattr(cql_engine, "_parse_morphism", counting)
    return calls


def _eager_reference(object_info: dict[str, Any]) -> dict[str, cql_engine.Morphism]:
    """Every class parsed up front, the way the graph used to build itself."""
    out: dict[str, cql_engine.Morphism] = {}
    for node_id, raw in object_info.items():
        if not isinstance(raw, dict):
            continue
        m = cql_engine._parse_morphism(node_id, raw)
        cql_engine._unconstrain_frontend_combos(m)
        out[node_id] = m
    return out


def _discovery_snapshot(g: Graph) -> dict[str, Any]:
    names = [m.id for m in g.all_nodes()]
    return {
        "all_nodes": [g.morphism_to_dict(m) for m in g.all_nodes()],
        "node_count": g.node_count(),
        "types": g.list_types(),
        "upstream": {n: [m.id for m in g.upstream(n)] for n in names},
        "downstream": {n: [m.id for m in g.downstream(n)] for n in names},
        "free_types": sorted(g.free_types()),
        "paths": g.find_paths("MODEL", "IMAGE"),
        "packs": g.packs(),
        "categories": g.category_tree(),
        "cloud_enabled": [m.id for m in g.cloud_enabled_nodes()],
    }


class TestLookupParsesOnlyWhatItTouches:
    def test_load_parses_nothing(self, parse_calls):
        Graph.from_object_info(_padded_sd15_object_info())
        assert parse_calls == []

    def test_node_lookup_parses_once(self, parse_calls):
        g = Graph.from_object_info(_padded_sd15_object_info())
        first = g.node("KSampler")
        assert g.node("KSampler") is first
        assert parse_calls == ["KSampler"]

    def test_validate_parses_only_workflow_classes(self, parse_calls):
        g = Graph.from_object_info(_padded_sd15_object_info())
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
            },
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["4", 2]}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        }
        g.validate_workflow(workflow)
        assert set(parse_calls) <= {"CheckpointLoaderSimple", "SaveImage", "VAEDecode", "EmptyLatentImage"}
        assert not any(c.startswith("Filler") for c in parse_calls)

    def test_workflow_print_parses_only_workflow_classes(self, parse_calls, tmp_path, capsys):
        oi = tmp_path / "object_info.json"
        oi.write_text(json.dumps(_padded_sd15_object_info()))
        reset_renderer_for_testing()
        r = Renderer.resolve(
            is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
        )
        r.mode = OutputMode.JSON
        set_renderer(r)
        try:
            result = CliRunner().invoke(
                workflow_cmd.app, ["print", str(SD15_UI), "--input", str(oi)], standalone_mode=False
            )
        finally:
            reset_renderer_for_testing()
        assert result.exception is None, result.exception
        assert parse_calls, "print should look up the workflow's classes"
        assert set(parse_calls) <= SD15_CLASSES
        assert len(parse_calls) == len(set(parse_calls))

    def test_discovery_parses_everything(self, parse_calls):
        info = _padded_sd15_object_info(20)
        g = Graph.from_object_info(info)
        g.upstream("VAEDecode")
        assert sorted(parse_calls) == sorted(info)

    def test_widget_catalog_parses_everything(self, parse_calls):
        # The agent refuses to start on a catalog `nodes widget-catalog` cannot
        # build, which holds only while building it parses every class.
        info = _padded_sd15_object_info(20)
        build_types(Graph.from_object_info(info))
        assert sorted(parse_calls) == sorted(info)


class TestMatchesFullyParsedGraph:
    @pytest.mark.parametrize("fixture", [SD15_OI, PATH_OI], ids=["sd15", "path"])
    def test_discovery_identical_whichever_is_touched_first(self, fixture):
        info = json.loads(fixture.read_text())
        index_first = Graph.from_object_info(info)
        index_first.list_types()
        lookups_first = Graph.from_object_info(info)
        for name in reversed(list(info)):
            lookups_first.node(name)
        assert _discovery_snapshot(index_first) == _discovery_snapshot(lookups_first)

    @pytest.mark.parametrize("fixture", [SD15_OI, PATH_OI], ids=["sd15", "path"])
    def test_morphisms_match_eager_parse(self, fixture):
        info = json.loads(fixture.read_text())
        g = Graph.from_object_info(info)
        ref = _eager_reference(info)
        assert [m.id for m in g.all_nodes()] == sorted(ref)
        for name, m in ref.items():
            assert g.node(name) == m

    def test_type_indexes_match_eager_build(self):
        info = json.loads(PATH_OI.read_text())
        info["OldDecoder"] = {**info["VAEDecode"], "deprecated": True}
        g = Graph.from_object_info(info)
        ref = _eager_reference(info)
        producers: dict[str, list[str]] = {}
        consumers: dict[str, list[str]] = {}
        types: set[str] = set()
        for m in ref.values():
            for t in m.output_types():
                if not m.deprecated:
                    producers.setdefault(t, []).append(m.id)
                types.add(t)
            for t in m.input_link_types():
                if not m.deprecated:
                    consumers.setdefault(t, []).append(m.id)
                types.add(t)
        assert {t: [m.id for m in ms] for t, ms in g._producers.items()} == {t: sorted(v) for t, v in producers.items()}
        assert {t: [m.id for m in ms] for t, ms in g._consumers.items()} == {t: sorted(v) for t, v in consumers.items()}
        assert g._types == types
        assert "OldDecoder" not in [m.id for m in g.downstream("CheckpointLoaderSimple")]
        assert g.node("OldDecoder") is not None and g.node("OldDecoder").deprecated

    def test_non_object_entries_are_absent(self):
        info = json.loads(SD15_OI.read_text())
        info["NotAClass"] = ["not", "an", "object"]
        info["AlsoNot"] = None
        g = Graph.from_object_info(info)
        assert g.node("NotAClass") is None and g.node("AlsoNot") is None
        assert g.node_count() == len(info) - 2
        assert "NotAClass" not in [m.id for m in g.all_nodes()]
        assert "NotAClass" not in g._nodes
        assert list(g._nodes) == [k for k, v in info.items() if isinstance(v, dict)]
        with pytest.raises(KeyError):
            g._nodes["NotAClass"]

    def test_unhashable_name_is_absent(self):
        g = Graph.from_object_info(json.loads(SD15_OI.read_text()))
        assert g.node(["KSampler"]) is None  # type: ignore[arg-type]

    def test_frontend_defined_combo_unconstrained_on_lookup(self):
        info = {
            "CustomCombo": {
                "input": {"required": {"choice": [["a", "b"]]}},
                "output": ["STRING"],
                "output_name": ["STRING"],
            }
        }
        port = Graph.from_object_info(info).node("CustomCombo").inputs[0]
        assert port.enum_values == [] and port.enum_declared is False

    def test_object_info_is_the_payload_passed_in(self):
        info = json.loads(SD15_OI.read_text())
        assert Graph.from_object_info(info).object_info is info
        assert Graph().object_info == {} and Graph().node_count() == 0


SUPPORTED = b"""
node_packs:
  - name: filler-pack
    node_labels:
      Filler000: [NetworkAccess]
      KSampler: [Sampler]
"""
DISABLED = b"""
disable_nodes:
  or:
    - NetworkAccess: true
"""


def _annotation_view(g: Graph) -> dict[str, tuple]:
    return {m.id: (m.pack, m.labels, m.cloud_disabled) for m in g.all_nodes()}


class TestAnnotationsReachLazilyParsedClasses:
    @pytest.fixture(autouse=True)
    def _no_annotation_cache(self, monkeypatch):
        monkeypatch.setenv("COMFY_NO_CACHE", "1")

    def test_annotate_before_and_after_parse_agree(self):
        info = _padded_sd15_object_info(5)
        before = Graph.from_object_info(copy.deepcopy(info))
        before.annotate(SUPPORTED, DISABLED)
        after = Graph.from_object_info(copy.deepcopy(info))
        after.all_nodes()
        after.annotate(SUPPORTED, DISABLED)
        mixed = Graph.from_object_info(copy.deepcopy(info))
        mixed.node("KSampler")
        mixed.annotate(SUPPORTED, DISABLED)
        view = _annotation_view(after)
        assert _annotation_view(before) == view
        assert _annotation_view(mixed) == view
        assert view["Filler000"][2] is True and view["Filler001"][2] is False

    def test_annotation_layers_replay_in_order(self):
        info = _padded_sd15_object_info(3)
        other = b"""
node_packs:
  - name: other-pack
    node_labels:
      Filler000: []
      Filler001: [Sampler]
"""
        eager = Graph.from_object_info(copy.deepcopy(info))
        eager.all_nodes()
        eager.annotate(SUPPORTED, DISABLED)
        eager.annotate(other, None)
        lazy = Graph.from_object_info(copy.deepcopy(info))
        lazy.annotate(SUPPORTED, DISABLED)
        lazy.annotate(other, None)
        view = _annotation_view(lazy)
        assert view == _annotation_view(eager)
        assert view["Filler000"] == ("other-pack", [], False)
        assert view["KSampler"][:2] == ("filler-pack", ["Sampler"])
