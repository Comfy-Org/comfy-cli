"""Promoted subgraph widgets: the host-owned value model (the "agent editing subgraph" report, scenarios 1/2).

The frontend (``ComfyUI_frontend`` ADR 0009, ``SubgraphNode.ts``) represents a
promoted widget as a *linked subgraph input*: the subgraph definition declares
an input, a boundary link feeds it into an interior node's widget-backed input,
and the HOST instance owns the value — ``widgets_values[i]`` on the instance,
consumed positionally by the i-th subgraph input that resolves to a widget
(``_applyPromotedWidgetValues`` / ``serializeFromStoreState``). Socket-only
inputs (``VIDEO``, ``MODEL``) own no slot. The interior widget is only a
schema/default provider: *"the host/exterior value wins over the
interior/source value during repair, persistence, and prompt serialization."*

Before this model existed in the CLI, every read and write followed the legacy
``properties.proxyWidgets`` list to the interior node — so ``set-widget
57.width 768`` edited a value the frontend never serializes, and ``comfy run``
submitted the interior prompt on post-migration templates whose real prompt
lives on the host.

Fixtures are verbatim gallery templates (``tests/comfy_cli/fixtures/gallery``):

* ``image_z_image_turbo.json`` — pre-migration save (frontend 0.3.73): proxies
  already backed by linked inputs, no host values materialized.
* ``audio_minimax_music_3.json`` — post-migration save: 8 host values that
  differ from the interior defaults (interior caption is ``''``).
* ``api_seedance2_5_video_extend.json`` — mixed: two ``VIDEO`` socket inputs
  (no slot) and four widget inputs with host values.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from comfy_cli.cql import promoted
from comfy_cli.cql.engine import Graph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_GALLERY = _FIXTURES / "gallery"
OBJECT_INFO = _FIXTURES / "object_info_subgraph_promoted.json"

Z_IMAGE_SG = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(OBJECT_INFO.read_text()))


def _instance(wf: dict, node_id: int) -> dict:
    return next(n for n in wf["nodes"] if n["id"] == node_id)


def _def_of(wf: dict, instance: dict) -> dict:
    return next(d for d in wf["definitions"]["subgraphs"] if d["id"] == instance["type"])


# --------------------------------------------------------------------------- #
# the model: which subgraph inputs own a host value slot, and where they land
# --------------------------------------------------------------------------- #


def test_z_image_every_declared_input_is_a_promoted_widget():
    wf = _load("image_z_image_turbo.json")
    sg = _def_of(wf, _instance(wf, 57))
    pis = promoted.promoted_inputs(sg, promoted.defs_by_id(wf))
    assert [p.name for p in pis] == ["text", "width", "height", "seed", "steps", "unet_name", "clip_name", "vae_name"]
    assert [p.value_index for p in pis] == list(range(8))
    width = next(p for p in pis if p.name == "width")
    assert width.source_node == "13"
    assert width.source_widget == "width"
    assert width.type == "INT"


def test_socket_inputs_own_no_host_slot():
    wf = _load("api_seedance2_5_video_extend.json")
    inst = _instance(wf, 39)
    pis = promoted.promoted_inputs(_def_of(wf, inst), promoted.defs_by_id(wf))
    assert [(p.name, p.value_index) for p in pis] == [
        ("clip_to_resize", None),
        ("base_video", None),
        ("pad_second_video", 0),
        ("interpolation", 1),
        ("padding_color", 2),
        ("drop_audio", 3),
    ]
    # the captured frontend save has exactly one value per widget-backed input
    assert len(inst["widgets_values"]) == len([p for p in pis if p.value_index is not None])


# --------------------------------------------------------------------------- #
# reads: host value wins, interior is the fallback
# --------------------------------------------------------------------------- #


def test_effective_value_is_the_host_value_when_materialized(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    caption = promoted.effective_value(wf, inst, "caption", graph)
    assert caption.startswith("Global Metadata: Lo-fi hip-hop")
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 60
    # interior default is NOT what the frontend runs
    interior = next(n for n in _def_of(wf, inst)["nodes"] if n["id"] == 13)
    assert interior["widgets_values"][0] == ""


def test_effective_value_falls_back_to_the_interior_widget(graph):
    wf = _load("image_z_image_turbo.json")
    inst = _instance(wf, 57)
    assert inst["widgets_values"] == []
    assert promoted.effective_value(wf, inst, "width", graph) == 1024
    assert promoted.effective_value(wf, inst, "steps", graph) == 8
    assert promoted.effective_value(wf, inst, "unet_name", graph) == "z_image_turbo_bf16.safetensors"


def test_quarantined_host_value_wins_by_name(graph):
    """ADR 0009: a repaired-but-unresolved legacy entry keeps its host value in
    ``proxyWidgetErrorQuarantine``; the frontend reads it before ``widgets_values``."""
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    inst["properties"]["proxyWidgetErrorQuarantine"] = [
        {
            "originalEntry": ["-1", "max_duration"],
            "reason": "missingSourceWidget",
            "hostValue": 12,
            "attemptedAtVersion": 1,
        }
    ]
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 12


# --------------------------------------------------------------------------- #
# writes: materialize the host array in subgraph-input order, never the interior
# --------------------------------------------------------------------------- #


def test_set_host_value_materializes_the_full_array_in_input_order(graph):
    wf = _load("image_z_image_turbo.json")
    inst = _instance(wf, 57)
    before = copy.deepcopy(_def_of(wf, inst))
    promoted.set_host_value(wf, inst, "width", 768, graph)
    assert inst["widgets_values"] == [
        "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up. ",
        768,
        1024,
        0,
        8,
        "z_image_turbo_bf16.safetensors",
        "qwen_3_4b.safetensors",
        "ae.safetensors",
    ]
    assert _def_of(wf, inst) == before  # interior untouched


def test_set_host_value_updates_one_slot_of_an_existing_array(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    before = list(inst["widgets_values"])
    promoted.set_host_value(wf, inst, "max_duration", 90, graph)
    assert inst["widgets_values"][2] == 90
    assert inst["widgets_values"][:2] == before[:2]
    assert inst["widgets_values"][3:] == before[3:]


def test_set_host_value_rewrites_a_shadowing_quarantine_value(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    inst["properties"]["proxyWidgetErrorQuarantine"] = [
        {
            "originalEntry": ["-1", "max_duration"],
            "reason": "missingSourceWidget",
            "hostValue": 12,
            "attemptedAtVersion": 1,
        }
    ]
    promoted.set_host_value(wf, inst, "max_duration", 90, graph)
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 90


# --------------------------------------------------------------------------- #
# slots: the advertised surface reports the value the frontend will run
# --------------------------------------------------------------------------- #


def test_slots_report_host_values_on_a_post_migration_template(graph):
    wf = _load("audio_minimax_music_3.json")
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert slots["37.caption"]["current_value"].startswith("Global Metadata: Lo-fi hip-hop")
    assert slots["37.max_duration"]["current_value"] == 60
    assert slots["37.switch"]["current_value"] is True


def test_slots_skip_socket_inputs_and_flag_external_links(graph):
    wf = _load("api_seedance2_5_video_extend.json")
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert "39.clip_to_resize" not in slots
    assert slots["39.pad_second_video"]["current_value"] is False
    assert slots["39.interpolation"]["current_value"] == "lanczos"
