"""Legacy ``properties.proxyWidgets`` → linked-input repair: a port of the
frontend's forward migration (``proxyWidgetMigration.ts``, ADR-SUBGRAPH-PROMOTION-0009).

The frontend runs ``flushProxyWidgetMigration`` on every subgraph host at load
time. The CLI runs the same flush on the instance whose legacy promotion a
write is about to edit, so the write can land on the HOST value the way it
does for every other promoted widget. These tests pin the port against real
gallery templates (``tests/comfy_cli/fixtures/gallery``):

* ``templates_graphic_design_recomposer.json`` — instance 143: a value widget
  with a labelled backing slot (``['135','choice']``) plus two ``PrimitiveNode``
  fan-outs (``['156','value']`` → two targets, ``['142','value']`` → two
  targets), both primitives renamed by the user.
* ``template_horizontal_vertical_extension.json`` — instance 252: three
  ``$$canvas-image-preview`` exposures plus a value widget whose interior node
  serializes no ``inputs[]`` at all (``['285','aspect_ratio']``).
* ``flux_dev_checkpoint_example.json`` — instance 56: seven entries already
  backed by linked inputs, ``['52','seed']`` repairable, and
  ``['52','control_after_generate']`` — a frontend-only widget with no backing
  input slot, which the frontend quarantines.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql import promoted
from comfy_cli.cql.engine import Graph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_GALLERY = _FIXTURES / "gallery"
OBJECT_INFO = _FIXTURES / "object_info_legacy_proxy.json"

RECOMPOSER = "templates_graphic_design_recomposer.json"
EXTENSION = "template_horizontal_vertical_extension.json"
FLUX = "flux_dev_checkpoint_example.json"

FLUX_SEED = 53943644181156


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(OBJECT_INFO.read_text()))


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


def _instance(wf: dict, node_id: int) -> dict:
    return next(n for n in wf["nodes"] if n["id"] == node_id)


def _def_of(wf: dict, instance: dict) -> dict:
    return next(d for d in wf["definitions"]["subgraphs"] if d["id"] == instance["type"])


def _inner(sg: dict, node_id: int) -> dict:
    return next(n for n in sg["nodes"] if n["id"] == node_id)


def _link(sg: dict, link_id) -> dict | None:
    return next((x for x in sg["links"] if x["id"] == link_id), None)


def _plans(wf: dict, instance_id: int, graph) -> dict[tuple, promoted.LegacyEntry]:
    return {tuple(e.original): e for e in promoted.plan_proxy_migration(wf, _instance(wf, instance_id), graph)}


OUTER = "0f6b0a4e-outer-0000-0000-000000000001"
INNER = "0f6b0a4e-inner-0000-0000-000000000002"


def _nested_legacy_workflow(entries: list[list[str]]) -> dict:
    """Synthetic: a legacy promotion whose SOURCE is a nested subgraph instance
    (top 10 → OUTER's node 5, an INNER instance) and whose projecting input
    (``prompt_text``) is named differently from the widget it projects
    (CLIPTextEncode 27's ``text``). Node 5 serializes no ``inputs[]`` — the
    older-save shape — so the repair must synthesize the slot. Node/link
    shapes are copied from ``flux_dev_checkpoint_example.json``."""
    return {
        "last_node_id": 10,
        "last_link_id": 0,
        "nodes": [{"id": 10, "type": OUTER, "inputs": [], "outputs": [], "properties": {"proxyWidgets": entries}}],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": OUTER,
                    "name": "Outer",
                    "state": {"lastGroupId": 0, "lastNodeId": 5, "lastLinkId": 0, "lastRerouteId": 0},
                    "inputs": [],
                    "outputs": [],
                    "nodes": [{"id": 5, "type": INNER, "inputs": [], "outputs": [], "widgets_values": []}],
                    "links": [],
                },
                {
                    "id": INNER,
                    "name": "Inner",
                    "state": {"lastGroupId": 0, "lastNodeId": 27, "lastLinkId": 1, "lastRerouteId": 0},
                    "inputs": [{"id": "i1", "name": "prompt_text", "type": "STRING", "linkIds": [1]}],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 27,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 1},
                            ],
                            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
                            "widgets_values": ["hi"],
                        }
                    ],
                    "links": [
                        {
                            "id": 1,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 27,
                            "target_slot": 1,
                            "type": "STRING",
                        }
                    ],
                },
            ]
        },
    }


# --------------------------------------------------------------------------- #
# nextUniqueName — the frontend's collision rule, ported verbatim
# --------------------------------------------------------------------------- #


def test_next_unique_name_appends_underscore_counter():
    assert promoted.next_unique_name("seed", []) == "seed"
    assert promoted.next_unique_name("seed", ["seed"]) == "seed_1"
    assert promoted.next_unique_name("seed", ["seed", "seed_1"]) == "seed_2"
    # the counter restarts from the BASE name, never from an existing suffix
    assert promoted.next_unique_name("seed", ["seed", "seed_2"]) == "seed_1"


# --------------------------------------------------------------------------- #
# classification (``classify`` in the frontend)
# --------------------------------------------------------------------------- #


def test_flux_plan_classifies_linked_repairable_and_unrepairable(graph):
    wf = _load(FLUX)
    plans = _plans(wf, 56, graph)
    linked = [e for e in plans.values() if e.plan == "alreadyLinked"]
    assert [e.name for e in linked] == ["text", "width", "height", "unet_name", "clip_name1", "clip_name2", "vae_name"]
    seed = plans[("52", "seed")]
    assert seed.plan == "createSubgraphInput"
    assert seed.name == "seed"
    assert seed.type == "INT"
    assert seed.host_value is promoted.UNSET  # widgets_values is [] — a hole
    cag = plans[("52", "control_after_generate")]
    assert cag.plan == "quarantine"
    assert cag.reason == "missingSubgraphInput"  # a widget with no backing input slot


def test_recomposer_plan_repairs_primitive_fanouts_under_the_user_title(graph):
    wf = _load(RECOMPOSER)
    plans = _plans(wf, 143, graph)
    choice = plans[("135", "choice")]
    assert choice.plan == "createSubgraphInput" and choice.name == "choice" and choice.type == "COMBO"
    resolution = plans[("156", "value")]
    assert resolution.plan == "primitiveBypass"
    assert resolution.name == "resolution"  # the primitive's user title, not its widget name
    assert resolution.type == "COMBO"
    assert resolution.targets == [("130", 4), ("157", 2)]
    aspect = plans[("142", "value")]
    assert aspect.plan == "primitiveBypass" and aspect.name == "aspect_ratio"
    assert aspect.targets == [("130", 3), ("157", 1)]


def test_extension_plan_keeps_previews_out_of_the_value_path(graph):
    wf = _load(EXTENSION)
    plans = _plans(wf, 252, graph)
    previews = [e for e in plans.values() if e.plan == "previewExposure"]
    assert [tuple(e.original) for e in previews] == [
        ("233", "$$canvas-image-preview"),
        ("232", "$$canvas-image-preview"),
        ("234", "$$canvas-image-preview"),
    ]
    aspect = plans[("285", "aspect_ratio")]
    assert aspect.plan == "createSubgraphInput" and aspect.type == "COMBO"


def test_plan_quarantines_missing_node_missing_widget_and_host_self_reference(graph):
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    inst["properties"]["proxyWidgets"] = [["999", "seed"], ["52", "nope"], ["-1", "width"]]
    inst["widgets_values"] = [1, 2, 640]
    plans = _plans(wf, 56, graph)
    assert plans[("999", "seed")].plan == "quarantine" and plans[("999", "seed")].reason == "missingSourceNode"
    assert plans[("52", "nope")].reason == "missingSourceWidget"
    # ``-1`` names the host itself: no interior node has that id, so the
    # frontend quarantines it too — but its host value is preserved.
    minus_one = plans[("-1", "width")]
    assert minus_one.reason == "missingSourceNode"
    assert minus_one.host_value == 640


def test_plan_quarantines_a_primitive_cohort_with_mixed_widget_names(graph):
    """``['156','value']`` + ``['156','control_after_generate']`` name ONE
    primitive with two widgets: the frontend's cohort validation fails and
    every entry of the cohort is quarantined (all-or-quarantine)."""
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    inst["properties"]["proxyWidgets"].append(["156", "control_after_generate"])
    plans = _plans(wf, 143, graph)
    assert plans[("156", "value")].plan == "quarantine"
    assert plans[("156", "value")].reason == "primitiveBypassFailed"
    assert plans[("156", "control_after_generate")].reason == "primitiveBypassFailed"
    # the other primitive is unaffected
    assert plans[("142", "value")].plan == "primitiveBypass"


def test_plan_quarantines_a_primitive_with_no_fanout(graph):
    wf = _load(RECOMPOSER)
    sg = _def_of(wf, _instance(wf, 143))
    sg["links"] = [x for x in sg["links"] if x["origin_id"] != 156]
    _inner(sg, 156)["outputs"][0]["links"] = []
    assert _plans(wf, 143, graph)[("156", "value")].reason == "unlinkedSourceWidget"


def test_plan_uses_a_unique_name_when_the_widget_name_is_taken(graph):
    wf = _load(FLUX)
    sg = _def_of(wf, _instance(wf, 56))
    sg["inputs"].append({"id": "x", "name": "seed", "type": "INT", "linkIds": []})
    assert _plans(wf, 56, graph)[("52", "seed")].name == "seed_1"


def test_plan_normalizes_a_node_prefixed_widget_name(graph):
    """An older save could prefix the widget name with its node id
    (``normalizeLegacyProxyWidgetEntry``): the prefix is stripped and kept as
    the disambiguator; the repair claims the bare widget."""
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    inst["properties"]["proxyWidgets"] = [["52", "52:seed"]]
    entry = _plans(wf, 56, graph)[("52", "52:seed")]
    assert entry.plan == "createSubgraphInput"
    assert entry.widget == "seed"
    assert entry.disambiguator == "52"
    assert entry.name == "seed"
    assert entry.key == "52.seed"


def test_plan_resolves_a_nested_instance_source_by_disambiguator(graph):
    """Through a nested instance the disambiguator must name the interior
    node the projecting input lands on (``resolveSourceWidget``)."""
    wf = _nested_legacy_workflow([["5", "27:text"], ["5", "99:text"]])
    plans = _plans(wf, 10, graph)
    hit = plans[("5", "27:text")]
    assert hit.plan == "createSubgraphInput"
    assert (hit.widget, hit.disambiguator, hit.source_input, hit.type) == ("text", "27", "prompt_text", "STRING")
    assert plans[("5", "99:text")].reason == "missingSourceWidget"


# --------------------------------------------------------------------------- #
# flush: the structural repair
# --------------------------------------------------------------------------- #


def test_flush_creates_the_input_the_boundary_link_and_the_host_value(graph):
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    sg = _def_of(wf, inst)
    ids = promoted.plan_repair_ids(["56"], promoted.plan_proxy_migration(wf, inst, graph))
    report = promoted.flush_proxy_migration(wf, inst, graph, ids=ids)

    assert report.created == ["seed"]
    new_input = sg["inputs"][-1]
    link_id = ids["52.seed"]["links"][0]
    assert new_input == {"id": ids["52.seed"]["input"], "name": "seed", "type": "INT", "linkIds": [link_id]}
    assert _link(sg, link_id) == {
        "id": link_id,
        "origin_id": -10,
        "origin_slot": 7,
        "target_id": 52,
        "target_slot": 4,
        "type": "INT",
    }
    # the interior input entry is created the way the frontend serializes a
    # converted widget (flux_dev_checkpoint_example.json node 50 ``width``)
    ksampler = _inner(sg, 52)
    assert ksampler["inputs"][4] == {
        "localized_name": "seed",
        "name": "seed",
        "type": "INT",
        "widget": {"name": "seed"},
        "link": link_id,
    }
    assert ksampler["widgets_values"][0] == FLUX_SEED  # interior untouched
    assert sg["state"]["lastLinkId"] >= link_id
    # host: the instance carries the value, materialized in subgraph-input order
    pis = promoted.promoted_inputs(sg, promoted.defs_by_id(wf))
    assert [p.name for p in pis if p.is_widget][-1] == "seed"
    assert inst["widgets_values"][-1] == FLUX_SEED
    assert inst["inputs"][-1] == {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None}
    # consumed: canonical saves do not re-emit repaired entries
    assert "proxyWidgets" not in inst["properties"]
    assert inst["properties"]["proxyWidgetErrorQuarantine"] == [
        {"originalEntry": ["52", "control_after_generate"], "reason": "missingSubgraphInput", "attemptedAtVersion": 1}
    ]


def test_flush_reads_legacy_host_values_by_proxy_position(graph):
    """A pre-migration save stores host values positionally by ``proxyWidgets``
    order (``pickHostValue``): those win over the interior defaults, for
    already-linked and freshly created inputs alike."""
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    # proxy order: text, width, height, unet_name, clip_name1, clip_name2, vae_name, seed, c_a_g
    inst["widgets_values"] = [None, 640]
    promoted.flush_proxy_migration(wf, inst, graph)
    sg = _def_of(wf, inst)
    by_name = {p.name: p for p in promoted.promoted_inputs(sg, promoted.defs_by_id(wf))}
    assert promoted.host_value(inst, by_name["width"]) == 640
    assert promoted.host_value(inst, by_name["text"]) is None  # an explicit null is a value, not a hole
    assert promoted.host_value(inst, by_name["height"]) == 1024  # hole → interior default
    assert promoted.host_value(inst, by_name["seed"]) == FLUX_SEED  # hole → seeded from the source widget

    wf = _load(FLUX)
    inst = _instance(wf, 56)
    inst["widgets_values"] = [None] * 7 + [777]
    promoted.flush_proxy_migration(wf, inst, graph)
    sg = _def_of(wf, inst)
    by_name = {p.name: p for p in promoted.promoted_inputs(sg, promoted.defs_by_id(wf))}
    assert promoted.host_value(inst, by_name["seed"]) == 777  # a legacy host value beats the interior


def test_flush_preserves_the_source_slot_label(graph):
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    promoted.flush_proxy_migration(wf, inst, graph)
    sg = _def_of(wf, inst)
    choice = next(i for i in sg["inputs"] if i["name"] == "choice")
    assert choice["label"] == "model"
    assert promoted.instance_input(inst, "choice")["label"] == "model"


def test_flush_synthesizes_the_interior_input_entry_when_none_is_serialized(graph):
    wf = _load(EXTENSION)
    inst = _instance(wf, 252)
    sg = _def_of(wf, inst)
    assert _inner(sg, 285).get("inputs") == []
    ids = promoted.plan_repair_ids(["252"], promoted.plan_proxy_migration(wf, inst, graph))
    promoted.flush_proxy_migration(wf, inst, graph, ids=ids)
    link_id = ids["285.aspect_ratio"]["links"][0]
    assert _inner(sg, 285)["inputs"] == [
        {
            "localized_name": "aspect_ratio",
            "name": "aspect_ratio",
            "type": "COMBO",
            "widget": {"name": "aspect_ratio"},
            "link": link_id,
        }
    ]
    assert _link(sg, link_id)["target_slot"] == 0
    assert inst["widgets_values"] == ["9:16 (Portrait Widescreen)"]


def test_flush_synthesizes_a_nested_instance_slot_under_its_projecting_input(graph):
    """The slot a nested instance backs a promoted widget with is its own
    host input — named after the PROJECTING INPUT, not the interior widget
    (``getSlotFromWidget(promotedInputWidget(input))``). Only that name lets
    the outer definition resolve the promotion through the inner one."""
    wf = _nested_legacy_workflow([["5", "text"]])
    inst = _instance(wf, 10)
    outer = _def_of(wf, inst)
    ids = promoted.plan_repair_ids(["10"], promoted.plan_proxy_migration(wf, inst, graph))
    promoted.flush_proxy_migration(wf, inst, graph, ids=ids)
    link_id = ids["5.text"]["links"][0]
    assert _inner(outer, 5)["inputs"] == [
        {"name": "prompt_text", "type": "STRING", "widget": {"name": "prompt_text"}, "link": link_id}
    ]
    assert _link(outer, link_id)["target_slot"] == 0
    pis = promoted.promoted_inputs(outer, promoted.defs_by_id(wf))
    assert [(p.name, p.value_index, p.nested, p.source_node, p.source_input) for p in pis] == [
        ("text", 0, True, "5", "prompt_text")
    ]
    assert inst["widgets_values"] == ["hi"]
    promoted.set_host_value(wf, inst, "text", "yo", graph)
    assert promoted.effective_value(wf, inst, "text", graph) == "yo"
    # the outer promotion resolves through the inner one to the concrete widget
    assert promoted.deepest_source(outer, pis[0], promoted.defs_by_id(wf)) == (["5", "27"], "text")


def test_flush_ignores_a_malformed_link_entry(graph):
    """A stray non-dict among ``links`` is skipped like everywhere else in the
    module, not stack-traced — the primitive repair scans links too."""
    clean = _load(RECOMPOSER)
    promoted.flush_proxy_migration(clean, _instance(clean, 143), graph)
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    sg["links"].insert(0, ["not", "a", "link"])
    report = promoted.flush_proxy_migration(wf, inst, graph)
    assert report.created == ["choice", "resolution", "aspect_ratio"]
    assert sg["links"][0] == ["not", "a", "link"]
    sg["links"].pop(0)
    assert json.dumps(wf, sort_keys=True) == json.dumps(clean, sort_keys=True)


def _wire_interior_feeder_into_choice(sg: dict, link_id: int = 999001) -> None:
    """Feed CustomCombo 135's ``choice`` slot from interior node 2
    (PrimitiveStringMultiline) — the shape ``resolve_write`` refuses to
    widget-write when no legacy entry names the slot."""
    sg["links"].append(
        {"id": link_id, "origin_id": 2, "origin_slot": 0, "target_id": 135, "target_slot": 0, "type": "STRING"}
    )
    _inner(sg, 135)["inputs"][0]["link"] = link_id
    _inner(sg, 2)["outputs"][0]["links"].append(link_id)


def test_flush_replaces_an_interior_link_feeding_the_source_slot(graph):
    """A legacy entry whose backing slot is already fed by an interior link is
    still repaired, and the boundary link REPLACES that link — exactly what
    the frontend does: ``repairCreateSubgraphInput`` calls
    ``SubgraphInput.connect(slot, node)`` unconditionally, and ``connect``
    resolves ``inputLink(...)`` for the slot, ``replaceLinkTopology``s it and
    ``_disconnectNodeInput``s the incumbent (``SubgraphInput.ts``). After the
    frontend loads such a file the interior feeder is gone and the host value
    runs, so this is the state a write has to target."""
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    _wire_interior_feeder_into_choice(sg)
    assert _plans(wf, 143, graph)[("135", "choice")].plan == "createSubgraphInput"
    ids = promoted.plan_repair_ids(["143"], promoted.plan_proxy_migration(wf, inst, graph))
    promoted.flush_proxy_migration(wf, inst, graph, ids=ids)
    boundary = ids["135.choice"]["links"][0]
    assert _link(sg, 999001) is None
    assert _inner(sg, 2)["outputs"][0]["links"] == [211, 247]
    assert _inner(sg, 135)["inputs"][0]["link"] == boundary
    assert _link(sg, boundary)["origin_id"] == -10
    assert inst["widgets_values"][0] == "Nano Banana 2"


def test_flush_reuses_a_slot_serialized_by_name_only(graph):
    """An older save can serialize a widget's input slot with ``name`` but no
    ``widget`` marker: that slot is the backing slot (matched by name), gets
    its marker back, and is never duplicated."""
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    del _inner(sg, 135)["inputs"][0]["widget"]
    entry = _plans(wf, 143, graph)[("135", "choice")]
    assert entry.plan == "createSubgraphInput" and entry.slot_index == 0 and entry.label == "model"
    ids = promoted.plan_repair_ids(["143"], promoted.plan_proxy_migration(wf, inst, graph))
    promoted.flush_proxy_migration(wf, inst, graph, ids=ids)
    assert _inner(sg, 135)["inputs"] == [
        {
            "label": "model",
            "localized_name": "choice",
            "name": "choice",
            "type": "COMBO",
            "widget": {"name": "choice"},
            "link": ids["135.choice"]["links"][0],
        }
    ]
    assert next(i for i in sg["inputs"] if i["name"] == "choice")["label"] == "model"


def test_planned_repair_refuses_an_ambiguous_legacy_alias(graph):
    """Two entries share the legacy widget name ``value`` (both primitives):
    the alias is refused with the minted names, never resolved to one of them."""
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    with pytest.raises(ValueError, match=r"143\.resolution.*143\.aspect_ratio"):
        promoted.planned_repair(wf, inst, graph, "value")
    inst["properties"]["proxyWidgets"] = [e for e in inst["properties"]["proxyWidgets"] if e != ["142", "value"]]
    assert promoted.planned_repair(wf, inst, graph, "value").name == "resolution"


def test_flush_leaves_preview_exposures_in_place(graph):
    """``$$`` entries are display-only: no input, no value, and the entry stays
    in ``proxyWidgets`` for the frontend's own preview-exposure migration
    (which also auto-exposes preview nodes when ``previewExposures`` is unset)."""
    wf = _load(EXTENSION)
    inst = _instance(wf, 252)
    before = copy.deepcopy(inst["properties"]["proxyWidgets"])
    report = promoted.flush_proxy_migration(wf, inst, graph)
    assert report.remaining == before[:3]
    assert inst["properties"]["proxyWidgets"] == before[:3]
    assert "previewExposures" not in inst["properties"]
    assert "proxyWidgetErrorQuarantine" not in inst["properties"]
    assert [i["name"] for i in _def_of(wf, inst)["inputs"]] == ["image", "aspect_ratio"]


def test_flush_bypasses_a_primitive_fanout_through_one_input(graph):
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    ids = promoted.plan_repair_ids(["143"], promoted.plan_proxy_migration(wf, inst, graph))
    report = promoted.flush_proxy_migration(wf, inst, graph, ids=ids)

    assert report.created == ["choice", "resolution", "aspect_ratio"]  # creates first, then primitive cohorts
    assert [i["name"] for i in sg["inputs"]] == ["images", "choice", "resolution", "aspect_ratio"]
    resolution = sg["inputs"][2]
    links = ids["156.value"]["links"]
    assert resolution == {"id": ids["156.value"]["input"], "name": "resolution", "type": "COMBO", "linkIds": links}
    # former primitive links are gone; every target is reconnected in target order
    assert _link(sg, 243) is None and _link(sg, 246) is None
    assert _link(sg, links[0]) == {
        "id": links[0],
        "origin_id": -10,
        "origin_slot": 2,
        "target_id": 130,
        "target_slot": 4,
        "type": "COMBO",
    }
    assert _link(sg, links[1])["target_id"] == 157 and _link(sg, links[1])["target_slot"] == 2
    assert _inner(sg, 130)["inputs"][4]["link"] == links[0]
    assert _inner(sg, 157)["inputs"][2]["link"] == links[1]
    # the primitive stays, disconnected and inert, marked with its bypass
    primitive = _inner(sg, 156)
    assert primitive["outputs"][0]["links"] == []
    assert primitive["properties"]["proxyBypassedToSubgraphInput"] == "resolution"
    assert primitive["widgets_values"] == ["2K", "fixed", ""]
    # host values: seeded from the source widgets (the primitive's own value)
    assert inst["widgets_values"] == ["Nano Banana 2", "2K", "16:9"]
    assert "proxyWidgets" not in inst["properties"]


def test_flush_repairs_creates_before_primitive_cohorts(graph):
    """Entry order is the frontend's: value widgets first, primitive cohorts
    after — so a value entry naming a primitive's own target takes that
    target over (its link is replaced), the cohort is re-collected against
    the mutated graph, and the primitive's input is named around the
    collision (``resolution`` → ``resolution_1``)."""
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    inst["properties"]["proxyWidgets"].insert(1, ["130", "resolution"])
    promoted.flush_proxy_migration(wf, inst, graph)
    assert [i["name"] for i in sg["inputs"]] == ["images", "choice", "resolution", "resolution_1", "aspect_ratio"]
    by_name = {i["name"]: i for i in sg["inputs"]}
    assert _inner(sg, 130)["inputs"][4]["link"] == by_name["resolution"]["linkIds"][0]
    assert [_link(sg, x)["target_id"] for x in by_name["resolution_1"]["linkIds"]] == [157]
    assert _link(sg, 243) is None and _link(sg, 246) is None
    assert inst["widgets_values"] == ["Nano Banana 2", "2K", "2K", "16:9"]


def test_flush_coalesces_duplicate_primitive_entries(graph):
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    inst["properties"]["proxyWidgets"].append(["156", "value"])
    promoted.flush_proxy_migration(wf, inst, graph)
    sg = _def_of(wf, inst)
    assert [i["name"] for i in sg["inputs"]].count("resolution") == 1
    assert "proxyWidgetErrorQuarantine" not in inst["properties"]


def test_flush_quarantines_a_type_incompatible_primitive_target(graph):
    wf = _load(RECOMPOSER)
    inst = _instance(wf, 143)
    sg = _def_of(wf, inst)
    _inner(sg, 157)["inputs"][2]["type"] = "STRING"
    promoted.flush_proxy_migration(wf, inst, graph)
    assert [i["name"] for i in sg["inputs"]] == ["images", "choice", "aspect_ratio"]
    assert _link(sg, 243) is not None and _link(sg, 246) is not None  # untouched: all-or-quarantine
    assert {tuple(q["originalEntry"]): q["reason"] for q in inst["properties"]["proxyWidgetErrorQuarantine"]} == {
        ("156", "value"): "primitiveBypassFailed"
    }


def test_flush_is_idempotent_and_deterministic(graph):
    wf = _load(RECOMPOSER)
    a, b = copy.deepcopy(wf), copy.deepcopy(wf)
    promoted.flush_proxy_migration(a, _instance(a, 143), graph)
    promoted.flush_proxy_migration(b, _instance(b, 143), graph)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    once = json.dumps(a, sort_keys=True)
    report = promoted.flush_proxy_migration(a, _instance(a, 143), graph)
    assert report.created == [] and report.quarantined == []
    assert json.dumps(a, sort_keys=True) == once


def test_flush_consumes_an_entry_a_previous_repair_already_linked(graph):
    """Replaying a repair against a document another replica already repaired
    finds the entry ``alreadyLinked`` and changes nothing structural."""
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    promoted.flush_proxy_migration(wf, inst, graph)
    inst["properties"]["proxyWidgets"] = [["52", "seed"]]
    report = promoted.flush_proxy_migration(wf, inst, graph)
    assert report.created == []
    assert report.consumed == [["52", "seed"]]
    assert [i["name"] for i in _def_of(wf, inst)["inputs"]].count("seed") == 1


# --------------------------------------------------------------------------- #
# flush: a catalog gap is not a final quarantine
# --------------------------------------------------------------------------- #


_UNKNOWN_SOURCE_OUTER = "0f6b0a4e-unknown-0000-0000-000000000003"


def _unknown_node_legacy_workflow(host_value: float = 2.0) -> dict:
    """Synthetic: a legacy promotion whose source is an interior node of a
    type the (local) catalog has never heard of — an uninstalled custom node.
    ``_resolve_source_widget`` cannot enumerate its widgets at all (``graph``
    has no entry for the type), which is indistinguishable, from inside
    ``plan_proxy_migration`` alone, from a genuine name mismatch. Only
    ``_source_widget_resolution_is_catalog_gap`` tells them apart."""
    return {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": _UNKNOWN_SOURCE_OUTER,
                "inputs": [],
                "outputs": [],
                "properties": {"proxyWidgets": [["5", "scale"]]},
                "widgets_values": [host_value],
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": _UNKNOWN_SOURCE_OUTER,
                    "name": "Outer",
                    "state": {"lastGroupId": 0, "lastNodeId": 5, "lastLinkId": 0, "lastRerouteId": 0},
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 5,
                            "type": "CustomNodeNotInstalled",
                            "inputs": [],
                            "outputs": [],
                            "widgets_values": [host_value],
                        }
                    ],
                    "links": [],
                }
            ]
        },
    }


def test_plan_marks_a_missing_source_widget_as_a_catalog_gap_only_when_the_type_is_unknown(graph):
    wf = _unknown_node_legacy_workflow()
    entry = _plans(wf, 1, graph)[("5", "scale")]
    assert entry.plan == "quarantine" and entry.reason == "missingSourceWidget"
    assert entry.catalog_unresolvable is True


def test_flush_leaves_a_catalog_gap_quarantine_pending_instead_of_permanent(graph):
    """Before this fix every ``missingSourceWidget`` quarantine — including one
    that exists only because THIS replica's catalog doesn't know the source
    node's type — was moved into ``proxyWidgetErrorQuarantine`` and dropped
    from ``proxyWidgets`` for good. A replica with that custom node installed
    could still have migrated it, so the entry must stay pending instead.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4623941923
    """
    wf = _unknown_node_legacy_workflow()
    inst = _instance(wf, 1)
    report = promoted.flush_proxy_migration(wf, inst, graph)
    assert report.created == [] and report.quarantined == []
    assert inst["properties"]["proxyWidgets"] == [["5", "scale"]]
    assert "proxyWidgetErrorQuarantine" not in inst["properties"]


def test_flush_still_permanently_quarantines_a_known_node_missing_widget(graph):
    """A genuine name mismatch on a node type the catalog DOES know (the FLUX
    fixture's ``control_after_generate``, a real frontend-only widget with no
    backing input slot) is not a catalog gap, and still quarantines for
    good — the fix narrows, it does not widen, what stays pending."""
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    inst["properties"]["proxyWidgets"] = [["52", "control_after_generate"]]
    entry = _plans(wf, 56, graph)[("52", "control_after_generate")]
    assert entry.reason == "missingSubgraphInput"
    assert entry.catalog_unresolvable is False
    promoted.flush_proxy_migration(wf, inst, graph)
    assert "proxyWidgets" not in inst["properties"]
    assert inst["properties"]["proxyWidgetErrorQuarantine"] == [
        {"originalEntry": ["52", "control_after_generate"], "reason": "missingSubgraphInput", "attemptedAtVersion": 1}
    ]


_MIXED_GAP_OUTER = "0f6b0a4e-mixedgap-0000-000000000004"


def _mixed_catalog_gap_and_create_workflow(scale_value: float = 2.0, width_value: int = 768) -> dict:
    """Synthetic: TWO legacy entries in one ``proxyWidgets`` — ``scale``
    backed by an uninstalled custom node (a catalog gap, left pending) and
    ``width`` backed by a known node (migrated in this same flush)."""
    return {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": _MIXED_GAP_OUTER,
                "inputs": [],
                "outputs": [],
                "properties": {"proxyWidgets": [["5", "scale"], ["6", "width"]]},
                "widgets_values": [scale_value, width_value],
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": _MIXED_GAP_OUTER,
                    "name": "Outer",
                    "state": {"lastGroupId": 0, "lastNodeId": 6, "lastLinkId": 0, "lastRerouteId": 0},
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 5,
                            "type": "CustomNodeNotInstalled",
                            "inputs": [],
                            "outputs": [],
                            "widgets_values": [scale_value],
                        },
                        {
                            "id": 6,
                            "type": "EmptySD3LatentImage",
                            "inputs": [],
                            "outputs": [],
                            "widgets_values": [width_value, 1024, 1],
                        },
                    ],
                    "links": [],
                }
            ]
        },
    }


def test_flush_preserves_a_retained_catalog_gaps_value_when_a_sibling_migrates(graph):
    """Before this fix, flushing an instance with BOTH a catalog-gap entry
    (left pending in ``proxyWidgets``) and a repairable sibling dropped the
    pending entry's own saved value: the final ``widgets_values`` rebuild is
    positional over DECLARED widget inputs only, which has no slot left for
    an entry still sitting in ``proxyWidgets``. A LATER
    ``plan_proxy_migration`` call then read the pending entry's host value
    off the rebuilt array by its (now unrelated) position, landing on the
    migrated sibling's value instead of its own.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4670514857
    """
    wf = _mixed_catalog_gap_and_create_workflow(scale_value=2, width_value=768)
    inst = _instance(wf, 1)
    report = promoted.flush_proxy_migration(wf, inst, graph)

    assert report.created == ["width"]
    assert inst["properties"]["proxyWidgets"] == [["5", "scale"]]
    assert inst["widgets_values"] == [768]  # only the migrated widget is declared now

    later = promoted.plan_proxy_migration(wf, inst, graph)
    scale = next(e for e in later if e.original == ["5", "scale"])
    assert scale.host_value == 2  # not 768 — the sibling's value a positional re-read would grab


# --------------------------------------------------------------------------- #
# apply: the repair gate at the engine's own write site (``apply_slots``)
# --------------------------------------------------------------------------- #


def test_apply_slot_does_not_fork_a_shared_definition_for_an_unrepairable_plan(graph):
    """``_apply_one_slot_impl`` used to fork+flush whenever ANY pending entry
    was not a preview — which includes a plan of only ``alreadyLinked``
    and/or ``quarantine`` entries. ``flush_proxy_migration`` never changes
    the DEFINITION for those (only the instance's own ``properties``/
    ``widgets_values``), so forking a SHARED definition for one is pure cost:
    the instance permanently stops sharing definition edits with its
    sibling. A write to an unrelated, already-linked widget (``width``) must
    not trigger a fork — but the flush itself still must run (see
    ``test_apply_slot_flushes_an_alreadylinked_only_plan_without_forking``
    for why skipping it entirely corrupts a sibling widget's value).
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4623941911
    """
    wf = _load(FLUX)
    inst = _instance(wf, 56)
    # Drop the one repairable entry (``seed``): what's left is only
    # ``alreadyLinked`` entries plus the unrepairable ``control_after_generate``
    # quarantine — no entry for which the flush would change the DEFINITION.
    inst["properties"]["proxyWidgets"] = [e for e in inst["properties"]["proxyWidgets"] if e != ["52", "seed"]]
    sibling = copy.deepcopy(inst)
    sibling["id"] = 156
    wf["nodes"].append(sibling)
    plan = promoted.plan_proxy_migration(wf, inst, graph)
    assert plan and not any(e.repairable for e in plan)
    original_type = inst["type"]

    wf, warnings = graph.apply_slots(wf, {"56.width": 768})

    assert warnings == []
    assert _instance(wf, 56)["type"] == original_type  # not forked
    assert _instance(wf, 156)["type"] == original_type  # sibling untouched
    assert _instance(wf, 156)["properties"]["proxyWidgets"]  # sibling's own entries untouched
    # The written instance's plan DID flush: the alreadyLinked entries are
    # consumed and the genuinely unrepairable one is now permanently
    # quarantined — just without forking the (shared, untouched) definition.
    assert "proxyWidgets" not in _instance(wf, 56)["properties"]
    assert _instance(wf, 56)["properties"]["proxyWidgetErrorQuarantine"] == [
        {"originalEntry": ["52", "control_after_generate"], "reason": "missingSubgraphInput", "attemptedAtVersion": 1}
    ]


def test_isolate_shared_subgraph_forks_the_same_instance_regardless_of_apply_order(graph):
    """Two instances (56, 156) share FLUX's definition and EACH needs its own
    legacy proxy repair: 56 keeps its original, repairable ``seed`` entry;
    156's entry is swapped for an independent, also-repairable ``steps``
    entry (KSampler node 52's ``steps`` widget, never promoted in the
    fixture) so neither instance's repair can structurally satisfy the
    other's — this isolates the ORDER-DEPENDENCE bug from the (separate,
    intentional) optimization where a sibling with NOTHING left to repair
    just inherits an already-migrated shared definition (see
    ``test_shared_definition_is_forked_not_mutated`` in
    ``test_workflow_edit_legacy_proxy.py``, which pins that behavior for one
    specific apply order on one replica).

    Before this fix, ``_isolate_shared_subgraph`` forked whichever instance
    applied its write FIRST (the definition was still shared at that moment),
    and left the SECOND instance — now the definition's sole remaining
    sharer — on the original, mutated in place. Reversing the apply order
    flipped which instance forked, so two replicas receiving the same pair of
    writes in opposite delivery orders ended up with different definition
    ids for the SAME hosts: final widget values agreed, but the documents
    did not converge, violating the invariant this function's own docstring
    and ADR-OPERATIONS-0018 both document.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4693109354
    """
    base = _load(FLUX)
    sibling = copy.deepcopy(_instance(base, 56))
    sibling["id"] = 156
    entries = sibling["properties"]["proxyWidgets"]
    entries[entries.index(["52", "seed"])] = ["52", "steps"]
    base["nodes"].append(sibling)

    forward, warnings_f = graph.apply_slots(base, {"56.width": 768, "156.width": 800})
    backward, warnings_b = graph.apply_slots(base, {"156.width": 800, "56.width": 768})

    assert warnings_f == [] and warnings_b == []
    # Order independence: each instance ends up on the SAME definition id
    # whichever order the pair of writes is applied in.
    assert _instance(forward, 56)["type"] == _instance(backward, 56)["type"]
    assert _instance(forward, 156)["type"] == _instance(backward, 156)["type"]
    # The two instances really did end up on two distinct definitions (one
    # forked off the other) — this isn't two orders converging on a no-op.
    assert _instance(forward, 56)["type"] != _instance(forward, 156)["type"]
    # Not just the same id: the same STRUCTURE too (both got their own
    # repair applied to their own copy, independent of apply order).
    def_56_fwd = _def_of(forward, _instance(forward, 56))
    def_56_bwd = _def_of(backward, _instance(backward, 56))
    assert json.dumps(def_56_fwd, sort_keys=True) == json.dumps(def_56_bwd, sort_keys=True)
    def_156_fwd = _def_of(forward, _instance(forward, 156))
    def_156_bwd = _def_of(backward, _instance(backward, 156))
    assert json.dumps(def_156_fwd, sort_keys=True) == json.dumps(def_156_bwd, sort_keys=True)
    # Each instance's OWN repair actually landed: 56 still resolves its seed
    # promotion, 156 resolves its independent steps promotion.
    assert [i["name"] for i in def_56_fwd["inputs"]][-1] == "seed"
    assert [i["name"] for i in def_156_fwd["inputs"]][-1] == "steps"


def test_apply_slot_flushes_an_alreadylinked_only_plan_without_forking(graph):
    """Before this fix, a plan of only ``alreadyLinked`` entries (no
    ``repairable`` one at all) never flushed, because both ``_apply_set_widget``
    and ``_apply_one_slot_impl`` gated the FLUSH ITSELF on ``e.repairable`` —
    not just the fork. ``widgets_values`` then stayed in legacy ``proxyWidgets``
    POSITION order forever, while ``set_host_value`` writes by DECLARED-input
    ``value_index`` — a different order whenever the two orders disagree. A
    write to ``width`` (declared second, proxy first) landed in the slot
    DECLARATION order assigns it, silently overwriting ``text``'s own value
    (declared first, proxy second) with whatever ``width``'s legacy slot held.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4670514870
    """
    outer = "test-outer-alreadylinked-0000-000000000005"
    wf = {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": outer,
                "inputs": [],
                "outputs": [],
                # proxy order: width, text — the REVERSE of declaration order.
                "properties": {"proxyWidgets": [["20", "width"], ["10", "text"]]},
                "widgets_values": [640, "legacy prompt"],
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": outer,
                    "name": "Outer",
                    "state": {"lastGroupId": 0, "lastNodeId": 20, "lastLinkId": 2, "lastRerouteId": 0},
                    # declaration order: text, width
                    "inputs": [
                        {"id": "i-text", "name": "text", "type": "STRING", "linkIds": [1]},
                        {"id": "i-width", "name": "width", "type": "INT", "linkIds": [2]},
                    ],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 10,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 1},
                            ],
                            "outputs": [],
                            "widgets_values": ["interior default text"],
                        },
                        {
                            "id": 20,
                            "type": "EmptySD3LatentImage",
                            "inputs": [{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 2}],
                            "outputs": [],
                            "widgets_values": [1024, 1024, 1],
                        },
                    ],
                    "links": [
                        {
                            "id": 1,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 10,
                            "target_slot": 1,
                            "type": "STRING",
                        },
                        {"id": 2, "origin_id": -10, "origin_slot": 1, "target_id": 20, "target_slot": 0, "type": "INT"},
                    ],
                }
            ]
        },
    }
    inst = _instance(wf, 1)
    plan = promoted.plan_proxy_migration(wf, inst, graph)
    assert [e.plan for e in plan] == ["alreadyLinked", "alreadyLinked"]
    assert not any(e.repairable for e in plan)

    wf, warnings = graph.apply_slots(wf, {"1.width": 800})

    assert warnings == []
    inst = _instance(wf, 1)
    sg = _def_of(wf, inst)
    by_name = {p.name: p for p in promoted.promoted_inputs(sg, promoted.defs_by_id(wf)) if p.is_widget}
    assert promoted.host_value(inst, by_name["width"]) == 800
    assert promoted.host_value(inst, by_name["text"]) == "legacy prompt"  # not 640


def test_apply_op_replays_a_pending_alreadylinked_write_offline_without_crashing(graph):
    """A write to a declared widget while an ``alreadyLinked`` entry is still
    pending now always runs ``flush_proxy_migration`` (see the fix above), to
    keep ``widgets_values`` from staying stuck in legacy proxy order. That
    flush raises ``ValueError`` ("needs the node catalog") when ``graph`` is
    ``None`` -- the honest answer when it actually can't resolve anything.
    But offline merge replay (no catalog available at all) now reaches that
    raise for an ORDINARY declared-widget write, not just a write to the
    legacy widget itself: before this fix, only the latter hit this path at
    all. Every other offline-degradation site in ``workflow_ops``
    (``_apply_inputcount_bump``, ``_autogrow_group_port``) no-ops instead of
    raising when ``graph`` is unavailable; applying this op during an offline
    merge must do the same instead of crashing the merge.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4623941930
    """
    outer = "test-outer-alreadylinked-offline-0000000006"
    wf = {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": outer,
                "inputs": [],
                "outputs": [],
                # proxy order: width, text — the REVERSE of declaration order.
                "properties": {"proxyWidgets": [["20", "width"], ["10", "text"]]},
                "widgets_values": [640, "legacy prompt"],
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": outer,
                    "name": "Outer",
                    "state": {"lastGroupId": 0, "lastNodeId": 20, "lastLinkId": 2, "lastRerouteId": 0},
                    # declaration order: text, width
                    "inputs": [
                        {"id": "i-text", "name": "text", "type": "STRING", "linkIds": [1]},
                        {"id": "i-width", "name": "width", "type": "INT", "linkIds": [2]},
                    ],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 10,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 1},
                            ],
                            "outputs": [],
                            "widgets_values": ["interior default text"],
                        },
                        {
                            "id": 20,
                            "type": "EmptySD3LatentImage",
                            "inputs": [{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 2}],
                            "outputs": [],
                            "widgets_values": [1024, 1024, 1],
                        },
                    ],
                    "links": [
                        {
                            "id": 1,
                            "origin_id": -10,
                            "origin_slot": 0,
                            "target_id": 10,
                            "target_slot": 1,
                            "type": "STRING",
                        },
                        {"id": 2, "origin_id": -10, "origin_slot": 1, "target_id": 20, "target_slot": 0, "type": "INT"},
                    ],
                }
            ]
        },
    }
    # Minted ONLINE (a replica with a real catalog) building the op.
    _, op = workflow_ops.set_widget(copy.deepcopy(wf), graph, "1", "width", 800)

    # Replayed OFFLINE on a peer merging without a catalog at all.
    replayed = workflow_ops.apply_op(copy.deepcopy(wf), op, None)  # must not raise

    inst = _instance(replayed, 1)
    assert inst["widgets_values"] is not None  # the op applied, degraded or not


def test_pending_value_lookup_ignores_malformed_originalentry_shapes():
    """``_pending_value_lookup`` built its ``{originalEntry tuple: record}``
    map with ``tuple(e["originalEntry"])`` after checking only that
    ``originalEntry`` was *a* list — not that it had the 2/3-STRING-tuple
    shape ``_proxy_tuples`` enforces everywhere else. A malformed record
    (e.g. a nested list inside ``originalEntry``, written by some other
    process sharing the document) made ``tuple(...)`` produce a tuple
    containing an unhashable element, and using it as a dict key raised
    ``TypeError: unhashable type: 'list'`` -- an attacker- or bug-controlled
    stash entry could crash every subsequent ``plan_proxy_migration`` call
    on that instance. Validate each entry the same way ``_proxy_tuples``
    does and skip the ones that don't fit, instead of trusting the stash
    blindly.
    https://github.com/Comfy-Org/comfy-cli/pull/1168#discussion_r4680810815
    """
    instance = {
        "properties": {
            promoted.PENDING_VALUE_PROPERTY: [
                {"originalEntry": ["52", "seed"], "hostValue": 42},  # valid 2-tuple
                {"originalEntry": ["52", "seed", "x"], "hostValue": 99},  # valid 3-tuple
                {"originalEntry": ["52", ["nested"]], "hostValue": 1},  # malformed: nested list element
                {"originalEntry": ["52"], "hostValue": 2},  # malformed: too short
                {"originalEntry": ["a", "b", "c", "d"], "hostValue": 3},  # malformed: too long
                {"originalEntry": [1, 2], "hostValue": 4},  # malformed: non-string elements
                {"originalEntry": "not-a-list", "hostValue": 5},  # malformed: not a list at all
                "not-a-dict",  # malformed: not a record at all
            ]
        }
    }

    out = promoted._pending_value_lookup(instance)

    assert out == {
        ("52", "seed"): {"originalEntry": ["52", "seed"], "hostValue": 42},
        ("52", "seed", "x"): {"originalEntry": ["52", "seed", "x"], "hostValue": 99},
    }


def test_repair_ids_are_stable_across_processes():
    ids = promoted.repair_ids(["56"], "52", "seed", 1)
    again = promoted.repair_ids(["56"], "52", "seed", 1)
    assert ids == again
    assert ids["links"][0] >= 1 << 40  # the op model's leaderless id range
    assert promoted.repair_ids(["156"], "52", "seed", 1) != ids  # another instance, other ids
    assert promoted.repair_ids(["56"], "52", "seed", 2)["links"][0] == ids["links"][0]
