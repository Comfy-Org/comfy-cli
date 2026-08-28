"""Legacy ``properties.proxyWidgets`` → linked-input repair: a port of the
frontend's forward migration (``proxyWidgetMigration.ts``, ADR 0009).

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


def test_repair_ids_are_stable_across_processes():
    ids = promoted.repair_ids(["56"], "52", "seed", 1)
    again = promoted.repair_ids(["56"], "52", "seed", 1)
    assert ids == again
    assert ids["links"][0] >= 1 << 40  # the op model's leaderless id range
    assert promoted.repair_ids(["156"], "52", "seed", 1) != ids  # another instance, other ids
    assert promoted.repair_ids(["56"], "52", "seed", 2)["links"][0] == ids["links"][0]
