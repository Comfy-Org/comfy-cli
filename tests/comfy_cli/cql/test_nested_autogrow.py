"""Nested auto-grow groups under a dynamic combo (``model.images``, ``model.reference_videos``).

Partner nodes such as ``GeminiNanoBanana2V2`` and ``MinimaxHailuo03ReferenceNode``
declare their reference-media inputs as ``COMFY_AUTOGROW_V3`` groups *inside*
the selected option of a ``COMFY_DYNAMICCOMBO_V3`` selector. The frontend names
the grown slots ``<combo>.<group>.<names[i]>`` (``model.images.image_1``) and
writes **no** ``widgets_values`` entry for the group. Two engine defects made
those nodes unbuildable from the CLI:

* ``widget_order_default`` / ``widget_defaults`` counted every sub-input of the
  selected option — link-only groups included — so a fresh node carried more
  positional values than the frontend serializes, and every widget after the
  groups (``seed``, ``watermark``, ``temperature``…) was read one or more slots
  off. The published widget catalog is what the CRDT doc host uses to name
  positions, so a UI-built node's ``seed`` was force-mapped onto
  ``model.reference_images``.
* Nothing in the engine could name a nested group at all: only a top-level
  ``COMFY_AUTOGROW_V3`` port was visible, so ``connect`` had nothing to grow.

Fixture: ``tests/comfy_cli/fixtures/object_info_nested_autogrow.json`` — the
production catalog's entries with tooltips stripped (structure verbatim).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli.cql.engine import Graph

FIXTURE = Path(__file__).resolve().parents[2] / "comfy_cli" / "fixtures" / "object_info_nested_autogrow.json"

# ``widgets_values`` the frontend wrote for ``api_minimax_h3_r2v`` (the gallery
# template), captured verbatim: selector, prompt, resolution, ratio, duration,
# seed, control_after_generate, watermark. No slot for the reference groups.
MINIMAX_UI_WIDGETS = ["MiniMax H3", "storyboard prompt", "768P", "adaptive", 5, 42, "randomize", False]


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(FIXTURE.read_text()))


# --------------------------------------------------------------------------- #
# (B) link-only sub-inputs of a dynamic combo own no widgets_values slot
# --------------------------------------------------------------------------- #


def test_default_order_skips_link_only_dynamic_subs(graph):
    order = graph.widget_order_default("MinimaxHailuo03ReferenceNode")
    assert order == [
        "model",
        "model.prompt",
        "model.resolution",
        "model.ratio",
        "model.duration",
        "seed",
        "control_after_generate",
        "watermark",
    ]


def test_default_order_matches_captured_frontend_shape(graph):
    order = graph.widget_order_default("MinimaxHailuo03ReferenceNode")
    assert len(order) == len(MINIMAX_UI_WIDGETS)
    by_name = dict(zip(order, MINIMAX_UI_WIDGETS))
    assert by_name["seed"] == 42
    assert by_name["watermark"] is False


def test_nano_banana_default_order_has_no_group_or_file_slots(graph):
    order = graph.widget_order_default("GeminiNanoBanana2V2")
    assert "model.images" not in order
    assert "model.files" not in order
    assert order.index("seed") == order.index("model.thinking_level") + 1


def test_widget_defaults_round_trip_through_value_aware_order(graph):
    """The layout ``add_node`` writes (defaults over the default order) must
    be exactly what ``set-widget`` indexes against on that same fresh node."""
    for cls in ("MinimaxHailuo03ReferenceNode", "GeminiNanoBanana2V2", "GrokImageEditNodeV2"):
        order = graph.widget_order_default(cls)
        defaults = graph.widget_defaults(cls)
        values = [defaults.get(name) for name in order]
        assert graph.widget_order_for_node(cls, values) == order, cls
        assert set(defaults) == set(order), cls


# --------------------------------------------------------------------------- #
# (A) the engine names every auto-grow group the selected option carries
# --------------------------------------------------------------------------- #


def test_autogrow_groups_lists_nested_groups_for_selected_option(graph):
    groups = graph.autogrow_groups("MinimaxHailuo03ReferenceNode", MINIMAX_UI_WIDGETS)
    assert [(p.name, p.autogrow_element_type) for p in groups] == [
        ("model.reference_images", "IMAGE"),
        ("model.reference_videos", "VIDEO"),
        ("model.reference_audios", "AUDIO"),
    ]


def test_autogrow_groups_lists_top_level_group(graph):
    groups = graph.autogrow_groups("BatchImagesNode", [])
    assert [(p.name, p.autogrow_element_type) for p in groups] == [("images", "IMAGE")]


def test_autogrow_groups_follow_the_node_selection(graph):
    lite = graph.widget_defaults("GeminiNanoBanana2V2")
    order = graph.widget_order_default("GeminiNanoBanana2V2")
    values = [lite.get(n) for n in order]
    values[order.index("model")] = "Nano Banana 2 Lite"
    groups = graph.autogrow_groups("GeminiNanoBanana2V2", values)
    assert [p.name for p in groups] == ["model.images"]
    assert graph.autogrow_groups("GeminiNanoBanana2V2", ["", "no such option"]) == []


def test_autogrow_template_and_limits_come_from_the_schema(graph):
    videos = next(
        p
        for p in graph.autogrow_groups("MinimaxHailuo03ReferenceNode", MINIMAX_UI_WIDGETS)
        if p.name.endswith("videos")
    )
    assert videos.autogrow_template == {"names": ["video_1", "video_2", "video_3"]}
    assert videos.autogrow_limits == (0, 3)
    images = graph.autogrow_groups("BatchImagesNode", [])[0]
    assert images.autogrow_template == {"prefix": "image"}
    assert images.autogrow_limits == (1, 50)
    grok_defaults = graph.widget_defaults("GrokImageEditNodeV2")
    grok_values = [grok_defaults.get(n) for n in graph.widget_order_default("GrokImageEditNodeV2")]
    grok = graph.autogrow_groups("GrokImageEditNodeV2", grok_values)[0]
    assert grok.autogrow_limits == (1, 3)


# --------------------------------------------------------------------------- #
# (C) `nodes show` payload names the group, its element type and next slot
# --------------------------------------------------------------------------- #


def test_show_payload_expands_dynamic_combo_options_with_autogrow_groups(graph):
    payload = graph.morphism_to_dict(graph.node("MinimaxHailuo03ReferenceNode"))
    model = next(i for i in payload["inputs"] if i["name"] == "model")
    assert [o["key"] for o in model["dynamic_options"]] == ["MiniMax H3"]
    subs = {s["name"]: s for s in model["dynamic_options"][0]["inputs"]}
    images = subs["model.reference_images"]
    assert images["is_link"] is True
    assert images["autogrow"] is True
    assert images["element_type"] == "IMAGE"
    assert images["slots"]["names"][:2] == ["image_1", "image_2"]
    assert images["slots"]["max"] == 9
    assert images["wire_as"].startswith("model.reference_images.image_1")
    assert subs["model.reference_videos"]["element_type"] == "VIDEO"
    assert subs["model.prompt"]["is_link"] is False


def test_show_payload_top_level_autogrow_uses_schema_names(graph):
    payload = graph.morphism_to_dict(graph.node("BatchImagesNode"))
    images = payload["inputs"][0]
    assert images["autogrow"] is True
    assert images["element_type"] == "IMAGE"
    assert images["wire_as"].startswith("images.image0")
