"""``comfy workflow print`` on dynamic-combo / auto-grow nodes and on subgraph
instances with promoted widgets.

Two things the printer has to get right, both of which the slot surface
(``comfy workflow slots``) and the editors (``set-widget`` / ``connect``)
already model:

* **Dynamic combos and auto-grow groups.** Widgets print by their value-aware
  names (``model``, ``model.aspect_ratio``, …) at the frontend's positions —
  never shifted, never a phantom widget slot for a link-only sub-input such as
  a ``COMFY_AUTOGROW_V3`` group. The group itself is a *link* surface: its
  grown slots print as links, and when nothing is wired the group is still
  visible with its element type so a reader knows the address exists.

* **Subgraph instances with promoted widgets.** A promoted widget's value
  lives on the HOST instance (``cql.promoted``) — the instance line prints
  the EFFECTIVE value (host value, else interior default) under the name the
  agent edits (``57.width``), an outside link into it prints as a link, and
  the interior line keeps ``IN.<name>`` so nobody edits ``57/13.width``.

Fixtures: ``object_info_nested_autogrow.json`` (dynamic combos + auto-grow)
and the verbatim gallery templates under ``fixtures/gallery`` with
``object_info_subgraph_promoted.json``.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer
from comfy_cli.workflow_print import render_py

FIXTURES = Path(__file__).parent / "fixtures"
GALLERY = FIXTURES / "gallery"


@pytest.fixture(scope="module")
def autogrow_graph() -> Graph:
    return Graph.from_object_info(json.loads((FIXTURES / "object_info_nested_autogrow.json").read_text()))


@pytest.fixture(scope="module")
def promoted_graph() -> Graph:
    return Graph.from_object_info(json.loads((FIXTURES / "object_info_subgraph_promoted.json").read_text()))


def _empty() -> dict[str, Any]:
    return {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [], "version": 0.4}


def _add(wf: dict, graph: Graph, class_type: str) -> tuple[dict, int]:
    wf, op = workflow_ops.add_node(wf, graph, class_type)
    return wf, op["node_id"]


def _line(source: str, node_id: Any) -> str:
    """The printed line whose trailing comment names ``node_id`` — the whole
    id, so ``# 4`` never matches a minted id that merely starts with 4."""
    pattern = re.compile(rf"  # {re.escape(str(node_id))}(?!\d)")
    return next(ln for ln in source.splitlines() if pattern.search(ln))


def _kwargs(line: str) -> str:
    """The trailing ``**{...}`` dict of a printed line (dotted names live there)."""
    return line.split("**{", 1)[1] if "**{" in line else ""


def _gallery(name: str) -> dict:
    return json.loads((GALLERY / name).read_text())


def _parses(source: str) -> None:
    """Every printed statement is valid Python on its own. (A definition
    block is indented under its ``# subgraph`` header, so the source as a
    whole is deliberately not one parseable module — the lines are.)"""
    for ln in source.splitlines():
        stripped = ln.strip()
        if stripped and not stripped.startswith("#"):
            ast.parse(stripped)


# --------------------------------------------------------------------------- #
# 1. Dynamic combos + auto-grow groups
# --------------------------------------------------------------------------- #


def test_dynamic_combo_widgets_print_by_value_aware_names(autogrow_graph):
    wf, b = _add(_empty(), autogrow_graph, "GeminiNanoBanana2V2")
    res = render_py(wf, autogrow_graph)
    _parses(res.source)
    line = _line(res.source, b)
    assert 'model="Nano Banana 2 (Gemini 3.1 Flash Image)"' in line
    assert "seed=42" in line and 'control_after_generate="fixed"' in line
    assert "temperature=1.0" in line and "top_p=0.95" in line
    kwargs = _kwargs(line)
    for frag in ('"model.aspect_ratio": "auto"', '"model.resolution": "1K"', '"model.thinking_level": "MINIMAL"'):
        assert frag in kwargs
    # the link-only sub-input owns no widget slot: never printed as a value
    assert '"model.images":' not in line
    assert res.warnings == []


def test_switching_the_selector_reexpands_the_sub_widgets_without_shifting(autogrow_graph):
    wf, b = _add(_empty(), autogrow_graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.set_widget(wf, autogrow_graph, b, "model", "Nano Banana 2 Lite")
    wf, _ = workflow_ops.set_widget(wf, autogrow_graph, b, "seed", 7)
    line = _line(render_py(wf, autogrow_graph).source, b)
    assert 'model="Nano Banana 2 Lite"' in line
    assert "seed=7" in line
    assert '"model.resolution": "1K"' in _kwargs(line)


def test_unwired_autogrow_group_is_visible_with_its_element_type(autogrow_graph):
    wf, b = _add(_empty(), autogrow_graph, "GeminiNanoBanana2V2")
    res = render_py(wf, autogrow_graph)
    _parses(res.source)
    line = _line(res.source, b)
    # the group's first slot is shown open, under the name `connect` accepts
    assert '"model.images.image_1": None' in _kwargs(line)
    # ...and the comment names the group, its element type and its capacity
    assert "model.images grows IMAGE (max 14)" in line.split("  # ", 1)[1]


def test_wired_autogrow_slots_print_as_links_then_the_next_free_slot(autogrow_graph):
    wf, a = _add(_empty(), autogrow_graph, "LoadImage")
    wf, c = _add(wf, autogrow_graph, "LoadImage")
    wf, b = _add(wf, autogrow_graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.connect(wf, autogrow_graph, a, "IMAGE", b, "model.images.image_1")
    wf, _ = workflow_ops.connect(wf, autogrow_graph, c, "IMAGE", b, "model.images")
    res = render_py(wf, autogrow_graph)
    _parses(res.source)
    line = _line(res.source, b)
    kwargs = _kwargs(line)
    # each grown slot references the loader wired into it (binding names are
    # assigned in print order, so resolve them through `bindings`)
    refs = dict(re.findall(r'"(model\.images\.image_\d+)": (\w+)\.IMAGE', kwargs))
    assert res.bindings[refs["model.images.image_1"]] == str(a)
    assert res.bindings[refs["model.images.image_2"]] == str(c)
    assert '"model.images.image_3": None' in kwargs
    assert '"model.images.image_4"' not in kwargs
    # the wired group shifted nothing: the widgets still read at their names
    assert "seed=42" in line and '"model.resolution": "1K"' in kwargs
    assert "model.images grows IMAGE (max 14)" in line
    assert res.warnings == []


def test_top_level_autogrow_base_entry_is_the_group_not_a_phantom_link_input(autogrow_graph):
    wf, a = _add(_empty(), autogrow_graph, "LoadImage")
    wf, d = _add(wf, autogrow_graph, "BatchImagesNode")
    line = _line(render_py(wf, autogrow_graph).source, d)
    assert "images=None" not in line
    assert '"images.image0": None' in _kwargs(line)
    assert "images grows IMAGE (max 50)" in line
    wf, _ = workflow_ops.connect(wf, autogrow_graph, a, "IMAGE", d, "images")
    line = _line(render_py(wf, autogrow_graph).source, d)
    assert "images=None" not in line
    assert '"images.image0": load_image' in _kwargs(line)
    assert '"images.image1": None' in _kwargs(line)


def _minimax_ui_workflow() -> dict[str, Any]:
    """A UI-built ``MinimaxHailuo03ReferenceNode``: ``image_1`` wired, one free
    trailing slot per group (the frontend's shape), 8 widget values."""
    return {
        "last_node_id": 4,
        "last_link_id": 4,
        "version": 0.4,
        "groups": [],
        "nodes": [
            {
                "id": 2,
                "type": "LoadImage",
                "pos": [0, 0],
                "size": [300, 300],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [4]},
                    {"name": "MASK", "type": "MASK", "links": []},
                ],
                "properties": {},
                "widgets_values": ["storyboard.png", "image"],
            },
            {
                "id": 4,
                "type": "MinimaxHailuo03ReferenceNode",
                "pos": [400, 0],
                "size": [400, 300],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [
                    {
                        "label": "image_1",
                        "name": "model.reference_images.image_1",
                        "shape": 7,
                        "type": "IMAGE",
                        "link": 4,
                    },
                    {
                        "label": "image_2",
                        "name": "model.reference_images.image_2",
                        "shape": 7,
                        "type": "IMAGE",
                        "link": None,
                    },
                    {
                        "label": "video_1",
                        "name": "model.reference_videos.video_1",
                        "shape": 7,
                        "type": "VIDEO",
                        "link": None,
                    },
                    {
                        "label": "audio_1",
                        "name": "model.reference_audios.audio_1",
                        "shape": 7,
                        "type": "AUDIO",
                        "link": None,
                    },
                ],
                "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": []}],
                "properties": {},
                "widgets_values": ["MiniMax H3", "storyboard prompt", "768P", "adaptive", 5, 42, "randomize", False],
            },
        ],
        "links": [[4, 2, 0, 4, 0, "IMAGE"]],
    }


def test_ui_built_free_slots_are_reused_not_duplicated(autogrow_graph):
    res = render_py(_minimax_ui_workflow(), autogrow_graph)
    _parses(res.source)
    line = _line(res.source, 4)
    kwargs = _kwargs(line)
    assert '"model.reference_images.image_1": load_image.IMAGE' in kwargs
    assert '"model.reference_images.image_2": None' in kwargs
    assert '"model.reference_images.image_3"' not in kwargs  # a free slot already exists
    assert '"model.reference_videos.video_1": None' in kwargs
    assert '"model.reference_audios.audio_1": None' in kwargs
    comment = line.split("  # ", 1)[1]
    assert "model.reference_images grows IMAGE (max 9)" in comment
    assert "model.reference_videos grows VIDEO (max 3)" in comment
    assert "model.reference_audios grows AUDIO (max 3)" in comment
    # widgets at their frontend positions, unshifted by the groups
    assert 'model="MiniMax H3"' in line and "seed=42" in line and "watermark=False" in line
    assert '"model.prompt": "storyboard prompt"' in kwargs and '"model.resolution": "768P"' in kwargs


def test_full_group_prints_no_free_slot(autogrow_graph):
    wf = _minimax_ui_workflow()
    wf, v = _add(wf, autogrow_graph, "LoadVideo")
    for _ in range(3):
        wf, _ = workflow_ops.connect(wf, autogrow_graph, v, "VIDEO", 4, "model.reference_videos")
    line = _line(render_py(wf, autogrow_graph).source, 4)
    kwargs = _kwargs(line)
    assert '"model.reference_videos.video_3": load_video' in kwargs
    assert '"model.reference_videos.video_4"' not in kwargs
    assert "model.reference_videos grows VIDEO (max 3)" in line


def test_autogrow_group_follows_the_current_selection(autogrow_graph):
    """A node switched to an option without the group stops advertising it."""
    wf, b = _add(_empty(), autogrow_graph, "GeminiNanoBanana2V2")
    node = next(n for n in wf["nodes"] if n["id"] == b)
    node["widgets_values"][1] = "not an option"  # no option → no sub-inputs, no group
    line = _line(render_py(wf, autogrow_graph).source, b)
    assert "model.images" not in line


# --------------------------------------------------------------------------- #
# 2. Subgraph instances with promoted widgets
# --------------------------------------------------------------------------- #


def test_pre_migration_instance_prints_interior_defaults_at_the_host_address(promoted_graph):
    """``image_z_image_turbo``: the host has no values yet, so the instance
    line shows the interior defaults — under the names ``set-widget 57.<name>``
    takes, not buried in the interior block."""
    wf = _gallery("image_z_image_turbo.json")
    res = render_py(wf, promoted_graph)
    _parses(res.source)
    line = _line(res.source, "57 subgraph")
    assert line.startswith('text_to_image_z_image_turbo = Subgraph["Text to Image (Z-Image-Turbo)"](')
    assert 'text="Latina female with thick wavy hair' in line
    for frag in (
        "width=1024",
        "height=1024",
        "seed=0",
        "steps=8",
        'unet_name="z_image_turbo_bf16.safetensors"',
        'clip_name="qwen_3_4b.safetensors"',
        'vae_name="ae.safetensors"',
    ):
        assert frag in line
    assert "text=None" not in line
    assert res.bindings["text_to_image_z_image_turbo"] == "57"
    assert res.warnings == []


def test_post_migration_host_values_win_over_the_interior(promoted_graph):
    """``audio_minimax_music_3``: the host materialized every promoted value;
    the interior caption is ``''`` but the host's is the real prompt. The
    values must land under the RIGHT names — not the instance's serialized
    ``inputs[]`` order, which only lists the one input the UI showed."""
    wf = _gallery("audio_minimax_music_3.json")
    line = _line(render_py(wf, promoted_graph).source, "37 subgraph")
    assert 'caption="Global Metadata: Lo-fi hip-hop' in line
    assert 'lyrics="[Intro]' in line
    assert "max_duration=60" in line
    assert "seed=197122968890040" in line
    assert 'unet_name="minimax_music3_dit_fp16.safetensors"' in line
    assert "switch=True" in line
    assert 'switch="Global Metadata' not in line
    assert 'caption=""' not in line


def test_promoted_values_are_never_positionally_shifted(promoted_graph):
    """``api_seedance2_5_video_extend``: two VIDEO sockets precede the widget
    inputs, and the host's ``widgets_values`` cover ONLY the widget-backed ones
    (in declaration order). The old positional read printed
    ``drop_audio="lanczos"``."""
    wf = _gallery("api_seedance2_5_video_extend.json")
    res = render_py(wf, promoted_graph)
    _parses(res.source)
    line = _line(res.source, "39 subgraph")
    assert "clip_to_resize=load_video" in line
    assert "base_video=byte_dance2_reference_node_v2" in line
    for frag in ("pad_second_video=False", 'interpolation="lanczos"', 'padding_color="white"', "drop_audio=False"):
        assert frag in line
    assert 'drop_audio="lanczos"' not in line


def test_interior_promoted_widgets_point_at_the_host_not_at_a_value(promoted_graph):
    wf = _gallery("image_z_image_turbo.json")
    src = render_py(wf, promoted_graph).source
    interior = _line(src, "13")
    assert interior.strip().startswith("empty_sd3_latent_image = EmptySD3LatentImage(")
    # the promoted widget is a reference to the instance's value, not a literal
    assert "width=IN.width" in interior and "height=IN.height" in interior
    assert "width=1024" not in interior
    # the unpromoted widget still prints as a plain, editable value
    assert "batch_size=1" in interior
    # the definition block says where those IN.* values live and how to edit them
    header_idx = next(i for i, ln in enumerate(src.splitlines()) if ln.startswith("# subgraph f2fdebf6"))
    promoted_line = src.splitlines()[header_idx + 1]
    assert promoted_line.startswith("#")
    assert "promoted" in promoted_line
    for name in ("text", "width", "height", "seed", "steps", "unet_name", "clip_name", "vae_name"):
        assert f"IN.{name}" in promoted_line
    assert "57.<name>" in promoted_line
    assert "57/<id>.<name>" in promoted_line


def test_socket_only_subgraph_inputs_are_not_listed_as_promoted_widgets(promoted_graph):
    wf = _gallery("api_seedance2_5_video_extend.json")
    src = render_py(wf, promoted_graph).source
    header_idx = next(i for i, ln in enumerate(src.splitlines()) if ln.startswith("# subgraph d9aa59a4"))
    promoted_line = src.splitlines()[header_idx + 1]
    assert "IN.interpolation" in promoted_line and "IN.drop_audio" in promoted_line
    assert "IN.clip_to_resize" not in promoted_line and "IN.base_video" not in promoted_line


def test_outside_link_into_a_promoted_input_prints_as_a_link(promoted_graph):
    wf = _gallery("image_z_image_turbo.json")
    wf, prim = _add(wf, promoted_graph, "PrimitiveInt")
    wf, _ = workflow_ops.set_widget(wf, promoted_graph, prim, "value", 640)
    wf, _ = workflow_ops.connect(wf, promoted_graph, prim, "INT", 57, "width")
    res = render_py(wf, promoted_graph)
    _parses(res.source)
    line = _line(res.source, "57 subgraph")
    assert "width=primitive_int" in line
    assert "width=1024" not in line
    assert "height=1024" in line  # the sibling is still the host/interior value
    # the primitive prints before the instance that consumes it
    assert res.source.index("primitive_int = PrimitiveInt(") < res.source.index("text_to_image_z_image_turbo = ")


def test_host_value_edited_by_set_widget_is_what_prints(promoted_graph):
    wf = _gallery("image_z_image_turbo.json")
    wf, _ = workflow_ops.set_widget(wf, promoted_graph, 57, "width", 768)
    line = _line(render_py(wf, promoted_graph).source, "57 subgraph")
    assert "width=768" in line
    assert "height=1024" in line


def test_nested_promotion_prints_the_effective_value_and_legacy_proxies(promoted_graph):
    """``subgraph_template_ui``: instance 10's ``value`` is promoted THROUGH a
    nested instance (10/3) — it shows on the instance line with the value the
    frontend runs; the nested instance's line inside the definition keeps
    referencing the outer proxy. Its legacy ``proxyWidgets`` route for
    ``seed`` (interior node 9, never declared as an input) is not a surface
    slot — ``slots`` does not advertise it, so neither does ``print``."""
    wf = json.loads((FIXTURES / "subgraph_template_ui.json").read_text())
    graph = Graph.from_object_info(json.loads((FIXTURES / "subgraph_object_info.json").read_text()))
    res = render_py(wf, graph)
    _parses(res.source)
    inst = _line(res.source, "10 subgraph")
    assert 'value=""' in inst
    assert 'aspect_ratio="16:9"' in inst
    assert "seed=" not in inst
    assert '**{"images.image0": None}' in inst
    nested = _line(res.source, "3 subgraph da09b826")
    assert "value=IN.value" in nested


def test_self_referential_definition_is_bounded(promoted_graph):
    """A definition that instantiates itself and promotes through itself must
    print (with the nested value unresolved), not recurse forever."""
    uuid = "aaaaaaaa-0000-0000-0000-aaaaaaaaaaaa"
    sg = {
        "id": uuid,
        "name": "Ouroboros",
        "inputs": [{"name": "value", "type": "INT", "linkIds": [1]}],
        "outputs": [],
        "nodes": [
            {
                "id": 2,
                "type": uuid,
                "inputs": [{"name": "value", "type": "INT", "link": 1, "widget": {"name": "value"}}],
                "outputs": [],
                "widgets_values": [],
            }
        ],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 2, "target_slot": 0}],
    }
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": uuid,
                "inputs": [{"name": "value", "type": "INT", "link": None, "widget": {"name": "value"}}],
                "outputs": [],
                "widgets_values": [3],
            }
        ],
        "links": [],
        "definitions": {"subgraphs": [sg]},
        "version": 0.4,
    }
    res = render_py(wf, promoted_graph)
    _parses(res.source)
    assert res.source.count("# subgraph aaaaaaaa-0000") == 1
    assert "value=IN.value" in _line(res.source, "2 subgraph")


# --------------------------------------------------------------------------- #
# 3. Through the CLI: the printed addresses are the ones the editors take
# --------------------------------------------------------------------------- #


def _run_print(path: Path, object_info: Path) -> dict:
    set_renderer(Renderer(OutputMode.JSON))
    try:
        result = CliRunner().invoke(workflow_cmd.app, ["print", str(path), "--input", str(object_info)])
    finally:
        set_renderer(Renderer(OutputMode.PRETTY))
    assert result.exit_code == 0, result.output
    return json.loads(result.output)["data"]


def test_cli_print_addresses_match_the_editors(tmp_path, promoted_graph):
    wf = _gallery("audio_minimax_music_3.json")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    data = _run_print(path, FIXTURES / "object_info_subgraph_promoted.json")
    line = _line(data["source"], "37 subgraph")
    assert 'caption="Global Metadata: Lo-fi hip-hop' in line
    # the binding resolves to the address `set-widget` takes: 37.caption
    assert data["bindings"]["text_to_music_mini_max_music_3"] == "37"
    wf2, op = workflow_ops.set_widget(wf, promoted_graph, 37, "caption", "new caption")
    assert op["node_id"] == 37 and op["widget"] == "caption"
    assert 'caption="new caption"' in _line(render_py(wf2, promoted_graph).source, "37 subgraph")


def test_cli_print_autogrow_addresses_match_connect(tmp_path, autogrow_graph):
    wf, a = _add(_empty(), autogrow_graph, "LoadImage")
    wf, b = _add(wf, autogrow_graph, "GeminiNanoBanana2V2")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    data = _run_print(path, FIXTURES / "object_info_nested_autogrow.json")
    line = _line(data["source"], b)
    assert '"model.images.image_1": None' in line
    # that printed slot name is exactly what connect grows
    wf2, _ = workflow_ops.connect(wf, autogrow_graph, a, "IMAGE", b, "model.images.image_1")
    node = next(n for n in wf2["nodes"] if n["id"] == b)
    assert [i["name"] for i in node["inputs"]] == ["model.images.image_1"]


def test_no_catalog_still_prints_the_instance_values(promoted_graph):
    """Without object_info nothing is value-aware — but promoted values that
    the host materialized are still the host's, by name."""
    wf = _gallery("audio_minimax_music_3.json")
    res = render_py(wf, None)
    line = _line(res.source, "37 subgraph")
    assert 'caption="Global Metadata: Lo-fi hip-hop' in line
    assert "switch=True" in line


def test_existing_workflow_is_not_mutated_by_printing(promoted_graph, autogrow_graph):
    wf = _gallery("image_z_image_turbo.json")
    before = copy.deepcopy(wf)
    render_py(wf, promoted_graph)
    assert wf == before
    wf2, _ = _add(_empty(), autogrow_graph, "GeminiNanoBanana2V2")
    before2 = copy.deepcopy(wf2)
    render_py(wf2, autogrow_graph)
    assert wf2 == before2


def test_frontend_injected_slots_are_not_printed_as_control_after_generate(autogrow_graph):
    """An older frontend serialized LoadImage as ``["a.png", "image"]`` — the
    second value is the injected ``upload`` button slot (``serialize:false``
    now). It owns no schema port, but it is NOT ``control_after_generate``,
    and it is not an editable value either: it must not print at all."""
    wf, nid = _add(_empty(), autogrow_graph, "LoadImage")
    node = next(n for n in wf["nodes"] if n["id"] == nid)
    node["widgets_values"] = ["a.png", "image"]
    source = render_py(wf, autogrow_graph).source
    line = _line(source, nid)
    assert 'image="a.png"' in line
    assert "control_after_generate" not in line
    assert "upload" not in line


def test_promoted_values_resolve_once_per_instance_not_per_widget(promoted_graph, monkeypatch):
    """Review (PR #816): the instance line used to call the public
    ``promoted.effective_value`` per promoted widget, which re-derived the
    definition index and re-located the ``PromotedInput`` the loop already
    held — 451 ``promoted_inputs`` calls for 51 needed on 50 instances. The
    loop's own ``pi`` and ``_State.promoted_defs`` are the whole answer."""
    from comfy_cli.cql import promoted as promoted_mod

    base = _gallery("image_z_image_turbo.json")
    template = next(n for n in base["nodes"] if n["id"] == 57)
    n_instances = 40
    wf = {**base, "nodes": [n for n in base["nodes"] if n["id"] != 57], "links": []}
    for k in range(n_instances):
        inst = copy.deepcopy(template)
        inst["id"] = 1000 + k
        inst["outputs"] = [{**o, "links": []} for o in inst.get("outputs") or []]
        wf["nodes"].append(inst)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] != 9]  # drop the SaveImage that consumed 57

    calls = {"promoted_inputs": 0, "defs_by_id": 0}
    real_pi, real_defs = promoted_mod.promoted_inputs, promoted_mod.defs_by_id

    def counting_pi(*a, **k):
        calls["promoted_inputs"] += 1
        return real_pi(*a, **k)

    def counting_defs(*a, **k):
        calls["defs_by_id"] += 1
        return real_defs(*a, **k)

    monkeypatch.setattr(promoted_mod, "promoted_inputs", counting_pi)
    monkeypatch.setattr(promoted_mod, "defs_by_id", counting_defs)
    res = render_py(wf, promoted_graph)
    assert res.warnings == []
    assert sum("width=1024" in ln for ln in res.source.splitlines()) == n_instances
    # one walk per instance line, one for the definition's promoted header —
    # never one per promoted widget (8 per instance here)
    assert calls["promoted_inputs"] <= n_instances + 2, calls
    assert calls["defs_by_id"] <= 2, calls
