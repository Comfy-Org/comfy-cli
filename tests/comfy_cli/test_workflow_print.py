from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_print import PrintUnsupported, binding_name, class_expr, py_literal, render_py

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sd15_graph() -> Graph:
    return Graph.from_object_info(json.loads((FIXTURES / "sd15_object_info.json").read_text()))


@pytest.fixture
def sd15_workflow() -> dict:
    return json.loads((FIXTURES / "sd15_ui_workflow.json").read_text())


SD15_GOLDEN = """\
checkpoint_loader_simple = CheckpointLoaderSimple(ckpt_name="v1-5-pruned-emaonly-fp16.safetensors")  # 4
empty_latent_image = EmptyLatentImage(width=512, height=512, batch_size=1)  # 5
clip_text_encode = CLIPTextEncode(clip=checkpoint_loader_simple.CLIP, text="beautiful scenery nature glass bottle landscape, purple galaxy bottle,")  # 6
clip_text_encode_2 = CLIPTextEncode(clip=checkpoint_loader_simple.CLIP, text="text, watermark")  # 7
ksampler = KSampler(model=checkpoint_loader_simple.MODEL, positive=clip_text_encode, negative=clip_text_encode_2, latent_image=empty_latent_image, seed=685468484323813, control_after_generate="randomize", steps=20, cfg=8, sampler_name="euler", scheduler="normal", denoise=1)  # 3
vae_decode = VAEDecode(samples=ksampler, vae=checkpoint_loader_simple.VAE)  # 8
save_image = SaveImage(images=vae_decode, filename_prefix="SD1.5")  # 9
"""


def test_sd15_golden_source(sd15_workflow, sd15_graph):
    res = render_py(sd15_workflow, sd15_graph)
    body = "\n".join(ln for ln in res.source.splitlines() if not ln.startswith("# note")) + "\n"
    assert body == SD15_GOLDEN
    assert res.bindings == {
        "checkpoint_loader_simple": "4",
        "empty_latent_image": "5",
        "clip_text_encode": "6",
        "clip_text_encode_2": "7",
        "ksampler": "3",
        "vae_decode": "8",
        "save_image": "9",
    }
    assert res.node_count == 7
    assert {s["type"] for s in res.skipped} == {"MarkdownNote"}
    assert len(res.skipped) == 4


def test_sd15_notes_print_as_comments(sd15_workflow, sd15_graph):
    src = render_py(sd15_workflow, sd15_graph).source
    assert '# note 11 "Note: Model link": [Tutorial](https://docs.comfy.org/tutorials/basic/text-to-image) (+' in src
    assert '# note 14 "Note: Output": Image will auto-save' in src  # single-line note: no (+N lines)
    assert "(+0 lines)" not in src


@pytest.mark.parametrize(
    "cls,expected",
    [
        ("KSampler", "ksampler"),
        ("CLIPTextEncode", "clip_text_encode"),
        ("CheckpointLoaderSimple", "checkpoint_loader_simple"),
        ("KSampler (Advanced)", "ksampler_advanced"),
        ("VHS_LoadVideo", "vhs_load_video"),
        ("class", "class_"),
        ("123Node", "n123_node"),
        ("", "node"),
    ],
)
def test_binding_name_shape(cls, expected):
    assert binding_name(cls, {}) == expected


def test_binding_name_dedupes_in_call_order():
    used = {}
    assert [binding_name("KSampler", used) for _ in range(3)] == ["ksampler", "ksampler_2", "ksampler_3"]


def test_class_expr():
    assert class_expr("KSampler") == "KSampler"
    assert class_expr("KSampler (Advanced)") == 'Node["KSampler (Advanced)"]'
    assert class_expr("VHS_LoadVideo") == "VHS_LoadVideo"


def test_py_literal():
    assert py_literal('a "quoted" str') == '"a \\"quoted\\" str"'
    assert (
        py_literal(8.0) == "8.0" and py_literal(8) == "8" and py_literal(True) == "True" and py_literal(None) == "None"
    )
    assert py_literal([1, "x"]) == '[1, "x"]' and py_literal({"k": 1}) == '{"k": 1}'


def _mini(nodes, links):
    return {"nodes": nodes, "links": links, "version": 0.4}


def _node(id_, type_, inputs=(), outputs=(), widgets=None, **extra):
    n = {
        "id": id_,
        "type": type_,
        "mode": 0,
        "inputs": [dict(i) for i in inputs],
        "outputs": [dict(o) for o in outputs],
        "widgets_values": list(widgets or []),
    }
    n.update(extra)
    return n


def test_mode_annotations_and_title(sd15_graph):
    wf = _mini(
        [
            _node(
                1,
                "EmptyLatentImage",
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}],
                widgets=[64, 64, 1],
                mode=4,
            ),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 1}, {"name": "vae", "type": "VAE", "link": None}],
                mode=2,
                title="Decode it",
            ),
        ],
        [[1, 1, 0, 2, 0, "LATENT"]],
    )
    src = render_py(wf, sd15_graph).source
    assert "empty_latent_image = EmptyLatentImage(width=64, height=64, batch_size=1)  # 1 mode=bypass" in src
    assert 'vae_decode = VAEDecode(samples=empty_latent_image, vae=None)  # 2 "Decode it" mode=mute' in src


def test_unknown_class_prints_positional_widgets(sd15_graph):
    wf = _mini(
        [
            _node(
                1, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}], widgets=[64, 64, 1]
            ),
            _node(2, "My Custom Thing", inputs=[{"name": "latent", "type": "LATENT", "link": 1}], widgets=["a", 2]),
        ],
        [[1, 1, 0, 2, 0, "LATENT"]],
    )
    res = render_py(wf, sd15_graph)
    assert (
        'my_custom_thing = Node["My Custom Thing"](latent=empty_latent_image, widgets=["a", 2])  # 2 class not in catalog'
        in res.source
    )
    assert res.warnings == ["node 2: class 'My Custom Thing' not in catalog; widgets printed positionally"]


def test_no_catalog_prints_everything_positionally(sd15_workflow):
    res = render_py(sd15_workflow, None)
    assert (
        'ksampler = KSampler(model=checkpoint_loader_simple.MODEL, positive=clip_text_encode, negative=clip_text_encode_2, latent_image=empty_latent_image, widgets=[685468484323813, "randomize", 20, 8, "euler", "normal", 1])  # 3 class not in catalog'
        in res.source
    )


def test_topological_order_ties_by_id(sd15_graph):
    # 9 feeds 3; both have no other deps -> 9 must print before 3 even though 3 < 9
    wf = _mini(
        [
            _node(
                3,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 1}, {"name": "vae", "type": "VAE", "link": None}],
            ),
            _node(
                9, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}], widgets=[1, 1, 1]
            ),
            _node(
                5, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": []}], widgets=[2, 2, 1]
            ),
        ],
        [[1, 9, 0, 3, 0, "LATENT"]],
    )
    lines = list(render_py(wf, sd15_graph).source.splitlines())
    assert [ln.rsplit("# ", 1)[1].split()[0] for ln in lines] == ["5", "9", "3"]


def test_multi_output_ref_uses_name_single_output_bare(sd15_graph):
    src = render_py(json.loads((FIXTURES / "sd15_ui_workflow.json").read_text()), sd15_graph).source
    assert "clip=checkpoint_loader_simple.CLIP" in src and "samples=ksampler," in src


def test_non_identifier_output_name_uses_out_index(sd15_graph):
    wf = _mini(
        [
            _node(
                1,
                "EmptyLatentImage",
                outputs=[
                    {"name": "LATENT", "type": "LATENT", "links": [1]},
                    {"name": "weird name", "type": "X", "links": []},
                ],
                widgets=[1, 1, 1],
            ),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 1}, {"name": "vae", "type": "VAE", "link": None}],
            ),
        ],
        [[1, 1, 0, 2, 0, "LATENT"]],
    )
    src = render_py(wf, sd15_graph).source
    assert "samples=empty_latent_image.LATENT" in src  # 2 outputs -> named
    wf["nodes"][0]["outputs"][0]["name"] = "has space"
    assert "samples=empty_latent_image.out[0]" in render_py(wf, sd15_graph).source


def test_cycle_is_refused(sd15_graph):
    wf = _mini(
        [
            _node(
                1,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 2}, {"name": "vae", "type": "VAE", "link": None}],
                outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
            ),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 1}, {"name": "vae", "type": "VAE", "link": None}],
                outputs=[{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
            ),
        ],
        [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 1, 0, "IMAGE"]],
    )
    with pytest.raises(PrintUnsupported) as e:
        render_py(wf, sd15_graph)
    assert e.value.reasons == ["link cycle among nodes 1, 2"]


def test_dangling_link_is_refused(sd15_graph):
    wf = _mini(
        [
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 1}, {"name": "vae", "type": "VAE", "link": None}],
            )
        ],
        [[1, 99, 0, 2, 0, "LATENT"]],
    )
    with pytest.raises(PrintUnsupported) as e:
        render_py(wf, sd15_graph)
    assert e.value.reasons == ["link 1 references missing node 99"]


def test_legacy_group_node_is_refused(sd15_graph):
    wf = _mini([_node(1, "workflow>MyGroup")], [])
    with pytest.raises(PrintUnsupported) as e:
        render_py(wf, sd15_graph)
    assert e.value.reasons == ["node 1 is a legacy group node (workflow>MyGroup)"]


def test_reroute_is_spliced(sd15_graph):
    wf = _mini(
        [
            _node(
                1, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}], widgets=[1, 1, 1]
            ),
            _node(
                5,
                "Reroute",
                inputs=[{"name": "", "type": "*", "link": 1}],
                outputs=[{"name": "", "type": "LATENT", "links": [2]}],
            ),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 2}, {"name": "vae", "type": "VAE", "link": None}],
            ),
        ],
        [[1, 1, 0, 5, 0, "LATENT"], [2, 5, 0, 2, 0, "LATENT"]],
    )
    res = render_py(wf, sd15_graph)
    assert "vae_decode = VAEDecode(samples=empty_latent_image, vae=None)  # 2" in res.source
    assert "Reroute" not in res.source
    assert {"id": "5", "type": "Reroute", "reason": "spliced"} in res.skipped


def test_reroute_chain_and_unlinked_reroute(sd15_graph):
    wf = _mini(
        [
            _node(
                1, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}], widgets=[1, 1, 1]
            ),
            _node(
                5,
                "Reroute",
                inputs=[{"name": "", "type": "*", "link": 1}],
                outputs=[{"name": "", "type": "LATENT", "links": [2]}],
            ),
            _node(
                6,
                "Reroute",
                inputs=[{"name": "", "type": "*", "link": 2}],
                outputs=[{"name": "", "type": "LATENT", "links": [3]}],
            ),
            _node(
                7,
                "Reroute",
                inputs=[{"name": "", "type": "*", "link": None}],
                outputs=[{"name": "", "type": "VAE", "links": [4]}],
            ),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 3}, {"name": "vae", "type": "VAE", "link": 4}],
            ),
        ],
        [[1, 1, 0, 5, 0, "LATENT"], [2, 5, 0, 6, 0, "LATENT"], [3, 6, 0, 2, 0, "LATENT"], [4, 7, 0, 2, 1, "VAE"]],
    )
    res = render_py(wf, sd15_graph)
    assert "vae_decode = VAEDecode(samples=empty_latent_image, vae=None)  # 2 vae unlinked via reroute 7" in res.source


def test_get_set_is_spliced(sd15_graph):
    wf = _mini(
        [
            _node(
                1, "EmptyLatentImage", outputs=[{"name": "LATENT", "type": "LATENT", "links": [1]}], widgets=[1, 1, 1]
            ),
            _node(5, "SetNode", inputs=[{"name": "LATENT", "type": "*", "link": 1}], widgets=["lat"]),
            _node(6, "GetNode", outputs=[{"name": "LATENT", "type": "LATENT", "links": [2]}], widgets=["lat"]),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 2}, {"name": "vae", "type": "VAE", "link": None}],
            ),
        ],
        [[1, 1, 0, 5, 0, "LATENT"], [2, 6, 0, 2, 0, "LATENT"]],
    )
    res = render_py(wf, sd15_graph)
    assert "vae_decode = VAEDecode(samples=empty_latent_image, vae=None)  # 2" in res.source
    assert {s["type"] for s in res.skipped} == {"SetNode", "GetNode"}


def test_get_without_set_warns(sd15_graph):
    wf = _mini(
        [
            _node(6, "GetNode", outputs=[{"name": "LATENT", "type": "LATENT", "links": [2]}], widgets=["nope"]),
            _node(
                2,
                "VAEDecode",
                inputs=[{"name": "samples", "type": "LATENT", "link": 2}, {"name": "vae", "type": "VAE", "link": None}],
            ),
        ],
        [[2, 6, 0, 2, 0, "LATENT"]],
    )
    res = render_py(wf, sd15_graph)
    assert "samples=None" in res.source
    assert "node 2: input 'samples' reads GetNode 6 variable 'nope' that no SetNode publishes" in res.warnings


def test_primitive_node_is_noted_on_target(sd15_graph):
    wf = _mini(
        [
            _node(
                12,
                "PrimitiveNode",
                outputs=[{"name": "STRING", "type": "STRING", "links": [1]}],
                widgets=["hello", "fixed"],
            ),
            _node(1, "EmptyLatentImage", outputs=[], widgets=[1, 1, 1]),
            _node(
                6,
                "CLIPTextEncode",
                inputs=[
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 1},
                ],
                widgets=["hello"],
            ),
        ],
        [[1, 12, 0, 6, 1, "STRING"]],
    )
    res = render_py(wf, sd15_graph)
    assert 'clip_text_encode = CLIPTextEncode(clip=None, text="hello")  # 6 text from primitive 12' in res.source
    assert {"id": "12", "type": "PrimitiveNode", "reason": "inlined into 6.text"} in res.skipped


def test_subgraph_instance_and_definition(sd15_graph):
    wf = json.loads((FIXTURES / "subgraph_template_ui.json").read_text())
    graph = Graph.from_object_info(json.loads((FIXTURES / "subgraph_object_info.json").read_text()))
    res = render_py(wf, graph)
    src = res.source
    # instance line: exposed inputs by name, non-identifier names via **{...}
    inst = next(ln for ln in src.splitlines() if "# 10 " in ln)
    assert inst.startswith('generate_image_1 = Subgraph["Generate Image 1"](')
    assert '**{"images.image0":' in inst
    # definition block printed once, names its instances and the address form
    assert "# subgraph d33c1791-dfd2-4102-8540-aa63e4434cd2" in src
    assert "instances: 10" in src and "address inner nodes as 10/<id>" in src
    assert "IN.value" in src and "OUT.IMAGE = " in src
    # inner nodes are addressable
    assert any(v.startswith("10/") or "/" in v for v in res.bindings.values())


def test_missing_subgraph_definition_is_refused(sd15_graph):
    wf = _mini([_node(10, "d33c1791-dfd2-4102-8540-aa63e4434cd2")], [])
    with pytest.raises(PrintUnsupported) as e:
        render_py(wf, sd15_graph)
    assert e.value.reasons == [
        "node 10 is a subgraph instance whose definition d33c1791-dfd2-4102-8540-aa63e4434cd2 is missing"
    ]
