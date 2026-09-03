"""``generate --emit-ops``: the emitter expressed as the frozen op vocabulary.

BE-11131: ``--emit-workflow`` writes an API-format file, which the canvas and
every edit tool refuse (``workflow_not_frontend_format`` — 48 refusals in one
staging day), and which the CRDT write path cannot attribute (no ops). Instead
of converting API→frontend after the fact — a second implementation of widget
order and layout — the emitter mints the graph as add_node/set_widget/connect
specs and lets ``workflow_ops.apply_specs`` materialize the frontend workflow,
exactly the machinery every hand edit already uses. One answer, not two.

The round-trip test is the contract: lowering the materialized frontend
workflow back to API format must reproduce the semantics of the API graph
``build_workflow`` has always emitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli import workflow_ops
from comfy_cli.command.generate import emit
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api

PARTNER_OBJECT_INFO = json.loads(
    (Path(__file__).parent / "fixtures" / "partner_nodes_object_info.json").read_text(encoding="utf-8")
)

# Core nodes build_workflow relies on that are not partner nodes. Minimal but
# faithful shapes: LoadImage's image is an upload-backed combo (so a local
# filename passes the enum gate), ImageBatch is the 2-input folder.
CORE_OBJECT_INFO = {
    "LoadImage": {
        "input": {"required": {"image": [["example.png"], {"image_upload": True}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
        "name": "LoadImage",
        "display_name": "Load Image",
        "category": "image",
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "input_order": {"required": ["images", "filename_prefix"]},
        "output": [],
        "output_name": [],
        "name": "SaveImage",
        "display_name": "Save Image",
        "category": "image",
    },
    "SaveVideo": {
        "input": {
            "required": {
                "video": ["VIDEO"],
                "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
                "format": [["auto", "mp4"], {"default": "auto"}],
                "codec": [["auto", "h264"], {"default": "auto"}],
            }
        },
        "input_order": {"required": ["video", "filename_prefix", "format", "codec"]},
        "output": [],
        "output_name": [],
        "name": "SaveVideo",
        "display_name": "Save Video",
        "category": "image/video",
    },
    "ImageBatch": {
        "input": {"required": {"image1": ["IMAGE"], "image2": ["IMAGE"]}},
        "input_order": {"required": ["image1", "image2"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "name": "ImageBatch",
        "display_name": "Batch Images",
        "category": "image",
    },
}


def _object_info() -> dict:
    merged = dict(CORE_OBJECT_INFO)
    merged.update(PARTNER_OBJECT_INFO)
    return merged


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


EMPTY_WF: dict = {"nodes": [], "links": [], "version": 0.4, "last_node_id": 0, "last_link_id": 0}


def _apply(specs: list[dict]) -> dict:
    wf, _ops, _aliases = workflow_ops.apply_specs(
        json.loads(json.dumps(EMPTY_WF)), _graph(), specs, actor="test", base_version=0
    )
    return wf


def _api_by_class(api_wf: dict) -> dict[str, dict]:
    """Index an API-format graph by class_type (unique per class in these
    graphs), normalizing link refs to the SOURCE class and autogrow element
    keys (``base.image_1``) to their base — the two representations that may
    legitimately differ between the legacy flat emitter and the schema-driven
    op path."""
    by_id = {nid: n["class_type"] for nid, n in api_wf.items()}
    out: dict[str, dict] = {}
    for n in api_wf.values():
        inputs: dict[str, object] = {}
        for key, value in (n.get("inputs") or {}).items():
            base = key.split(".")[0] if key.count(".") and key.rsplit(".", 1)[-1].split("_")[-1].isdigit() else key
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in by_id:
                inputs[base] = ("link", by_id[str(value[0])])
            else:
                inputs[base] = value
        out[n["class_type"]] = inputs
    return out


# ─── unit: ops_from_api_workflow ──────────────────────────────────────────


def test_ops_shape_for_an_image_edit_model():
    api = emit.build_workflow("nano-banana", {"prompt": "add sunglasses", "image": "cat.png"})
    specs = emit.ops_from_api_workflow(api, _graph())

    kinds = [s["op"] for s in specs]
    assert kinds == sorted(kinds, key=["add_node", "set_widget", "connect"].index), (
        "adds, then widgets, then connects — every endpoint exists before it is referenced"
    )
    adds = [s for s in specs if s["op"] == "add_node"]
    assert {s["class_type"] for s in adds} == {"LoadImage", "GeminiImageNode", "SaveImage"}
    assert all(s.get("as") for s in adds), "every add declares an alias for later specs to reference"
    prompts = [s for s in specs if s["op"] == "set_widget" and s["widget"] == "prompt"]
    assert len(prompts) == 1 and prompts[0]["value"] == "add sunglasses"
    connects = [s for s in specs if s["op"] == "connect"]
    assert len(connects) == 2, "loader→partner and partner→save"


def test_ops_apply_to_a_frontend_workflow_that_lowers_back_to_the_same_api_graph():
    api = emit.build_workflow("nano-banana", {"prompt": "add sunglasses", "image": "cat.png"})
    wf = _apply(emit.ops_from_api_workflow(api, _graph()))

    assert isinstance(wf.get("nodes"), list), "the materialized workflow is FRONTEND format"
    lowered = convert_ui_to_api(wf, _object_info())
    got, want = _api_by_class(lowered), _api_by_class(api)
    assert set(got) == set(want)
    for cls in want:
        for key, value in want[cls].items():
            assert got[cls].get(key) == value, f"{cls}.{key}: emitted {got[cls].get(key)!r}, want {value!r}"


def test_ops_roundtrip_for_a_video_model():
    api = emit.build_workflow("seedance", {"prompt": "drift", "image": "frame.png", "duration": 8})
    wf = _apply(emit.ops_from_api_workflow(api, _graph()))
    lowered = convert_ui_to_api(wf, _object_info())
    got, want = _api_by_class(lowered), _api_by_class(api)
    assert got["SaveVideo"]["video"] == ("link", "ByteDanceImageToVideoNode")
    assert got["ByteDanceImageToVideoNode"].get("duration") == want["ByteDanceImageToVideoNode"]["duration"]


def test_ops_roundtrip_with_no_image_params():
    api = emit.build_workflow("flux-2", {"prompt": "a fox", "width": 512, "height": 768})
    wf = _apply(emit.ops_from_api_workflow(api, _graph()))
    lowered = convert_ui_to_api(wf, _object_info())
    got = _api_by_class(lowered)
    assert got["Flux2ProImageNode"]["prompt"] == "a fox"
    assert got["Flux2ProImageNode"]["width"] == 512
    assert got["SaveImage"]["images"] == ("link", "Flux2ProImageNode")


def test_ops_fold_multiple_images_through_image_batch():
    api = emit.build_workflow("nano-banana", {"prompt": "merge", "image": ["a.png", "b.png"]})
    wf = _apply(emit.ops_from_api_workflow(api, _graph()))
    lowered = convert_ui_to_api(wf, _object_info())
    got = _api_by_class(lowered)
    assert got["GeminiImageNode"]["images"] == ("link", "ImageBatch")
    assert got["ImageBatch"]["image1"] == ("link", "LoadImage")


# ─── write path: frontend file + replace_ops envelope batch ───────────────


def test_write_frontend_workflow_writes_frontend_and_returns_replace_batch(tmp_path):
    out = tmp_path / "workflow.json"
    wf, ops = emit.write_frontend_workflow(
        "nano-banana",
        {"prompt": "add sunglasses", "image": "cat.png"},
        out,
        _graph(),
        actor="agent",
        base_version=3,
    )
    on_disk = json.loads(out.read_text())
    assert isinstance(on_disk.get("nodes"), list), "the file on disk is frontend format"
    assert on_disk == wf
    assert ops, "the envelope batch expresses the replacement as attributed ops"
    assert all(o.get("op_id") for o in ops), "ops are fully minted (dual-shape, replayable)"
    assert all(o.get("actor") == "agent" for o in ops)


def test_write_frontend_workflow_emits_delete_half_over_a_previous_graph(tmp_path):
    out = tmp_path / "workflow.json"
    _wf1, _ops1 = emit.write_frontend_workflow(
        "flux-2", {"prompt": "first"}, out, _graph(), actor="agent", base_version=0
    )
    _wf2, ops2 = emit.write_frontend_workflow(
        "nano-banana", {"prompt": "second", "image": "cat.png"}, out, _graph(), actor="agent", base_version=1
    )
    deletes = [o for o in ops2 if o["op"] == "delete_node"]
    assert deletes, "replacing an existing canvas must delete what it replaces, like templates fetch"


def test_write_frontend_workflow_unsupported_model_still_raises(tmp_path):
    with pytest.raises(emit.UnsupportedModelError):
        emit.write_frontend_workflow("no-such-model", {}, tmp_path / "w.json", _graph())


# ─── regression: the converter honors input_order like every other surface ──


def test_convert_ui_to_api_honors_input_order_over_dict_order():
    """A re-serialized object_info can alphabetize the input dicts; the
    ``input_order`` block exists to carry the real declaration order, and the
    cql engine already honors it. The converter pairing widgets positionally
    from DICT order silently swaps neighboring widget values (observed:
    prompt/model traded places on GeminiImageNode). One order, every surface."""
    object_info = {
        "Reordered": {
            # Dict order is alphabetical (b_model before a_prompt would sort
            # differently — use names whose sort order INVERTS input_order).
            "input": {
                "required": {
                    "alpha": ["STRING", {"default": ""}],
                    "beta": [["x", "y"], {"default": "x"}],
                }
            },
            "input_order": {"required": ["beta", "alpha"]},
            "output": [],
            "output_name": [],
            "name": "Reordered",
            "display_name": "Reordered",
            "category": "test",
        }
    }
    ui = {
        "nodes": [
            {
                "id": 1,
                "type": "Reordered",
                "pos": [0, 0],
                "size": [200, 100],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                # Positional per input_order: beta first, then alpha.
                "widgets_values": ["y", "hello"],
            }
        ],
        "links": [],
        "version": 0.4,
        "last_node_id": 1,
        "last_link_id": 0,
    }
    api = convert_ui_to_api(ui, object_info)
    node = api["1"]
    assert node["inputs"]["beta"] == "y", f"beta took {node['inputs'].get('beta')!r} — dict-order pairing"
    assert node["inputs"]["alpha"] == "hello"


# ─── CLI: the flag end to end ─────────────────────────────────────────────


def test_cli_emit_ops_writes_frontend_and_envelope_ops(tmp_path, monkeypatch, runner=None):
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    oi_path = tmp_path / "object_info.json"
    oi_path.write_text(json.dumps(_object_info()), encoding="utf-8")
    monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(oi_path))
    out = tmp_path / "workflow.json"

    result = CliRunner().invoke(
        app,
        [
            "--json",
            "generate",
            "nano-banana",
            "--prompt",
            "add sunglasses",
            "--image",
            "cat.png",
            "--emit-workflow",
            str(out),
            "--emit-ops",
            "--actor",
            "agent-user",
            "--base-version",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["data"]["format"] == "frontend"
    ops = envelope["data"]["ops"]
    assert ops and all(o.get("actor") == "agent-user" for o in ops)
    on_disk = json.loads(out.read_text())
    assert isinstance(on_disk.get("nodes"), list), "the written file is canvas-editable frontend format"


def test_cli_emit_ops_without_emit_workflow_is_an_error(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    result = CliRunner().invoke(app, ["--json", "generate", "nano-banana", "--prompt", "x", "--emit-ops"])
    assert result.exit_code != 0
    assert "generate_bad_args" in result.output
