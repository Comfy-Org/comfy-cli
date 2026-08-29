"""Regression sweep over REAL subgraphed gallery templates.

Subgraph gallery templates must convert fully through
``convert_ui_to_api`` and expose their interior slots via
``Graph.get_template_schema``, but the existing subgraph tests only use small
synthetic shapes. These tests pin that behavior against verbatim copies of
templates from Comfy-Org/workflow_templates (see ``fixtures/gallery/README.md``
for provenance + refresh command). Everything here is pure-Python and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api, is_subgraph_uuid

_GALLERY = Path(__file__).resolve().parent / "fixtures" / "gallery"

# Auto-discovered so a newly committed *_subgraphed.json fixture joins the sweep
# without editing this list; the explicit assert keeps the glob honest.
GALLERY_TEMPLATES = sorted(p.name for p in _GALLERY.glob("*_subgraphed.json"))
assert {"02_qwen_Image_edit_subgraphed.json", "05_audio_ace_step_1_t2a_song_subgraphed.json"} <= set(
    GALLERY_TEMPLATES
), f"expected gallery fixtures missing from {_GALLERY}: found {GALLERY_TEMPLATES}"

QWEN = "02_qwen_Image_edit_subgraphed.json"


def _load_template(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


def _iter_subgraph_defs(workflow_or_def: dict):
    """Yield every subgraph def reachable from a workflow or def's definitions."""
    definitions = workflow_or_def.get("definitions")
    if not isinstance(definitions, dict):
        return
    for sg in definitions.get("subgraphs") or []:
        if isinstance(sg, dict):
            yield sg
            yield from _iter_subgraph_defs(sg)


@pytest.mark.parametrize("template_name", GALLERY_TEMPLATES)
class TestGalleryTemplateConversion:
    def test_converts_with_empty_object_info(self, template_name):
        """The structural path (no object_info) must convert without raising."""
        wf = _load_template(template_name)
        api = convert_ui_to_api(wf, {})
        assert isinstance(api, dict)
        assert api, f"{template_name} converted to an empty prompt"

    def test_every_subgraph_instance_expanded(self, template_name):
        """No UUID class_type may survive conversion — expansion must be total."""
        wf = _load_template(template_name)
        # Sanity: the fixture really exercises subgraphs (a refresh that drops
        # them would silently gut this sweep).
        instance_types = {
            n.get("type") for n in wf.get("nodes", []) if isinstance(n, dict) and is_subgraph_uuid(n.get("type"))
        }
        assert instance_types, f"{template_name} no longer contains any subgraph instance"

        api = convert_ui_to_api(wf, {})
        unexpanded = {nid: n.get("class_type") for nid, n in api.items() if is_subgraph_uuid(n.get("class_type"))}
        assert not unexpanded, f"{template_name} left unexpanded subgraph instances in the API prompt: {unexpanded}"
        # Interior nodes surface under '<instance>:<interior>' ids — their
        # presence proves the instances were expanded, not just dropped.
        assert any(":" in nid for nid in api), (
            f"{template_name} produced no subgraph-interior nodes; "
            "subgraph instances appear to have been dropped instead of expanded"
        )

    def test_no_nested_definitions_blind_spot(self, template_name):
        """_collect_subgraph_defs reads only the workflow's top-level
        definitions.subgraphs. A def carrying its OWN nested definitions block
        would NOT be expanded today. No gallery fixture does this yet — if one
        ever does, _collect_subgraph_defs (comfy_cli/workflow_to_api.py) needs
        recursive collection before that fixture can be trusted to convert."""
        wf = _load_template(template_name)
        for sg in _iter_subgraph_defs(wf):
            nested = (sg.get("definitions") or {}).get("subgraphs") if isinstance(sg.get("definitions"), dict) else None
            assert not nested, (
                f"{template_name} subgraph def {sg.get('id')!r} carries nested definitions.subgraphs — "
                "_collect_subgraph_defs only reads top-level definitions and must learn to collect recursively"
            )


class TestGalleryTemplateSlots:
    """Slot extraction over the qwen fixture with a minimal realistic object_info."""

    @pytest.fixture(scope="class")
    def qwen_schema(self) -> dict:
        oi = json.loads((_GALLERY / "qwen_object_info.json").read_text(encoding="utf-8"))
        graph = Graph.from_object_info(oi)
        return graph.get_template_schema(QWEN, _load_template(QWEN))

    def test_surfaces_subgraph_interior_slots(self, qwen_schema):
        interior = [s for s in qwen_schema["slots"] if "/" in s["address"]]
        assert interior, f"no subgraph-interior slots surfaced; got {[s['address'] for s in qwen_schema['slots']]}"

    def test_promoted_prompt_and_seed_are_addressed_at_the_host(self, qwen_schema):
        """The slots an agent actually edits: the prompt + seed of the opaque
        'Qwen Image Edit 2509' subgraph instance (node 141). Both are promoted
        widgets, so they live at the instance address with the HOST value —
        the interior KSampler still carries a stale seed (1118877715456453)
        the frontend never runs (ADR 0009)."""
        by_addr = {s["address"]: s for s in qwen_schema["slots"]}
        assert "141.prompt" in by_addr, f"missing promoted prompt slot; got {sorted(by_addr)}"
        assert by_addr["141.prompt"]["current_value"].startswith("Change the style")
        assert by_addr["141.seed"]["current_value"] == 392667428726572
        assert "141/132.prompt" not in by_addr
        assert "141/137.seed" not in by_addr
        # unpromoted interior widgets stay reachable
        assert by_addr["141/137.steps"]["current_value"] == 4

    def test_top_level_slots_still_present(self, qwen_schema):
        by_addr = {s["address"]: s for s in qwen_schema["slots"]}
        assert by_addr["60.filename_prefix"]["current_value"] == "ComfyUI"
