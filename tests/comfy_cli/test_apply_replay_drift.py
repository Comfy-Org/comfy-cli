"""Replay totality under document shape drift, and convergence-oracle robustness.

The op plane's Totality guarantee says a merge consumer can replay any op
against any document state without a crash — a vanished *node* already
no-ops (delete wins). These tests pin the sibling cases that used to escape:

* a vanished/never-existed *slot* (``to_slot`` out of range) must no-op the
  same way, and must NOT claim the LWW register on the way out — the original
  bug committed the stamp, then raised ``IndexError`` before recording the
  ``op_id``, so a retry of the identical op lost to its own stamp and the
  connect was silently dropped forever;
* a vanished *source slot* keeps the documented "delete wins over the link,
  not the register claim" behavior — no crash, register claimed, input left
  empty;
* an exception escaping a handler must leave the LWW bookkeeping untouched
  (no stamp without a matching applied op_id);
* ``canonical()`` — the convergence oracle — must accept the id mix that
  amendment v1.2 declares legal (int and string node ids in one document);
* dict-shaped ``widgets_values`` (the VHS_* serialization) must not be
  corrupted by the inputcount bump, must not crash ``capture``, and known
  values must survive a positional rewrite instead of being dropped.
"""

from __future__ import annotations

import json

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    "Src": {
        "input": {"required": {}},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "test",
        "display_name": "Src",
        "python_module": "nodes",
    },
    "Dst": {
        "input": {"required": {"samples": ["LATENT"]}},
        "input_order": {"required": ["samples"]},
        "output": [],
        "output_name": [],
        "category": "test",
        "display_name": "Dst",
        "python_module": "nodes",
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            },
        },
        "input_order": {"required": ["width", "height", "batch_size"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "latent",
        "display_name": "Empty Latent Image",
        "python_module": "nodes",
    },
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _two_node_workflow() -> dict:
    return {
        "nodes": [
            {
                "id": 1,
                "type": "Src",
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": None}],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "Dst",
                "inputs": [{"name": "samples", "type": "LATENT", "link": None}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [],
    }


def _connect_op(to_slot: int, *, from_slot: int = 0, base_version: int = 1, actor: str = "agent") -> dict:
    return W._new_op(
        "connect",
        actor,
        base_version,
        link_id=900,
        from_node=1,
        from_slot=from_slot,
        to_node=2,
        to_slot=to_slot,
        link_type="LATENT",
    )


class TestConnectReplayAgainstDriftedDocuments:
    def test_out_of_range_to_slot_is_a_total_noop(self, graph: Graph):
        wf = _two_node_workflow()
        op = _connect_op(to_slot=5)
        W.apply_op(wf, op, graph)  # must not raise
        assert wf["links"] == []
        assert wf["nodes"][1]["inputs"][0]["link"] is None
        assert op["op_id"] in wf["_applied_ops"]

    def test_out_of_range_to_slot_does_not_claim_the_register(self, graph: Graph):
        wf = _two_node_workflow()
        op = _connect_op(to_slot=5)
        W.apply_op(wf, op, graph)
        register = json.dumps(W._write_target(op))
        assert register not in (wf.get("_widget_stamps") or {})

    def test_register_usable_after_slot_appears(self, graph: Graph):
        """The original bug: the failed op's own stamp poisoned the register, so
        even after the document was repaired a replay was silently dropped."""
        wf = _two_node_workflow()
        W.apply_op(wf, _connect_op(to_slot=1), graph)  # slot 1 doesn't exist -> no-op
        # Document repaired: the slot now exists.
        wf["nodes"][1]["inputs"].append({"name": "extra", "type": "LATENT", "link": None})
        later = _connect_op(to_slot=1, base_version=2)
        W.apply_op(wf, later, graph)
        assert wf["nodes"][1]["inputs"][1]["link"] == later["link_id"]
        assert any(ln[0] == later["link_id"] for ln in wf["links"])

    def test_malformed_slot_entry_is_a_total_noop(self, graph: Graph):
        wf = _two_node_workflow()
        wf["nodes"][1]["inputs"][0] = "not-a-slot"
        op = _connect_op(to_slot=0)
        W.apply_op(wf, op, graph)  # must not raise
        assert wf["links"] == []

    def test_vanished_source_slot_keeps_register_leaves_input_empty(self, graph: Graph):
        """Delete wins over the LINK, not over the register claim — the
        documented semantics for a concurrently-deleted source, extended to a
        source SLOT that is out of range on the replayed document."""
        wf = _two_node_workflow()
        op = _connect_op(to_slot=0, from_slot=7)
        W.apply_op(wf, op, graph)  # must not raise
        assert wf["links"] == []
        assert wf["nodes"][1]["inputs"][0]["link"] is None
        register = json.dumps(W._write_target(op))
        assert register in (wf.get("_widget_stamps") or {})

    def test_exception_in_a_handler_rolls_back_the_stamps(self, graph: Graph, monkeypatch):
        """Defense in depth: no handler exception may leave a stamp committed
        without its op_id recorded — that is the poison state."""
        wf = _two_node_workflow()
        op = _connect_op(to_slot=0)

        def _explodes_after_committing(workflow, op_, graph_):
            W._lww_commit(workflow, op_)
            raise RuntimeError("boom")

        monkeypatch.setattr(W, "_apply_connect", _explodes_after_committing)
        with pytest.raises(RuntimeError):
            W.apply_op(wf, op, graph)
        register = json.dumps(W._write_target(op))
        assert register not in (wf.get("_widget_stamps") or {})
        assert op["op_id"] not in (wf.get("_applied_ops") or [])


class TestCanonicalAcceptsLegalIdMixes:
    def test_mixed_node_id_types_do_not_raise(self):
        wf = {
            "nodes": [
                {"id": 7, "type": "Src", "inputs": [], "outputs": []},
                {"id": "57:3", "type": "Dst", "inputs": [], "outputs": []},
            ],
            "links": [],
        }
        first = W.canonical(wf)
        assert first == W.canonical(wf)

    def test_mixed_link_id_types_do_not_raise(self):
        wf = {
            "nodes": [],
            "links": [[1, 1, 0, 2, 0, "LATENT"], ["str-link", 1, 0, 2, 0, "LATENT"]],
        }
        assert W.canonical(wf) == W.canonical(wf)

    def test_link_slot_identity_survives_id_type_mismatch(self):
        """A link that stores the destination id as a string while the node
        carries an int (or vice versa) must still get its raw slot index
        rewritten to the position-independent identity."""
        wf = {
            "nodes": [
                {
                    "id": 7,
                    "type": "Dst",
                    "inputs": [{"name": "samples", "type": "LATENT", "link": 900}],
                    "outputs": [],
                }
            ],
            "links": [[900, 1, 0, "7", 0, "LATENT"]],
        }
        canon = W.canonical(wf)
        assert canon["links"][0][4] == ("name", "samples")


class TestDictWidgetsValuesSurvival:
    def test_inputcount_bump_normalizes_a_dict(self, graph: Graph):
        multi_info = dict(_OBJECT_INFO)
        multi_info["MultiIn"] = {
            "input": {"required": {"inputcount": ["INT", {"default": 2}]}},
            "input_order": {"required": ["inputcount"]},
            "output": [],
            "output_name": [],
            "category": "test",
            "display_name": "MultiIn",
            "python_module": "nodes",
        }
        g = Graph.from_object_info(multi_info)
        wf = {"nodes": [{"id": 9, "type": "MultiIn", "inputs": [], "outputs": []}]}
        dst = wf["nodes"][0]
        dst["widgets_values"] = {"inputcount": 2}
        op = W._new_op("connect", "agent", 1, link_id=901, from_node=1, from_slot=0, to_node=9, to_slot=None)
        W._apply_inputcount_bump(wf, dst, op, g, "inputcount", 3)
        assert dst["widgets_values"] == [3]

    def test_capture_tolerates_dict_widgets_and_keeps_values(self, graph: Graph):
        wf = {
            "nodes": [
                {
                    "id": 4,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": {"width": 768, "height": 512, "batch_size": 1},
                }
            ],
            "links": [],
        }
        recipe, _warnings = W.capture_recipe(wf, graph)  # must not raise
        sets = {(o["widget"], o["value"]) for o in recipe["ops"] if o["op"] == "set_widget"}
        assert ("width", 768) in sets  # non-default value survived the dict form

    def test_set_widget_preserves_dict_siblings(self, graph: Graph):
        wf = {
            "nodes": [
                {
                    "id": 4,
                    "type": "EmptyLatentImage",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": {"width": 768, "height": 512, "batch_size": 1},
                }
            ],
            "links": [],
        }
        new_wf, _op = W.set_widget(wf, graph, 4, "batch_size", 4)
        assert new_wf["nodes"][0]["widgets_values"] == [768, 512, 4]


class TestBatchFailureReportsNothingApplied:
    def test_applied_count_is_zero_on_failure(self, graph: Graph):
        """docs/op-vocabulary-v1.md: 'applied_count is always 0 on failure —
        nothing is written.' The whole batch is discarded, so reporting the
        specs that applied-then-were-discarded teaches a merge consumer that
        k-1 ops persisted when zero did."""
        wf = _two_node_workflow()
        specs = [
            {"op": "add_node", "class_type": "EmptyLatentImage", "as": "latent"},
            {"op": "connect", "from": "$latent.LATENT", "to": "$missing.samples"},
        ]
        with pytest.raises(ValueError) as exc:
            W.apply_specs(wf, graph, specs, actor="agent", base_version=1)
        assert getattr(exc.value, "applied_count", None) == 0
