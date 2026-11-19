"""Amendment A23: the name-keyed `widgets_values_named` register stays coherent.

A23 (`comfy-multi-player` `docs/multiplayer-schema.md`, 2026-10-02) specifies
this field's coherence across implementations. comfy-cli maintained it only for
DynamicGroup nodes, so a `set-widget` on any ordinary node published the new
value positionally and left the PRE-OP value in the register — and the frontend
both writes that register on every save and prefers it when restoring, so a
consumer reading back by name reverted the write.

The gates are A23's, not this package's invention, and the distinction that
makes them right is A23's too: an earlier occurrence writing the register would
turn a stale read into a CORRUPT one, so it must leave the register alone rather
than either updating or discarding it.
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli.cql.engine import Graph, _update_named_widget_value, _write_widget

_PLAIN = {
    "Plain": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        },
        "input_order": {"required": ["width", "height", "batch_size"]},
    }
}
_ORDER = ["width", "height", "batch_size"]


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_PLAIN)


# --------------------------------------------------------------------------- #
# The three gates, unit-tested against the helper so each one is pinned alone.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stored", [None, [], ["x"], "invalid", 7, 0.5, True])
def test_gate_1_a_register_that_is_not_a_name_keyed_object_is_left_alone(stored: Any) -> None:
    """A foreign producer's array is left alone rather than reshaped (A23 gate 1)."""
    node = {"widgets_values_named": stored}
    _update_named_widget_value(node, _ORDER, "width", 1024)
    assert node["widgets_values_named"] == stored


def test_gate_1_an_absent_register_is_not_invented() -> None:
    node: dict = {}
    _update_named_widget_value(node, _ORDER, "width", 1024)
    assert "widgets_values_named" not in node


def test_gate_2_a_name_the_register_does_not_carry_is_not_invented() -> None:
    """ "A register that does not hold the name is not stale for it" (A23 gate 2).

    Inventing the entry would publish a value the producer deliberately omitted.
    """
    node = {"widgets_values_named": {"height": 512}}
    _update_named_widget_value(node, _ORDER, "width", 1024)
    assert node["widgets_values_named"] == {"height": 512}


def test_gate_3_a_duplicated_name_leaves_the_register_untouched() -> None:
    """The final occurrence owns the slot; this package writes the first (A23 gate 3).

    Not updated AND not dropped: writing an earlier occurrence's value there
    would replace the final occurrence's, which A23 calls a corrupt read rather
    than a stale one.
    """
    node = {"widgets_values_named": {"control_after_generate": "fixed"}}
    order = ["seed", "control_after_generate", "noise_seed", "control_after_generate"]
    _update_named_widget_value(node, order, "control_after_generate", "randomize")
    assert node["widgets_values_named"] == {"control_after_generate": "fixed"}


@pytest.mark.parametrize("order", [[], None])
def test_gate_3_an_unresolvable_layout_is_not_final(order: Any) -> None:
    """ "A node whose layout this package cannot resolve has no projected position
    to be final at" (A23)."""
    node = {"widgets_values_named": {"width": 512}}
    _update_named_widget_value(node, order or [], "width", 1024)
    assert node["widgets_values_named"] == {"width": 512}


def test_all_gates_passing_updates_the_entry() -> None:
    node = {"widgets_values_named": {"width": 512, "height": 512}}
    _update_named_widget_value(node, _ORDER, "width", 1024)
    assert node["widgets_values_named"] == {"width": 1024, "height": 512}


# --------------------------------------------------------------------------- #
# Through the real write path.
# --------------------------------------------------------------------------- #


def _node(named: Any = "default") -> dict:
    node: dict = {"id": 1, "type": "Plain", "widgets_values": [512, 512, 1], "inputs": []}
    if named != "omit":
        node["widgets_values_named"] = {"width": 512, "height": 512, "batch_size": 1} if named == "default" else named
    return node


def test_an_edit_updates_the_register(graph: Graph) -> None:
    node = _node()
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    # Before A23 was implemented here this still read 512, so restoring by name
    # reverted the edit.
    assert node["widgets_values_named"] == {"width": 1024, "height": 512, "batch_size": 1}


def test_an_unwritten_name_keeps_its_entry(graph: Graph) -> None:
    node = _node()
    _write_widget(node, "batch_size", 4, graph, extend=False)
    assert node["widgets_values_named"] == {"width": 512, "height": 512, "batch_size": 4}


def test_a_node_with_no_register_does_not_gain_one(graph: Graph) -> None:
    node = _node(named="omit")
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    assert "widgets_values_named" not in node


@pytest.mark.parametrize("unusable", [None, [], "invalid", 7])
def test_an_unusable_register_does_not_refuse_the_edit(graph: Graph, unusable: Any) -> None:
    """This path has always been editable; a malformed register must not change that."""
    node = _node(named=unusable)
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    assert node["widgets_values_named"] == unusable
