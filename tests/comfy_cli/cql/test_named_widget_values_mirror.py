"""``widgets_values_named``, if present, must agree with ``widgets_values``.

ComfyUI_frontend writes that map on every save and prefers it over the
positional array when restoring a node, so a map an edit did not follow is a map
that lies about the node. Before this suite, comfy-cli maintained it only for
nodes carrying a DynamicGroup: editing a plain node moved the array and left the
map holding the value the edit replaced.

The contract established here is narrower and more useful than "always mirror":
**a map that is present agrees with the array.** Where an edit can keep that
true it mirrors; where it cannot it drops the field, because absence is
unambiguous (the frontend rebuilds from the array, converters read positionally)
while a half-updated map is not -- nothing downstream can tell which keys are
stale.

Schemas here are built inline rather than loaded from an ``object_info``
snapshot. An earlier revision pointed at a path on one developer's machine and
skipped the whole file everywhere else, so it asserted nothing in CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli.cql.engine import Graph, _write_widget

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


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_PLAIN)


def _node(named: Any = "default") -> dict:
    node: dict = {"id": 1, "type": "Plain", "widgets_values": [512, 512, 1], "inputs": []}
    if named == "default":
        node["widgets_values_named"] = {"width": 512, "height": 512, "batch_size": 1}
    elif named is not ...:
        node["widgets_values_named"] = named
    return node


def test_an_edit_mirrors_into_a_usable_map(graph: Graph) -> None:
    node = _node()
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    # Before the fix this still read 512, so a reader preferring the map got the
    # value the edit had just replaced.
    assert node["widgets_values_named"] == {"width": 1024, "height": 512, "batch_size": 1}


def test_an_unwritten_widget_keeps_its_entry(graph: Graph) -> None:
    node = _node()
    _write_widget(node, "batch_size", 4, graph, extend=False)
    assert node["widgets_values_named"] == {"width": 512, "height": 512, "batch_size": 4}


def test_a_node_without_a_map_does_not_grow_one(graph: Graph) -> None:
    """Absence is meaningful: a pre-#10392 save records no map and must not gain one.

    Inventing one would make the node claim a layout authority it never had, and
    a present map is read as authoritative.
    """
    node = _node(named=...)
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    assert "widgets_values_named" not in node


@pytest.mark.parametrize("unusable", [None, [], "invalid", 7])
def test_an_unusable_map_is_left_alone_and_does_not_refuse_the_edit(graph: Graph, unusable: Any) -> None:
    """A malformed map is skipped, not fatal.

    The hard guard stays scoped to grouped nodes, whose edits restructure the
    map by name and cannot proceed against an unusable one. A plain node's edit
    only mirrors a scalar, and this path has always been editable, so refusing it
    because a field the edit would otherwise skip is the wrong shape would be a
    regression rather than a fix.
    """
    node = _node(named=unusable)
    _write_widget(node, "width", 1024, graph, extend=False)
    assert node["widgets_values"] == [1024, 512, 1]
    assert node["widgets_values_named"] == unusable
