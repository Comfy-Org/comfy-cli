"""A leftover link row is inert when its value already reaches the target.

`_scope_findings` recognizes a stale or duplicate row whose source+slot
already feeds the target through a sibling link, and stays silent. That
dedup must hold when the source is the subgraph's input proxy (-10) too:
the same topology must produce the same (empty) findings whichever kind of
source the rows share.
"""

from __future__ import annotations

import pytest

from comfy_cli.link_integrity import _scope_findings


def _scope(src_id: int) -> tuple[list, list]:
    nodes = [
        {
            "id": 5,
            "type": "StringSource",
            "inputs": [],
            "outputs": [{"name": "STRING", "type": "STRING", "links": [1, 2]}],
        },
        {
            "id": 7,
            "type": "CLIPTextEncode",
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": None},
                {"name": "text", "type": "STRING", "link": 1},
            ],
            "outputs": [],
        },
    ]
    links = [
        # The live link: source slot 0 into input 1 (`text`).
        [1, src_id, 0, 7, 1, "STRING"],
        # A leftover row from the same source+slot, aimed past 7's inputs.
        [2, src_id, 0, 7, 6, "STRING"],
    ]
    return nodes, links


@pytest.mark.parametrize("src_id", [5, -10], ids=["real-node source", "subgraph-input proxy source"])
def test_a_leftover_row_whose_value_already_reaches_the_target_is_silent(src_id):
    nodes, links = _scope(src_id)
    errors, warnings = _scope_findings(nodes, links, "70")
    assert errors == []
    assert warnings == [], "the value already reaches input `text` through link 1"


def test_a_proxy_row_that_reaches_nothing_is_still_reported():
    nodes, links = _scope(-10)
    nodes[1]["inputs"][1]["link"] = None  # nothing else carries the proxy value
    links = [links[1]]
    errors, warnings = _scope_findings(nodes, links, "70")
    assert [f["code"] for f in errors + warnings] == ["link_slot_out_of_range"]


def test_print_names_a_missing_interior_source_by_its_qualified_address():
    from comfy_cli.workflow_print import _broken_links

    nodes = [
        {
            "id": 7,
            "type": "CLIPTextEncode",
            "inputs": [{"name": "text", "type": "STRING", "link": 3}],
            "outputs": [],
        }
    ]
    warnings, _, _ = _broken_links(nodes, [[3, 99, 0, 7, 0, "STRING"]], qualify=lambda nid: f"70/{nid}")
    text = " ".join(warnings)
    assert "source node 70/99 does not exist" in text, text
