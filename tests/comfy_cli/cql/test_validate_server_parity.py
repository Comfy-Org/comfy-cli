"""Cases where `workflow validate` disagreed with the server it models.

Each test here was first run against a real ComfyUI (`POST /prompt`) and then
against `Graph.validate_workflow` on the same catalog: the server's verdict is
quoted in the docstring, and the test asserts the validator now reaches it. A
graph the validator calls valid and the server rejects is the failure that
costs a submit; a crash is worse, because `--json` then emits nothing at all.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from comfy_cli.cql.engine import Graph


def _object_info() -> dict[str, Any]:
    """A small catalog: an image source, a 1-in/1-out filter, an output node,
    and the production shape of an autogrow group (BatchImagesNode)."""
    return {
        "MakeImage": {
            "input": {"required": {"width": ["INT", {"default": 64, "min": 16, "max": 4096, "step": 8}]}},
            "input_order": {"required": ["width"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
            "python_module": "nodes",
        },
        "InvertImage": {
            "input": {"required": {"image": "IMAGE"}},
            "input_order": {"required": ["image"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
            "python_module": "nodes",
        },
        "ShowImage": {
            "input": {"required": {"images": "IMAGE"}},
            "input_order": {"required": ["images"]},
            "output": [],
            "output_name": [],
            "output_node": True,
            "python_module": "nodes",
        },
        "BatchImagesNode": {
            "input": {
                "required": {
                    "images": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"image": ["IMAGE", {}]}},
                                "prefix": "image",
                                "min": 1,
                                "max": 50,
                            }
                        },
                    ]
                }
            },
            "input_order": {"required": ["images"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
            "python_module": "nodes",
        },
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _codes(result: dict) -> list[str]:
    return [e.get("code") for e in result["errors"]]


class TestDependencyCycles:
    """The server walks the graph and rejects any cycle it reaches from an
    output (`dependency_cycle`, "Dependency cycle detected"). The validator
    never walked it at all, so every cycle below validated clean."""

    def test_self_link_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "InvertImage", "inputs": {"image": ["1", 0]}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "dependency_cycle" in _codes(result)

    def test_two_node_cycle_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "InvertImage", "inputs": {"image": ["2", 0]}},
            "2": {"class_type": "InvertImage", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "dependency_cycle" in _codes(result)

    def test_three_node_cycle_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "InvertImage", "inputs": {"image": ["3", 0]}},
            "2": {"class_type": "InvertImage", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "InvertImage", "inputs": {"image": ["2", 0]}},
            "4": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "dependency_cycle" in _codes(result)

    def test_cycle_message_names_the_nodes_in_it(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "InvertImage", "inputs": {"image": ["2", 0]}},
            "2": {"class_type": "InvertImage", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        cycle = next(e for e in graph.validate_workflow(wf)["errors"] if e["code"] == "dependency_cycle")
        assert "1" in cycle["message"] and "2" in cycle["message"]

    def test_an_unreachable_cycle_is_not_an_error(self, graph: Graph) -> None:
        """The server prunes what no output reaches and never validates it, so
        a cycle off to the side must not hard-reject a runnable graph."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
            "8": {"class_type": "InvertImage", "inputs": {"image": ["9", 0]}},
            "9": {"class_type": "InvertImage", "inputs": {"image": ["8", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert result["valid"], result["errors"]


class TestMalformedLinks:
    """`[node_id, slot_index]` is the only link shape the server accepts:
    anything else is `bad_linked_input`, "must be a length-2 list". The
    validator only recognised length-2 lists as links and let the rest pass
    through as opaque literal values."""

    @pytest.mark.parametrize(
        "link",
        [
            pytest.param(["1"], id="one-element"),
            pytest.param(["1", 0, "extra"], id="three-element"),
            pytest.param([], id="empty"),
        ],
    )
    def test_wrong_length_link_is_rejected(self, graph: Graph, link: list) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": link}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "bad_link_shape" in _codes(result)

    def test_non_string_source_id_is_rejected(self, graph: Graph) -> None:
        """Prompt keys are strings, so the server's `prompt[1]` raises KeyError
        and returns 400. The validator coerced the id with str() and resolved
        the node, so an unsubmittable graph validated clean."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": [1, 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "bad_link_shape" in _codes(result)

    def test_string_source_id_still_validates(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"]


class TestAutogrowAnchor:
    """An autogrow group with `min: 1` requires the index-0 slot by name: the
    server answers `required_input_missing: image0` for a group wired from
    image1 upwards. The validator counted slots and never checked which."""

    def test_missing_index_zero_slot_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image1": ["1", 0], "images.image2": ["1", 0]},
            },
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "autogrow_missing_first_slot" in _codes(result)

    def test_slots_from_index_zero_validate(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image0": ["1", 0], "images.image1": ["1", 0]},
            },
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"]

    def test_a_gap_after_index_zero_validates(self, graph: Graph) -> None:
        """image0 + image2 (no image1) is accepted by the server."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image0": ["1", 0], "images.image2": ["1", 0]},
            },
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"]


class TestNonFiniteNumbers:
    """`json.loads` accepts the bare tokens NaN and Infinity, so a workflow
    file can carry them. The server answers 400 `invalid_input_type`; the
    validator raised ValueError/OverflowError out of Port.validate_shape,
    which `--json` cannot report."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(math.nan, id="nan"),
            pytest.param(math.inf, id="inf"),
            pytest.param(-math.inf, id="-inf"),
        ],
    )
    def test_non_finite_is_an_error_not_a_crash(self, graph: Graph, value: float) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": value}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "shape_mismatch" in _codes(result)


class TestErrorPointsAtTheRealProblem:
    """Two messages sent the reader to the wrong place. Both were found by
    comparing against the server, which names the malformed node itself and
    calls a wrong-typed index a type error."""

    @pytest.mark.parametrize(
        "index",
        [
            pytest.param("0", id="string"),
            pytest.param(0.0, id="float"),
            pytest.param(None, id="null"),
        ],
    )
    def test_wrong_typed_index_is_not_reported_as_out_of_range(self, graph: Graph, index: Any) -> None:
        """`["1", "0"]` names a valid index in the wrong JSON type. Reporting
        `output_index_out_of_range` claims index 0 is out of range on a node
        whose only valid index IS 0, which reads as nonsense."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", index]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "output_index_not_an_integer" in _codes(result)
        assert "out of range" not in " ".join(e["message"] for e in result["errors"])

    def test_missing_class_type_is_reported_on_that_node(self, graph: Graph) -> None:
        """The server answers `missing_node_type` for the malformed node. The
        validator called it a warning and hard-failed the CONSUMING node with a
        dangling_edge, so the reader fixed the wrong node."""
        wf = {
            "1": {"inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        missing = [e for e in result["errors"] if e["code"] == "missing_class_type"]
        assert missing, _codes(result)
        assert missing[0]["node_id"] == "1"

    def test_a_malformed_source_is_not_called_missing(self, graph: Graph) -> None:
        """The consumer also reported `dangling_edge`: "references node '1'
        which does not exist". It does exist — it has no class_type, which the
        node's own error already says."""
        wf = {
            "1": {"inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        errors = graph.validate_workflow(wf)["errors"]
        assert not [e for e in errors if e["code"] == "dangling_edge"], errors
