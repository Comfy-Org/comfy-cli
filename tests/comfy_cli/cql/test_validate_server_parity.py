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
        "NamedAutogrowNode": {
            # A group that names its slots explicitly (ClaudeNode.images ships
            # names image_1..image_20). There is no "image0" in this scheme.
            "input": {
                "required": {
                    "images": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"image": ["IMAGE", {}]}},
                                "names": ["image_1", "image_2", "image_3"],
                                "min": 1,
                                "max": 3,
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
        "OptionalAutogrowNode": {
            # GLSLShader.floats and friends sit under `required` but declare
            # min 0: using none of them is legitimate.
            "input": {
                "required": {
                    "image": "IMAGE",
                    "floats": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"float": ["FLOAT", {}]}},
                                "prefix": "u_float",
                                "min": 0,
                                "max": 20,
                            }
                        },
                    ],
                }
            },
            "input_order": {"required": ["image", "floats"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
            "python_module": "nodes",
        },
        "SocketlessNode": {
            # ImageCompare.compare_view: a display-only input with no socket and
            # no widget value; nothing is ever serialized for it.
            "input": {"required": {"image": "IMAGE", "compare_view": ["IMAGECOMPARE", {"socketless": True}]}},
            "input_order": {"required": ["image", "compare_view"]},
            "output": [],
            "output_name": [],
            "output_node": True,
            "python_module": "nodes",
        },
        "EmptyLoader": {
            # A model loader whose folder is empty: the catalog ships the choice
            # list inline and it has no entries. The server rejects any value
            # against it (value_not_in_list), verified live.
            "input": {"required": {"clip_name": [[]]}},
            "input_order": {"required": ["clip_name"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
            "python_module": "nodes",
        },
        "UserCombo": {
            # CustomCombo: a user-editable dropdown. The choices live in the
            # node's own widget, so the catalog declares options: [] by design
            # and the server accepts any value, verified live.
            "input": {"required": {"choice": ["COMBO", {"multiselect": False, "options": []}]}},
            "input_order": {"required": ["choice"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "output_node": False,
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

    def test_missing_class_type_is_an_error_even_when_unreachable(self, graph: Graph) -> None:
        """Unlike the checks the server prunes, class_type is read while the
        graph is built: the server answers `missing_node_type` for a node no
        output reaches ("Node 'ID #9' has no class_type"), so gating this one
        on reachability let a rejected workflow validate clean."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
            "9": {"inputs": {"width": 32}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        missing = [e for e in result["errors"] if e["code"] == "missing_class_type"]
        assert missing and missing[0]["node_id"] == "9", _codes(result)

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


class TestMalformedInputsBlock:
    """`inputs` must be an object. A scalar there crashed the validator with
    `TypeError: argument of type 'int' is not iterable`, so `--json` printed
    nothing; the graph is junk either way and has to come back as a verdict.
    (The server 500s on the same input, which is its own bug.)"""

    @pytest.mark.parametrize(
        "inputs",
        [
            pytest.param(42, id="int"),
            pytest.param("not-a-dict", id="string"),
            pytest.param([], id="list"),
            pytest.param(None, id="null"),
            pytest.param(True, id="bool"),
        ],
    )
    @pytest.mark.parametrize("class_type", ["BatchImagesNode", "MakeImage", "ShowImage"])
    def test_non_object_inputs_is_a_verdict_not_a_crash(self, graph: Graph, inputs: Any, class_type: str) -> None:
        """Every node shape: an autogrow group, ordinary required widgets, and
        an output node, because each reaches a different check."""
        wf = {
            "1": {"class_type": class_type, "inputs": inputs},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert _codes(result)


class TestAutogrowRealSchemas:
    """Shapes taken from the shipped template gallery, where an earlier version
    of the anchor check rejected 51 of 517 official templates."""

    def test_a_named_group_wired_from_its_first_name_validates(self, graph: Graph) -> None:
        """names: [image_1, …] has no image0; demanding one rejects the only
        correct wiring there is."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "NamedAutogrowNode", "inputs": {"images.image_1": ["1", 0]}},
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"], graph.validate_workflow(wf)["errors"]

    def test_a_named_group_skipping_its_first_name_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "NamedAutogrowNode", "inputs": {"images.image_2": ["1", 0]}},
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "autogrow_missing_first_slot" in _codes(result)

    def test_a_group_with_min_zero_may_have_no_slots(self, graph: Graph) -> None:
        """GLSLShader.floats sits under `required` with min 0: a shader using no
        float uniforms is legitimate and the server runs it."""
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "OptionalAutogrowNode", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "ShowImage", "inputs": {"images": ["2", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"], graph.validate_workflow(wf)["errors"]


class TestSocketlessInputs:
    """`socketless: true` marks a display-only input: no socket, no widget
    value, nothing serialized. Demanding it rejected every template ending in
    an ImageCompare (40 of 517), and 20 more classes declare one."""

    def test_a_socketless_required_input_may_be_absent(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "SocketlessNode", "inputs": {"image": ["1", 0]}},
        }
        assert graph.validate_workflow(wf)["valid"], graph.validate_workflow(wf)["errors"]

    def test_a_normal_required_input_is_still_demanded(self, graph: Graph) -> None:
        wf = {"2": {"class_type": "SocketlessNode", "inputs": {}}}
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "required_input_missing" in _codes(result)


class TestEmptyCombos:
    """Two combos with no choices in the catalog and OPPOSITE server verdicts,
    which object_info cannot tell apart.

    A loader with an empty list is rejected (`value_not_in_list`, verified live
    on UpscaleModelLoader and CLIPVisionLoader). CustomCombo, a user-editable
    dropdown, is accepted with whatever the author typed (verified live) —
    and it declares the identical schema, `["COMBO", {"multiselect": false,
    "options": []}]`, as 17 real loaders do. The server's leniency comes from
    the node's own VALIDATE_INPUTS, which object_info does not expose.

    So the check stays as it is: reporting the loaders is worth 22 false
    rejections on CustomCombo in the shipped templates, because silencing the
    shape would hide a missing model on every one of those 17 loaders.
    """

    def test_a_value_against_an_empty_installed_list_is_rejected(self, graph: Graph) -> None:
        wf = {
            "1": {"class_type": "EmptyLoader", "inputs": {"clip_name": "nope.safetensors"}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert not result["valid"]
        assert "no_options_available" in _codes(result)

    def test_a_user_editable_dropdown_is_reported_too_known_divergence(self, graph: Graph) -> None:
        """Documents the accepted cost above: the server takes this one."""
        wf = {
            "1": {"class_type": "UserCombo", "inputs": {"choice": "9:16"}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
        }
        result = graph.validate_workflow(wf)
        assert "no_options_available" in _codes(result)


class TestPrunedNodesStayAdvisory:
    """The server validates only what an output reaches and prunes the rest, so
    a malformed link on a pruned node does not stop the prompt: it is accepted
    and runs. The structural checks must follow the same rule the required,
    range and edge-type checks already follow, or they hard-reject workflows
    the server happily runs (21 of 638 fuzz cases did exactly that)."""

    def _wf(self, broken: dict) -> dict:
        return {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": {"images": ["1", 0]}},
            # node 8 is wired to nothing an output reaches
            "8": {"class_type": "InvertImage", "inputs": broken},
        }

    @pytest.mark.parametrize(
        "broken",
        [
            pytest.param({"image": ["1"]}, id="short-link"),
            pytest.param({"image": [1, 0]}, id="numeric-id"),
            pytest.param({"image": ["1", "0"]}, id="string-index"),
            pytest.param({"image": ["1", 7]}, id="index-out-of-range"),
            pytest.param({"image": ["999", 0]}, id="missing-source"),
        ],
    )
    def test_a_pruned_node_does_not_fail_the_workflow(self, graph: Graph, broken: dict) -> None:
        result = graph.validate_workflow(self._wf(broken))
        assert result["valid"], result["errors"]

    @pytest.mark.parametrize(
        "broken",
        [
            pytest.param({"images": ["1"]}, id="short-link"),
            pytest.param({"images": [1, 0]}, id="numeric-id"),
            pytest.param({"images": ["1", "0"]}, id="string-index"),
        ],
    )
    def test_the_same_break_on_a_reachable_node_still_fails(self, graph: Graph, broken: dict) -> None:
        wf = {
            "1": {"class_type": "MakeImage", "inputs": {"width": 64}},
            "2": {"class_type": "ShowImage", "inputs": broken},
        }
        assert not graph.validate_workflow(wf)["valid"]
