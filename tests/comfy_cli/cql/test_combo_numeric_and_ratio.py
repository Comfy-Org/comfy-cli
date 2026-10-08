"""Two COMBO false refusals seen on the production agent.

1. **Numeric options compare by number.** A float combo (``scale_factor`` with
   options ``0.25, 0.5, 1.0, 2.0, 4.0``) rejected the value ``1``: membership
   compared ``str(1)`` against ``str(1.0)``. A workflow saved by the frontend
   writes ``1.0`` as ``1`` (JavaScript has one number type), and the server's
   own check is ``1 in [.., 1.0, ..]`` — true. ``validate`` refused a graph the
   server runs.

2. **An aspect-ratio label with the right ratio is the same option.**
   ``set-widget <id>.aspect_ratio '16:9 (Landscape)'`` (or bare ``'16:9'``) on
   a node whose options are ``'16:9 (Widescreen)'``, ``'9:16 (Portrait
   Widescreen)'``, … The ratio IS the value; the parenthetical is a display
   label. When exactly one option carries that ratio the edit now writes it,
   with a ``normalized_value`` note, instead of refusing and costing the agent
   a round to copy the label back. Anything that is not a ``W:H`` ratio keeps
   the refusal (``best_match`` stays a hint there, see
   ``test_combo_best_match.py``).
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph, Port

RATIOS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


def _float_port() -> Port:
    return Port(name="scale_factor", type="COMBO", enum_values=[0.25, 0.5, 1.0, 2.0, 4.0])


@pytest.mark.parametrize("value", [1, 1.0, 2, 0.5])
def test_numeric_value_matches_float_option(value):
    assert _float_port().validate_catalog(value) == []


@pytest.mark.parametrize("value", [3, 1.5])
def test_numeric_value_absent_from_float_options_still_refused(value):
    [w] = _float_port().validate_catalog(value)
    assert w["code"] == "unknown_enum_value"


def test_bool_never_matches_numeric_option():
    # True == 1 in Python; a JSON boolean is a shape error, not option 1.0.
    port = _float_port()
    assert port.validate_shape(True) is not None


def _graph() -> Graph:
    return Graph.from_object_info(
        {
            "ResolutionSelector": {
                "input": {"required": {"aspect_ratio": [RATIOS, {}], "scale": [["1x", "2x"], {}]}},
                "input_order": {"required": ["aspect_ratio", "scale"]},
                "output": ["INT", "INT"],
                "output_name": ["width", "height"],
                "name": "ResolutionSelector",
                "display_name": "ResolutionSelector",
                "category": "test",
                "output_node": False,
            }
        }
    )


def _workflow() -> dict:
    return {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "ResolutionSelector",
                "inputs": [],
                "outputs": [],
                "widgets_values": ["1:1 (Square)", "1x"],
            }
        ],
        "links": [],
        "groups": [],
        "version": 0.4,
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        ("16:9 (Landscape)", "16:9 (Widescreen)"),
        ("3:4 (Portrait)", "3:4 (Portrait Standard)"),
        ("9:16 (Portrait)", "9:16 (Portrait Widescreen)"),
        ("16:9", "16:9 (Widescreen)"),
        ("2:3", "2:3 (Portrait Photo)"),
    ],
)
def test_set_widget_writes_the_option_with_the_same_ratio(value, expected):
    wf = _workflow()
    _, op = workflow_ops.set_widget(wf, _graph(), 1, "aspect_ratio", value)
    assert op["value"] == expected
    assert wf["nodes"][0]["widgets_values"][0] == expected
    [note] = [w for w in op.get("warnings") or [] if w.get("code") == "normalized_value"]
    assert note["from"] == value and note["to"] == expected


def test_set_widget_ratio_absent_from_options_still_refused():
    with pytest.raises(workflow_ops.FatalFindingError) as ei:
        workflow_ops.set_widget(_workflow(), _graph(), 1, "aspect_ratio", "4:5 (Portrait)")
    assert ei.value.finding["code"] == "unknown_enum_value"


def test_set_widget_ambiguous_ratio_still_refused():
    g = Graph.from_object_info(
        {
            "ResolutionSelector": {
                "input": {"required": {"aspect_ratio": [[*RATIOS, "16:9 (HD)"], {}]}},
                "input_order": {"required": ["aspect_ratio"]},
                "output": ["INT"],
                "output_name": ["width"],
                "name": "ResolutionSelector",
                "display_name": "ResolutionSelector",
                "category": "test",
                "output_node": False,
            }
        }
    )
    wf = _workflow()
    wf["nodes"][0]["widgets_values"] = ["1:1 (Square)"]
    with pytest.raises(workflow_ops.FatalFindingError):
        workflow_ops.set_widget(wf, g, 1, "aspect_ratio", "16:9 (Landscape)")


def test_set_widget_non_ratio_label_still_refused():
    """Only a W:H ratio is treated as the value; '2x (fast)' is a guess."""
    with pytest.raises(workflow_ops.FatalFindingError):
        workflow_ops.set_widget(_workflow(), _graph(), 1, "scale", "2x (fast)")
