"""An INT widget set to a string of digits is written as the number.

Models often send an integer widget value as a digit string (``"424242"`` for
a seed, ``"480"`` for a width). The edit paths refused it with ``seed:
expected INT, got str '424242'`` even though the meaning is unambiguous. They
now store the int, so the saved workflow holds a JSON number. A string that is
not a plain integer (``"high"``, ``"1.5"``, ``"+5"``) is still refused with the
same message, and FLOAT/COMBO/STRING/BOOLEAN handling is unchanged.
"""

from __future__ import annotations

import copy
import json

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    "SeededSampler": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}],
                "width": ["INT", {"default": 512, "min": 16, "max": 16384}],
                "offset": ["INT", {"default": 0, "min": -1000, "max": 1000}],
                "cfg": ["FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}],
                "mode": [["fast", "high"], {"default": "fast"}],
                "label": ["STRING", {"default": ""}],
            },
        },
        "input_order": {"required": ["seed", "width", "offset", "cfg", "mode", "label"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
        "display_name": "Seeded Sampler",
        "python_module": "nodes",
    },
}

# Positional layout: seed, its control_after_generate companion, then the rest.
_WIDGETS = [1, "fixed", 512, 0, 7.0, "fast", ""]
_INDEX = {"seed": 0, "width": 2, "offset": 3, "cfg": 4, "mode": 5, "label": 6}

_BIG = 271828182845904123  # above 2**53: a float round trip would change it


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _workflow() -> dict:
    return {"nodes": [{"id": 3, "type": "SeededSampler", "widgets_values": list(_WIDGETS)}], "links": []}


def _written(workflow: dict, widget: str):
    value = workflow["nodes"][0]["widgets_values"][_INDEX[widget]]
    # What a saved file holds: the JSON token, not just the Python value.
    return value, json.dumps(value)


def _note(warnings: list[dict]) -> dict | None:
    notes = [w for w in warnings if w.get("code") == "normalized_value"]
    assert len(notes) <= 1, notes
    return notes[0] if notes else None


def _expected_note(widget: str, text: str, number: int) -> dict:
    # Same shape as the COMBO/STRING rewrites (op-vocabulary §1.3, amendment v1.15).
    return {
        "code": "normalized_value",
        "field": widget,
        "message": f"{widget} is an INT widget; wrote {text!r} as the number {number}",
        "from": text,
        "to": number,
    }


def test_an_int_value_records_no_rewrite(graph):
    _, op = W.set_widget(_workflow(), graph, 3, "seed", 424242)
    assert _note(op.get("warnings", [])) is None
    _, warnings = graph.apply_slots(_workflow(), {"3.seed": 424242})
    assert _note(warnings) is None


@pytest.mark.parametrize(
    ("widget", "text", "expected"),
    [
        ("seed", "424242", 424242),
        ("width", "480", 480),
        ("offset", "-12", -12),
        ("seed", str(_BIG), _BIG),
        ("width", " 480 ", 480),
    ],
)
class TestDigitStringIsWrittenAsInt:
    def test_set_widget(self, graph, widget, text, expected):
        wf, op = W.set_widget(_workflow(), graph, 3, widget, text)
        value, token = _written(wf, widget)
        assert type(value) is int and value == expected
        assert token == str(expected)
        assert op["value"] == expected and type(op["value"]) is int
        assert _note(op.get("warnings", [])) == _expected_note(widget, text, expected)

    def test_apply_specs(self, graph, widget, text, expected):
        specs = [{"op": "set_widget", "node": 3, "widget": widget, "value": text}]
        wf, ops, _aliases = W.apply_specs(_workflow(), graph, specs)
        value, token = _written(wf, widget)
        assert type(value) is int and value == expected
        assert token == str(expected)
        assert _note(ops[0].get("warnings", [])) == _expected_note(widget, text, expected)

    def test_apply_slots(self, graph, widget, text, expected):
        wf, warnings = graph.apply_slots(_workflow(), {f"3.{widget}": text})
        value, token = _written(wf, widget)
        assert type(value) is int and value == expected
        assert token == str(expected)
        assert _note(warnings) == _expected_note(widget, text, expected)

    def test_replayed_op_with_a_string_value(self, graph, widget, text, expected):
        # An op minted elsewhere can still carry the string: the writer coerces too.
        _, op = W.set_widget(_workflow(), graph, 3, widget, expected)
        op = dict(op, value=text)
        wf = W.apply_op(_workflow(), op, graph)
        value, _token = _written(wf, widget)
        assert type(value) is int and value == expected


@pytest.mark.parametrize("text", ["high", "1.5", "+5", "4_2", "", "-", "0x10", "٤٢"])
class TestNonIntegerStringStillRefused:
    def test_set_widget(self, graph, text):
        wf = _workflow()
        before = copy.deepcopy(wf)
        with pytest.raises(ValueError) as exc:
            W.set_widget(wf, graph, 3, "seed", text)
        assert str(exc.value) == f"seed: expected INT, got str {text!r}"
        assert wf == before

    def test_apply_slots(self, graph, text):
        with pytest.raises(ValueError) as exc:
            graph.apply_slots(_workflow(), {"3.seed": text})
        assert str(exc.value) == f"seed: expected INT, got str {text!r}"


def test_bool_on_int_still_refused(graph):
    with pytest.raises(ValueError, match=r"seed: expected INT, got bool True"):
        W.set_widget(_workflow(), graph, 3, "seed", True)


class TestOtherTypesUnchanged:
    def test_digit_string_on_float_still_refused(self, graph):
        with pytest.raises(ValueError, match=r"cfg: expected FLOAT, got str"):
            W.set_widget(_workflow(), graph, 3, "cfg", "7")

    def test_digit_string_on_string_widget_stays_text(self, graph):
        wf, _ = W.set_widget(_workflow(), graph, 3, "label", "480")
        assert _written(wf, "label")[0] == "480"

    def test_combo_option_stays_a_string(self, graph):
        wf, _ = W.set_widget(_workflow(), graph, 3, "mode", "high")
        assert _written(wf, "mode")[0] == "high"


def test_coerce_only_touches_int_ports(graph):
    ports = {p.name: p for p in graph.node("SeededSampler").inputs}
    assert ports["seed"].coerce("424242") == 424242
    assert ports["seed"].coerce("high") == "high"
    assert ports["seed"].coerce(7) == 7
    assert ports["cfg"].coerce("7") == "7"
    assert ports["label"].coerce("7") == "7"
    assert ports["mode"].coerce("7") == "7"
