"""The implicit-seed companion guard must peek at the next WIDGET, not the next input.

``_has_control_after_generate_companion``'s implicit path refuses to consume a
control keyword when the NEXT widget is a COMBO that legitimately lists that
keyword as an option (the value is the combo's own selection, not a phantom
companion). The peek used to hand it the next *declared* input — so one
connection-only input sitting between an unflagged seed INT and the COMBO
defeated the guard and the combo's real value was consumed as a marker,
shifting every later widget.
"""

from __future__ import annotations

from comfy_cli.workflow_to_api import _schema_widget_pairs

# An unflagged seed-like INT, then a CONNECTION input, then a COMBO whose legal
# values include a control keyword. widgets_values only carries widget slots:
# [seed, mode] — "fixed" here is mode's real saved selection.
_SCHEMA = {
    "input": {
        "required": {
            "seed": ["INT", {"default": 0}],
            "mask": ["MASK"],
            "mode": [["fixed", "loop"], {}],
        },
    },
    "input_order": {"required": ["seed", "mask", "mode"]},
}


def test_connection_input_between_seed_and_combo_does_not_eat_the_combo_value():
    pairs = _schema_widget_pairs(_SCHEMA, [42, "fixed"])
    assert ("seed", 42) in pairs
    assert ("mode", "fixed") in pairs


def test_phantom_companion_still_dropped_when_next_widget_is_not_that_combo():
    # Same shape but the trailing value is NOT a legal option of the next
    # widget — that is a genuine frontend companion and must be dropped.
    schema = {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0}],
                "mask": ["MASK"],
                "steps": ["INT", {"default": 20}],
            },
        },
        "input_order": {"required": ["seed", "mask", "steps"]},
    }
    pairs = _schema_widget_pairs(schema, [42, "randomize", 20])
    assert ("seed", 42) in pairs
    assert ("steps", 20) in pairs
    assert not any(v == "randomize" for _n, v in pairs)
