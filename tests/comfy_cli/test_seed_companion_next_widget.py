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


# An unflagged ``seed`` INT followed by a COMBO whose options include a control
# keyword, then a trailing INT. The current frontend always saves a companion
# after ``seed``, so the stream is [seed, <companion>, sampler, steps]. Older
# streams without the companion are [seed, sampler, steps].
_SEED_THEN_KEYWORD_COMBO = {
    "input": {
        "required": {
            "seed": ["INT", {"default": 0}],
            "sampler": [["randomize", "euler", "ddim"], {}],
            "steps": ["INT", {"default": 20}],
        },
    },
    "input_order": {"required": ["seed", "sampler", "steps"]},
}


def test_current_stream_consumes_companion_even_when_next_combo_lists_it():
    # Companion slot present: "randomize" at index 1 is the seed's companion,
    # the COMBO's real value is "euler", and steps stays aligned.
    pairs = _schema_widget_pairs(_SEED_THEN_KEYWORD_COMBO, [42, "randomize", "euler", 30])
    assert pairs == [("seed", 42), ("sampler", "euler"), ("steps", 30)]


def test_current_stream_with_combo_also_set_to_the_keyword():
    pairs = _schema_widget_pairs(_SEED_THEN_KEYWORD_COMBO, [42, "fixed", "randomize", 30])
    assert pairs == [("seed", 42), ("sampler", "randomize"), ("steps", 30)]


def test_legacy_stream_without_companion_keeps_combo_value():
    # No companion slot: "randomize" is the COMBO's real saved value.
    pairs = _schema_widget_pairs(_SEED_THEN_KEYWORD_COMBO, [42, "randomize", 30])
    assert pairs == [("seed", 42), ("sampler", "randomize"), ("steps", 30)]


def test_current_stream_noise_seed_companion_consumed():
    schema = {
        "input": {
            "required": {
                "noise_seed": ["INT", {"default": 0}],
                "mode": [["fixed", "increment", "loop"], {}],
                "cfg": ["FLOAT", {"default": 7.0}],
            },
        },
        "input_order": {"required": ["noise_seed", "mode", "cfg"]},
    }
    pairs = _schema_widget_pairs(schema, [7, "fixed", "loop", 5.5])
    assert pairs == [("noise_seed", 7), ("mode", "loop"), ("cfg", 5.5)]
    legacy = _schema_widget_pairs(schema, [7, "fixed", 5.5])
    assert legacy == [("noise_seed", 7), ("mode", "fixed"), ("cfg", 5.5)]
