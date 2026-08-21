"""Unit tests for ``comfy_cli.selector`` — the ONE `--select` projection grammar.

Grammar under test (deliberately small, see the module docstring; no second
dialect will ever be added):

  - dot path:        ``a.b.c``
  - array index:     ``a.0.b``
  - array wildcard:  ``items.#.name``  -> array of per-element matches
  - multi-select:    ``name,inputs``   -> object keyed by each sub-expression

Fail-open semantics (malformed expression / zero matches) and the byte-count
envelope fields are covered here at the pure-function level; the per-command
wiring is covered in each command's test file.
"""

from __future__ import annotations

import json

import pytest

from comfy_cli import error_codes
from comfy_cli.selector import key_inventory, select, selected_payload

PAYLOAD = {
    "name": "KSampler",
    "category": "sampling",
    "inputs": [
        {"name": "seed", "type": "INT", "options": {"default": 0}},
        {"name": "steps", "type": "INT", "options": {"default": 20}},
        {"name": "sampler_name", "type": "COMBO"},
    ],
    "output_types": ["LATENT"],
    "nested": {"a": {"b": {"c": 42}}},
}


class TestPath:
    def test_top_level_key(self):
        result, matched = select(PAYLOAD, "name")
        assert matched is True
        assert result == "KSampler"

    def test_nested_dot_path(self):
        result, matched = select(PAYLOAD, "nested.a.b.c")
        assert matched is True
        assert result == 42

    def test_path_returns_subtree(self):
        result, matched = select(PAYLOAD, "nested.a")
        assert matched is True
        assert result == {"b": {"c": 42}}

    def test_missing_key_is_a_miss(self):
        result, matched = select(PAYLOAD, "nope")
        assert matched is False
        assert result is None

    def test_missing_nested_key_is_a_miss(self):
        _, matched = select(PAYLOAD, "nested.a.zzz")
        assert matched is False

    def test_traversal_into_scalar_is_a_miss(self):
        _, matched = select(PAYLOAD, "name.deeper")
        assert matched is False


class TestIndex:
    def test_array_index(self):
        result, matched = select(PAYLOAD, "inputs.1.name")
        assert matched is True
        assert result == "steps"

    def test_index_out_of_range_is_a_miss(self):
        _, matched = select(PAYLOAD, "inputs.99.name")
        assert matched is False

    def test_index_returns_element(self):
        result, matched = select(PAYLOAD, "output_types.0")
        assert matched is True
        assert result == "LATENT"

    def test_digit_segment_on_dict_is_key_lookup(self):
        result, matched = select({"0": "zero"}, "0")
        assert matched is True
        assert result == "zero"


class TestWildcard:
    def test_wildcard_projects_each_element(self):
        result, matched = select(PAYLOAD, "inputs.#.name")
        assert matched is True
        assert result == ["seed", "steps", "sampler_name"]

    def test_bare_wildcard_returns_whole_array(self):
        result, matched = select(PAYLOAD, "inputs.#")
        assert matched is True
        assert result == PAYLOAD["inputs"]

    def test_wildcard_drops_per_element_misses(self):
        result, matched = select(PAYLOAD, "inputs.#.options.default")
        assert matched is True
        assert result == [0, 20]

    def test_wildcard_on_non_array_is_a_miss(self):
        _, matched = select(PAYLOAD, "nested.#")
        assert matched is False

    def test_wildcard_zero_element_matches_is_a_miss(self):
        _, matched = select(PAYLOAD, "inputs.#.bogus")
        assert matched is False

    def test_wildcard_over_empty_array_matches_empty(self):
        result, matched = select({"items": []}, "items.#.name")
        assert matched is True
        assert result == []

    def test_nested_wildcards_compose(self):
        data = {"rows": [{"tags": ["a", "b"]}, {"tags": ["c"]}]}
        result, matched = select(data, "rows.#.tags.#")
        assert matched is True
        assert result == [["a", "b"], ["c"]]


class TestComma:
    def test_multi_select_returns_object_of_matches(self):
        result, matched = select(PAYLOAD, "name,category")
        assert matched is True
        assert result == {"name": "KSampler", "category": "sampling"}

    def test_multi_select_mixed_expressions(self):
        result, matched = select(PAYLOAD, "name,inputs.#.name")
        assert matched is True
        assert result == {"name": "KSampler", "inputs.#.name": ["seed", "steps", "sampler_name"]}

    def test_multi_select_drops_missing_parts(self):
        result, matched = select(PAYLOAD, "name,nope")
        assert matched is True
        assert result == {"name": "KSampler"}

    def test_multi_select_all_missing_is_a_miss(self):
        _, matched = select(PAYLOAD, "nope,alsonope")
        assert matched is False


class TestMalformed:
    @pytest.mark.parametrize("expr", ["", "   ", ".", "a..b", ".a", "a.", ",", ",,"])
    def test_malformed_expressions_are_misses(self, expr):
        result, matched = select(PAYLOAD, expr)
        assert matched is False
        assert result is None


# ---------------------------------------------------------------------------
# Fail-open inventory + byte accounting (the shared emit path)
# ---------------------------------------------------------------------------


def test_malformed_selector_fails_open_with_key_inventory():
    data, matched, meta = selected_payload(PAYLOAD, "a..b")
    assert matched is False
    inv = data["inventory"]
    # Top-level keys with value types; dicts/lists carry sizes; nested one
    # level of keys.
    assert set(inv) >= {"name", "inputs", "nested"}
    assert inv["inputs"]["type"] == "array"
    assert inv["inputs"]["size"] == 3
    assert inv["nested"]["type"] == "object"
    assert "a" in inv["nested"]["keys"]
    # Advisory hint under a registered code; the command still succeeds.
    warning = data["warnings"][0]
    assert warning["code"] == "select_no_match"
    registered = error_codes.get("select_no_match")
    assert registered is not None, "select_no_match must be registered in error_codes.py"
    assert warning["hint"] == registered.hint
    # Inventory stays bounded.
    assert len(json.dumps(inv)) <= 2048


def test_zero_match_selector_fails_open_like_malformed():
    data, matched, _ = selected_payload(PAYLOAD, "definitely.not.here")
    assert matched is False
    assert "inventory" in data
    assert data["warnings"][0]["code"] == "select_no_match"


def test_inventory_is_bounded_on_huge_payloads():
    huge = {f"key_{i}": {f"sub_{j}": "x" * 50 for j in range(50)} for i in range(200)}
    inv = key_inventory(huge)
    assert len(json.dumps(inv)) <= 2048


def test_envelope_reports_selected_and_total_bytes():
    data, matched, meta = selected_payload(PAYLOAD, "inputs.#.name")
    assert matched is True
    assert data == ["seed", "steps", "sampler_name"]
    assert meta["selected_bytes"] == len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert meta["total_bytes"] == len(json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8"))
    assert meta["selected_bytes"] < meta["total_bytes"]


def test_selected_bytes_counts_the_inventory_slice_on_miss():
    data, matched, meta = selected_payload(PAYLOAD, "nope")
    assert matched is False
    assert meta["selected_bytes"] == len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    assert meta["total_bytes"] == len(json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8"))


def test_select_is_pure_and_does_not_mutate():
    snapshot = json.dumps(PAYLOAD, sort_keys=True)
    select(PAYLOAD, "inputs.#.name")
    select(PAYLOAD, "nope")
    selected_payload(PAYLOAD, "a..b")
    assert json.dumps(PAYLOAD, sort_keys=True) == snapshot
