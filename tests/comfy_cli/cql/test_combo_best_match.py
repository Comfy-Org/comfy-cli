"""A rejected COMBO value whose leading token names exactly one option.

`workflow set-widget <id>.aspect_ratio '16:9 (Landscape)'` on a
ResolutionSelector, whose options are
'16:9 (Widescreen)', '9:16 (Portrait Widescreen)', '21:9 (Ultrawide)', ...
The did_you_mean list was right ('16:9 (Widescreen)' ranked first), but it
read like four equal guesses. One option has the same ratio, so that option
is the likely intent. The refusal now names it as `best_match` and leads the
hint with it. It still never writes it: a guessed value is the caller's call.
"""

from __future__ import annotations

from comfy_cli.cql.engine import Port

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


def _port(options=RATIOS) -> Port:
    return Port(name="aspect_ratio", type="COMBO", enum_values=list(options))


def test_unique_leading_token_is_the_best_match():
    [w] = _port().validate_catalog("16:9 (Landscape)")
    assert w["code"] == "unknown_enum_value"
    assert w["best_match"] == "16:9 (Widescreen)", w
    assert w["did_you_mean"][0] == "16:9 (Widescreen)", w
    assert "16:9 (Widescreen)" in w["message"], w


def test_ambiguous_leading_token_names_no_best_match():
    [w] = _port([*RATIOS, "16:9 (HD)"]).validate_catalog("16:9 (Landscape)")
    assert "best_match" not in w, w


def test_single_token_value_names_no_best_match():
    """A bare filename or word has no qualifier to disagree on — difflib only."""
    [w] = _port(["sd_xl_base.safetensors", "v1-5-pruned.safetensors"]).validate_catalog("v1-5-prund.safetensors")
    assert "best_match" not in w, w


def test_filename_value_names_no_best_match():
    """'flux dev.safetensors' sharing 'flux' with 'flux schnell.safetensors' is a
    different model, not a relabelled one — never promote it."""
    [w] = _port(["flux schnell.safetensors", "sdxl base.safetensors"]).validate_catalog("flux dev.safetensors")
    assert "best_match" not in w, w
