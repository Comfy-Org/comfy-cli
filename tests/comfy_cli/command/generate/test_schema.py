"""Tests for openapi schema → CLI flag conversion and argv parsing."""

import pytest

from comfy_cli.command.generate import schema, spec


def test_flags_for_bfl_classifies_types():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = {f.name: f for f in schema.flags_for(ep)}
    assert flags["prompt"].kind == "string"
    assert flags["prompt"].required
    assert flags["width"].kind == "integer"
    assert flags["prompt_upsampling"].kind == "boolean"
    assert flags["output_format"].kind == "enum"
    assert flags["output_format"].enum == ["jpeg", "png"]


def test_flags_for_multipart_finds_binary_fields():
    ep = spec.get_endpoint("ideogram/ideogram-v3/edit")
    flags = {f.name: f for f in schema.flags_for(ep)}
    assert flags["image"].kind == "binary"
    # style_reference_images is an array of binary file inputs.
    assert flags["style_reference_images"].kind == "array"
    assert flags["style_reference_images"].item_kind == "binary"


def test_parse_args_basic_coercion():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    values = schema.parse_args(
        flags,
        ["--prompt", "a cat", "--width", "1024", "--height", "1024", "--prompt_upsampling"],
    )
    assert values == {
        "prompt": "a cat",
        "width": 1024,
        "height": 1024,
        "prompt_upsampling": True,
    }


def test_parse_args_eq_form_and_enum():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    values = schema.parse_args(
        flags,
        ["--prompt=a", "--width=1", "--height=1", "--output_format=png"],
    )
    assert values["output_format"] == "png"


def test_parse_args_rejects_unknown_flag():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    with pytest.raises(schema.SchemaError, match="Unknown flag"):
        schema.parse_args(flags, ["--prompt", "a", "--width", "1", "--height", "1", "--bogus", "x"])


def test_parse_args_rejects_bad_int():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    with pytest.raises(schema.SchemaError, match="expected integer"):
        schema.parse_args(flags, ["--prompt", "a", "--width", "abc", "--height", "1"])


def test_parse_args_missing_required():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    with pytest.raises(schema.SchemaError, match="Missing required"):
        schema.parse_args(flags, ["--prompt", "a"])


def test_parse_args_enum_value_validated():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    with pytest.raises(schema.SchemaError, match="not one of"):
        schema.parse_args(
            flags,
            ["--prompt", "a", "--width", "1", "--height", "1", "--output_format", "tiff"],
        )


def test_parse_args_object_accepts_json():
    ep = spec.get_endpoint("ideogram/ideogram-v3/generate")
    flags = schema.flags_for(ep)
    values = schema.parse_args(
        flags,
        [
            "--prompt",
            "x",
            "--rendering_speed",
            "TURBO",
            "--color_palette",
            '{"name":"PASTEL"}',
        ],
    )
    assert values["color_palette"] == {"name": "PASTEL"}
