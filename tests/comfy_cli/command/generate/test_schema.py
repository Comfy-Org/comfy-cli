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


def _string_array_flag():
    return schema.FlagDef(
        name="image",
        kind="array",
        required=False,
        description="",
        default=None,
        enum=[],
        item_kind="string",
        upload_mode=None,
    )


def test_coerce_string_array_accepts_bare_value():
    # prod: --image 'Linked profile pic.jpeg' (spaces, no JSON) must not error
    assert schema._coerce(_string_array_flag(), "Linked profile pic.jpeg") == ["Linked profile pic.jpeg"]


def test_coerce_string_array_accepts_comma_list():
    assert schema._coerce(_string_array_flag(), "a.jpg, b.png") == ["a.jpg", "b.png"]


def test_coerce_string_array_json_still_works():
    assert schema._coerce(_string_array_flag(), '["a.jpg","b.png"]') == ["a.jpg", "b.png"]


def test_coerce_string_array_malformed_json_still_errors():
    # An explicit-JSON attempt ('[' prefix) that is broken must keep failing loudly,
    # not be silently reinterpreted as a filename starting with '['.
    with pytest.raises(schema.SchemaError):
        schema._coerce(_string_array_flag(), '["a.jpg",')


def test_coerce_string_array_empty_raises():
    # Empty string splits and strips to no items; should raise SchemaError, not return [].
    with pytest.raises(schema.SchemaError, match="expected at least one value"):
        schema._coerce(_string_array_flag(), "")


def test_coerce_string_array_commas_only_raises():
    # Commas and whitespace split/strip to no items; should raise SchemaError, not return [].
    with pytest.raises(schema.SchemaError, match="expected at least one value"):
        schema._coerce(_string_array_flag(), ", ,")


def test_flags_for_unwraps_single_branch_anyof():
    # BFL's Fill inputs wrap every optional field in a one-branch anyOf; the
    # wrapper is spec noise, not a real union, so the declared type must win.
    ep = spec.get_endpoint("bfl/flux-pro-1.0-fill/generate")
    flags = {f.name: f for f in schema.flags_for(ep)}
    assert flags["prompt"].kind == "string"
    assert flags["steps"].kind == "integer"
    assert flags["guidance"].kind == "number"
    assert flags["output_format"].kind == "enum"
    assert flags["output_format"].enum == ["jpeg", "png"]
    # The parent's own default survives the unwrap.
    assert flags["output_format"].default == "jpeg"


def test_flags_for_single_branch_anyof_keeps_base64_upload():
    # upload-mode detection only inspects `string` props, so a mis-classified
    # `object` silently loses it and forces the caller to base64 by hand.
    ep = spec.get_endpoint("bfl/flux-pro-1.0-fill/generate")
    flags = {f.name: f for f in schema.flags_for(ep)}
    assert flags["mask"].upload_mode == "base64"
    assert flags["image"].upload_mode == "base64"


def test_unwrap_single_variant_drops_null_branch():
    prop = {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "d"}
    assert schema._unwrap_single_variant(prop) == {"type": "string", "description": "d"}


def test_unwrap_single_variant_keeps_real_union_as_object():
    prop = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert schema._unwrap_single_variant(prop) == prop
    assert schema._classify(prop) == ("object", None)


def test_unwrap_single_variant_branch_overrides_parent_type():
    prop = {"anyOf": [{"type": "integer"}], "title": "outer", "default": 3}
    assert schema._unwrap_single_variant(prop) == {"type": "integer", "title": "outer", "default": 3}


def test_parse_args_accepts_plain_text_for_wrapped_prompt():
    # The regression this guards: `--prompt "a fox"` used to fail with
    # "expected JSON object" because the one-branch anyOf read as an object.
    ep = spec.get_endpoint("bfl/flux-pro-1.0-expand/generate")
    flags = schema.flags_for(ep)
    values = schema.parse_args(flags, ["--image", "in.png", "--prompt", "a fox", "--left", "236"])
    assert values["prompt"] == "a fox"
    assert values["left"] == 236
