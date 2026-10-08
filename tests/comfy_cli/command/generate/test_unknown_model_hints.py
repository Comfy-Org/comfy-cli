"""`generate` names a partner MODEL that is not an alias, instead of guessing aliases.

`comfy generate <name>` used to fail with
`Unknown model: 'gpt-image-1'` and `Unknown model: 'seedream'` ("Did you mean:
seedance, ideogram?"). Neither is a typo:

* gpt-image-1 is an OpenAI image model. The `dalle` alias
  (openai/images/generations) takes it as `--model`, but that schema types
  `model` as a free string, so nothing linked the name to the alias.
* seedream is ByteDance's IMAGE model family, served at
  byteplus/api/v3/images/generations (an enum of seedream-* ids in the spec).
  `comfy generate` has no alias for that route. The suggested `seedance` is
  the VIDEO model, which is the wrong substitute.
"""

from __future__ import annotations

from unittest import mock

import pytest

from comfy_cli.command.generate import spec


@pytest.fixture(autouse=True)
def _bundled_spec(monkeypatch, tmp_path):
    # Read the vendored spec, never a developer's ~/.comfy cache.
    monkeypatch.setattr(spec, "_USER_CACHE", tmp_path / "absent.yml")
    spec.load_raw_spec.cache_clear()
    spec._registry.cache_clear()
    yield
    spec.load_raw_spec.cache_clear()
    spec._registry.cache_clear()


@pytest.mark.parametrize("name", ["gpt-image-1", "gpt-image-1.5", "GPT-Image-2"])
def test_gpt_image_name_points_at_the_partner_node_first(name):
    """A caller building a workflow runs generate with --emit-workflow, and
    `dalle` cannot emit. So the workflow route (the
    partner node) comes first, and `generate dalle --model` is labelled as the
    direct, non-workflow route."""
    msg = spec._unknown_endpoint_message(name)
    assert "OpenAIGPTImageNodeV2" in msg, msg
    assert msg.index("OpenAIGPTImageNodeV2") < msg.index("comfy generate dalle"), msg
    assert f"--model {name.lower()}" in msg, msg
    assert "--emit-workflow" in msg, f"must say the direct route cannot emit a workflow: {msg}"


def test_dall_e_name_names_only_the_direct_route():
    msg = spec._unknown_endpoint_message("dall-e-3")
    assert "comfy generate dalle --model dall-e-3" in msg, msg


def test_seedream_names_its_route_and_partner_node_search():
    msg = spec._unknown_endpoint_message("seedream")
    assert "byteplus/api/v3/images/generations" in msg, msg
    assert "seedream-4-5-251128" in msg, msg
    assert "no `comfy generate` alias" in msg, msg
    assert "comfy nodes search seedream" in msg, msg


@pytest.mark.parametrize(
    ("name", "alias"),
    [("flux", "flux-pro"), ("stable", "stability-sd3"), ("minimax", "minimax/video_generation"), ("seed", "seedance")],
)
def test_alias_typos_keep_did_you_mean_even_when_a_model_hint_applies(name, alias):
    """A short name that prefixes some partner's model ids ("flux" → recraft's
    flux1dev) is still most likely an alias typo; the hint is appended, never
    replaces the alias suggestions."""
    msg = spec._unknown_endpoint_message(name)
    assert "Did you mean:" in msg, msg
    did_you_mean = msg.split("Did you mean:", 1)[1].split("\n", 1)[0]
    assert alias in did_you_mean, msg


def test_model_hint_for_an_alias_with_emit_names_the_emit_route():
    """seedance can emit a workflow, so its --model hint may say so."""
    msg = spec._unknown_endpoint_message("seedance-1-5-pro")
    assert "comfy generate seedance --model" in msg, msg


def test_plain_typo_keeps_did_you_mean():
    msg = spec._unknown_endpoint_message("flux-pr")
    assert "Did you mean:" in msg and "flux-pro" in msg, msg


@pytest.mark.parametrize(
    "request_body",
    [
        "invalid",
        {"content": ["invalid"]},
        {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Missing"}}}},
        {"content": {"application/json": {"schema": {"$ref": 1}}}},
    ],
)
def test_model_hint_skips_malformed_partner_request_bodies(monkeypatch, request_body):
    raw = {
        "paths": {
            "/proxy/malformed": {"post": {"requestBody": request_body}},
            "/proxy/valid": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"model_id": {"enum": ["example-model-v1"]}},
                                }
                            }
                        }
                    }
                }
            },
        }
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("example")
    assert hint is not None
    assert "example-model-v1" in hint
    assert "served at valid" in hint


def test_unhashable_ref_raises_the_schema_error():
    with pytest.raises(spec.SpecError, match="Invalid non-string \\$ref"):
        spec._resolve({}, {"$ref": ["not", "a", "reference"]})


def test_model_hint_falls_through_a_freeform_model_field(monkeypatch):
    raw = {
        "paths": {
            "/proxy/mixed": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"type": "string"},
                                        "model_id": {"enum": ["example-model-v1"]},
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("example")
    assert hint is not None
    assert "example-model-v1" in hint


def test_model_hint_searches_later_enum_fields_and_uses_their_real_flag(monkeypatch):
    raw = {
        "paths": {
            "/proxy/kling/v1/videos/text2video": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"enum": ["family-a"]},
                                        "model_name": {"enum": ["kling-v3"]},
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("kling-v3")
    assert hint is not None
    assert "--model-name <id>" in hint, hint
    assert "--model <id>" not in hint, hint


def test_model_hint_emits_one_route_when_multiple_fields_match(monkeypatch):
    raw = {
        "paths": {
            "/proxy/kling/v1/videos/text2video": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model": {"enum": ["kling-v3"]},
                                        "model_name": {"enum": ["kling-v3"]},
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("kling-v3")
    assert hint is not None
    assert hint.count("- kling-v3:") == 1, hint
    assert "--model <id>" in hint, hint
    assert "--model-name <id>" not in hint, hint


def test_extract_enum_memoizes_shared_schema_branches():
    schema: dict = {"enum": ["example-model-v1"]}
    depth = 32
    for _ in range(depth):
        schema = {"anyOf": [schema, schema]}

    with mock.patch.object(spec, "_extract_enum", wraps=spec._extract_enum) as extract:
        assert spec._extract_enum(schema) == ["example-model-v1"]

    assert extract.call_count <= depth * 2 + 1


def test_model_hint_skips_an_excessively_deep_ref_chain(monkeypatch):
    schemas = {f"S{i}": {"$ref": f"#/components/schemas/S{i + 1}"} for i in range(1100)}
    schemas["S1100"] = {"type": "object"}
    raw = {
        "components": {"schemas": schemas},
        "paths": {
            "/proxy/deep": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/S0"}}}}
                }
            },
            "/proxy/valid": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"model_id": {"enum": ["example-model-v1"]}},
                                }
                            }
                        }
                    }
                }
            },
        },
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("example")
    assert hint is not None
    assert "example-model-v1" in hint


def test_model_hint_memoizes_a_wide_ref_diamond(monkeypatch):
    schemas = {
        f"S{i}": {
            "allOf": [
                {"$ref": f"#/components/schemas/S{i + 1}"},
                {"$ref": f"#/components/schemas/S{i + 1}"},
            ]
        }
        for i in range(32)
    }
    schemas["S32"] = {"type": "object"}
    raw = {
        "components": {"schemas": schemas},
        "paths": {
            "/proxy/wide": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/S0"}}}}
                }
            },
            "/proxy/valid": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"model_id": {"enum": ["example-model-v1"]}},
                                }
                            }
                        }
                    }
                }
            },
        },
    }

    def load_raw_spec():
        return raw

    load_raw_spec.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(spec, "load_raw_spec", load_raw_spec)

    hint = spec._model_name_hint("example")
    assert hint is not None
    assert "example-model-v1" in hint


def test_resolve_memoizes_shared_inline_alias_branches():
    schema: dict = {"type": "string"}
    depth = 32
    for _ in range(depth):
        schema = {"anyOf": [schema, schema]}

    with mock.patch.object(spec, "_resolve_schema", wraps=spec._resolve_schema) as resolve:
        resolved = spec._resolve({}, schema)

    assert isinstance(resolved, dict)
    assert resolve.call_count <= depth * 3 + 2


def test_resolve_shares_cycle_free_targets_across_different_ref_ancestries():
    schemas: dict[str, dict] = {"S40": {"type": "string"}}
    for level in reversed(range(40)):
        schemas[f"S{level}"] = {
            "anyOf": [
                {"$ref": f"#/components/schemas/A{level}"},
                {"$ref": f"#/components/schemas/B{level}"},
            ]
        }
        schemas[f"A{level}"] = {"$ref": f"#/components/schemas/S{level + 1}"}
        schemas[f"B{level}"] = {"$ref": f"#/components/schemas/S{level + 1}"}
    raw = {"components": {"schemas": schemas}}

    with mock.patch.object(spec, "_resolve_schema", wraps=spec._resolve_schema) as resolve:
        resolved = spec._resolve(raw, {"$ref": "#/components/schemas/S0"})

    assert isinstance(resolved, dict)
    assert resolve.call_count <= 40 * 10


def test_extract_enum_deduplicates_repeated_cached_branches_linearly():
    enum = {"enum": [f"model-{index}" for index in range(2_000)]}
    schema = {"anyOf": [enum] * 2_000}

    with mock.patch.object(spec, "_extract_enum", wraps=spec._extract_enum) as extract:
        values = spec._extract_enum(schema)

    assert values == enum["enum"]
    assert extract.call_count <= 2_001


def test_ref_memo_does_not_reuse_a_cycle_pruned_resolution():
    outer_ref = "#/components/schemas/Outer"
    inner_ref = "#/components/schemas/Inner"
    raw = {
        "components": {
            "schemas": {
                "Outer": {
                    "type": "object",
                    "properties": {
                        "model": {"enum": ["example-model-v1"]},
                        "child": {"$ref": inner_ref},
                    },
                },
                "Inner": {"allOf": [{"$ref": outer_ref}]},
            }
        }
    }
    resolved = spec._resolve(raw, {"anyOf": [{"$ref": outer_ref}, {"$ref": inner_ref}]})

    sibling_inner = resolved["anyOf"][1]
    assert spec._extract_enum(spec._find_property(sibling_inner, "model")) == ["example-model-v1"]
