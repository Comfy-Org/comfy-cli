"""Tests for the openapi registry — verify the curated image allowlist resolves
against the vendored spec and classifies each endpoint correctly."""

from unittest import mock

import pytest
import yaml

from comfy_cli.command.generate import spec

# A minimal JSON spec body — the shape api.comfy.org/openapi actually serves
# (JSON, not YAML). JSON is a subset of YAML 1.2 so it loads via _YamlLoader.
_VALID_JSON_SPEC = (
    '{"openapi":"3.1.0","servers":[{"url":"https://api.comfy.org"}],'
    '"paths":{"/proxy/openai/images/generations":{"post":{"summary":"Create image",'
    '"requestBody":{"content":{"application/json":{"schema":{"type":"object",'
    '"properties":{"prompt":{"type":"string"}}}}}},'
    '"responses":{"200":{"content":{"application/json":{"schema":{"type":"object"}}}}}}}}}'
)


def test_validate_spec_text_accepts_json_and_rejects_bad_bodies():
    parsed = spec.validate_spec_text(_VALID_JSON_SPEC)
    assert isinstance(parsed["paths"], dict)
    for bad in ("not: [valid: yaml", '{"openapi":"3.1.0"}', "null", "[]"):
        try:
            spec.validate_spec_text(bad)
        except spec.SpecError:
            pass
        else:
            raise AssertionError(f"expected SpecError for {bad!r}")


def test_cached_json_spec_round_trips(monkeypatch, tmp_path):
    """(d) a cached JSON body round-trips through load_raw_spec() and
    get_endpoint() resolves an endpoint from it."""
    cache = tmp_path / "openapi-cache.yml"
    cache.write_text(_VALID_JSON_SPEC, encoding="utf-8")
    monkeypatch.setattr(spec, "_USER_CACHE", cache)
    spec.load_raw_spec.cache_clear()
    spec._registry.cache_clear()
    try:
        assert spec.active_spec_path() == cache
        raw = spec.load_raw_spec()
        assert isinstance(raw["paths"], dict)
        ep = spec.get_endpoint("openai/images/generations")
        assert ep.path == "/proxy/openai/images/generations"
        assert ep.method == "post"
    finally:
        # Don't leak the temp spec into other tests via the module-level caches.
        spec.load_raw_spec.cache_clear()
        spec._registry.cache_clear()


def test_registry_loads_and_has_entries():
    eps = spec.list_endpoints()
    assert len(eps) > 20, "expected the v1 allowlist to resolve >20 endpoints"


def test_registry_skips_an_endpoint_whose_schema_cannot_be_resolved(monkeypatch):
    bad_id = "bad/endpoint"
    good_id = "good/endpoint"
    raw = {
        "paths": {
            f"/proxy/{bad_id}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"bad": True}}}},
                    "responses": {},
                }
            },
            f"/proxy/{good_id}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {},
                }
            },
        }
    }
    real_resolve = spec._resolve

    def resolve(raw_spec, schema, *args, **kwargs):
        if schema.get("bad"):
            raise spec.SpecError("schema resolution exceeded its safe traversal limit")
        return real_resolve(raw_spec, schema, *args, **kwargs)

    monkeypatch.setattr(spec, "load_raw_spec", lambda: raw)
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(bad_id, "test", None), (good_id, "test", None)])
    monkeypatch.setattr(spec, "_resolve", resolve)
    spec._registry.cache_clear()
    try:
        assert list(spec._registry()) == [good_id]
    finally:
        spec._registry.cache_clear()


@pytest.mark.parametrize(
    "path_item",
    ["invalid", {"parameters": []}, {"post": "invalid"}, {"post": {"requestBody": "invalid"}}],
)
def test_registry_skips_malformed_path_and_operation_shapes(monkeypatch, path_item):
    endpoint_id = "bad/endpoint"
    monkeypatch.setattr(spec, "load_raw_spec", lambda: {"paths": {f"/proxy/{endpoint_id}": path_item}})
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(endpoint_id, "test", None)])
    spec._registry.cache_clear()
    try:
        assert spec._registry() == {}
    finally:
        spec._registry.cache_clear()


@pytest.mark.parametrize(
    "response",
    [
        "invalid",
        {"content": "invalid"},
        {"content": {"application/json": "invalid"}},
        {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Missing"}}}},
    ],
)
def test_registry_keeps_endpoint_when_only_response_schema_is_malformed(monkeypatch, response):
    endpoint_id = "usable/endpoint"
    raw = {
        "paths": {
            f"/proxy/{endpoint_id}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": response},
                }
            }
        }
    }
    monkeypatch.setattr(spec, "load_raw_spec", lambda: raw)
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(endpoint_id, "test", None)])
    spec._registry.cache_clear()
    try:
        assert list(spec._registry()) == [endpoint_id]
        assert spec._registry()[endpoint_id].response_schema == {}
    finally:
        spec._registry.cache_clear()


def test_registry_computes_one_resolution_budget_per_spec(monkeypatch):
    endpoint_ids = ["one/endpoint", "two/endpoint"]
    raw = {
        "paths": {
            f"/proxy/{endpoint_id}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {"200": {"content": {"application/json": {"schema": {"type": "object"}}}}},
                }
            }
            for endpoint_id in endpoint_ids
        }
    }
    monkeypatch.setattr(spec, "load_raw_spec", lambda: raw)
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(endpoint_id, "test", None) for endpoint_id in endpoint_ids])
    spec._registry.cache_clear()
    try:
        with mock.patch.object(spec, "_schema_resolution_budget", wraps=spec._schema_resolution_budget) as budget:
            assert list(spec._registry()) == endpoint_ids
        assert budget.call_count == 1
    finally:
        spec._registry.cache_clear()


def test_registry_gives_each_endpoint_a_fresh_decrementing_resolution_budget(monkeypatch):
    endpoint_ids = ["one/endpoint", "two/endpoint"]
    raw = {
        "paths": {
            f"/proxy/{endpoint_id}": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                    "responses": {},
                }
            }
            for endpoint_id in endpoint_ids
        }
    }
    monkeypatch.setattr(spec, "load_raw_spec", lambda: raw)
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(endpoint_id, "test", None) for endpoint_id in endpoint_ids])
    budgets: list[list[int]] = []
    real_resolve = spec._resolve

    def resolve(*args, **kwargs):
        budgets.append(kwargs["budget"])
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(spec, "_resolve", resolve)
    spec._registry.cache_clear()
    try:
        assert list(spec._registry()) == endpoint_ids
        assert len(budgets) == len(endpoint_ids)
        assert budgets[0] is not budgets[1]
    finally:
        spec._registry.cache_clear()


def test_registry_ignores_non_string_path_item_method_keys(monkeypatch):
    endpoint_id = "one/endpoint"
    raw = {
        "paths": {
            f"/proxy/{endpoint_id}": {
                200: {"unexpected": True},
                "post": {"requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}}},
            }
        }
    }
    monkeypatch.setattr(spec, "load_raw_spec", lambda: raw)
    monkeypatch.setattr(spec, "_ENDPOINT_ALLOWLIST", [(endpoint_id, "test", None)])
    spec._registry.cache_clear()
    try:
        assert list(spec._registry()) == [endpoint_id]
    finally:
        spec._registry.cache_clear()


def test_get_endpoint_round_trip():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    assert ep.partner == "bfl"
    assert ep.path == "/proxy/bfl/flux-pro-1.1/generate"
    assert ep.method == "post"
    assert ep.polling == "bfl"
    assert ep.category == "text-to-image"


def test_unknown_endpoint_suggests_close_match():
    try:
        spec.get_endpoint("bfl/flux-pro-1.1/genrate")  # typo
    except spec.SpecError as e:
        assert "Did you mean" in str(e)
        assert "bfl/flux-pro-1.1/generate" in str(e)
    else:
        raise AssertionError("expected SpecError")


def test_request_schema_resolved_no_refs():
    ep = spec.get_endpoint("ideogram/ideogram-v3/generate")
    props = ep.request_schema["properties"]
    # `rendering_speed` was a $ref in source; should now be inlined.
    assert isinstance(props["rendering_speed"], dict)
    assert "$ref" not in props["rendering_speed"]


def test_multipart_endpoints_detected():
    ep = spec.get_endpoint("ideogram/ideogram-v3/edit")
    assert ep.request_content_type == "multipart/form-data"


def test_json_endpoints_detected():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    assert ep.request_content_type == "application/json"


def test_sync_endpoints_have_no_polling():
    ep = spec.get_endpoint("openai/images/generations")
    assert ep.polling is None


def test_filter_by_partner_and_category():
    bfl = spec.list_endpoints(partner="bfl")
    assert bfl and all(e.partner == "bfl" for e in bfl)
    t2i = spec.list_endpoints(category="text-to-image")
    assert all(e.category == "text-to-image" for e in t2i)


def test_proxy_prefix_accepted():
    ep = spec.get_endpoint("/proxy/bfl/flux-pro-1.1/generate")
    assert ep.id == "bfl/flux-pro-1.1/generate"


def test_unknown_model_suggests_leading_token_family(monkeypatch):
    """Test that unknown-model errors suggest family members keyed on leading token.

    Regression test for difflib alone ranking cross-partner shape-alikes over
    the caller's intended family. Monkeypatches both _registry and _ALIASES to
    fully isolate the candidate pool — difflib returns no close matches, so only
    the family-finding code contributes suggestions.
    """
    # Mock registry with kling-extend, kling-lipsync (kling family), plus runway
    mock_registry = {
        "kling/v1/videos/video-extend": spec.Endpoint(
            id="kling/v1/videos/video-extend",
            path="/proxy/kling/v1/videos/video-extend",
            method="post",
            partner="kling",
            summary="Extend video",
            category="video-extend",
            request_schema={},
            request_content_type="application/json",
            response_schema={},
            polling="kling",
        ),
        "kling/v1/videos/lip-sync": spec.Endpoint(
            id="kling/v1/videos/lip-sync",
            path="/proxy/kling/v1/videos/lip-sync",
            method="post",
            partner="kling",
            summary="Lip sync",
            category="lipsync",
            request_schema={},
            request_content_type="application/json",
            response_schema={},
            polling="kling",
        ),
        "runway/image_to_video": spec.Endpoint(
            id="runway/image_to_video",
            path="/proxy/runway/image_to_video",
            method="post",
            partner="runway",
            summary="Image to video",
            category="image-to-video",
            request_schema={},
            request_content_type="application/json",
            response_schema={},
            polling=None,
        ),
    }
    # Fully isolate candidates: mock both _registry and _ALIASES so the test
    # is not driven by real-world aliases that happen to match the input.
    mock_aliases = {}  # Empty: no alias coincidences to mask the family logic
    spec._registry.cache_clear()
    monkeypatch.setattr(spec, "_registry", lambda: mock_registry)
    monkeypatch.setattr(spec, "_ALIASES", mock_aliases)

    msg = spec._unknown_endpoint_message("kling-image-to-video")

    # difflib finds no close matches (cutoff=0.5, minimal overlap).
    # Only the family-finding code (leading token="kling") contributes suggestions.
    # Exactly two kling family members should appear, sorted, in the message.
    assert msg.startswith("Unknown model: 'kling-image-to-video'")
    assert "Did you mean:" in msg
    assert "kling/v1/videos/lip-sync" in msg
    assert "kling/v1/videos/video-extend" in msg
    assert "comfy generate list" in msg


def test_unknown_model_no_family_still_helpful(monkeypatch):
    """Test that errors are helpful even when there's no family match."""
    # Isolate to ensure _ALIASES is controlled
    mock_aliases = {"some-unrelated": "foo/bar/baz"}
    spec._registry.cache_clear()
    monkeypatch.setattr(spec, "_ALIASES", mock_aliases)

    msg = spec._unknown_endpoint_message("krea-2")
    assert msg.startswith("Unknown model: 'krea-2'")
    assert "comfy generate list" in msg


@pytest.mark.parametrize("text", ["1e+16", "1e-07", "-2e+5"])
def test_pointless_exponent_parses_as_float(text):
    """Regression: the spec is now fetched as JSON, and `json.dumps`
    emits exponents without a decimal point for very large/small floats. PyYAML's
    YAML 1.1 resolver leaves those as strings, which would leak into flag schemas.
    """
    assert isinstance(yaml.load(text, Loader=spec._YamlLoader), float)


@pytest.mark.parametrize("text", ["3.0.2", "on", "off", "v1e5"])
def test_float_resolver_does_not_over_match(text):
    """The added exponent resolver must not swallow version strings or the
    string enums the spec relies on."""
    assert isinstance(yaml.load(text, Loader=spec._YamlLoader), str)


def test_validate_spec_text_accepts_openapi_mapping():
    parsed = spec.validate_spec_text('{"openapi": "3.0.2", "paths": {}}')
    assert parsed["openapi"] == "3.0.2"


@pytest.mark.parametrize(
    "text",
    [
        "<html><body>Just a moment...</body></html>",  # interstitial → str
        "[1, 2, 3]",  # JSON array
        "42",  # JSON scalar
        '{"foo": 1}',  # mapping, but not an OpenAPI doc
    ],
)
def test_validate_spec_text_rejects_non_spec_bodies(text):
    """A non-spec 200 must be refused, not cached for the 7-day TTL."""
    with pytest.raises(spec.SpecError):
        spec.validate_spec_text(text)


# ── model_enum — spec-derived partner model lists ─────────────────────────


def test_model_enum_from_vendored_spec():
    models = spec.model_enum("byteplus/api/v3/contents/generations/tasks")
    assert models, "expected the byteplus tasks request schema to carry a model enum"
    assert all(m.startswith("seedance-") for m in models)


def test_model_enum_returns_none_without_enum():
    # Gemini's model variant is a path param, not a request-body property.
    assert spec.model_enum("vertexai/gemini/{model}") is None
    # Property exists but carries no enum.
    assert spec.model_enum("openai/images/generations", field="prompt") is None
    # Unknown endpoint / unknown field — no exception, just None.
    assert spec.model_enum("nope/nope") is None
    assert spec.model_enum("openai/images/generations", field="nope") is None


def test_extract_enum_walks_items_and_variants():
    assert spec._extract_enum({"enum": ["a", "b"]}) == ["a", "b"]
    assert spec._extract_enum({"type": "array", "items": {"enum": ["x"]}}) == ["x"]
    assert spec._extract_enum({"anyOf": [{"type": "integer"}, {"enum": ["y"]}]}) == ["y"]
    assert spec._extract_enum({"anyOf": [{"type": "null"}, {"enum": ["optional"]}]}) == ["optional"]
    assert spec._extract_enum({"oneOf": [{"items": {"enum": ["z"]}}]}) == ["z"]
    # Numeric members coerce to their string form (unquoted YAML values);
    # bools and enum-less schemas don't count.
    assert spec._extract_enum({"enum": [1, 2.5]}) == ["1", "2.5"]
    assert spec._extract_enum({"enum": [True, False]}) is None
    assert spec._extract_enum({"type": "string"}) is None


def test_extract_enum_unions_anyof_and_intersects_allof():
    # A spec that splits the model set across anyOf/oneOf branches surfaces
    # every branch, deduped, not just the first.
    assert spec._extract_enum({"anyOf": [{"enum": ["a", "b"]}, {"enum": ["b", "c"]}]}) == ["a", "b", "c"]
    assert spec._extract_enum({"oneOf": [{"enum": ["x"]}, {"enum": ["y"]}]}) == ["x", "y"]
    # allOf branches are constraints: only values valid in every branch count.
    assert spec._extract_enum({"allOf": [{"enum": ["a", "b", "c"]}, {"enum": ["b", "c", "d"]}]}) == ["b", "c"]
    # An empty allOf intersection means no usable enum.
    assert spec._extract_enum({"allOf": [{"enum": ["a"]}, {"enum": ["b"]}]}) is None
    # An enum-less allOf branch constrains nothing.
    assert spec._extract_enum({"allOf": [{"type": "string"}, {"enum": ["k"]}]}) == ["k"]
    assert spec._extract_enum({"enum": ["a", "b"], "allOf": [{"enum": ["a"]}]}) == ["a"]
    assert spec._extract_enum(
        {"anyOf": [{"type": "string"}, {"enum": ["ignored"]}], "allOf": [{"enum": ["kept"]}]}
    ) == ["kept"]


def test_find_property_descends_top_level_composition():
    # A request body composed via top-level allOf/anyOf/oneOf still surfaces
    # its model property instead of silently falling back to hardcoded lists.
    composed = {"allOf": [{"type": "object"}, {"properties": {"model": {"enum": ["m1"]}}}]}
    assert spec._find_property(composed, "model") == {"enum": ["m1"]}
    nested = {"anyOf": [{"oneOf": [{"properties": {"model": {"enum": ["m2"]}}}]}]}
    assert spec._find_property(nested, "model") == {"enum": ["m2"]}
    assert spec._find_property({"allOf": [{"type": "object"}]}, "model") is None


def test_find_property_preserves_combinator_enum_semantics():
    all_of = {
        "allOf": [
            {"properties": {"model": {"enum": ["allowed", "forbidden"]}}},
            {"properties": {"model": {"enum": ["allowed"]}}},
        ]
    }
    any_of = {
        "anyOf": [
            {"properties": {"model": {"enum": ["v1"]}}},
            {"properties": {"model": {"enum": ["v2"]}}},
        ]
    }

    assert spec._extract_enum(spec._find_property(all_of, "model")) == ["allowed"]
    assert spec._extract_enum(spec._find_property(any_of, "model")) == ["v1", "v2"]


def test_find_property_treats_a_non_declaring_union_branch_as_unconstrained():
    schema = {
        "properties": {"model": {"enum": ["a", "b"]}},
        "anyOf": [
            {"properties": {"model": {"enum": ["a"]}}},
            {"properties": {"other": {"type": "string"}}},
        ],
    }

    assert spec._extract_enum(spec._find_property(schema, "model")) == ["a", "b"]


def test_find_property_treats_an_enumless_declared_union_branch_as_unconstrained():
    schema = {
        "anyOf": [
            {"properties": {"model": {"enum": ["v1"]}}},
            {"properties": {"model": {"type": "string"}}},
        ]
    }

    assert spec._extract_enum(spec._find_property(schema, "model")) is None
