"""Tests for the openapi registry — verify the curated image allowlist resolves
against the vendored spec and classifies each endpoint correctly."""

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


def test_find_property_descends_top_level_composition():
    # A request body composed via top-level allOf/anyOf/oneOf still surfaces
    # its model property instead of silently falling back to hardcoded lists.
    composed = {"allOf": [{"type": "object"}, {"properties": {"model": {"enum": ["m1"]}}}]}
    assert spec._find_property(composed, "model") == {"enum": ["m1"]}
    nested = {"anyOf": [{"oneOf": [{"properties": {"model": {"enum": ["m2"]}}}]}]}
    assert spec._find_property(nested, "model") == {"enum": ["m2"]}
    assert spec._find_property({"allOf": [{"type": "object"}]}, "model") is None
