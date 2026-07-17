"""Tests for the openapi registry — verify the curated image allowlist resolves
against the vendored spec and classifies each endpoint correctly."""

import pytest
import yaml

from comfy_cli.command.generate import spec


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


@pytest.mark.parametrize("text", ["1e+16", "1e-07", "-2e+5"])
def test_pointless_exponent_parses_as_float(text):
    """Regression (BE-2982): the spec is now fetched as JSON, and `json.dumps`
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
