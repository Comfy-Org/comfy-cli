"""Tests for the openapi registry — verify the curated image allowlist resolves
against the vendored spec and classifies each endpoint correctly."""

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


def test_unknown_model_suggests_leading_token_family(monkeypatch):
    """Test that unknown-model errors suggest family members keyed on leading token."""
    # Mock registry with kling-extend, kling-lipsync (kling family), plus some other endpoints
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
    # Clear the cache before patching so the mock takes effect
    spec._registry.cache_clear()
    monkeypatch.setattr(spec, "_registry", lambda: mock_registry)

    msg = spec._unknown_endpoint_message("kling-image-to-video")

    # Should contain kling family members
    assert "kling/v1/videos/video-extend" in msg or "kling-extend" in msg or "video-extend" in msg
    assert "kling/v1/videos/lip-sync" in msg or "kling-lipsync" in msg or "lip-sync" in msg
    # Should start correctly
    assert msg.startswith("Unknown model: 'kling-image-to-video'")
    # Should end with the help message
    assert "comfy generate list" in msg


def test_unknown_model_no_family_still_helpful():
    """Test that errors are helpful even when there's no family match."""
    msg = spec._unknown_endpoint_message("krea-2")
    assert msg.startswith("Unknown model: 'krea-2'")
    assert "comfy generate list" in msg
