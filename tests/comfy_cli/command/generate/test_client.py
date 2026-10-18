"""Tests for the httpx client wrapper — auth header, payload split."""

import httpx
import pytest

from comfy_cli.command.generate import client, schema, spec


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "  sk-abc  ")
    assert client.resolve_api_key() == "sk-abc"


def test_resolve_api_key_explicit_wins(monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "env-key")
    assert client.resolve_api_key("flag-key") == "flag-key"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    # No OAuth session in the isolated test config and no env key -> "no credential"
    # path. OAuth-over-env precedence is intentional; assert the current message.
    with pytest.raises(client.ApiError, match="Not signed in"):
        client.resolve_api_key()


def test_split_payload_json_pass_through():
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    json_body, files, data = client._split_payload(
        {"prompt": "x", "width": 1024, "height": 1024},
        flags,
        ep.request_content_type,
    )
    assert json_body == {"prompt": "x", "width": 1024, "height": 1024}
    assert files is None and data is None


def test_split_payload_multipart_separates_files(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")
    ep = spec.get_endpoint("ideogram/ideogram-v3/edit")
    flags = schema.flags_for(ep)
    json_body, files, data = client._split_payload(
        {"prompt": "edit", "rendering_speed": "TURBO", "image": img, "num_images": 2},
        flags,
        ep.request_content_type,
    )
    assert json_body is None
    field_names = [name for name, _ in files]
    assert "image" in field_names
    assert data["prompt"] == "edit"
    assert data["num_images"] == "2"
    # Close any file handles we opened.
    for _name, payload in files:
        payload[1].close()


def test_send_request_encodes_form_urlencoded_values(monkeypatch):
    ep = spec.get_endpoint("pika/generate/2.2/t2v")
    flags = schema.flags_for(ep)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(200)

    monkeypatch.setattr(client.httpx, "post", fake_post)

    client.send_request(ep, {"promptText": "hello"}, flags, api_key="comfyui-test")

    assert captured["data"] == {"promptText": "hello"}
    assert captured.get("files") is None
    assert captured.get("json") is None


def test_multipart_without_a_file_stays_multipart(monkeypatch):
    ep = spec.Endpoint(
        id="example/multipart",
        path="/proxy/example/multipart",
        method="post",
        partner="example",
        summary="",
        category="test",
        request_schema={"type": "object"},
        request_content_type="multipart/form-data",
        response_schema={},
        polling=None,
    )
    captured = {}
    monkeypatch.setattr(
        client.httpx,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or httpx.Response(200),
    )

    client.send_request(ep, {"prompt": "hello"}, [], api_key="comfyui-test")

    assert captured["files"] == [("prompt", (None, "hello"))]
    assert captured["data"] is None


def test_form_urlencoded_arrays_repeat_keys_and_binary_is_refused(tmp_path):
    flags = [
        schema.FlagDef(name="tags", kind="array", item_kind="string", required=False),
        schema.FlagDef(name="image", kind="binary", required=False),
    ]
    data = client._split_payload({"tags": ["a", "b"]}, flags, "application/x-www-form-urlencoded")[2]
    assert data == {"tags": ["a", "b"]}
    request = httpx.Request("POST", "https://example.test", data=data)
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert request.content == b"tags=a&tags=b"
    image = tmp_path / "image.png"
    image.write_bytes(b"x")
    with pytest.raises(client.ApiError, match="require multipart"):
        client._split_payload({"image": image}, flags, "application/x-www-form-urlencoded")


def test_form_urlencoded_rejects_non_array_and_nested_values():
    flags = [schema.FlagDef(name="tags", kind="array", item_kind="string", required=False)]
    with pytest.raises(client.ApiError, match="expected an array"):
        client._split_payload({"tags": "not-an-array"}, flags, "application/x-www-form-urlencoded")
    with pytest.raises(client.ApiError, match="nested form values"):
        client._split_payload({"tags": [{"secret": "value"}]}, flags, "application/x-www-form-urlencoded")


def test_empty_multipart_payload_forces_multipart_encoding():
    payload = client._split_payload({}, [], "multipart/form-data")
    assert payload == (None, [], None)


def test_send_empty_multipart_payload_builds_a_zero_part_wire_body(monkeypatch):
    ep = spec.Endpoint(
        id="example/multipart",
        path="/proxy/example/multipart",
        method="post",
        partner="example",
        summary="",
        category="test",
        request_schema={"type": "object"},
        request_content_type="multipart/form-data",
        response_schema={},
        polling=None,
    )
    captured = {}
    monkeypatch.setattr(
        client.httpx,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or httpx.Response(200),
    )

    client.send_request(ep, {}, [], api_key="comfyui-test")

    assert captured["headers"]["Content-Type"] == "multipart/form-data; boundary=comfy-cli-empty"
    assert captured["content"] == b"--comfy-cli-empty--\r\n"
    assert "files" not in captured and "data" not in captured


def test_parameterized_json_media_type_uses_json_body(monkeypatch):
    ep = spec.Endpoint(
        id="example/json",
        path="/proxy/example/json",
        method="post",
        partner="example",
        summary="",
        category="test",
        request_schema={"type": "object"},
        request_content_type="application/problem+json; charset=utf-8",
        response_schema={},
        polling=None,
    )
    captured = {}
    monkeypatch.setattr(
        client.httpx,
        "post",
        lambda url, **kwargs: captured.update(url=url, **kwargs) or httpx.Response(200),
    )

    client.send_request(ep, {"prompt": "hello"}, [], api_key="comfyui-test")

    assert captured["json"] == {"prompt": "hello"}
    assert "files" not in captured and "data" not in captured


def _capture_post(monkeypatch):
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(200, json={"id": "abc", "polling_url": "https://x"})

    monkeypatch.setattr(client.httpx, "post", fake_post)
    return captured


def test_send_request_uses_x_api_key_for_comfyui_keys(monkeypatch):
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    captured = _capture_post(monkeypatch)
    client.send_request(ep, {"prompt": "x", "width": 1, "height": 1}, flags, api_key="comfyui-abc")
    assert captured["headers"]["X-API-Key"] == "comfyui-abc"
    assert "Authorization" not in captured["headers"]
    assert captured["headers"]["Comfy-Env"] == "comfy-cli"


def test_send_request_uses_bearer_for_firebase_tokens(monkeypatch):
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    captured = _capture_post(monkeypatch)
    client.send_request(ep, {"prompt": "x", "width": 1, "height": 1}, flags, api_key="eyJhbGciOi.foo.bar")
    assert captured["headers"]["Authorization"] == "Bearer eyJhbGciOi.foo.bar"
    assert "X-API-Key" not in captured["headers"]
    assert captured["url"].endswith("/proxy/bfl/flux-pro-1.1/generate")


def test_send_request_dispatches_non_get_post_methods(monkeypatch):
    ep = spec.Endpoint(
        id="example/update",
        path="/proxy/example/update",
        method="put",
        partner="example",
        summary="",
        category="test",
        request_schema={"type": "object"},
        request_content_type="application/json",
        response_schema={},
        polling=None,
    )
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200)

    monkeypatch.setattr(client.httpx, "request", fake_request)
    client.send_request(ep, {"prompt": "x"}, [], api_key="comfyui-test")

    assert captured["method"] == "PUT"
    assert captured["json"] == {"prompt": "x"}


def test_raise_for_status_includes_body():
    resp = httpx.Response(400, json={"error": "bad prompt"})
    with pytest.raises(client.ApiError) as exc:
        client.raise_for_status(resp)
    assert exc.value.status == 400
    assert "bad prompt" in exc.value.body
