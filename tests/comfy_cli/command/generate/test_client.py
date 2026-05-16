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
    with pytest.raises(client.ApiError, match="No API key"):
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


def test_send_request_sets_bearer_header(monkeypatch):
    ep = spec.get_endpoint("bfl/flux-pro-1.1/generate")
    flags = schema.flags_for(ep)
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(200, json={"id": "abc", "polling_url": "https://x"})

    monkeypatch.setattr(client.httpx, "post", fake_post)
    resp = client.send_request(
        ep, {"prompt": "x", "width": 1, "height": 1}, flags, api_key="sk-test"
    )
    assert resp.status_code == 200
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["X-Comfy-Env"] == "comfy-cli"
    assert captured["url"].endswith("/proxy/bfl/flux-pro-1.1/generate")
    assert captured["json"]["prompt"] == "x"


def test_raise_for_status_includes_body():
    resp = httpx.Response(400, json={"error": "bad prompt"})
    with pytest.raises(client.ApiError) as exc:
        client.raise_for_status(resp)
    assert exc.value.status == 400
    assert "bad prompt" in exc.value.body
