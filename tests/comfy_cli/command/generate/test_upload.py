"""Tests for /customers/storage upload helpers."""

import httpx
import pytest

from comfy_cli.command.generate import client, upload


def test_request_signed_url_posts_hash(monkeypatch):
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(
            200,
            json={
                "upload_url": "https://signed/up",
                "download_url": "https://signed/down",
                "expires_at": "2099-01-01T00:00:00Z",
                "existing_file": False,
            },
        )

    monkeypatch.setattr(upload.httpx, "post", fake_post)
    body = upload._request_signed_url("cat.png", "image/png", "deadbeef", "comfyui-test")
    assert body["upload_url"] == "https://signed/up"
    assert captured["json"] == {"file_name": "cat.png", "content_type": "image/png", "file_hash": "deadbeef"}
    assert captured["headers"]["X-API-Key"] == "comfyui-test"
    assert captured["headers"]["X-Comfy-Env"] == "comfy-cli"
    assert captured["url"].endswith("/customers/storage")


def test_upload_bytes_dedupe_skips_put(monkeypatch):
    monkeypatch.setattr(
        upload,
        "_request_signed_url",
        lambda **kw: {
            "download_url": "https://cached/x.png",
            "existing_file": True,
            "expires_at": "2099-01-01T00:00:00Z",
        },
    )
    called = {"put": False}

    def boom(*a, **kw):
        called["put"] = True

    monkeypatch.setattr(upload, "_put_bytes", boom)
    result = upload.upload_bytes(b"hello", "x.png", "comfyui-test")
    assert result.url == "https://cached/x.png"
    assert result.existing_file is True
    assert called["put"] is False


def test_upload_bytes_new_file_puts(monkeypatch):
    monkeypatch.setattr(
        upload,
        "_request_signed_url",
        lambda **kw: {
            "upload_url": "https://signed/up",
            "download_url": "https://signed/down",
            "existing_file": False,
            "expires_at": None,
        },
    )
    captured = {}

    def fake_put(upload_url, data, content_type):
        captured["upload_url"] = upload_url
        captured["data"] = data
        captured["content_type"] = content_type

    monkeypatch.setattr(upload, "_put_bytes", fake_put)
    result = upload.upload_bytes(b"raw-bytes", "cat.png", "comfyui-test")
    assert result.url == "https://signed/down"
    assert result.existing_file is False
    assert captured["data"] == b"raw-bytes"
    assert captured["content_type"] == "image/png"


def test_upload_path_reads_file(monkeypatch, tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"jpeg-bytes")
    called = {}

    def fake_upload_bytes(data, file_name, api_key, content_type=None):
        called["data"] = data
        called["file_name"] = file_name
        called["content_type"] = content_type
        return upload.UploadResult(url="https://x/a", expires_at=None, existing_file=False)

    monkeypatch.setattr(upload, "upload_bytes", fake_upload_bytes)
    upload.upload_path(img, "comfyui-test")
    assert called["data"] == b"jpeg-bytes"
    assert called["file_name"] == "x.jpg"


def test_upload_path_missing_file(tmp_path):
    with pytest.raises(client.ApiError, match="not found"):
        upload.upload_path(tmp_path / "nope.png", "comfyui-test")


def test_upload_remote_url_rehosts(monkeypatch):
    monkeypatch.setattr(
        upload,
        "upload_bytes",
        lambda data, file_name, api_key, content_type=None: upload.UploadResult(
            url=f"https://rehosted/{file_name}", expires_at=None, existing_file=False
        ),
    )

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url):
            return httpx.Response(
                200,
                content=b"png-bytes",
                headers={"content-type": "image/png"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(upload.httpx, "Client", FakeClient)
    result = upload.upload_remote_url("https://example.com/photo.png", "comfyui-test")
    assert result.url == "https://rehosted/photo.png"


def test_upload_target_dispatches_on_scheme(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(upload, "upload_path", lambda p, api_key: called.setdefault("path", p))
    monkeypatch.setattr(upload, "upload_remote_url", lambda u, api_key: called.setdefault("url", u))
    upload.upload_target("https://example.com/x.png", "k")
    upload.upload_target("/tmp/x.png", "k")
    assert called["url"] == "https://example.com/x.png"
    assert called["path"] == "/tmp/x.png"


def test_put_bytes_raises_on_error(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def put(self, *a, **kw):
            return httpx.Response(500, text="boom", request=httpx.Request("PUT", "https://x"))

    monkeypatch.setattr(upload.httpx, "Client", FakeClient)
    with pytest.raises(client.ApiError, match="HTTP 500"):
        upload._put_bytes("https://x", b"data", "image/png")
