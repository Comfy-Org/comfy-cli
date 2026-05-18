"""Tests for the per-endpoint adapters: Gemini (nano-banana) and Seedance."""

from __future__ import annotations

import base64

import httpx
import pytest

from comfy_cli.command.generate import adapters, client, output, schema, spec

# ── Gemini / nano-banana ─────────────────────────────────────────────────


def test_nano_banana_alias_resolves():
    ep = spec.get_endpoint("nano-banana")
    assert ep.id == "vertexai/gemini/{model}"
    assert ep.polling is None
    assert ep.partner == "vertexai"


def test_gemini_adapter_overrides_schema_flags():
    """Schema-derived flags would be `contents`/`tools`/…; the adapter
    swaps in a friendlier `prompt`/`image`/`model` triple."""
    ep = spec.get_endpoint("nano-banana")
    names = [f.name for f in schema.flags_for(ep)]
    assert names == ["prompt", "image", "model"]


def test_gemini_build_body_text_only():
    body = adapters._gemini_build_body({"prompt": "a fox"}, api_key="k")
    assert body["contents"][0]["role"] == "user"
    parts = body["contents"][0]["parts"]
    assert parts == [{"text": "a fox"}]
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]


def test_gemini_build_body_inlines_local_image(tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n-bytes-")
    body = adapters._gemini_build_body(
        {"prompt": "add hat", "image": [str(img)]},
        api_key="k",
    )
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "add hat"}
    inline = parts[1]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert base64.b64decode(inline["data"]) == b"\x89PNG\r\n\x1a\n-bytes-"


def test_gemini_build_body_inlines_remote_url(monkeypatch):
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
                content=b"jpeg-bytes",
                headers={"content-type": "image/jpeg"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(adapters.httpx, "Client", FakeClient)
    body = adapters._gemini_build_body(
        {"prompt": "x", "image": ["https://example.com/a.jpg"]},
        api_key="k",
    )
    inline = body["contents"][0]["parts"][1]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"
    assert base64.b64decode(inline["data"]) == b"jpeg-bytes"


def test_gemini_build_body_inlines_data_uri():
    blob = base64.b64encode(b"png-bytes").decode("ascii")
    body = adapters._gemini_build_body(
        {"prompt": "x", "image": f"data:image/png;base64,{blob}"},
        api_key="k",
    )
    inline = body["contents"][0]["parts"][1]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert inline["data"] == blob


def test_gemini_inline_image_missing_path_raises(tmp_path):
    with pytest.raises(client.ApiError, match="not found"):
        adapters._inline_image(str(tmp_path / "nope.png"))


def test_gemini_decode_sync_saves_inline_blobs(tmp_path):
    blob = base64.b64encode(b"png-payload").decode("ascii")
    body = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": blob}}]}}]}
    out = tmp_path / "out.png"
    saved = adapters._gemini_decode_sync(body, str(out), "req-1")
    assert saved == [out]
    assert out.read_bytes() == b"png-payload"


def test_gemini_decode_sync_handles_snake_case_keys(tmp_path):
    """Gemini responses are sometimes serialized as inline_data/mime_type.
    With a directory-shorthand template, the mime drives the extension."""
    blob = base64.b64encode(b"webp-payload").decode("ascii")
    body = {"candidates": [{"content": {"parts": [{"inline_data": {"mime_type": "image/webp", "data": blob}}]}}]}
    saved = adapters._gemini_decode_sync(body, str(tmp_path) + "/", "r")
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"webp-payload"
    assert saved[0].suffix == ".webp"


def test_gemini_decode_sync_returns_empty_when_blocked(tmp_path):
    body = {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}
    saved = adapters._gemini_decode_sync(body, str(tmp_path / "x.png"), "r")
    assert saved == []


def test_gemini_resolve_path_substitutes_model():
    ep = spec.get_endpoint("nano-banana")
    adapter = adapters.get(ep.id)
    url = adapters.resolve_path(ep.path, {"model": "gemini-2.5-flash-image"}, adapter)
    assert url == "/proxy/vertexai/gemini/gemini-2.5-flash-image"


def test_gemini_send_request_hits_substituted_path(monkeypatch):
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_kw):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"candidates": []})

    monkeypatch.setattr(client.httpx, "post", fake_post)
    ep = spec.get_endpoint("nano-banana")
    flags = schema.flags_for(ep)
    client.send_request(
        ep,
        {"prompt": "hi", "model": "gemini-2.5-flash-image"},
        flags,
        api_key="comfyui-test",
    )
    assert captured["url"].endswith("/proxy/vertexai/gemini/gemini-2.5-flash-image")
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "hi"


# ── Seedance ──────────────────────────────────────────────────────────────


def test_seedance_alias_resolves():
    ep = spec.get_endpoint("seedance")
    assert ep.id == "byteplus/api/v3/contents/generations/tasks"
    assert ep.polling == "seedance"
    assert ep.partner == "byteplus"


def test_seedance_adapter_overrides_flags():
    ep = spec.get_endpoint("seedance")
    names = [f.name for f in schema.flags_for(ep)]
    assert "prompt" in names
    assert "model" in names
    assert "resolution" in names
    assert "ratio" in names
    assert "duration" in names


def test_seedance_build_body_text_only():
    body = adapters._seedance_build_body({"prompt": "a wave"}, api_key="k")
    assert body["model"] == "seedance-1-0-pro-250528"
    assert body["content"] == [{"type": "text", "text": "a wave"}]


def test_seedance_build_body_inlines_knobs_into_text():
    body = adapters._seedance_build_body(
        {
            "prompt": "a boat",
            "resolution": "720p",
            "ratio": "16:9",
            "duration": 5,
            "fps": 24,
            "camerafixed": True,
        },
        api_key="k",
    )
    text = body["content"][0]["text"]
    assert text.startswith("a boat ")
    assert "--resolution 720p" in text
    assert "--ratio 16:9" in text
    assert "--duration 5" in text
    assert "--fps 24" in text
    assert "--camerafixed true" in text


def test_seedance_build_body_uploads_local_image(monkeypatch, tmp_path):
    """Local paths get pushed through /customers/storage and replaced with the
    returned signed URL — we shouldn't see the path appear in the body."""
    img = tmp_path / "ref.png"
    img.write_bytes(b"ref")

    from comfy_cli.command.generate import upload

    def fake_upload_path(path, api_key):
        return upload.UploadResult(url="https://cdn/signed-ref.png", expires_at=None, existing_file=False)

    monkeypatch.setattr(upload, "upload_path", fake_upload_path)
    body = adapters._seedance_build_body(
        {"prompt": "wave", "image": str(img)},
        api_key="comfyui-test",
    )
    image_part = body["content"][1]
    assert image_part == {"type": "image_url", "image_url": {"url": "https://cdn/signed-ref.png"}}


def test_seedance_build_body_keeps_remote_url_verbatim(monkeypatch):
    """Remote URLs and data: URIs are pass-through — no re-upload."""
    from comfy_cli.command.generate import upload

    def boom(*a, **kw):
        raise AssertionError("upload should not be called for remote URLs")

    monkeypatch.setattr(upload, "upload_path", boom)
    body = adapters._seedance_build_body(
        {"prompt": "x", "image": "https://example.com/a.jpg"},
        api_key="k",
    )
    assert body["content"][1]["image_url"]["url"] == "https://example.com/a.jpg"


def test_seedance_build_body_includes_audio_flag():
    body = adapters._seedance_build_body(
        {"prompt": "x", "model": "seedance-1-5-pro-251215", "generate_audio": True},
        api_key="k",
    )
    assert body["generate_audio"] is True
    assert body["model"] == "seedance-1-5-pro-251215"


def test_seedance_send_request_passes_through_body(monkeypatch):
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None, **_kw):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"id": "task-1"})

    monkeypatch.setattr(client.httpx, "post", fake_post)
    ep = spec.get_endpoint("seedance")
    flags = schema.flags_for(ep)
    client.send_request(ep, {"prompt": "x"}, flags, api_key="comfyui-test")
    assert captured["url"].endswith("/proxy/byteplus/api/v3/contents/generations/tasks")
    assert captured["json"]["model"]
    assert captured["json"]["content"][0]["type"] == "text"


# ── Seedance polling ──────────────────────────────────────────────────────


def test_seedance_poll_url_and_success_extraction(monkeypatch):
    """Driver should hit the task-status endpoint and pluck the video_url."""
    from comfy_cli.command.generate import poll

    monkeypatch.setattr(poll, "_sleep", lambda *_: None)
    captured = {}

    def fake_get(url, **_kw):
        captured["url"] = url
        return httpx.Response(
            200,
            json={"id": "t1", "status": "succeeded", "content": {"video_url": "https://cdn/v.mp4"}},
        )

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    result = poll.get_poller("seedance")({"id": "t1"}, api_key="k")
    assert captured["url"] == "/proxy/byteplus/api/v3/contents/generations/tasks/t1"
    assert result.status == "succeeded"
    assert "https://cdn/v.mp4" in result.image_urls


def test_seedance_poll_failure(monkeypatch):
    from comfy_cli.command.generate import poll

    monkeypatch.setattr(poll, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        lambda *a, **kw: httpx.Response(200, json={"id": "t1", "status": "failed", "error": {"code": "x"}}),
    )
    result = poll.get_poller("seedance")({"id": "t1"}, api_key="k")
    assert result.status == "failed"
    assert "failed" in (result.error or "")


def test_seedance_resume_helper_round_trip():
    """`comfy generate resume` reverses the create-response into something the
    poller can read — make sure that helper knows about seedance."""
    from comfy_cli.command.generate import poll

    body = poll.build_synthetic_initial("seedance", "t-42")
    assert poll.extract_job_id("seedance", body) == "t-42"


# ── Inline blob saving ────────────────────────────────────────────────────


def test_save_inline_blobs_auto_indexes_multi(tmp_path):
    blobs = [("image/png", b"a"), ("image/png", b"b")]
    saved = output.save_inline_blobs(blobs, str(tmp_path / "out.png"), "req")
    assert len(saved) == 2
    assert saved[0].name == "out_0.png"
    assert saved[1].name == "out_1.png"
    assert saved[0].read_bytes() == b"a"
    assert saved[1].read_bytes() == b"b"


def test_save_inline_blobs_picks_extension_from_mime(tmp_path):
    saved = output.save_inline_blobs([("image/webp", b"x")], str(tmp_path) + "/", "req")
    assert saved[0].suffix == ".webp"
