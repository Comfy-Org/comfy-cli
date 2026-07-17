"""Unit tests for the shared cloud error → envelope mapper (BE-3266)."""

from __future__ import annotations

import io
import urllib.error

import pytest
import typer

from comfy_cli.command._cloud_errors import _MAX_ERROR_BODY_BYTES, handle_cloud_http_error


class _FakeRenderer:
    """Captures the single ``renderer.error`` call the mapper is expected to make."""

    def __init__(self):
        self.calls: list[dict] = []

    def error(self, **kwargs):
        self.calls.append(kwargs)


def _http_error(code: int, body: bytes = b'{"error":"boom"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://cloud.example.com/x", code, "err", {}, io.BytesIO(body))


def _handle(renderer, e: Exception) -> typer.Exit:
    return handle_cloud_http_error(
        renderer,
        e,
        operation="cancel",
        not_found_code="prompt_not_found",
        not_found_message="no cloud job with id 'p1'",
        not_found_hint="check `comfy jobs ls --where cloud`",
        id_label="prompt_id",
        resource_id="p1",
    )


@pytest.mark.parametrize("code", [401, 403, 500])
def test_error_body_read_is_capped(code: int):
    """``base_url`` is env-configurable, so a hostile endpoint returning a huge
    error body must not be buffered unbounded into memory."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(code, b"A" * (5 * 1024 * 1024)))

    body = renderer.calls[0]["details"]["body"]
    assert len(body) <= _MAX_ERROR_BODY_BYTES
    assert body == "A" * _MAX_ERROR_BODY_BYTES


class _ExplodingBody(io.BytesIO):
    def read(self, *args):
        raise ConnectionResetError("stream reset mid-read")


@pytest.mark.parametrize("code", [401, 403, 500])
def test_body_read_failure_still_emits_envelope(code: int):
    """A reset/truncated body stream must degrade to an empty body, not escape
    as an unhandled traceback in place of the structured envelope."""
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", code, "err", {}, _ExplodingBody(b"ignored"))

    exit_exc = _handle(renderer, e)

    assert isinstance(exit_exc, typer.Exit)
    assert len(renderer.calls) == 1
    assert renderer.calls[0]["details"]["body"] == ""


def test_404_does_not_consume_body():
    """The 404 envelope discards the body, so it should not read the stream."""
    renderer = _FakeRenderer()
    e = urllib.error.HTTPError("https://x/y", 404, "err", {}, _ExplodingBody(b"ignored"))

    _handle(renderer, e)

    call = renderer.calls[0]
    assert call["code"] == "prompt_not_found"
    assert "body" not in call["details"]


@pytest.mark.parametrize("code", [401, 403])
def test_unauthorized_details_carry_body_and_resource_id(code: int):
    """A 403 for a non-auth reason (forbidden resource, quota, finished job)
    must not lose the server's explanation or the id being operated on."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(code, b'{"error":"quota exceeded"}'))

    call = renderer.calls[0]
    assert call["code"] == "cloud_unauthorized"
    assert call["details"]["status"] == code
    assert "quota exceeded" in call["details"]["body"]
    assert call["details"]["prompt_id"] == "p1"
    assert call["details"]["operation"] == "cancel"


def test_401_hint_points_at_relogin():
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(401))
    assert renderer.calls[0]["hint"] == "re-run `comfy cloud login`"


def test_403_hint_is_softened():
    """403 is not unambiguously an auth failure, so the hint must not assert
    re-login as the remediation the way 401's does."""
    renderer = _FakeRenderer()
    _handle(renderer, _http_error(403))

    hint = renderer.calls[0]["hint"]
    assert "details.body" in hint
    assert hint != "re-run `comfy cloud login`"


def test_url_error_surfaces_network_hint():
    renderer = _FakeRenderer()
    _handle(renderer, urllib.error.URLError("connection refused"))

    call = renderer.calls[0]
    assert call["code"] == "cloud_http_error"
    assert "comfy auth whoami" in call["hint"]
