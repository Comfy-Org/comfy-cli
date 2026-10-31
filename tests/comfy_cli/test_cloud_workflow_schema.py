"""Acceptance parity with the actual matching frontend saved-canvas consumer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = json.loads((Path(__file__).parent / "fixtures/cloud_workflow_consumer/corpus.json").read_text())


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["id"])
def test_saved_canvas_matches_frontend_consumer(case):
    from comfy_cli.cloud_workflow_schema import is_supported_canvas

    assert is_supported_canvas(case["input"]) is case["accepted"]


def test_url_validation_does_not_resolve_or_contact_the_host(monkeypatch):
    import socket

    from comfy_cli.cloud_workflow_schema import _uri

    def network_forbidden(*args, **kwargs):
        pytest.fail("URL syntax validation must not perform network I/O")

    monkeypatch.setattr(socket, "getaddrinfo", network_forbidden)
    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    assert _uri("https://never-resolve-this-name.invalid/path?q=1#fragment") is True


def test_configured_cloud_origin_is_not_an_implicit_url_base(monkeypatch):
    from comfy_cli.cloud_workflow_schema import _uri

    monkeypatch.setenv("COMFY_CLOUD_BASE_URL", "https://cloud.test")
    assert _uri("/relative/model.safetensors") is False
    assert _uri("//example.com/model.safetensors") is False


def test_missing_native_parser_cannot_verify_a_url(monkeypatch):
    import builtins

    from comfy_cli.cloud_workflow_schema import _uri

    original = builtins.__import__

    def without_parser(name, *args, **kwargs):
        if name == "ada_url":
            raise ImportError("simulated unavailable native parser")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_parser)
    assert _uri("https://example.com/model.safetensors") is False
