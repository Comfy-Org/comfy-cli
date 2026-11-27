"""Tests for the Router client accessor — base-URL bridge and timeout pass-through.

A fake ``comfy_sdk`` stands in for the real one so these record what the
accessor hands ``Comfy()``; the real import is covered in
``tests/comfy_cli/test_tracking_lazy_import.py``.
"""

import sys
import types

import pytest

from comfy_cli.command.generate import router_client


class _FakeComfy:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        _FakeComfy.calls.append(kwargs)
        self.models = ("models", kwargs)


@pytest.fixture
def fake_sdk(monkeypatch):
    _FakeComfy.calls = []
    monkeypatch.setitem(sys.modules, "comfy_sdk", types.SimpleNamespace(Comfy=_FakeComfy))
    # setenv first so teardown restores the pre-test state even after
    # router_models writes the var itself.
    monkeypatch.setenv("COMFY_ROUTER_BASE_URL", "")
    monkeypatch.delenv("COMFY_ROUTER_BASE_URL")
    monkeypatch.delenv("COMFY_API_BASE_URL", raising=False)
    return _FakeComfy


def test_returns_the_clients_models_api(fake_sdk):
    assert router_client.router_models("comfyui-k") == ("models", {"api_key": "comfyui-k"})


def test_omitted_timeout_keeps_the_sdk_default(fake_sdk):
    router_client.router_models("comfyui-k")
    assert "timeout" not in fake_sdk.calls[-1]


def test_explicit_timeout_is_passed_through(fake_sdk):
    router_client.router_models("comfyui-k", timeout=5.0)
    assert fake_sdk.calls[-1]["timeout"] == 5.0


def test_api_base_url_is_bridged_onto_the_router_var(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    router_client.router_models("comfyui-k")
    assert router_client.os.environ["COMFY_ROUTER_BASE_URL"] == "https://stagingapi.example"


def test_explicit_router_base_url_wins(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    monkeypatch.setenv("COMFY_ROUTER_BASE_URL", "https://router.example")
    router_client.router_models("comfyui-k")
    assert router_client.os.environ["COMFY_ROUTER_BASE_URL"] == "https://router.example"


def test_no_base_url_leaves_the_router_var_unset(fake_sdk):
    router_client.router_models("comfyui-k")
    assert "COMFY_ROUTER_BASE_URL" not in router_client.os.environ
