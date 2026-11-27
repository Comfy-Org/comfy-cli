"""Tests for the Router client accessor — base-URL bridge and timeout pass-through.

A fake ``comfy_sdk`` stands in for the real one so these record what the
accessor hands ``Comfy()``; the real import is covered in
``tests/comfy_cli/test_tracking_lazy_import.py``.
"""

import os
import sys
import threading
import time
import types

import pytest

from comfy_cli.command.generate import router_client


class _FakeComfy:
    calls: list[dict] = []

    router_urls: list[str | None] = []

    def __init__(self, **kwargs):
        _FakeComfy.calls.append(kwargs)
        # The real SDK reads the Router host at construction, so record it here.
        _FakeComfy.router_urls.append(os.environ.get("COMFY_ROUTER_BASE_URL"))
        self.models = ("models", kwargs)


@pytest.fixture
def fake_sdk(monkeypatch):
    _FakeComfy.calls = []
    _FakeComfy.router_urls = []
    monkeypatch.setitem(
        sys.modules,
        "comfy_sdk",
        types.SimpleNamespace(Comfy=_FakeComfy, ROUTER_BASE_URL_ENV_VAR="COMFY_ROUTER_BASE_URL"),
    )
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
    assert fake_sdk.router_urls[-1] == "https://stagingapi.example"


def test_bridge_is_scoped_to_construction(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    router_client.router_models("comfyui-k")
    assert "COMFY_ROUTER_BASE_URL" not in os.environ
    # A later change to COMFY_API_BASE_URL is honoured, not shadowed by the first call.
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://other.example")
    router_client.router_models("comfyui-k")
    assert fake_sdk.router_urls[-1] == "https://other.example"


def test_bridged_url_is_normalised(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "  https://stagingapi.example/  ")
    router_client.router_models("comfyui-k")
    assert fake_sdk.router_urls[-1] == "https://stagingapi.example"


def test_blank_api_base_url_is_not_bridged(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "   ")
    router_client.router_models("comfyui-k")
    assert fake_sdk.router_urls[-1] is None


def test_explicit_router_base_url_wins(fake_sdk, monkeypatch):
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    monkeypatch.setenv("COMFY_ROUTER_BASE_URL", "https://router.example")
    router_client.router_models("comfyui-k")
    assert fake_sdk.router_urls[-1] == "https://router.example"
    assert os.environ["COMFY_ROUTER_BASE_URL"] == "https://router.example"


def test_no_base_url_leaves_the_router_var_unset(fake_sdk):
    router_client.router_models("comfyui-k")
    assert fake_sdk.router_urls[-1] is None
    assert "COMFY_ROUTER_BASE_URL" not in os.environ


def test_bridge_is_restored_when_construction_raises(fake_sdk, monkeypatch):
    def boom(**kwargs):
        raise ValueError("bad url")

    monkeypatch.setattr(sys.modules["comfy_sdk"], "Comfy", boom)
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    with pytest.raises(ValueError):
        router_client.router_models("comfyui-k")
    assert "COMFY_ROUTER_BASE_URL" not in os.environ


def test_real_sdk_reads_the_router_var_the_bridge_sets():
    """The bridge writes whatever variable the installed SDK names; pin that the
    SDK still reads it, so an SDK rename cannot silently route to production."""
    comfy_sdk = pytest.importorskip("comfy_sdk")
    assert comfy_sdk.ROUTER_BASE_URL_ENV_VAR == "COMFY_ROUTER_BASE_URL"


def test_overlapping_calls_each_construct_against_the_bridged_url(fake_sdk, monkeypatch):
    # Without serialisation, B sees A's override as a user-set Router var, skips its
    # own, and then constructs after A has restored the env — on the SDK default.
    a_in_construction = threading.Event()
    a_done = threading.Event()
    seen: dict[str, str | None] = {}

    def comfy(**kwargs):
        name = threading.current_thread().name
        if name == "a":
            a_in_construction.set()
            time.sleep(0.2)  # give B time to reach the bridge while A holds the override
        else:
            a_done.wait(timeout=5)
        seen[name] = os.environ.get("COMFY_ROUTER_BASE_URL")
        return types.SimpleNamespace(models="models")

    def run_a():
        router_client.router_models("comfyui-k")
        a_done.set()

    monkeypatch.setattr(sys.modules["comfy_sdk"], "Comfy", comfy)
    monkeypatch.setenv("COMFY_API_BASE_URL", "https://stagingapi.example")
    a = threading.Thread(target=run_a, name="a")
    b = threading.Thread(target=router_client.router_models, args=("comfyui-k",), name="b")
    a.start()
    assert a_in_construction.wait(timeout=5)
    b.start()
    a.join(timeout=10)
    b.join(timeout=10)
    assert seen == {"a": "https://stagingapi.example", "b": "https://stagingapi.example"}
    assert "COMFY_ROUTER_BASE_URL" not in os.environ
