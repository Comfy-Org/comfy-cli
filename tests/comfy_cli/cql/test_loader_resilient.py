"""Resilience tests for ``resilient_load_object_info``.

Covers the four behaviours A3 promises:

  (a) successful fetch populates the per-host cache,
  (b) a failed fetch triggers ``ensure_fresh_session`` + a single retry that
      then succeeds (and caches),
  (c) a persistently-failing fetch falls back to the cached dump with a warning,
  (d) a persistently-failing fetch with no cache re-raises the original error.

The network fetch (``engine._load_from_target``) and the session refresh
(``oauth.ensure_fresh_session``) are always mocked — no sockets are opened.
"""

from __future__ import annotations

import json

import pytest

from comfy_cli.cql import loader
from comfy_cli.cql.engine import LoadError

OBJECT_INFO = {
    "KSampler": {
        "input": {"required": {"seed": ["INT", {"default": 0}]}},
        "output": ["LATENT"],
        "category": "sampling",
    }
}

STALE_OBJECT_INFO = {
    "OldNode": {
        "input": {"required": {}},
        "output": [],
        "category": "legacy",
    }
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the cache dir at a throwaway tmp dir for every test."""
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    # Make the host-key resolution deterministic and I/O-free.
    monkeypatch.setattr(loader, "_resolve_host_key", lambda mode, host, port: "https://test.comfy.org")
    return cache_root


def _fake_refresh(monkeypatch):
    """Install a no-op ``ensure_fresh_session`` and count its calls."""
    calls = {"n": 0}

    def _refresh(**_kw):
        calls["n"] += 1
        return None

    import comfy_cli.cloud.oauth as oauth

    monkeypatch.setattr(oauth, "ensure_fresh_session", _refresh)
    return calls


# ---------------------------------------------------------------------------
# (a) success → cache populated
# ---------------------------------------------------------------------------


def test_success_populates_cache(monkeypatch):
    import comfy_cli.cql.engine as engine

    monkeypatch.setattr(engine, "_load_from_target", lambda **kw: OBJECT_INFO)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)
    assert result == OBJECT_INFO

    # Cache file written for this host key.
    path = loader.object_info_cache_path("https://test.comfy.org")
    assert path.is_file()
    assert json.loads(path.read_text()) == OBJECT_INFO


# ---------------------------------------------------------------------------
# (b) 401 → refresh + retry → success
# ---------------------------------------------------------------------------


def test_failure_then_refresh_retry_succeeds(monkeypatch):
    import comfy_cli.cql.engine as engine

    refresh = _fake_refresh(monkeypatch)
    attempts = {"n": 0}

    def _flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LoadError("HTTP 401 from /object_info", details={"status": 401})
        return OBJECT_INFO

    monkeypatch.setattr(engine, "_load_from_target", _flaky)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)

    assert result == OBJECT_INFO
    assert attempts["n"] == 2  # original + exactly one retry
    assert refresh["n"] == 1  # refresh attempted between the two
    # Retry success also caches.
    assert loader.object_info_cache_path("https://test.comfy.org").is_file()


def test_401_retry_forces_refresh(monkeypatch):
    """The reactive retry must FORCE the refresh (force=True): a server 401 is
    authoritative, so the token must be spent even if the local expiry check
    still thinks the access token is valid."""
    import comfy_cli.cloud.oauth as oauth
    import comfy_cli.cql.engine as engine

    seen_kwargs: list[dict] = []

    def _refresh(**kw):
        seen_kwargs.append(kw)
        return None

    monkeypatch.setattr(oauth, "ensure_fresh_session", _refresh)

    attempts = {"n": 0}

    def _flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LoadError("HTTP 401 from /object_info", details={"status": 401})
        return OBJECT_INFO

    monkeypatch.setattr(engine, "_load_from_target", _flaky)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)
    assert result == OBJECT_INFO
    # Forced, exactly once. The loader is a foreground command, so it keeps the
    # default allow_clear=True (a genuine token-endpoint invalid_grant should
    # still clear the dead session).
    assert seen_kwargs == [{"force": True, "allow_clear": True}]


def test_persistent_401_no_cache_retries_once_then_raises(monkeypatch):
    """Graceful failure: a refresh token that can't rescue the session must
    surface the original error exactly once — no infinite refresh/retry loop."""
    import comfy_cli.cloud.oauth as oauth
    import comfy_cli.cql.engine as engine

    refresh = {"n": 0}

    def _refresh(**kw):
        refresh["n"] += 1
        return None

    monkeypatch.setattr(oauth, "ensure_fresh_session", _refresh)

    attempts = {"n": 0}
    original = LoadError("HTTP 401 from /object_info", details={"status": 401, "hint": "run `comfy cloud login`"})

    def _always_401(**kw):
        attempts["n"] += 1
        raise original

    monkeypatch.setattr(engine, "_load_from_target", _always_401)

    with pytest.raises(LoadError) as excinfo:
        loader.resilient_load_object_info(mode="cloud", host="h", port=1, _warn=lambda m: None)

    assert excinfo.value is original  # the cloud_unauthorized-style hint survives
    assert attempts["n"] == 2  # original + exactly one retry — no loop
    assert refresh["n"] == 1  # refreshed exactly once


# ---------------------------------------------------------------------------
# (c) persistent failure + cache present → stale fallback with warning
# ---------------------------------------------------------------------------


def test_persistent_failure_falls_back_to_cache_with_warning(monkeypatch):
    import comfy_cli.cql.engine as engine

    _fake_refresh(monkeypatch)

    # Seed the cache with a (stale) dump.
    loader.write_object_info_cache("https://test.comfy.org", STALE_OBJECT_INFO)

    def _always_fail(**kw):
        raise LoadError("HTTP 401 from /object_info", details={"status": 401})

    monkeypatch.setattr(engine, "_load_from_target", _always_fail)

    warnings: list[str] = []
    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1, _warn=warnings.append)

    assert result == STALE_OBJECT_INFO
    assert len(warnings) == 1
    assert "cached" in warnings[0].lower()
    assert "stale" in warnings[0].lower()


def test_connection_error_also_falls_back_to_cache(monkeypatch):
    import comfy_cli.cql.engine as engine

    _fake_refresh(monkeypatch)
    loader.write_object_info_cache("https://test.comfy.org", STALE_OBJECT_INFO)

    def _offline(**kw):
        raise LoadError("cannot reach https://test.comfy.org/object_info: offline")

    monkeypatch.setattr(engine, "_load_from_target", _offline)

    warnings: list[str] = []
    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1, _warn=warnings.append)
    assert result == STALE_OBJECT_INFO
    assert warnings


# ---------------------------------------------------------------------------
# (d) persistent failure + no cache → original error re-raised
# ---------------------------------------------------------------------------


def test_persistent_failure_no_cache_raises_original(monkeypatch):
    import comfy_cli.cql.engine as engine

    _fake_refresh(monkeypatch)

    original = LoadError("HTTP 401 from /object_info", details={"status": 401, "hint": "run `comfy cloud login`"})

    def _always_fail(**kw):
        raise original

    monkeypatch.setattr(engine, "_load_from_target", _always_fail)

    warnings: list[str] = []
    with pytest.raises(LoadError) as excinfo:
        loader.resilient_load_object_info(mode="cloud", host="h", port=1, _warn=warnings.append)

    # The *original* error (with its hint) is what propagates.
    assert excinfo.value is original
    assert excinfo.value.details.get("hint") == "run `comfy cloud login`"
    assert warnings == []  # no stale-cache warning when there's nothing to fall back to


# ---------------------------------------------------------------------------
# explicit --input always wins and is never cached
# ---------------------------------------------------------------------------


def test_input_path_wins_and_is_not_cached(monkeypatch, tmp_path):
    import comfy_cli.cql.engine as engine

    # If the live fetch were consulted it would blow up — assert it isn't.
    def _boom(**kw):
        raise AssertionError("network fetch must not run when --input is given")

    monkeypatch.setattr(engine, "_load_from_target", _boom)

    dump = tmp_path / "object_info.json"
    dump.write_text(json.dumps(OBJECT_INFO), encoding="utf-8")

    result = loader.resilient_load_object_info(mode="cloud", input_path=str(dump))
    assert result == OBJECT_INFO
    # Explicit input is not auto-cached.
    assert not loader.object_info_cache_path("https://test.comfy.org").exists()


# ---------------------------------------------------------------------------
# cache helpers: corrupt / missing handled gracefully
# ---------------------------------------------------------------------------


def test_read_cache_missing_returns_none():
    assert loader.read_object_info_cache("https://nope.example") is None


def test_read_cache_corrupt_returns_none():
    path = loader.object_info_cache_path("https://test.comfy.org")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert loader.read_object_info_cache("https://test.comfy.org") is None


def test_per_host_cache_keys_do_not_collide(monkeypatch):
    a = loader.object_info_cache_path("https://a.comfy.org")
    b = loader.object_info_cache_path("http://127.0.0.1:8188")
    assert a != b


# ---------------------------------------------------------------------------
# (e) stale-cache fallback fires the on_stale callback
# ---------------------------------------------------------------------------


def test_resilient_load_reports_stale_via_callback(monkeypatch):
    from comfy_cli.cql import loader
    from comfy_cli.cql.engine import LoadError

    def boom(**kw):
        raise LoadError("server down")

    monkeypatch.setattr("comfy_cli.cql.engine._load_from_target", boom)
    monkeypatch.setattr(loader, "read_object_info_cache", lambda key: {"NodeA": {}})
    # Suppress the stderr warning during the test.
    monkeypatch.setattr(loader, "_default_warn", lambda msg: None)

    stale = {}
    data = loader.resilient_load_object_info(mode="cloud", on_stale=lambda key, err: stale.update(host=key, error=err))
    assert data == {"NodeA": {}}
    assert stale.get("host")  # callback fired with the host key
