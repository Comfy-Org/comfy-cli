"""Cache-first TTL tests for ``resilient_load_object_info``.

The loader serves a per-host cache entry younger than the TTL (default 10
minutes, ``COMFY_OBJECT_INFO_TTL`` seconds to override, ``0`` = always fetch)
without any network call. These tests pin that policy:

  - a fresh cache hit never touches the network,
  - an expired entry refetches live,
  - TTL=0 bypasses the gate entirely,
  - entries are keyed per target base URL (cloud vs local never collide),
  - a fetch failure still falls back to the stale cache even past the TTL.

``engine._load_from_target`` is always mocked — no sockets are opened.
"""

from __future__ import annotations

import os

import pytest

from comfy_cli.cql import loader
from comfy_cli.cql.engine import LoadError

CACHED = {"CachedNode": {"input": {"required": {}}, "output": [], "category": "cached"}}
LIVE = {"LiveNode": {"input": {"required": {}}, "output": [], "category": "live"}}

CLOUD_KEY = "https://cloud.test.comfy.org"
LOCAL_KEY = "http://127.0.0.1:8188"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Throwaway cache dir; no TTL override leaking in from the dev env."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv(loader.OBJECT_INFO_TTL_ENV, raising=False)


def _pin_host_key(monkeypatch, key: str) -> None:
    monkeypatch.setattr(loader, "_resolve_host_key", lambda mode, host, port: key)


def _expire_cache(host_key: str, age_seconds: float) -> None:
    """Backdate the cache file's mtime so the entry reads as ``age_seconds`` old."""
    path = loader.object_info_cache_path(host_key)
    stamp = path.stat().st_mtime - age_seconds
    os.utime(path, (stamp, stamp))


def _forbid_network(monkeypatch):
    """Fail loudly if the live fetch runs; return the call counter."""
    import comfy_cli.cql.engine as engine

    calls = {"n": 0}

    def _boom(**kw):
        calls["n"] += 1
        raise AssertionError("network fetch must not run on a fresh cache hit")

    monkeypatch.setattr(engine, "_load_from_target", _boom)
    return calls


# ---------------------------------------------------------------------------
# fresh cache hit → no network call
# ---------------------------------------------------------------------------


def test_fresh_cache_hit_skips_network(monkeypatch):
    _pin_host_key(monkeypatch, CLOUD_KEY)
    calls = _forbid_network(monkeypatch)
    loader.write_object_info_cache(CLOUD_KEY, CACHED)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)

    assert result == CACHED
    assert calls["n"] == 0


def test_fresh_hit_respects_custom_ttl(monkeypatch):
    """An entry older than the default TTL is still fresh under a larger one."""
    _pin_host_key(monkeypatch, CLOUD_KEY)
    calls = _forbid_network(monkeypatch)
    loader.write_object_info_cache(CLOUD_KEY, CACHED)
    _expire_cache(CLOUD_KEY, age_seconds=3600)  # 1h old — past the 10m default
    monkeypatch.setenv(loader.OBJECT_INFO_TTL_ENV, "7200")

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)

    assert result == CACHED
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# expired entry → live refetch (and cache rewrite)
# ---------------------------------------------------------------------------


def test_expired_ttl_refetches(monkeypatch):
    import comfy_cli.cql.engine as engine

    _pin_host_key(monkeypatch, CLOUD_KEY)
    loader.write_object_info_cache(CLOUD_KEY, CACHED)
    _expire_cache(CLOUD_KEY, age_seconds=loader.DEFAULT_OBJECT_INFO_TTL_SECONDS + 1)

    calls = {"n": 0}

    def _live(**kw):
        calls["n"] += 1
        return LIVE

    monkeypatch.setattr(engine, "_load_from_target", _live)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)

    assert result == LIVE
    assert calls["n"] == 1
    # The refetch rewrote the cache with the live payload.
    assert loader.read_object_info_cache(CLOUD_KEY) == LIVE


# ---------------------------------------------------------------------------
# TTL=0 → always fetch live, even with a brand-new cache entry
# ---------------------------------------------------------------------------


def test_ttl_zero_bypasses_cache(monkeypatch):
    import comfy_cli.cql.engine as engine

    _pin_host_key(monkeypatch, CLOUD_KEY)
    loader.write_object_info_cache(CLOUD_KEY, CACHED)  # fresh, would hit
    monkeypatch.setenv(loader.OBJECT_INFO_TTL_ENV, "0")

    calls = {"n": 0}

    def _live(**kw):
        calls["n"] += 1
        return LIVE

    monkeypatch.setattr(engine, "_load_from_target", _live)

    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1)

    assert result == LIVE
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# per-target keying: a fresh cloud entry must not satisfy a local lookup
# ---------------------------------------------------------------------------


def test_fresh_entry_for_other_target_does_not_hit(monkeypatch):
    import comfy_cli.cql.engine as engine

    loader.write_object_info_cache(CLOUD_KEY, CACHED)  # fresh, but for cloud
    _pin_host_key(monkeypatch, LOCAL_KEY)  # this call targets local

    calls = {"n": 0}

    def _live(**kw):
        calls["n"] += 1
        return LIVE

    monkeypatch.setattr(engine, "_load_from_target", _live)

    result = loader.resilient_load_object_info(mode="local", host="127.0.0.1", port=8188)

    assert result == LIVE  # served live, never the cloud entry
    assert calls["n"] == 1
    # Each target keeps its own entry.
    assert loader.read_object_info_cache(CLOUD_KEY) == CACHED
    assert loader.read_object_info_cache(LOCAL_KEY) == LIVE


def test_local_never_serves_fresh_cache(monkeypatch):
    """Cache-first TTL is cloud-only: a fresh LOCAL entry is NOT served, so a
    node just installed into the user's own server is visible immediately. The
    live fetch still runs (and rewrites the cache for the failure fallback)."""
    import comfy_cli.cql.engine as engine

    _pin_host_key(monkeypatch, LOCAL_KEY)
    loader.write_object_info_cache(LOCAL_KEY, CACHED)  # fresh local entry

    calls = {"n": 0}

    def _live(**kw):
        calls["n"] += 1
        return LIVE

    monkeypatch.setattr(engine, "_load_from_target", _live)

    result = loader.resilient_load_object_info(mode="local", host="127.0.0.1", port=8188)

    assert result == LIVE  # live, not the fresh cache
    assert calls["n"] == 1
    assert loader.read_object_info_cache(LOCAL_KEY) == LIVE  # cache rewritten for failure fallback


# ---------------------------------------------------------------------------
# expired entry + fetch failure → stale fallback still works
# ---------------------------------------------------------------------------


def test_expired_entry_still_serves_as_stale_fallback(monkeypatch):
    import comfy_cli.cloud.oauth as oauth
    import comfy_cli.cql.engine as engine

    _pin_host_key(monkeypatch, CLOUD_KEY)
    monkeypatch.setattr(oauth, "ensure_fresh_session", lambda **kw: None)
    loader.write_object_info_cache(CLOUD_KEY, CACHED)
    _expire_cache(CLOUD_KEY, age_seconds=loader.DEFAULT_OBJECT_INFO_TTL_SECONDS + 1)

    def _offline(**kw):
        raise LoadError("cannot reach the server: offline")

    monkeypatch.setattr(engine, "_load_from_target", _offline)

    warnings: list[str] = []
    result = loader.resilient_load_object_info(mode="cloud", host="h", port=1, _warn=warnings.append)

    assert result == CACHED
    assert len(warnings) == 1
    assert "stale" in warnings[0].lower()


# ---------------------------------------------------------------------------
# TTL env parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, loader.DEFAULT_OBJECT_INFO_TTL_SECONDS),  # unset → default
        ("", loader.DEFAULT_OBJECT_INFO_TTL_SECONDS),  # blank → default
        ("   ", loader.DEFAULT_OBJECT_INFO_TTL_SECONDS),  # whitespace → default
        ("garbage", loader.DEFAULT_OBJECT_INFO_TTL_SECONDS),  # unparseable → default
        ("0", 0.0),
        ("-5", 0.0),  # negative clamps to bypass
        ("30", 30.0),
        ("1.5", 1.5),
    ],
)
def test_object_info_cache_ttl_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(loader.OBJECT_INFO_TTL_ENV, raising=False)
    else:
        monkeypatch.setenv(loader.OBJECT_INFO_TTL_ENV, raw)
    assert loader.object_info_cache_ttl() == expected


def test_read_fresh_missing_file_returns_none():
    assert loader.read_fresh_object_info_cache("https://nope.example", ttl=600) is None


def test_read_fresh_future_mtime_treated_as_expired(monkeypatch):
    """Clock skew: an mtime in the future must not count as fresh."""
    loader.write_object_info_cache(CLOUD_KEY, CACHED)
    _expire_cache(CLOUD_KEY, age_seconds=-3600)  # 1h in the future
    assert loader.read_fresh_object_info_cache(CLOUD_KEY, ttl=600) is None
