"""Tests for node-annotation resolution (cache → fetch → bundled fallback)."""

from __future__ import annotations

import time

import pytest

from comfy_cli.cql import annotations_source as src


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir and default to network-off for safety."""
    monkeypatch.setattr(src, "_cache_dir", lambda: tmp_path / "comfy-complete")
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "1")
    yield


def test_network_disabled_env_parsing(monkeypatch):
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "1")
    assert src._network_disabled() is True
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")
    assert src._network_disabled() is False
    monkeypatch.delenv("COMFY_CLI_NO_REMOTE_REFRESH", raising=False)
    assert src._network_disabled() is False


def test_falls_back_to_bundled_when_offline():
    """Network disabled + empty cache → the package-bundled snapshot is used."""
    sup, dis = src.load_annotation_bytes()
    assert sup is not None and b"node_packs" in sup
    assert dis is not None and b"disable_nodes" in dis


def test_fresh_cache_wins_without_network(tmp_path, monkeypatch):
    cache_dir = tmp_path / "comfy-complete"
    cache_dir.mkdir(parents=True)
    (cache_dir / "supported_nodes.yaml").write_bytes(b"cached-sup")
    (cache_dir / "cloud_disable_config.yaml").write_bytes(b"cached-dis")

    # Network must never be called; assert that by making _fetch explode.
    monkeypatch.setattr(src, "_fetch", lambda name: pytest.fail("network used"))
    sup, dis = src.load_annotation_bytes()
    assert sup == b"cached-sup"
    assert dis == b"cached-dis"


def test_stale_cache_used_when_fetch_fails(tmp_path, monkeypatch):
    cache_dir = tmp_path / "comfy-complete"
    cache_dir.mkdir(parents=True)
    f = cache_dir / "supported_nodes.yaml"
    f.write_bytes(b"stale-sup")
    # Make it stale (older than the TTL).
    old = time.time() - src._CACHE_TTL_SECONDS - 100
    import os

    os.utime(f, (old, old))

    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")  # allow network path

    def boom(name):
        raise RuntimeError("network down")

    monkeypatch.setattr(src, "_fetch", boom)
    sup = src._resolve_one("supported_nodes.yaml", refresh=False)
    assert sup == b"stale-sup"  # stale cache beats nothing


def test_fetch_success_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")
    monkeypatch.setattr(src, "_fetch", lambda name: b"fresh-" + name.encode())

    sup, dis = src.load_annotation_bytes(refresh=True)
    assert sup == b"fresh-supported_nodes.yaml"
    assert dis == b"fresh-cloud_disable_config.yaml"
    # Written through to the cache.
    cache_dir = tmp_path / "comfy-complete"
    assert (cache_dir / "supported_nodes.yaml").read_bytes() == sup


def test_refresh_annotations_reports_bundled_when_disabled():
    results = src.refresh_annotations()
    assert {r["name"] for r in results} == set(src._FILES)
    assert all(r["source"] == "bundled" for r in results)
    assert all("disabled" in r["error"] for r in results)


def test_refresh_annotations_reports_remote_on_success(monkeypatch):
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")
    monkeypatch.setattr(src, "_fetch", lambda name: b"x" * 10)
    results = src.refresh_annotations()
    assert all(r["source"] == "remote" and r["bytes"] == 10 for r in results)
