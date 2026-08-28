"""Tests for the comfy-knowledge bundle loader and the ``comfy knowledge`` verbs.

Contract under test:
  * load order: ``COMFY_KNOWLEDGE_FILE`` (authoritative, never cached) >
    fresh cache > fetch (``COMFY_KNOWLEDGE_URL``, else the cloud default) >
    stale cache > nothing.
  * a broken or missing bundle returns ``None``; nothing here raises.
  * manifest ``schema_version`` and ``sha256`` reject a mismatched bundle.
  * the index is a tolerant reader: unknown keys/fields are ignored.
  * the three verbs emit one ``envelope/1`` line whose ``data`` validates
    against ``comfy_cli/schemas/knowledge.json``.

Network is never touched: an autouse fixture replaces ``knowledge._http_get``
with a guard that fails the test if reached.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli import knowledge
from comfy_cli.caller import Caller
from comfy_cli.command import knowledge as knowledge_cmd
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"
FIXTURE_KNOWLEDGE = FIXTURES / "knowledge.json"
FIXTURE_MANIFEST = FIXTURES / "manifest.json"
SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "comfy_cli" / "schemas" / "knowledge.json").read_text())

_REAL_HTTP_GET = knowledge._http_get


def _network_guard(url: str) -> bytes:
    raise AssertionError(f"network touched: {url}")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for var in (knowledge.ENV_FILE, knowledge.ENV_URL, knowledge.ENV_TTL):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(knowledge, "_http_get", _network_guard)
    knowledge._reset_for_testing()
    yield
    knowledge._reset_for_testing()


def _env_bundle(tmp_path: Path, monkeypatch, *, manifest: dict | None | str = "fixture") -> Path:
    """Copy the fixture to a tmp dir and point COMFY_KNOWLEDGE_FILE at it.

    ``manifest``: ``"fixture"`` copies the real one, ``None`` writes none,
    a dict writes that dict as the sibling manifest.
    """
    d = tmp_path / "env"
    d.mkdir()
    shutil.copy(FIXTURE_KNOWLEDGE, d / "knowledge.json")
    if manifest == "fixture":
        shutil.copy(FIXTURE_MANIFEST, d / "manifest.json")
    elif manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv(knowledge.ENV_FILE, str(d / "knowledge.json"))
    return d / "knowledge.json"


def _seed_cache(age_seconds: float = 0.0) -> tuple[Path, Path]:
    k_path, m_path = knowledge.cache_paths()
    k_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_KNOWLEDGE, k_path)
    shutil.copy(FIXTURE_MANIFEST, m_path)
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(k_path, (stamp, stamp))
    return k_path, m_path


def _fake_http(monkeypatch, *, knowledge_bytes: bytes | None = None, manifest_bytes: bytes | None = None):
    """Install a fake ``_http_get``; records calls. ``None`` for either body means raise."""
    calls: list[str] = []

    def fake(url: str) -> bytes:
        calls.append(url)
        body = manifest_bytes if url.endswith("manifest.json") else knowledge_bytes
        if body is None:
            raise RuntimeError(f"fake fetch failure: {url}")
        return body

    monkeypatch.setattr(knowledge, "_http_get", fake)
    return calls


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class TestLoadOrder:
    def test_env_file_loads_and_never_fetches(self, tmp_path, monkeypatch):
        path = _env_bundle(tmp_path, monkeypatch)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        b = knowledge.load_bundle()
        assert b is not None
        assert b.source == "env"
        assert b.stale is False
        assert b.path == str(path)
        assert b.version == "0.1.0-fixture"
        assert knowledge.last_reason() is None
        assert not knowledge.cache_paths()[0].exists()

    def test_env_file_missing_returns_none_without_fallthrough(self, tmp_path, monkeypatch):
        _seed_cache()
        monkeypatch.setenv(knowledge.ENV_FILE, str(tmp_path / "nope.json"))
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_ENV_FILE

    def test_env_file_rejected_by_manifest_schema_version(self, tmp_path, monkeypatch):
        _env_bundle(tmp_path, monkeypatch, manifest={"schema_version": 2, "version": "9"})
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_ENV_FILE

    def test_env_file_rejected_by_manifest_sha(self, tmp_path, monkeypatch):
        bad = {"schema_version": 1, "version": "x", "files": {"knowledge.json": {"sha256": "00" * 32}}}
        _env_bundle(tmp_path, monkeypatch, manifest=bad)
        assert knowledge.load_bundle() is None

    def test_env_file_without_manifest_loads_as_unknown_version(self, tmp_path, monkeypatch):
        _env_bundle(tmp_path, monkeypatch, manifest=None)
        b = knowledge.load_bundle()
        assert b is not None
        assert b.version == "unknown"

    def test_manifest_without_sha_is_tolerated(self, tmp_path, monkeypatch):
        _env_bundle(tmp_path, monkeypatch, manifest={"schema_version": 1, "version": "1.2.3", "files": {}})
        b = knowledge.load_bundle()
        assert b is not None
        assert b.version == "1.2.3"

    def test_unparseable_manifest_is_treated_as_absent(self, tmp_path, monkeypatch):
        path = _env_bundle(tmp_path, monkeypatch, manifest=None)
        (path.parent / "manifest.json").write_text("not json {")
        b = knowledge.load_bundle()
        assert b is not None
        assert b.version == "unknown"

    def test_bundle_with_a_non_finite_constant_is_rejected(self, tmp_path, monkeypatch):
        p = tmp_path / "k.json"
        p.write_text('{"models": {"a": {"id": "a", "score": NaN}}}')
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        assert knowledge.load_bundle() is None
        knowledge._reset_for_testing()
        p.write_text('{"models": {"a": {"id": "a", "score": 1e400}}}')
        assert knowledge.load_bundle() is None

    def test_env_file_not_a_bundle_returns_none(self, tmp_path, monkeypatch):
        p = tmp_path / "k.json"
        p.write_text(json.dumps({"hello": "world"}))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        assert knowledge.load_bundle() is None
        p.write_text("[1, 2]")
        knowledge._reset_for_testing()
        assert knowledge.load_bundle() is None

    def test_fresh_cache_serves_without_fetch(self, monkeypatch):
        _seed_cache()
        assert knowledge.load_bundle().source == "cache"
        knowledge._reset_for_testing()
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        b = knowledge.load_bundle()
        assert b.source == "cache"
        assert b.stale is False

    def test_expired_cache_fetches_and_rewrites_cache(self, monkeypatch):
        k_path, m_path = _seed_cache(age_seconds=2 * 24 * 3600)
        m_path.unlink()
        manifest_bytes = FIXTURE_MANIFEST.read_bytes()
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=manifest_bytes)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/k/knowledge.json")
        b = knowledge.load_bundle()
        assert b.source == "fetch"
        assert b.stale is False
        assert calls == ["https://example.com/k/knowledge.json", "https://example.com/k/manifest.json"]
        assert time.time() - k_path.stat().st_mtime < 60
        assert m_path.read_bytes() == manifest_bytes

    def test_fetch_without_manifest_drops_stale_cached_manifest(self, monkeypatch):
        _k_path, m_path = _seed_cache(age_seconds=2 * 24 * 3600)
        _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        b = knowledge.load_bundle()
        assert b.source == "fetch"
        assert b.version == "unknown"
        assert not m_path.exists()

    def test_expired_cache_with_failed_fetch_serves_stale(self, monkeypatch):
        _seed_cache(age_seconds=2 * 24 * 3600)
        _fake_http(monkeypatch, knowledge_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        b = knowledge.load_bundle()
        assert b.source == "stale-cache"
        assert b.stale is True

    def test_fetch_rejecting_validation_falls_back_to_stale(self, monkeypatch):
        _seed_cache(age_seconds=2 * 24 * 3600)
        _fake_http(monkeypatch, knowledge_bytes=b'{"models": []}', manifest_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle().source == "stale-cache"

    def test_no_cache_and_no_url_fetches_the_default_and_reports_signed_out(self, monkeypatch):
        calls = _fake_http(monkeypatch, knowledge_bytes=None)
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_SIGNED_OUT
        assert calls == [knowledge.default_url()]

    def test_no_cache_failed_fetch_is_none(self, monkeypatch):
        _fake_http(monkeypatch, knowledge_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_FETCH_FAILED

    def test_default_url_follows_the_cloud_base_url(self, monkeypatch):
        monkeypatch.setenv("COMFY_CLOUD_BASE_URL", "https://staging.example/")
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        assert knowledge.load_bundle().source == "fetch"
        assert calls == [
            "https://staging.example/api/knowledge/knowledge.json",
            "https://staging.example/api/knowledge/manifest.json",
        ]

    def test_explicit_url_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("COMFY_CLOUD_BASE_URL", "https://staging.example")
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/k/knowledge.json")
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        assert knowledge.load_bundle().source == "fetch"
        assert calls == ["https://example.com/k/knowledge.json", "https://example.com/k/manifest.json"]

    def test_env_file_skips_the_default_fetch(self, tmp_path, monkeypatch):
        _env_bundle(tmp_path, monkeypatch)
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes())
        assert knowledge.load_bundle().source == "env"
        assert calls == []

    def test_unsafe_url_is_a_fetch_failure(self, monkeypatch):
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes())
        monkeypatch.setenv(knowledge.ENV_URL, "http://example.com/knowledge.json")
        assert knowledge.load_bundle() is None
        assert calls == []

    def test_ttl_zero_always_fetches(self, monkeypatch):
        _seed_cache()
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        monkeypatch.setenv(knowledge.ENV_TTL, "0")
        assert knowledge.load_bundle().source == "fetch"
        assert len(calls) == 2

    def test_ttl_env_parsing(self, monkeypatch):
        assert knowledge.ttl_seconds() == knowledge.DEFAULT_TTL_SECONDS
        monkeypatch.setenv(knowledge.ENV_TTL, "  ")
        assert knowledge.ttl_seconds() == knowledge.DEFAULT_TTL_SECONDS
        monkeypatch.setenv(knowledge.ENV_TTL, "abc")
        assert knowledge.ttl_seconds() == knowledge.DEFAULT_TTL_SECONDS
        monkeypatch.setenv(knowledge.ENV_TTL, "-5")
        assert knowledge.ttl_seconds() == 0.0
        monkeypatch.setenv(knowledge.ENV_TTL, "90")
        assert knowledge.ttl_seconds() == 90.0
        for raw in ("nan", "inf", "-inf", "1e400"):
            monkeypatch.setenv(knowledge.ENV_TTL, raw)
            assert knowledge.ttl_seconds() == knowledge.DEFAULT_TTL_SECONDS

    def test_force_fetch_without_a_fetch_keeps_fresh_cache_unstale(self, monkeypatch):
        _seed_cache()
        b = knowledge.load_bundle(force_fetch=True)
        assert b.source == "cache"
        assert b.stale is False
        knowledge._reset_for_testing()
        _fake_http(monkeypatch, knowledge_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        b = knowledge.load_bundle(force_fetch=True)
        assert b.source == "cache"
        assert b.stale is False

    def test_cache_survives_failed_manifest_write(self, monkeypatch):
        _k_path, m_path = _seed_cache(age_seconds=2 * 24 * 3600)
        new_raw = FIXTURE_KNOWLEDGE.read_bytes() + b"\n"
        manifest = json.loads(FIXTURE_MANIFEST.read_text())
        manifest["files"]["knowledge.json"]["sha256"] = hashlib.sha256(new_raw).hexdigest()
        _fake_http(monkeypatch, knowledge_bytes=new_raw, manifest_bytes=json.dumps(manifest).encode())
        real_write = knowledge.atomic_write_bytes

        def flaky_write(path, data, **kw):
            if path.name == "manifest.json":
                raise OSError("disk full")
            real_write(path, data, **kw)

        monkeypatch.setattr(knowledge, "atomic_write_bytes", flaky_write)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle().source == "fetch"
        assert not m_path.exists()
        knowledge._reset_for_testing()
        monkeypatch.delenv(knowledge.ENV_URL)
        b = knowledge.load_bundle()
        assert b is not None
        assert b.source == "cache"
        assert b.version == "unknown"

    def test_memo_and_force_fetch(self, monkeypatch):
        _seed_cache()
        first = knowledge.load_bundle()
        assert knowledge.load_bundle() is first
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle() is first
        assert calls == []
        forced = knowledge.load_bundle(force_fetch=True)
        assert forced is not first
        assert forced.source == "fetch"
        assert knowledge.load_bundle() is forced


class TestAuthRouting:
    @staticmethod
    def _fake_resp(body: bytes, final_url: str = "https://example.com/knowledge.json"):
        class _Resp:
            status = 200
            url = final_url

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n: int = -1) -> bytes:
                return body

        return _Resp()

    def test_cloud_url_uses_authed_opener(self, monkeypatch):
        import comfy_cli.target
        from comfy_cli.cloud import get_base_url

        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        # The real resolve_target can refresh an OAuth token; keep credential code out of the test.
        monkeypatch.setattr(comfy_cli.target, "resolve_target", lambda **kw: SimpleNamespace(kind="cloud", **kw))
        seen: dict[str, Any] = {}

        def fake_authed(url, target, **kw):
            seen["url"] = url
            seen["target_kind"] = target.kind
            # A failed token refresh during this background fetch must not wipe the stored session.
            seen["allow_clear"] = target.allow_clear
            return self._fake_resp(b'{"models": {}}')

        monkeypatch.setattr(knowledge, "authed_urlopen", fake_authed)
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda *a, **kw: pytest.fail("plain opener used"))
        url = f"{get_base_url()}/api/knowledge/knowledge.json"
        assert knowledge._http_get(url) == b'{"models": {}}'
        assert seen == {"url": url, "target_kind": "cloud", "allow_clear": False}

    def test_signed_out_401_on_the_default_url_degrades_quietly(self, monkeypatch):
        import comfy_cli.target

        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        monkeypatch.setattr(comfy_cli.target, "resolve_target", lambda **kw: SimpleNamespace(kind="cloud", **kw))
        seen: list[str] = []

        def unauthorized(url, target, **kw):
            seen.append(url)
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(knowledge, "authed_urlopen", unauthorized)
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda *a, **kw: pytest.fail("plain opener used"))
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_SIGNED_OUT
        assert seen == [knowledge.default_url()]

    def test_lookalike_host_does_not_get_credentials(self, monkeypatch):
        from comfy_cli.cloud import get_base_url

        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        monkeypatch.setattr(knowledge, "authed_urlopen", lambda *a, **kw: pytest.fail("authed opener used"))
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda req, **kw: self._fake_resp(b"{}"))
        assert knowledge._http_get(get_base_url() + ".evil.example/knowledge.json") == b"{}"

    def test_other_url_uses_plain_opener(self, monkeypatch):
        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        seen: dict[str, Any] = {}

        def fake_plain(req, **kw):
            seen["url"] = req.full_url
            seen["ua"] = req.get_header("User-agent")
            return self._fake_resp(b'{"models": {}}')

        monkeypatch.setattr(knowledge, "plain_urlopen", fake_plain)
        monkeypatch.setattr(knowledge, "authed_urlopen", lambda *a, **kw: pytest.fail("authed opener used"))
        assert knowledge._http_get("https://example.com/knowledge.json") == b'{"models": {}}'
        assert seen == {"url": "https://example.com/knowledge.json", "ua": "comfy-cli"}

    def test_non_200_and_oversize_raise(self, monkeypatch):
        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        resp = self._fake_resp(b"{}")
        resp.status = 404
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda req, **kw: resp)
        with pytest.raises(RuntimeError):
            knowledge._http_get("https://example.com/knowledge.json")
        monkeypatch.setattr(knowledge, "MAX_BUNDLE_BYTES", 4)
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda req, **kw: self._fake_resp(b"x" * 10))
        with pytest.raises(ResponseTooLarge):
            knowledge._http_get("https://example.com/knowledge.json")

    def test_plain_fetch_rejects_redirect_to_http(self, monkeypatch):
        monkeypatch.setattr(knowledge, "_http_get", _REAL_HTTP_GET)
        resp = self._fake_resp(b"{}", final_url="http://mirror.example/knowledge.json")
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda req, **kw: resp)
        with pytest.raises(ValueError):
            knowledge._http_get("https://example.com/knowledge.json")


class TestRefreshIfStale:
    """The warm-path refresh. `attach` never calls it: a discovery turn must not
    wait on the network, which left the TTL decorative on a local install."""

    def test_stale_cache_is_refetched(self, tmp_path, monkeypatch):
        _seed_cache(age_seconds=10 * 24 * 60 * 60)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        calls = _fake_http(
            monkeypatch,
            knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(),
            manifest_bytes=FIXTURE_MANIFEST.read_bytes(),
        )
        knowledge.refresh_if_stale()
        assert "https://example.com/knowledge.json" in calls

    def test_fresh_cache_is_left_alone(self, monkeypatch):
        _seed_cache()
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes())
        knowledge.refresh_if_stale()
        assert calls == []

    def test_env_file_is_authoritative_and_never_refetched(self, tmp_path, monkeypatch):
        _env_bundle(tmp_path, monkeypatch)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes())
        knowledge.refresh_if_stale()
        assert calls == []

    def test_a_failed_refresh_is_silent(self, monkeypatch, capsys):
        _seed_cache(age_seconds=10 * 24 * 60 * 60)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        _fake_http(monkeypatch)  # every body None: the fetch raises
        knowledge.refresh_if_stale()
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""


class TestIndex:
    @pytest.fixture
    def bundle(self, tmp_path, monkeypatch) -> knowledge.Bundle:
        _env_bundle(tmp_path, monkeypatch)
        b = knowledge.load_bundle()
        assert b is not None
        return b

    def test_tolerant_reader(self, bundle, tmp_path, monkeypatch):
        assert "future_field" in bundle.models["testvid"]
        assert len(bundle.models) == 5
        knowledge._reset_for_testing()
        p = tmp_path / "min.json"
        p.write_text(json.dumps({"models": {"a": {"id": "a"}}, "aliases": {"A1": "a"}}))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        b = knowledge.load_bundle()
        assert b.capabilities == {}
        assert b.deprecations == {}
        assert b.aliases == {"a1": "a", "a": "a"}

    def test_malformed_rows_are_skipped(self, tmp_path, monkeypatch):
        data = {
            "models": {
                "good": {"id": "good", "aliases": ["G", 7], "resolves": "nope", "routing": [{"use": 3}]},
                "bad": "not a row",
            },
            "aliases": {"x": 1, "G2": "good"},
            "capabilities": {"c": {"picks": "nope"}, "d": 5},
            "deprecations": [{"id": "good"}, "junk", {"no_id": 1}],
        }
        p = tmp_path / "k.json"
        p.write_text(json.dumps(data))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        b = knowledge.load_bundle()
        assert set(b.models) == {"good"}
        assert b.aliases == {"g2": "good", "good": "good", "g": "good"}
        assert set(b.capabilities) == {"c"}
        assert set(b.deprecations) == {"good"}
        assert b.templates == {} and b.nodes == {}

    def test_alias_index(self, bundle):
        assert bundle.aliases["halo 03"] == "acme-h3"
        assert bundle.aliases["acme h3"] == "acme-h3"
        assert bundle.aliases["acme-h3"] == "acme-h3"
        assert bundle.aliases["testvid"] == "testvid"
        assert bundle.aliases["h3"] == "acme-h3"

    def test_template_and_node_index(self, bundle):
        assert bundle.templates["video_acme_h3_t2v"] == ["acme-h3"]
        assert bundle.templates["api_lipco_lip_sync_video"] == ["lipco-3"]
        # Capability pick template for a model outside the trimmed set still maps.
        assert bundle.templates["video_testlx_ia2v"] == ["testlx"]
        assert bundle.nodes["TestvidLipSyncAudioToVideoNode"] == ["testvid"]
        assert all(len(v) == len(set(v)) for v in bundle.templates.values())

    def test_deprecations_keyed_by_id(self, bundle):
        assert bundle.deprecations["testvid-avatar-2"]["superseded_by"] == "lipco-3"
        assert set(bundle.deprecations) == {"testvid-avatar-2", "retired-model-2"}

    def test_resolve(self, bundle):
        assert knowledge.resolve(bundle, "  HALO 03 ")["id"] == "acme-h3"
        assert knowledge.resolve(bundle, "Testvid-Avatar-2")["id"] == "testvid-avatar-2"
        assert knowledge.resolve(bundle, "nope-xyz") is None

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Halo 3", "acme-h3"),
            ("halo3", "acme-h3"),
            ("HALO-03", "acme-h3"),
            ("Ac Me H3", "acme-h3"),
            ("lipco.3", "lipco-3"),
            ("testvidg", None),
        ],
    )
    def test_resolve_accepts_spelling_variants(self, bundle, query, expected):
        row = knowledge.resolve(bundle, query)
        assert (row["id"] if row else None) == expected
        assert knowledge.resolve_id(bundle, query) == expected

    def test_two_ids_sharing_a_normalized_key_cancel(self):
        """`model-1` and `model01` both normalize to `model1`. Letting the last one
        win would answer a spelling neither id owns; the exact spellings still work."""
        data = {"models": {"model-1": {"id": "model-1"}, "model01": {"id": "model01"}}}
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        assert knowledge.resolve_id(b, "model 1") is None
        assert knowledge.resolve_id(b, "model-1") == "model-1"
        assert knowledge.resolve_id(b, "model01") == "model01"

    def test_capability_id_beats_a_colliding_alias(self):
        """One capability's alias spelled like another capability's id used to
        delete both normalized keys, so `pick` stopped resolving either."""
        data = {
            "models": {},
            "capabilities": {
                "text-to-video": {"id": "text-to-video", "picks": []},
                "lipsync": {"id": "lipsync", "aliases": ["Text to Video"], "picks": []},
            },
        }
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        for spelling in ("text-to-video", "text to video", "Text To Video"):
            assert knowledge.pick(b, spelling)["id"] == "text-to-video", spelling
        assert knowledge.pick(b, "lipsync")["id"] == "lipsync"

    def test_manifest_version_is_bounded(self):
        b = knowledge._index(
            {"models": {}},
            {"schema_version": 1, "version": "v" * 5000},
            source="env",
            stale=False,
            path="p",
            mtime=0.0,
        )
        assert len(b.version) == knowledge.MAX_VERSION_CHARS

    def test_model_id_beats_colliding_alias(self):
        data = {
            "models": {"testvid": {"id": "testvid"}, "testvid-3": {"id": "testvid-3"}},
            "aliases": {"Testvid": "testvid-3"},
        }
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        assert b.aliases["testvid"] == "testvid"
        assert knowledge.resolve(b, "TESTVID")["id"] == "testvid"

    def test_resolve_id_requires_a_row(self):
        data = {"models": {"a": {"id": "a"}}, "aliases": {"ghost": "missing"}}
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        assert knowledge.resolve_id(b, "ghost") is None
        assert knowledge.resolve(b, "ghost") is None
        assert knowledge.resolve_id(b, " A ") == "a"

    def test_normalize(self):
        assert knowledge._normalize("Halo 03") == "halo3"
        assert knowledge._normalize("image to video") == "imagetovideo"
        assert knowledge._normalize("Flux 1.10") == "flux110"
        assert knowledge._normalize("v2.0") == "v20"

    def test_normalized_collision_falls_back_to_exact_only(self):
        data = {
            "models": {"a": {"id": "a", "aliases": ["Model 01"]}, "b": {"id": "b", "aliases": ["model-1"]}},
        }
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        assert "model1" not in b.normalized_aliases
        assert knowledge.resolve(b, "model 1") is None
        assert knowledge.resolve(b, "Model 01")["id"] == "a"
        assert knowledge.resolve(b, "model-1")["id"] == "b"
        assert knowledge.resolve(b, "A")["id"] == "a"

    def test_pick(self, bundle):
        cap = knowledge.pick(bundle, " LIPSYNC ")
        assert cap is not None
        assert [p["rank"] for p in cap["picks"]] == [1, 2, 3, 4, 5, 6, 7]
        assert cap is not bundle.capabilities["lipsync"]
        assert knowledge.pick(bundle, "nope") is None
        assert knowledge.pick(bundle, "Lip Sync")["id"] == "lipsync"
        assert knowledge.pick(bundle, "audio generation")["id"] == "audio-generation"

    def test_pick_keeps_the_resolved_id_when_the_row_omits_it(self, bundle):
        # The reader is tolerant, so a row need not repeat its own key. The id
        # the caller reports must still be what spelling and wording resolved to,
        # never the raw phrase.
        bundle.capabilities["lipsync"].pop("id", None)
        assert knowledge.pick(bundle, "Lip Sync")["id"] == "lipsync"
        assert knowledge.pick(bundle, "make me a talking head clip")["id"] == "lipsync"

    def test_pick_sorts_missing_rank_last(self):
        b = knowledge._index(
            {
                "models": {},
                "capabilities": {
                    "c": {"picks": [{"model": "x"}, {"model": "y", "rank": 2}, {"model": "z", "rank": 1}]}
                },
            },
            None,
            source="env",
            stale=False,
            path="p",
            mtime=0.0,
        )
        assert [p["model"] for p in knowledge.pick(b, "c")["picks"]] == ["z", "y", "x"]
        assert b.as_of == "1970-01-01T00:00:00Z"

    def test_pick_attaches_the_model_fits(self):
        fits = {"vram_gb": {"fp8": 12, "bf16": 24}, "credits_per_image": 0.5, "max_refs": 3, "source": "measured"}
        data = {
            "models": {"a": {"id": "a", "fits": fits}, "b": {"id": "b"}, "c": {"id": "c", "fits": "12 GB"}},
            "capabilities": {
                "cap": {
                    "picks": [
                        {"model": "a", "rank": 1, "caveat": "fits"},
                        {"model": "b", "rank": 2},
                        {"model": "c", "rank": 3},
                        {"model": "ghost", "rank": 4},
                    ]
                }
            },
        }
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        picks = knowledge.pick(b, "cap")["picks"]
        assert [p["model"] for p in picks] == ["a", "b", "c", "ghost"]
        assert picks[0] == {"model": "a", "rank": 1, "caveat": "fits", "fits": fits}
        # No fits on the row, a non-dict fits, and a model the bundle lacks all pass through untouched.
        assert picks[1:] == data["capabilities"]["cap"]["picks"][1:]
        assert all("fits" not in p for p in picks[1:])

    def test_pick_copies_fits_instead_of_touching_the_bundle(self):
        data = {
            "models": {"a": {"id": "a", "fits": {"vram_gb": {"fp8": 12}}}},
            "capabilities": {"cap": {"picks": [{"model": "a", "rank": 1}]}},
        }
        snapshot = copy.deepcopy(data)
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        top = knowledge.pick(b, "cap")["picks"][0]
        top["fits"]["vram_gb"]["fp8"] = 0
        top["fits"]["extra"] = True
        assert data == snapshot
        assert "fits" not in b.capabilities["cap"]["picks"][0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _run(args: list[str], capsys) -> tuple[int, dict[str, Any]]:
    _force_json_renderer()
    result = CliRunner().invoke(knowledge_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    # With standalone_mode=False click hands back a typer.Exit code as the return value.
    exit_code = result.return_value if isinstance(result.return_value, int) else result.exit_code
    for line in reversed(captured.strip().splitlines()):
        try:
            return exit_code, json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


def _validate(data: dict) -> None:
    jsonschema.Draft202012Validator(SCHEMA).validate(data)


class TestCli:
    def test_status_with_env_file(self, tmp_path, monkeypatch, capsys):
        path = _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["status"], capsys)
        assert rc == 0
        assert env["ok"] is True
        assert env["schema"] == "envelope/1"
        assert env["command"] == "knowledge status"
        data = env["data"]
        assert data["loaded"] is True
        assert data["source"] == "env"
        assert data["version"] == "0.1.0-fixture"
        assert data["schema_version"] == 1
        assert data["env_file"] == str(path)
        assert data["url"] == knowledge.default_url()
        assert data["ttl_seconds"] == knowledge.DEFAULT_TTL_SECONDS
        assert data["counts"] == {"models": 5, "capabilities": 2, "aliases": 23, "deprecations": 2}
        _validate(data)

    def test_status_with_nothing(self, capsys):
        rc, env = _run(["status"], capsys)
        assert rc == 0
        assert env["ok"] is True
        data = env["data"]
        assert data["loaded"] is False
        assert data["reason"] == knowledge.REASON_SIGNED_OUT
        assert data["url"] == knowledge.default_url()
        assert data["cache_path"].endswith(os.path.join("knowledge", "knowledge.json"))
        _validate(data)

    def test_status_redacts_url_credentials(self, monkeypatch, capsys):
        monkeypatch.setenv(knowledge.ENV_URL, "https://user:secret@cdn.example/k/knowledge.json?token=abc#frag")
        _rc, env = _run(["status"], capsys)
        assert env["data"]["url"] == "https://cdn.example/k/knowledge.json"
        _validate(env["data"])

    def test_status_refresh_forces_fetch(self, monkeypatch, capsys):
        _seed_cache()
        calls = _fake_http(monkeypatch, knowledge_bytes=FIXTURE_KNOWLEDGE.read_bytes(), manifest_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        _rc, env = _run(["status", "--refresh"], capsys)
        assert env["data"]["source"] == "fetch"
        assert calls
        _validate(env["data"])

    def test_resolve_hit(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["resolve", "Halo 03"], capsys)
        assert rc == 0
        assert env["command"] == "knowledge resolve"
        data = env["data"]
        assert data["query"] == "Halo 03"
        assert data["id"] == "acme-h3"
        assert data["model"]["id"] == "acme-h3"
        assert data["deprecation"] is None
        assert data["bundle_version"] == "0.1.0-fixture"
        assert data["stale"] is False
        _validate(data)

    def test_resolve_deprecated(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        _rc, env = _run(["resolve", "testvid-avatar-2"], capsys)
        data = env["data"]
        assert data["deprecation"]["superseded_by"] == "lipco-3"
        _validate(data)

    def test_resolve_miss(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["resolve", "nope-xyz"], capsys)
        assert rc == 1
        assert env["ok"] is False
        assert env["error"]["code"] == "knowledge_unknown_model"
        assert isinstance(env["error"]["details"]["close_matches"], list)
        assert env["error"]["details"]["query"] == "nope-xyz"

    def test_resolve_spelling_variant_is_a_hit(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["resolve", "Halo 3"], capsys)
        assert rc == 0
        assert env["data"]["id"] == "acme-h3"
        assert env["data"]["query"] == "Halo 3"
        _validate(env["data"])

    def test_resolve_close_matches(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        _rc, env = _run(["resolve", "testvidg"], capsys)
        assert env["error"]["code"] == "knowledge_unknown_model"
        assert "testvid" in env["error"]["details"]["close_matches"]

    def test_resolve_without_bundle(self, capsys):
        rc, env = _run(["resolve", "x"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"
        assert "comfy cloud login" in env["error"]["hint"]

    def test_resolve_after_explicit_url_failure_points_at_the_url(self, monkeypatch, capsys):
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        rc, env = _run(["resolve", "x"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"
        assert knowledge.REASON_FETCH_FAILED in env["error"]["message"]
        assert "COMFY_KNOWLEDGE_URL" in env["error"]["hint"]
        assert "comfy cloud login" not in env["error"]["hint"]

    def test_pick_hit(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "lipsync"], capsys)
        assert rc == 0
        assert env["command"] == "knowledge pick"
        data = env["data"]
        assert data["capability"] == "lipsync"
        ranks = [p["rank"] for p in data["picks"]]
        assert ranks == sorted(ranks) and len(ranks) == 7
        assert all("status" in p and "superseded_by" in p for p in data["picks"])
        by_model = {p["model"]: p for p in data["picks"]}
        assert by_model["testvid"]["status"] == "available"
        assert by_model["testlx"]["status"] is None
        assert by_model["testlx"]["template"] == "video_testlx_ia2v"
        assert by_model["testvid"]["template"] is None
        _validate(data)

    def test_pick_spelling_variant(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "Audio Generation"], capsys)
        assert rc == 0
        assert env["data"]["capability"] == "audio-generation"
        _validate(env["data"])

    def test_pick_miss_is_a_zero_hit_answer_not_an_error(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "nope"], capsys)
        assert rc == 0
        assert env["ok"] is True
        data = env["data"]
        assert data["zero_hit"] is True
        assert data["query"] == "nope"
        assert "nope" in data["nudge"]
        assert [c["id"] for c in data["capabilities"]] == ["audio-generation", "lipsync"]
        _validate(data)

    def test_pick_miss_clips_a_long_query(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "x" * 500], capsys)
        assert rc == 0
        assert len(env["data"]["query"]) == knowledge.MAX_QUERY_CHARS

    def test_pick_hit_reports_zero_hit_false(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "lipsync"], capsys)
        assert rc == 0
        assert env["data"]["zero_hit"] is False

    def test_pick_resolves_a_phrased_request(self, tmp_path, monkeypatch, capsys):
        # Rule 1 sends the user's own words here, so a sentence must reach the
        # capability its alias is worded inside.
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "make me a talking head clip"], capsys)
        assert rc == 0
        assert env["data"]["capability"] == "lipsync"
        assert env["data"]["zero_hit"] is False

    def test_pick_with_a_blank_capability_lists_them(self, tmp_path, monkeypatch, capsys):
        # A blank argument means "no capability given", not a curation gap.
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "   "], capsys)
        assert rc == 0
        assert env["data"]["zero_hit"] is False
        assert "query" not in env["data"]

    def test_pick_without_a_capability_lists_them(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick"], capsys)
        assert rc == 0
        assert env["command"] == "knowledge pick"
        data = env["data"]
        assert [c["id"] for c in data["capabilities"]] == ["audio-generation", "lipsync"]
        assert all("description" in c for c in data["capabilities"])
        assert data["zero_hit"] is False
        _validate(data)

    def test_pick_without_a_capability_needs_a_bundle(self, capsys):
        rc, env = _run(["pick"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"

    def test_pick_payload_normalizes_rank_and_model(self, tmp_path, monkeypatch, capsys):
        picks = [
            {"model": "x", "rank": "1"},
            {"model": 7, "rank": 2},
            {"model": "y", "rank": 1.5, "route": 7, "template": [], "caveat": {}},
        ]
        cap = {"id": "c", "description": ["not", "text"], "as_of": 7, "picks": picks}
        models = {"y": {"id": "y", "status": 1, "superseded_by": ["z"]}}
        p = tmp_path / "k.json"
        p.write_text(json.dumps({"models": models, "capabilities": {"c": cap}}))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        rc, env = _run(["pick", "c"], capsys)
        assert rc == 0
        got = [(q["model"], q["rank"]) for q in env["data"]["picks"]]
        assert got == [("y", 1.5), (None, 2), ("x", None)]
        assert env["data"]["description"] is None and env["data"]["as_of"] is None
        top = env["data"]["picks"][0]
        assert [top[k] for k in ("route", "template", "caveat", "status", "superseded_by")] == [None] * 5
        _validate(env["data"])

    def test_pick_payload_carries_fits_verbatim(self, tmp_path, monkeypatch, capsys):
        fits = {"vram_gb": {"fp8": 12}, "credits_per_sec": 0.02, "max_refs": 1, "source": "measured"}
        models = {"x": {"id": "x", "fits": fits}, "y": {"id": "y"}}
        cap = {"id": "c", "picks": [{"model": "x", "rank": 1}, {"model": "y", "rank": 2}]}
        p = tmp_path / "k.json"
        p.write_text(json.dumps({"models": models, "capabilities": {"c": cap}}))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        rc, env = _run(["pick", "c"], capsys)
        assert rc == 0
        x, y = env["data"]["picks"]
        assert x["fits"] == fits
        assert "fits" not in y
        _validate(env["data"])

    def test_resolve_pretty_tolerates_malformed_row_fields(self, tmp_path, monkeypatch, pretty_no_stdout):
        p = tmp_path / "k.json"
        p.write_text(json.dumps({"models": {"m": {"id": "m", "best_for": 5, "pitfalls": "ouch"}}}))
        monkeypatch.setenv(knowledge.ENV_FILE, str(p))
        result = CliRunner().invoke(knowledge_cmd.app, ["resolve", "m"], standalone_mode=False)
        assert result.exception is None, result.exception

    def test_pick_without_bundle(self, capsys):
        rc, env = _run(["pick", "lipsync"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"

    def test_pretty_mode_writes_nothing_to_stdout(self, tmp_path, monkeypatch, pretty_no_stdout):
        _env_bundle(tmp_path, monkeypatch)
        for args in (["status"], ["resolve", "testvid"], ["pick", "lipsync"]):
            result = CliRunner().invoke(knowledge_cmd.app, args, standalone_mode=False)
            assert result.exception is None, result.exception

    def test_error_codes_registered(self):
        from comfy_cli import error_codes

        for code in ("knowledge_unavailable", "knowledge_unknown_model"):
            assert error_codes.is_registered(code)

    def test_fixture_manifest_matches_fixture(self):
        manifest = json.loads(FIXTURE_MANIFEST.read_text())
        raw = FIXTURE_KNOWLEDGE.read_bytes()
        assert manifest["files"]["knowledge.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert manifest["files"]["knowledge.json"]["bytes"] == len(raw)


class TestVerbQueryLog:
    """SKILL.md rule 1 tells the agent a miss records the gap. The verbs are
    what it runs, so they have to feed the same log enrichment feeds."""

    @pytest.fixture
    def queries(self, monkeypatch):
        """Only the miss log. `track_command` fires its own event either way."""
        from comfy_cli import tracking

        seen: list[dict] = []
        monkeypatch.setattr(
            tracking,
            "track_event",
            lambda name, props=None, **kw: seen.append(props) if name == "knowledge_query" else None,
        )
        return seen

    def test_pick_logs_a_hit_as_a_namespaced_capability(self, tmp_path, monkeypatch, capsys, queries):
        _env_bundle(tmp_path, monkeypatch)
        _run(["pick", "lipsync"], capsys)
        assert len(queries) == 1
        assert queries[0]["command"] == "knowledge pick"
        assert queries[0]["queries"] == ["lipsync"]
        assert queries[0]["hit_ids"] == ["cap:lipsync"]
        assert queries[0]["zero_hit"] is False

    def test_pick_logs_a_phrase_under_the_capability_it_resolved_to(self, tmp_path, monkeypatch, capsys, queries):
        # The curation feed wants the phrase verbatim, filed under the id it
        # reached. Filing it under the phrase would make every wording its own row.
        _env_bundle(tmp_path, monkeypatch)
        _, env = _run(["pick", "make me a talking head clip"], capsys)
        assert env["data"]["capability"] == "lipsync"
        assert queries[0]["queries"] == ["make me a talking head clip"]
        assert queries[0]["hit_ids"] == ["cap:lipsync"]
        assert queries[0]["zero_hit"] is False

    def test_pick_logs_a_miss_with_the_clipped_query(self, tmp_path, monkeypatch, capsys, queries):
        _env_bundle(tmp_path, monkeypatch)
        _run(["pick", "z" * 500], capsys)
        assert queries[0]["hit_ids"] == []
        assert queries[0]["zero_hit"] is True
        assert queries[0]["queries"] == ["z" * knowledge.MAX_QUERY_CHARS]

    def test_resolve_logs_both_a_hit_and_a_miss(self, tmp_path, monkeypatch, capsys, queries):
        _env_bundle(tmp_path, monkeypatch)
        _run(["resolve", "testvid"], capsys)
        _run(["resolve", "zzzz"], capsys)
        assert len(queries) == 2
        assert queries[0]["command"] == "knowledge resolve"
        assert queries[0]["hit_ids"] == ["testvid"]
        assert queries[0]["zero_hit"] is False
        assert queries[1]["hit_ids"] == []
        assert queries[1]["zero_hit"] is True

    def test_the_bare_listing_asks_nothing_so_logs_nothing(self, tmp_path, monkeypatch, capsys, queries):
        _env_bundle(tmp_path, monkeypatch)
        _run(["pick"], capsys)
        assert queries == []
