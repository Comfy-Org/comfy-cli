"""Tests for the comfy-knowledge bundle loader and the ``comfy knowledge`` verbs.

Contract under test:
  * load order: ``COMFY_KNOWLEDGE_FILE`` (authoritative, never cached) >
    fresh cache > ``COMFY_KNOWLEDGE_URL`` fetch > stale cache > nothing.
  * a broken or missing bundle returns ``None``; nothing here raises.
  * manifest ``schema_version`` and ``sha256`` reject a mismatched bundle.
  * the index is a tolerant reader: unknown keys/fields are ignored.
  * the three verbs emit one ``envelope/1`` line whose ``data`` validates
    against ``comfy_cli/schemas/knowledge.json``.

Network is never touched: an autouse fixture replaces ``knowledge._http_get``
with a guard that fails the test if reached.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli import knowledge
from comfy_cli.caller import Caller
from comfy_cli.command import knowledge as knowledge_cmd
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

    def test_no_cache_no_url_is_none(self):
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_NO_URL

    def test_no_cache_failed_fetch_is_none(self, monkeypatch):
        _fake_http(monkeypatch, knowledge_bytes=None)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.com/knowledge.json")
        assert knowledge.load_bundle() is None
        assert knowledge.last_reason() == knowledge.REASON_FETCH_FAILED

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
    def _fake_resp(body: bytes):
        class _Resp:
            status = 200

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
            return self._fake_resp(b'{"models": {}}')

        monkeypatch.setattr(knowledge, "authed_urlopen", fake_authed)
        monkeypatch.setattr(knowledge, "plain_urlopen", lambda *a, **kw: pytest.fail("plain opener used"))
        url = f"{get_base_url()}/api/knowledge/knowledge.json"
        assert knowledge._http_get(url) == b'{"models": {}}'
        assert seen == {"url": url, "target_kind": "cloud"}

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
        with pytest.raises(Exception):
            knowledge._http_get("https://example.com/knowledge.json")


class TestIndex:
    @pytest.fixture
    def bundle(self, tmp_path, monkeypatch) -> knowledge.Bundle:
        _env_bundle(tmp_path, monkeypatch)
        b = knowledge.load_bundle()
        assert b is not None
        return b

    def test_tolerant_reader(self, bundle, tmp_path, monkeypatch):
        assert "future_field" in bundle.models["kling"]
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
        assert bundle.aliases["hailuo 03"] == "minimax-h3"
        assert bundle.aliases["minimax h3"] == "minimax-h3"
        assert bundle.aliases["minimax-h3"] == "minimax-h3"
        assert bundle.aliases["kling"] == "kling"
        assert bundle.aliases["h3"] == "minimax-h3"

    def test_template_and_node_index(self, bundle):
        assert bundle.templates["video_minimax_h3_t2v"] == ["minimax-h3"]
        assert bundle.templates["api_sync_so_lip_sync_video"] == ["sync-3"]
        # Capability pick template for a model outside the trimmed set still maps.
        assert bundle.templates["video_ltx2_3_ia2v"] == ["ltx"]
        assert bundle.nodes["KlingLipSyncAudioToVideoNode"] == ["kling"]
        assert all(len(v) == len(set(v)) for v in bundle.templates.values())

    def test_deprecations_keyed_by_id(self, bundle):
        assert bundle.deprecations["kling-avatar-2"]["superseded_by"] == "sync-3"
        assert set(bundle.deprecations) == {"kling-avatar-2", "sora-2"}

    def test_resolve(self, bundle):
        assert knowledge.resolve(bundle, "  HAILUO 03 ")["id"] == "minimax-h3"
        assert knowledge.resolve(bundle, "Kling-Avatar-2")["id"] == "kling-avatar-2"
        assert knowledge.resolve(bundle, "nope-xyz") is None

    def test_resolve_accepts_spelling_variants(self, bundle):
        for query in ("Hailuo 3", "hailuo3", "HAILUO-03", "Mini Max H3", "sync.3"):
            assert knowledge.resolve(bundle, query)["id"] == "minimax-h3" if "sync" not in query else True
        assert knowledge.resolve(bundle, "sync.3")["id"] == "sync-3"
        assert knowledge.resolve(bundle, "klingg") is None

    def test_normalize(self):
        assert knowledge._normalize("Hailuo 03") == "hailuo3"
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
        assert data["url"] is None
        assert data["ttl_seconds"] == knowledge.DEFAULT_TTL_SECONDS
        assert data["counts"] == {"models": 5, "capabilities": 2, "aliases": 22, "deprecations": 2}
        _validate(data)

    def test_status_with_nothing(self, capsys):
        rc, env = _run(["status"], capsys)
        assert rc == 0
        assert env["ok"] is True
        data = env["data"]
        assert data["loaded"] is False
        assert data["reason"]
        assert data["cache_path"].endswith(os.path.join("knowledge", "knowledge.json"))
        _validate(data)

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
        rc, env = _run(["resolve", "Hailuo 03"], capsys)
        assert rc == 0
        assert env["command"] == "knowledge resolve"
        data = env["data"]
        assert data["query"] == "Hailuo 03"
        assert data["id"] == "minimax-h3"
        assert data["model"]["id"] == "minimax-h3"
        assert data["deprecation"] is None
        assert data["bundle_version"] == "0.1.0-fixture"
        assert data["stale"] is False
        _validate(data)

    def test_resolve_deprecated(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        _rc, env = _run(["resolve", "kling-avatar-2"], capsys)
        data = env["data"]
        assert data["deprecation"]["superseded_by"] == "sync-3"
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
        rc, env = _run(["resolve", "Hailuo 3"], capsys)
        assert rc == 0
        assert env["data"]["id"] == "minimax-h3"
        assert env["data"]["query"] == "Hailuo 3"
        _validate(env["data"])

    def test_resolve_close_matches(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        _rc, env = _run(["resolve", "klingg"], capsys)
        assert env["error"]["code"] == "knowledge_unknown_model"
        assert "kling" in env["error"]["details"]["close_matches"]

    def test_resolve_without_bundle(self, capsys):
        rc, env = _run(["resolve", "x"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"

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
        assert by_model["kling"]["status"] == "available"
        assert by_model["ltx"]["status"] is None
        assert by_model["ltx"]["template"] == "video_ltx2_3_ia2v"
        assert by_model["kling"]["template"] is None
        _validate(data)

    def test_pick_spelling_variant(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "Audio Generation"], capsys)
        assert rc == 0
        assert env["data"]["capability"] == "audio-generation"
        _validate(env["data"])

    def test_pick_miss(self, tmp_path, monkeypatch, capsys):
        _env_bundle(tmp_path, monkeypatch)
        rc, env = _run(["pick", "nope"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unknown_capability"
        assert env["error"]["details"]["known"] == ["audio-generation", "lipsync"]

    def test_pick_without_bundle(self, capsys):
        rc, env = _run(["pick", "lipsync"], capsys)
        assert rc == 1
        assert env["error"]["code"] == "knowledge_unavailable"

    def test_pretty_mode_writes_nothing_to_stdout(self, tmp_path, monkeypatch, pretty_no_stdout):
        _env_bundle(tmp_path, monkeypatch)
        for args in (["status"], ["resolve", "kling"], ["pick", "lipsync"]):
            result = CliRunner().invoke(knowledge_cmd.app, args, standalone_mode=False)
            assert result.exception is None, result.exception

    def test_error_codes_registered(self):
        from comfy_cli import error_codes

        for code in ("knowledge_unavailable", "knowledge_unknown_model", "knowledge_unknown_capability"):
            assert error_codes.is_registered(code)

    def test_fixture_manifest_matches_fixture(self):
        import hashlib

        manifest = json.loads(FIXTURE_MANIFEST.read_text())
        raw = FIXTURE_KNOWLEDGE.read_bytes()
        assert manifest["files"]["knowledge.json"]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert manifest["files"]["knowledge.json"]["bytes"] == len(raw)
