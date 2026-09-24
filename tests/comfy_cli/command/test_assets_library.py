"""``comfy assets library`` error envelopes.

Pins the 404 mapping for ``assets library ensure``. Found in prod (Langfuse
2026-08-25, ``use_asset_as_input``): an agent passed a FILE NAME
(``comfyorg_logo.png``) where the content hash belongs, the API answered 404,
and the CLI reported ``workflow_not_found`` / "workflow not found (ensure)"
with a hint to list *workflows* — the shared cloud-HTTP helper's 404 branch
was written for the saved-workflow commands and hardcoded their vocabulary.
The agent had to guess its way past a message about the wrong resource.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import error_codes
from comfy_cli.caller import Caller
from comfy_cli.command import assets_library
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch):
    from comfy_cli.target import Target

    fake = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    return fake


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(assets_library.app, args, standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    return json.loads(captured.strip().splitlines()[-1])


def _http_error(code: int, body: bytes = b""):
    return urllib.error.HTTPError("https://cloud.example.com/api/assets/from-hash", code, "err", {}, io.BytesIO(body))


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, outcome):
    calls: list[dict] = []

    class _Resp:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return json.dumps(outcome).encode()

    def _fake(req, timeout=None):
        calls.append({"url": req.full_url, "method": req.get_method(), "body": req.data})
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


class TestEnsure:
    def test_404_is_asset_not_found_not_workflow_not_found(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(404))
        env = _run(["ensure", "--hash", "comfyorg_logo.png", "--where", "cloud"], capsys)
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "asset_not_found"
        assert error_codes.is_registered(err["code"])
        assert "workflow" not in err["message"].lower()
        assert "comfyorg_logo.png" in err["message"]
        # The remediation names the asset commands, not `workflow list`.
        assert "assets library ls" in err["hint"]
        assert "workflow list" not in err["hint"]
        assert err["details"]["hash"] == "comfyorg_logo.png"
        assert err["details"]["operation"] == "ensure"

    def test_401_is_still_cloud_unauthorized(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(401))
        env = _run(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert env["error"]["code"] == "cloud_unauthorized"

    def test_success_reports_id_hash_and_created_new(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1", "hash": "a" * 64})
        env = _run(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"] == {"id": "asset-1", "hash": "a" * 64, "created_new": True}
        assert calls[0]["url"].endswith("/api/assets/from-hash") and calls[0]["method"] == "POST"


class TestRateLimited:
    """`assets library ls` maps through `cloud_http`'s mapper, which must also
    classify a 429 as throttling rather than a generic HTTP failure."""

    def test_ls_429_is_cloud_rate_limited(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError(
            "https://cloud.example.com/api/assets", 429, "Too Many Requests", {"Retry-After": "7"}, io.BytesIO(b"")
        )
        _patch_urlopen(monkeypatch, err)
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["error"]["code"] == "cloud_rate_limited"
        assert error_codes.is_registered("cloud_rate_limited")
        assert env["error"]["details"]["retry_after"] == 7
        assert env["error"]["details"]["status"] == 429

    def test_ls_500_is_still_cloud_http_error(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(500, b"boom"))
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["error"]["code"] == "cloud_http_error"


_HEX = "ab" * 32


class TestEnsureHashWithExtension:
    """An agent passed the stored file name (`<hash>.png`) where the content hash
    belongs and got `asset_not_found`. A `<64-hex>.<ext>` is sent as
    the canonical `blake3:<hex>` wire hash — which the cloud matches against
    every storage-key shape, including `<hex>.<ext>` — and only that shape."""

    def test_hash_with_extension_is_sent_as_canonical_hash(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1", "hash": f"blake3:{_HEX}"})
        env = _run(["ensure", "--hash", f"{_HEX}.png", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert json.loads(calls[0]["body"])["hash"] == f"blake3:{_HEX}"

    def test_canonical_hash_with_extension_drops_the_extension(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1"})
        _run(["ensure", "--hash", f"blake3:{_HEX}.mp4", "--where", "cloud"], capsys)
        assert json.loads(calls[0]["body"])["hash"] == f"blake3:{_HEX}"

    @pytest.mark.parametrize("value", [_HEX, f"blake3:{_HEX}"])
    def test_bare_hash_is_sent_unchanged(self, cloud_target, monkeypatch, capsys, value):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1", "hash": value})
        env = _run(["ensure", "--hash", value, "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert json.loads(calls[0]["body"])["hash"] == value

    @pytest.mark.parametrize("value", ["comfyorg_logo.png", f"{_HEX[:-1]}.png", "photo.final.jpg"])
    def test_non_hash_name_with_a_dot_is_not_mangled(self, cloud_target, monkeypatch, capsys, value):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1"})
        _run(["ensure", "--hash", value, "--where", "cloud"], capsys)
        assert json.loads(calls[0]["body"])["hash"] == value

    def test_unknown_hash_with_extension_still_asset_not_found(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _http_error(404))
        env = _run(["ensure", "--hash", f"{_HEX}.png", "--where", "cloud"], capsys)
        err = env["error"]
        assert err["code"] == "asset_not_found"
        # The message names what the caller passed, so they can recognise it.
        assert f"{_HEX}.png" in err["message"]


# Synthetic: an asset's real hash, and a garbled 65-char copy of it that keeps
# only the first four characters — the shape an agent's mangled copy takes.
_REAL = "ab12" + "cd" * 30
_GARBLED = "ab12" + "ef" * 30 + "0"


def _route_urlopen(monkeypatch: pytest.MonkeyPatch, *, ensure_outcome, library):
    """from-hash → ``ensure_outcome``; the library listing → ``library`` (rows or an exception)."""
    calls: list[str] = []

    class _Resp:
        status = 200

        def __init__(self, payload):
            self._raw = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return self._raw

    def _fake(req, timeout=None):
        calls.append(req.full_url)
        outcome = ensure_outcome if "/from-hash" in req.full_url else library
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome if "/from-hash" in req.full_url else {"assets": outcome})

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


def _asset(hash_value: str, name: str) -> dict:
    return {"id": f"id-{name}", "name": name, "hash": hash_value}


class TestEnsureSuggestsNearHashes:
    """When the hash is not found and looks like a (garbled) hex hash, suggest
    the library assets sharing its longest prefix — from ONE bounded listing."""

    def test_garbled_hash_suggests_the_real_one(self, cloud_target, monkeypatch, capsys):
        library = [
            _asset("blake3:" + "ffff" + "0" * 60, "other.png"),
            _asset(f"{_REAL}.png", "beach.png"),
            _asset("ab1" + "1" * 61, "near-miss-3-chars.png"),
        ]
        calls = _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", f"{_GARBLED}.png", "--where", "cloud"], capsys)

        err = env["error"]
        assert err["code"] == "asset_not_found"
        suggestions = err["details"]["suggestions"]
        assert [s["hash"] for s in suggestions] == [f"{_REAL}.png"]
        assert suggestions[0]["name"] == "beach.png"
        assert f"did you mean {_REAL}.png (beach.png)?" in err["hint"]
        assert "exactly" in err["hint"]
        # Exactly one bounded listing, after the from-hash miss.
        assert len(calls) == 2
        assert "/api/assets?" in calls[1] and "limit=500" in calls[1]

    def test_suggestions_rank_by_longest_prefix_and_cap_at_three(self, cloud_target, monkeypatch, capsys):
        target = "abcdef" + "0" * 58
        library = [_asset(p + "9" * (64 - len(p)), f"{p}.png") for p in ("abcd", "abcde", "abcdef1", "abcd1", "abc")]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", target, "--where", "cloud"], capsys)

        names = [s["name"] for s in env["error"]["details"]["suggestions"]]
        assert names == ["abcdef1.png", "abcde.png", "abcd.png"]

    def test_no_prefix_match_gives_no_suggestions(self, cloud_target, monkeypatch, capsys):
        library = [_asset("ffff" + "0" * 60, "a.png"), _asset("ab1" + "0" * 61, "b.png")]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", f"{_GARBLED}.png", "--where", "cloud"], capsys)

        err = env["error"]
        assert err["code"] == "asset_not_found"
        assert "suggestions" not in err["details"]
        assert "did you mean" not in err["hint"]

    def test_non_hex_name_makes_no_library_call(self, cloud_target, monkeypatch, capsys):
        calls = _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=[_asset(_REAL, "x.png")])
        env = _run(["ensure", "--hash", "comfyorg_logo.png", "--where", "cloud"], capsys)

        assert env["error"]["code"] == "asset_not_found"
        assert "suggestions" not in env["error"]["details"]
        assert len(calls) == 1

    def test_a_failed_listing_still_reports_asset_not_found(self, cloud_target, monkeypatch, capsys):
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=_http_error(500))
        env = _run(["ensure", "--hash", f"{_GARBLED}.png", "--where", "cloud"], capsys)

        assert env["error"]["code"] == "asset_not_found"
        assert "suggestions" not in env["error"]["details"]

    def test_other_errors_make_no_library_call(self, cloud_target, monkeypatch, capsys):
        calls = _route_urlopen(monkeypatch, ensure_outcome=_http_error(500), library=[_asset(_REAL, "x.png")])
        env = _run(["ensure", "--hash", _GARBLED, "--where", "cloud"], capsys)

        assert env["error"]["code"] == "cloud_http_error"
        assert len(calls) == 1


class TestLongExtensions:
    """Model files carry extensions longer than image ones (`.safetensors` is 11)."""

    def test_hash_with_safetensors_extension_is_canonicalized(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1"})
        _run(["ensure", "--hash", f"{_HEX}.safetensors", "--where", "cloud"], capsys)
        assert json.loads(calls[0]["body"])["hash"] == f"blake3:{_HEX}"

    def test_library_hash_with_safetensors_extension_is_suggested(self, cloud_target, monkeypatch, capsys):
        library = [_asset(f"{_REAL}.safetensors", "model.safetensors")]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", f"{_GARBLED}.safetensors", "--where", "cloud"], capsys)
        assert [s["name"] for s in env["error"]["details"]["suggestions"]] == ["model.safetensors"]


class TestSuggestionRequestIsShortLived:
    def test_listing_uses_a_short_timeout(self, cloud_target, monkeypatch, capsys):
        """The listing only feeds an optional hint; a stalled one must not hold
        the asset_not_found envelope for the default 30s request timeout."""
        timeouts: dict[str, float | None] = {}

        def _fake(req, timeout=None):
            if "/from-hash" in req.full_url:
                timeouts["ensure"] = timeout
                raise _http_error(404)
            timeouts["listing"] = timeout
            raise TimeoutError("stalled")

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        env = _run(["ensure", "--hash", _GARBLED, "--where", "cloud"], capsys)

        assert env["error"]["code"] == "asset_not_found"
        assert timeouts["listing"] is not None and timeouts["listing"] <= 5


class TestUppercaseHex:
    @pytest.mark.parametrize("value", [f"{_HEX.upper()}.png", f"blake3:{_HEX.upper()}.PNG"])
    def test_uppercase_digest_with_extension_is_canonicalized_lowercase(self, cloud_target, monkeypatch, capsys, value):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1"})
        _run(["ensure", "--hash", value, "--where", "cloud"], capsys)
        assert json.loads(calls[0]["body"])["hash"] == f"blake3:{_HEX}"

    def test_bare_uppercase_digest_is_sent_unchanged(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"id": "asset-1"})
        _run(["ensure", "--hash", _HEX.upper(), "--where", "cloud"], capsys)
        assert json.loads(calls[0]["body"])["hash"] == _HEX.upper()
