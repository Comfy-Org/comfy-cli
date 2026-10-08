"""``comfy assets library`` envelopes.

Pins two things: the ``assets library ls`` pagination passthrough, and the 404
mapping for ``assets library ensure``.

``ls`` forwards the server's ``has_more``/``total`` so a downstream consumer
(the cloud agent's asset gate) reads an authoritative truncation signal instead
of inferring one from a page that came back exactly full — which both misses a
short truncated page and misreads a library of exactly ``--limit`` assets. Both
fields are ``required`` on the cloud API's ``ListAssetsResponse``, but a server
that omits them must leave the keys ABSENT rather than emit ``null``, because
the consumer type-asserts them out of the decoded envelope.

For ``ensure``: when a caller passed a FILE NAME (``comfyorg_logo.png``) where
the content hash belongs, the API answered 404, and the CLI reported
``workflow_not_found`` / "workflow not found (ensure)"
with a hint to list *workflows* — the shared cloud-HTTP helper's 404 branch
was written for the saved-workflow commands and hardcoded their vocabulary.
The caller had to guess its way past a message about the wrong resource.
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


def _patch_urlopen_raw(monkeypatch: pytest.MonkeyPatch, raw: bytes):
    """Like ``_patch_urlopen`` but serves ``raw`` verbatim, so a test can send a
    body that is not valid JSON at all (or is genuinely empty) rather than one
    that round-trips through ``json.dumps``."""

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return raw

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp())


class TestLsPagination:
    """`ls` forwards the server's truncation signal, and only when it sent one."""

    def _ls(self, monkeypatch, capsys, body: dict) -> dict[str, Any]:
        _patch_urlopen(monkeypatch, body)
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        return env["data"]

    def test_forwards_has_more_and_total(self, cloud_target, monkeypatch, capsys):
        data = self._ls(
            monkeypatch,
            capsys,
            {"assets": [{"id": "a1", "name": "cat.png", "hash": "h1"}], "has_more": True, "total": 1234},
        )
        assert data["has_more"] is True
        assert data["total"] == 1234
        # Alongside, not instead of, the existing shape.
        assert data["count"] == 1
        assert data["assets"][0]["id"] == "a1"

    def test_forwards_has_more_false(self, cloud_target, monkeypatch, capsys):
        # `false` is a real answer, not a missing one — the falsy value must
        # survive, otherwise the consumer cannot distinguish "not truncated"
        # from "server did not say".
        data = self._ls(monkeypatch, capsys, {"assets": [], "has_more": False, "total": 0})
        assert data["has_more"] is False
        assert data["total"] == 0

    def test_omits_keys_when_server_does_not_send_them(self, cloud_target, monkeypatch, capsys):
        data = self._ls(monkeypatch, capsys, {"assets": [{"id": "a1"}]})
        assert "has_more" not in data
        assert "total" not in data
        assert data["count"] == 1

    def test_omits_keys_when_server_sends_nulls(self, cloud_target, monkeypatch, capsys):
        # A forwarded `null` would poison the consumer's type assertion, so a
        # null is treated exactly like an absent key.
        data = self._ls(monkeypatch, capsys, {"assets": [], "has_more": None, "total": None})
        assert "has_more" not in data
        assert "total" not in data

    def test_omits_cross_typed_values(self, cloud_target, monkeypatch, capsys):
        # `bool` is a subclass of `int` in Python, so a shared bool-or-int check
        # would let `has_more: 0` / `total: false` through and emit an envelope
        # that violates this command's own published schema. Each field is
        # validated against its own type instead, and a mistyped value is
        # dropped exactly like an absent one.
        data = self._ls(monkeypatch, capsys, {"assets": [], "has_more": 0, "total": False})
        assert "has_more" not in data
        assert "total" not in data

    def test_omits_a_negative_total(self, cloud_target, monkeypatch, capsys):
        # The schema publishes `total` as `minimum: 0`, so forwarding a negative
        # server value would emit an envelope violating this command's own
        # contract — the same class of bug as the cross-typed case above.
        data = self._ls(monkeypatch, capsys, {"assets": [], "has_more": False, "total": -1})
        assert "total" not in data
        assert data["has_more"] is False

    def test_omits_values_of_the_wrong_json_type(self, cloud_target, monkeypatch, capsys):
        data = self._ls(monkeypatch, capsys, {"assets": [], "has_more": "true", "total": "1234"})
        assert "has_more" not in data
        assert "total" not in data

    def test_non_object_body_is_an_error_envelope_not_a_traceback(self, cloud_target, monkeypatch, capsys):
        # A proxy or error page can answer 200 with valid JSON that is not an
        # object. `b.get(...)` would raise a bare AttributeError past the
        # HTTPError/URLError/OSError handler, so the user would see a traceback
        # rather than an envelope.
        _patch_urlopen(monkeypatch, [1, 2, 3])
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert error_codes.is_registered(env["error"]["code"])
        assert env["error"]["details"]["got_type"] == "list"

    def test_non_list_assets_is_an_error_envelope_not_a_traceback(self, cloud_target, monkeypatch, capsys):
        # Same failure one level down: `len(rows)` on a scalar raises TypeError.
        _patch_urlopen(monkeypatch, {"assets": 42})
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert env["error"]["details"]["got_type"] == "int"

    def test_json_null_body_is_an_empty_listing_not_an_error(self, cloud_target, monkeypatch, capsys):
        data = self._ls(monkeypatch, capsys, None)
        assert data["count"] == 0
        assert data["assets"] == []

    def test_truly_empty_body_is_an_empty_listing_not_an_error(self, cloud_target, monkeypatch, capsys):
        # Zero bytes: the server had nothing to say, which is a legitimate
        # empty library and must stay a success envelope.
        _patch_urlopen_raw(monkeypatch, b"")
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["count"] == 0
        assert env["data"]["assets"] == []

    def test_whitespace_only_body_is_an_empty_listing_not_an_error(self, cloud_target, monkeypatch, capsys):
        # A body of just a newline is "nothing to say", not a malformed answer;
        # `strict_json` must not turn it into an error envelope.
        _patch_urlopen_raw(monkeypatch, b"\n")
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["count"] == 0

    def test_invalid_json_body_is_an_error_envelope_not_an_empty_listing(self, cloud_target, monkeypatch, capsys):
        # `http_request` collapses a JSONDecodeError to `None` by default, which
        # is indistinguishable from the empty body above — so a proxy or
        # captive-portal error page answering 200 rendered as a successful EMPTY
        # library. `ls` opts into `strict_json` so the two stay distinct.
        _patch_urlopen_raw(monkeypatch, b"<html><body>502 Bad Gateway</body></html>")
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert error_codes.is_registered(env["error"]["code"])

    def test_invalid_utf8_body_is_an_error_envelope_not_a_traceback(self, cloud_target, monkeypatch, capsys):
        # Handed raw bytes, `json.loads` rejects a NON-UTF-8 body with
        # `UnicodeDecodeError`, not `JSONDecodeError` — so a binary error page (or
        # a gzip/TLS fragment from a misbehaving proxy) escaped the `strict_json`
        # catch entirely and crashed the CLI with a traceback, past the very
        # handler added to stop invalid JSON from masquerading as an empty
        # library. Same malformed answer as the test above, different bytes.
        #
        # These bytes are chosen to be invalid UTF-8 that is ALSO not a BOM:
        # `json.loads` sniffs raw bytes for UTF-16/32 (RFC 4627), so a body
        # starting `\xff\x00` is guessed as UTF-16-LE, decodes to garbage text and
        # fails as an ordinary `JSONDecodeError` — which the narrow catch already
        # handled, making it a test that passes with or without the fix.
        _patch_urlopen_raw(monkeypatch, b"\x80\x81\x82 binary garbage")
        env = _run(["ls", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert error_codes.is_registered(env["error"]["code"])

    def test_count_matches_the_emitted_assets_and_projection_is_unchanged(self, cloud_target, monkeypatch, capsys):
        rows = [
            {
                "id": "a1",
                "name": "cat.png",
                "hash": "h1",
                "mime_type": "image/png",
                "size": 12,
                "tags": ["input"],
                "preview_url": "https://example.com/p.png",
                "job_id": "j1",
                "created_at": "2026-01-01T00:00:00Z",
                "extra_server_field": "dropped",
            },
            "not-a-dict",
        ]
        data = self._ls(monkeypatch, capsys, {"assets": rows, "has_more": False, "total": 1})
        # `count` describes the array actually emitted: the non-dict row is
        # dropped from `assets`, so counting it too would report more items than
        # the payload carries — misleading in general, and self-defeating beside
        # a forwarded `total` whose whole purpose is an authoritative count.
        assert data["count"] == 1
        assert data["total"] == 1
        assert data["assets"] == [
            {
                "id": "a1",
                "name": "cat.png",
                "hash": "h1",
                "mime_type": "image/png",
                "size": 12,
                "tags": ["input"],
                "preview_url": "https://example.com/p.png",
                "job_id": "j1",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

    def test_envelope_validates_against_the_published_schema(self, cloud_target, monkeypatch, capsys):
        import json as _json
        from pathlib import Path

        import jsonschema

        data = self._ls(monkeypatch, capsys, {"assets": [{"id": "a1"}], "has_more": True, "total": 7})
        schema_path = Path(assets_library.__file__).resolve().parents[1] / "schemas" / "assets_library.json"
        jsonschema.Draft202012Validator(_json.loads(schema_path.read_text())).validate(data)


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


class TestHttpRequestRejectsEveryUnparseableBody:
    """`http_request`'s decode guard is keyed on the malformed body, not on which
    exception the parser happened to pick.

    `json.loads` rejects a malformed body with three different exceptions and only
    one is a `JSONDecodeError`; the other two are exercised here through a patched
    `json.loads` rather than through pathological input, because the thresholds
    that produce them (CPython's 4300-digit int limit, the recursion limit) move
    between interpreter versions and would make the test a platform coin-flip.
    The invalid-UTF-8 case is deterministic everywhere and is pinned end-to-end
    in `TestLsPagination` above.
    """

    @staticmethod
    def _request(monkeypatch, target, exc: BaseException, *, strict_json: bool):
        from comfy_cli.command import cloud_http

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=None):
                return b'{"assets": []}'

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp())

        def _boom(_raw):
            raise exc

        monkeypatch.setattr(cloud_http.json, "loads", _boom)
        return cloud_http.http_request(target.url("assets"), target, strict_json=strict_json)

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(ValueError("Exceeds the limit for integer string conversion"), id="bare-ValueError"),
            pytest.param(RecursionError("maximum recursion depth exceeded"), id="RecursionError"),
        ],
    )
    def test_strict_json_raises_response_unparseable(self, cloud_target, monkeypatch, exc):
        from comfy_cli.command.cloud_http import ResponseUnparseable

        # Not `pytest.raises(type(exc))`: the point is that the raw exception is
        # translated, so a call site's `except ResponseUnparseable` sees it.
        with pytest.raises(ResponseUnparseable):
            self._request(monkeypatch, cloud_target, exc, strict_json=True)

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(ValueError("Exceeds the limit for integer string conversion"), id="bare-ValueError"),
            pytest.param(RecursionError("maximum recursion depth exceeded"), id="RecursionError"),
        ],
    )
    def test_default_path_still_collapses_to_none(self, cloud_target, monkeypatch, exc):
        # The default path's documented contract is that an undecodable body
        # collapses to `None`; broadening the catch makes that true for these two
        # as well, instead of letting them escape as a traceback.
        assert self._request(monkeypatch, cloud_target, exc, strict_json=False) == (200, None)


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

    def test_response_without_hash_reports_the_normalized_hash(self, cloud_target, monkeypatch, capsys):
        # The success payload falls back to the hash that was SENT, not the raw
        # `<hex>.<ext>` input, when the server omits `hash`.
        _patch_urlopen(monkeypatch, {"id": "asset-1"})
        env = _run(["ensure", "--hash", f"{_HEX}.png", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["hash"] == f"blake3:{_HEX}"

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


# Synthetic: an asset's real hash and the shapes a mangled copy of it takes.
# A 4-char shared prefix is weak evidence (common by chance across a library
# page), so it must NOT produce a suggestion; the strong forms must.
_REAL = "0123456789abcdef" * 4
_GARBLED = _REAL[:10] + "f" + _REAL[10:]  # one extra char mid-hash (65 chars)
_DROPPED = _REAL[:30] + _REAL[31:]  # one char lost (63 chars)
_TRUNCATED_TAIL = _REAL[:20] + "e" * 45  # first 20 chars kept, tail garbled (65 chars)
_WEAK = _REAL[:4] + "e" * 61  # only 4 chars shared (65 chars)


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
    """When the hash is not found and looks mangled, suggest library assets it
    is almost certainly a copy of, from ONE bounded listing. Evidence must be
    strong: within one inserted/dropped/changed char of a stored hash, or a
    shared prefix of >= 16 hex chars."""

    @pytest.mark.parametrize("mangled", [_GARBLED, _DROPPED, _TRUNCATED_TAIL])
    def test_mangled_hash_suggests_the_real_one(self, cloud_target, monkeypatch, capsys, mangled):
        library = [
            _asset("blake3:" + "ffff" + "0" * 60, "other.png"),
            _asset(f"{_REAL}.png", "beach.png"),
            _asset(_REAL[:4] + "1" * 60, "weak-4-char-prefix.png"),
        ]
        calls = _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", f"{mangled}.png", "--where", "cloud"], capsys)

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

    def test_bare_uppercase_digest_suggests_its_lowercase_form(self, cloud_target, monkeypatch, capsys):
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=[_asset(_REAL, "beach.png")])
        env = _run(["ensure", "--hash", _REAL.upper(), "--where", "cloud"], capsys)
        assert [s["hash"] for s in env["error"]["details"]["suggestions"]] == [_REAL]

    def test_short_shared_prefix_is_not_evidence(self, cloud_target, monkeypatch, capsys):
        library = [_asset(f"{_REAL}.png", "beach.png"), _asset(_REAL[:15] + "e" * 49, "fifteen.png")]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", _WEAK, "--where", "cloud"], capsys)

        err = env["error"]
        assert err["code"] == "asset_not_found"
        assert "suggestions" not in err["details"]
        assert "did you mean" not in err["hint"]

    def test_suggestions_rank_by_longest_prefix_and_cap_at_three(self, cloud_target, monkeypatch, capsys):
        wanted = "a" * 22 + "0" * 43  # 65 chars: mangled length
        library = [_asset("a" * k + "b" * (64 - k), f"{k}.png") for k in (16, 18, 20, 17, 15)]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", wanted, "--where", "cloud"], capsys)

        names = [s["name"] for s in env["error"]["details"]["suggestions"]]
        assert names == ["20.png", "18.png", "17.png"]

    def test_one_edit_match_outranks_a_long_prefix(self, cloud_target, monkeypatch, capsys):
        library = [_asset(_REAL[:40] + "e" * 24, "long-prefix.png"), _asset(_REAL, "exact-but-one.png")]
        _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=library)
        env = _run(["ensure", "--hash", _DROPPED, "--where", "cloud"], capsys)

        names = [s["name"] for s in env["error"]["details"]["suggestions"]]
        assert names[0] == "exact-but-one.png"

    @pytest.mark.parametrize("value", [_REAL, f"blake3:{_REAL}", f"{_REAL}.png", f"blake3:{_REAL}.png"])
    def test_well_formed_digest_that_is_missing_makes_no_library_call(self, cloud_target, monkeypatch, capsys, value):
        """A full lowercase 64-hex digest is not mangled; it simply is not in the
        library. Listing would cost a request and could only guess."""
        calls = _route_urlopen(monkeypatch, ensure_outcome=_http_error(404), library=[_asset(_REAL, "x.png")])
        env = _run(["ensure", "--hash", value, "--where", "cloud"], capsys)

        assert env["error"]["code"] == "asset_not_found"
        assert "suggestions" not in env["error"]["details"]
        assert len(calls) == 1

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
