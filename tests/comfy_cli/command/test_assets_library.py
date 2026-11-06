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

import base64
import hashlib
import http.client
import io
import json
import os
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from blake3 import blake3
from typer.testing import CliRunner

from comfy_cli import error_codes
from comfy_cli.caller import Caller
from comfy_cli.command import assets_library
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch, tmp_path):
    from comfy_cli import constants

    monkeypatch.setattr(
        constants, "DEFAULT_CONFIG", {key: str(tmp_path / "config") for key in constants.DEFAULT_CONFIG}
    )
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture(scope="module")
def root_cli_ready():
    """Load the actual CLI before a test's target fake can bind into lazy modules."""
    from click import Context
    from typer.main import get_command

    from comfy_cli.cmdline import app as root_app

    root = get_command(root_app)
    with Context(root) as context:
        for name in root.list_commands(context):
            root.get_command(context, name)


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch, root_cli_ready):
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
    return _run_with_rc(args, capsys)[0]


def _run_with_rc(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], int]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(assets_library.app, args, standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    # Under `standalone_mode=False` click returns a `typer.Exit` code as the
    # return value instead of exiting, so `exit_code` alone reads 0.
    rc = result.exit_code or (result.return_value if isinstance(result.return_value, int) else 0)
    return json.loads(captured.strip().splitlines()[-1]), rc


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


def _patch_urlopen_raw(monkeypatch: pytest.MonkeyPatch, raw: bytes, status: int = 200):
    """Serve ``raw`` verbatim, unlike ``_patch_urlopen`` which JSON-encodes it.

    Needed for the bodies a real proxy/gateway returns — an HTML error page,
    non-UTF-8 bytes, nothing at all — which never survive a ``json.dumps``
    round trip.
    """
    calls: list[dict] = []

    class _Resp:
        def __init__(self):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=None):
            return raw

    def _fake(req, timeout=None):
        calls.append({"url": req.full_url, "method": req.get_method(), "body": req.data})
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


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

    def _assert_unconfirmed_borrow(self, capsys, *, got_type: str, status: int, missing: str | None = None):
        env, rc = _run_with_rc(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert rc == 1
        assert env["ok"] is False
        err = env["error"]
        assert err["code"] == "cloud_http_error"
        assert error_codes.is_registered(err["code"])
        assert err["details"]["operation"] == "ensure"
        assert err["details"]["hash"] == "a" * 64
        assert err["details"]["got_type"] == got_type
        assert err["details"]["status"] == status
        assert err["details"]["missing"] == missing
        assert "created_new" not in json.dumps(env)

    def test_non_object_body_is_cloud_http_error(self, cloud_target, monkeypatch, capsys):
        # Pre-fix: `AttributeError: 'list' object has no attribute 'get'` and no envelope at all.
        _patch_urlopen(monkeypatch, [1, 2, 3])
        self._assert_unconfirmed_borrow(capsys, got_type="list", status=201)

    def test_unparseable_body_is_cloud_http_error_not_fake_success(self, cloud_target, monkeypatch, capsys):
        # Pre-fix: `{"ok": true, "data": {"id": null, "hash": <the caller's own hash>, "created_new": true}}`
        # — a borrow the server never confirmed, reported as done.
        _patch_urlopen_raw(monkeypatch, b"<html><body>502 Bad Gateway</body></html>", status=201)
        self._assert_unconfirmed_borrow(capsys, got_type="NoneType", status=201)

    def test_non_utf8_body_is_cloud_http_error_not_traceback(self, cloud_target, monkeypatch, capsys):
        # `json.loads(bytes)` raises `UnicodeDecodeError`, not `JSONDecodeError`.
        _patch_urlopen_raw(monkeypatch, b"\xff", status=201)
        self._assert_unconfirmed_borrow(capsys, got_type="NoneType", status=201)

    def test_empty_body_is_cloud_http_error(self, cloud_target, monkeypatch, capsys):
        # A POST with no body cannot confirm the borrow: unlike `ls`, empty is an error here.
        _patch_urlopen_raw(monkeypatch, b"", status=200)
        self._assert_unconfirmed_borrow(capsys, got_type="NoneType", status=200)

    def test_whitespace_body_is_cloud_http_error(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen_raw(monkeypatch, b"\n", status=201)
        self._assert_unconfirmed_borrow(capsys, got_type="NoneType", status=201)

    @pytest.mark.skipif(not getattr(sys, "get_int_max_str_digits", lambda: 0)(), reason="no int-conversion digit limit")
    def test_oversized_integer_body_is_cloud_http_error_not_traceback(self, cloud_target, monkeypatch, capsys):
        # Past the int-conversion limit `json.loads` raises a bare `ValueError`, not `JSONDecodeError`.
        _patch_urlopen_raw(monkeypatch, b'{"id": ' + b"1" * (sys.get_int_max_str_digits() + 1) + b"}", status=201)
        self._assert_unconfirmed_borrow(capsys, got_type="NoneType", status=201)

    @pytest.mark.parametrize(
        "body",
        [{}, {"detail": "upstream timeout"}, {"id": None, "hash": "a" * 64}, {"id": ""}, {"id": True}, {"id": [1]}],
    )
    def test_object_without_id_is_cloud_http_error(self, cloud_target, monkeypatch, capsys, body):
        # A 2xx object with no `id` confirms nothing — it must not echo the caller's hash back as success.
        _patch_urlopen(monkeypatch, body)
        self._assert_unconfirmed_borrow(capsys, got_type="dict", status=201, missing="id")

    def test_null_hash_in_response_falls_back_to_requested_hash(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"id": "asset-1", "hash": None})
        env = _run(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["hash"] == "a" * 64

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


# --- assets library upload ---------------------------------------------------

_SIGNED = "https://storage.example.com/assets/temp/model-uploads/x.safetensors?X-Goog-Signature=SECRETSIG"
_PUT_HEADERS = {
    "Content-Type": "application/octet-stream",
    "x-goog-content-length-range": "0,16",
    "x-goog-if-generation-match": "0",
}
_FOLDERS = {"model_upload_folders": ["checkpoints", "embeddings", "loras", "vae"]}
# What the `model_file` fixture holds, and its hashes.
_MODEL_BYTES = b"0123456789abcdef"
_MODEL_HASH = "blake3:" + blake3(_MODEL_BYTES).hexdigest()
_MODEL_SHA256 = hashlib.sha256(_MODEL_BYTES).hexdigest()
_DONE = {
    "id": "task-1",
    "status": "completed",
    "result": {"success": True, "asset_id": "asset-9", "hash": _MODEL_HASH, "filename": "style.safetensors"},
}
# No model in the folder has the file's name yet.
_NONE_NAMED = {"assets": [], "has_more": False}


def _started(n: int = 1) -> dict:
    return {
        "upload_id": f"00000000-0000-0000-0000-00000000000{n}",
        "upload_url": f"{_SIGNED}{n}",
        "method": "PUT",
        "headers": dict(_PUT_HEADERS),
        "expires_at": "2026-11-02T00:15:00Z",
        "max_size_bytes": 2147483648,
    }


class _Cloud:
    """Answers each ``(method, path)`` from a queue: a dict body, raw bytes, an exception, or a response factory."""

    def __init__(self, routes: dict):
        self.routes = {k: list(v) if isinstance(v, list) else [v] for k, v in routes.items()}
        self.calls: list[tuple[str, str, Any]] = []

    def __call__(self, req, timeout=None):
        import urllib.parse

        method, path = req.get_method(), urllib.parse.urlsplit(req.full_url).path
        self.calls.append((method, path, json.loads(req.data) if req.data else None))
        queue = self.routes[(method, path)]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        raw = outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=None):
                return raw

        return _Resp()

    def paths(self, method: str | None = None) -> list[str]:
        return [p for m, p, _ in self.calls if method in (None, m)]


class _Storage:
    """Stands in for ``requests.put``: answers each PUT from a queue of statuses or exceptions."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.puts: list[dict] = []

    def __call__(self, url, data=None, headers=None, timeout=None, allow_redirects=True):
        sent = b""
        while block := data.read(4):
            sent += block
        self.puts.append({"url": url, "headers": headers, "body": sent, "allow_redirects": allow_redirects})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class _Resp:
            status_code = outcome

        return _Resp()


class _SentSoFar(_Storage):
    """A ``_Storage`` that records each block as it reads it, so a PUT whose body read
    raises still shows the bytes a socket would already have sent."""

    def __call__(self, url, data=None, headers=None, timeout=None, allow_redirects=True):
        put = {"url": url, "headers": headers, "body": b"", "allow_redirects": allow_redirects}
        self.puts.append(put)
        while block := data.read(4):
            put["body"] += block
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class _Resp:
            status_code = outcome

        return _Resp()


def _cloud_error(code: int, server_code: str, message: str):
    body = json.dumps({"code": server_code, "message": message}).encode()
    return urllib.error.HTTPError("https://cloud.example.com/api/x", code, "err", {}, io.BytesIO(body))


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "style.safetensors"
    path.write_bytes(_MODEL_BYTES)
    return path


@pytest.fixture
def no_wait(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(assets_library, "_clock", lambda: now[0])
    monkeypatch.setattr(assets_library, "_sleep", lambda s: now.__setitem__(0, now[0] + s))
    return now


def _upload(
    monkeypatch,
    capsys,
    args: list[str],
    routes: dict,
    puts: list,
    cloud: _Cloud | None = None,
    storage: _Storage | None = None,
    app: Any = None,
) -> tuple[dict, int, _Cloud, _Storage, str]:
    cloud, storage = cloud or _Cloud(routes), storage or _Storage(puts)
    monkeypatch.setattr("urllib.request.urlopen", cloud)
    monkeypatch.setattr("requests.put", storage)
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(app or assets_library.app, ["upload", *args, "--where", "cloud"], standalone_mode=False)
    captured = capsys.readouterr()
    everything = captured.out + captured.err + result.output
    rc = result.exit_code or (result.return_value if isinstance(result.return_value, int) else 0)
    lines = (captured.out or result.stdout).strip().splitlines()
    assert lines, f"no envelope on stdout (rc={rc}, exc={result.exception})"
    return json.loads(lines[-1]), rc, cloud, storage, everything


_UPLOAD_ROUTES = {
    ("GET", "/api/features"): _FOLDERS,
    ("GET", "/api/assets"): _NONE_NAMED,
    ("POST", "/api/assets/uploads"): [_started(1), _started(2)],
    ("POST", "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete"): {
        "task_id": "task-1",
        "status": "created",
    },
    ("POST", "/api/assets/uploads/00000000-0000-0000-0000-000000000002/complete"): {
        "task_id": "task-1",
        "status": "created",
    },
    ("GET", "/api/tasks/task-1"): [{"id": "task-1", "status": "running"}, _DONE],
    # Read after the hash check, for Comfy Cloud's loader_path.
    ("GET", "/api/assets/asset-9"): {"id": "asset-9", "hash": _MODEL_HASH, "loader_path": "style.safetensors"},
}


class TestUpload:
    def test_uploads_completes_and_reports_the_stored_asset(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        import jsonschema

        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [200]
        )

        assert rc == 0 and env["ok"] is True
        assert env["data"] == {
            "id": "asset-9",
            "name": "style.safetensors",
            "folder": "loras",
            "loader_path": "style.safetensors",
            "hash": _MODEL_HASH,
            "sha256": _MODEL_SHA256,
            "size": 16,
            "workspace_id": None,
            "task_id": "task-1",
            "uploaded": True,
            "already_present": False,
            "dry_run": False,
        }
        schema = json.loads(
            (Path(assets_library.__file__).resolve().parents[1] / "schemas" / "assets_library.json").read_text()
        )
        jsonschema.Draft202012Validator(schema).validate(env["data"])
        assert cloud.paths() == [
            "/api/features",
            "/api/assets",
            "/api/assets/uploads",
            "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete",
            "/api/tasks/task-1",
            "/api/tasks/task-1",
            "/api/assets/asset-9",
        ]
        assert cloud.calls[2][2] == {
            "filename": "style.safetensors",
            "size_bytes": 16,
            "tags": ["models", "loras"],
            "resumable": True,
        }
        assert cloud.calls[3][2] == {"filename": "style.safetensors", "tags": ["models", "loras"]}
        (put,) = storage.puts
        assert put["url"] == f"{_SIGNED}1" and put["headers"] == _PUT_HEADERS and put["allow_redirects"] is False
        assert put["body"] == model_file.read_bytes()
        assert "SECRETSIG" not in everything

    @pytest.mark.parametrize(
        ("name", "size", "reason"),
        [
            ("style.ckpt", 16, "unsupported_file_type"),
            ("style.pt", 16, "unsupported_file_type"),
            (".hidden.safetensors", 16, "bad_filename"),
            ("a..b.safetensors", 16, "bad_filename"),
            ("x" * 250 + ".safetensors", None, "bad_filename"),
            ("style.safetensors", None, "file_not_found"),
            ("style.safetensors", 0, "empty_file"),
        ],
    )
    def test_a_file_comfy_cloud_would_refuse_is_refused_before_any_request(
        self, cloud_target, monkeypatch, capsys, tmp_path, name, size, reason
    ):
        path = tmp_path / name
        if size is not None:
            with path.open("wb") as f:
                f.truncate(size)  # sparse: no 2 GB written to disk

        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(path), "--type", "loras"], {}, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == reason
        assert cloud.calls == [] and storage.puts == []

    @pytest.mark.parametrize("size", [2 * 1024**3 + 1, 25 * 1024**3, 25 * 1024**3 + 1])
    def test_size_limit_is_left_to_the_server(self, tmp_path, size):
        path = tmp_path / "big.SAFETENSORS"
        with path.open("wb") as f:
            f.truncate(size)
        assert assets_library._local_refusal(path) is None

    def test_unknown_type_is_refused_listing_the_valid_ones_before_the_upload(
        self, cloud_target, monkeypatch, capsys, model_file
    ):
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "lora"], dict(_UPLOAD_ROUTES), []
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["valid_types"] == _FOLDERS["model_upload_folders"]
        assert "checkpoints, embeddings, loras, vae" in env["error"]["message"]
        assert cloud.paths() == ["/api/features"] and storage.puts == []

    def test_without_upload_folders_the_start_call_decides(self, cloud_target, monkeypatch, capsys, model_file):
        routes = {
            ("GET", "/api/features"): {"max_upload_size": 1},
            ("GET", "/api/assets"): _NONE_NAMED,
            ("POST", "/api/assets/uploads"): _cloud_error(
                403, "FEATURE_DISABLED", "Uploading a model file is not enabled for this workspace"
            ),
        }
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_refused"
        assert env["error"]["message"] == "Uploading a model file is not enabled for this workspace"
        assert env["error"]["details"]["server_code"] == "FEATURE_DISABLED"
        assert storage.puts == []

    def test_storage_cap_refusal_prints_the_server_message(self, cloud_target, monkeypatch, capsys, model_file):
        routes = dict(_UPLOAD_ROUTES)
        routes[("POST", "/api/assets/uploads")] = _cloud_error(
            403, "MODEL_STORAGE_LIMIT_REACHED", "Your workspace has used its 1 TB of model storage."
        )
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, []
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_refused"
        assert env["error"]["message"] == "Your workspace has used its 1 TB of model storage."
        assert env["error"]["details"] == {
            "status": 403,
            "server_code": "MODEL_STORAGE_LIMIT_REACHED",
            "operation": "start",
        }
        assert storage.puts == [] and "/complete" not in "".join(cloud.paths())

    def test_412_means_the_bytes_already_landed(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [412]
        )

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert len(storage.puts) == 1 and cloud.paths("POST").count("/api/assets/uploads") == 1

    def test_a_put_with_no_answer_starts_one_fresh_upload(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        import requests

        lost = requests.ConnectionError(f"Max retries exceeded with url: {_SIGNED}1")
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [lost, 200]
        )

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert [p["url"] for p in storage.puts] == [f"{_SIGNED}1", f"{_SIGNED}2"]
        assert "/api/assets/uploads/00000000-0000-0000-0000-000000000002/complete" in cloud.paths()
        assert "SECRETSIG" not in everything

    def test_the_fresh_upload_is_bounded(self, cloud_target, monkeypatch, capsys, model_file):
        import requests

        lost = requests.ConnectionError(f"Max retries exceeded with url: {_SIGNED}1")
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [lost, 503]
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert env["error"]["details"]["status"] == 503 and env["error"]["details"]["attempts"] == 2
        assert len(storage.puts) == 2 and not any(p.endswith("/complete") for p in cloud.paths())
        assert "SECRETSIG" not in everything

    def test_a_refused_put_is_not_retried(self, cloud_target, monkeypatch, capsys, model_file):
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [403]
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert len(storage.puts) == 1 and "SECRETSIG" not in everything

    def test_a_failed_task_exits_non_zero_with_its_message(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = {
            "id": "task-1",
            "status": "failed",
            "error_message": "This file is not a valid .safetensors file.",
            "result": {"success": False, "error": "This file is not a valid .safetensors file."},
        }
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert "This file is not a valid .safetensors file." in env["error"]["message"]
        assert env["error"]["details"] == {"task_id": "task-1", "status": "failed"}

    def test_task_reads_ride_out_a_brief_5xx(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [_http_error(503), urllib.error.URLError("reset"), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.paths().count("/api/tasks/task-1") == 3

    def test_task_reads_give_up_after_the_window(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = _http_error(503)
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "cloud_http_error"
        assert env["error"]["details"]["status"] == 503
        assert no_wait[0] >= assets_library._UNANSWERED_WINDOW_SECONDS
        assert no_wait[0] < assets_library._UNANSWERED_WINDOW_SECONDS + 2 * assets_library._TASK_POLL_SECONDS

    def test_a_task_4xx_ends_the_wait_at_once(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = _http_error(404)
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and cloud.paths().count("/api/tasks/task-1") == 1

    def test_a_task_can_take_longer_than_fifteen_minutes(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [{"id": "task-1", "status": "running"}] * 500 + [_DONE]
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert no_wait[0] > 15 * 60

    @pytest.mark.parametrize(
        ("route", "answer"),
        [
            (("POST", "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete"), b"<html>proxy</html>"),
            (("POST", "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete"), {"status": "created"}),
            (("GET", "/api/tasks/task-1"), {"id": "task-1", "status": "completed", "result": {"success": True}}),
            (("POST", "/api/assets/uploads"), {"upload_id": "u", "method": "PUT", "headers": {}}),
        ],
    )
    def test_an_unusable_answer_is_an_error_not_a_success(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, route, answer
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[route] = answer
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["ok"] is False and env["error"]["code"] == "cloud_http_error"

    def test_new_codes_are_registered(self):
        for code in (
            "asset_upload_invalid",
            "asset_upload_refused",
            "asset_upload_failed",
            "asset_upload_timeout",
            "asset_upload_unverified",
            "asset_upload_wrong_workspace",
            "model_upload_needs_confirm",
        ):
            assert error_codes.is_registered(code), code


class _ExpiringCloud(_Cloud):
    """``_Cloud`` that answers 401 to ``Bearer old-token`` from its ``expires_at``-th call on."""

    def __init__(self, routes: dict, expires_at: int):
        super().__init__(routes)
        self.expires_at = expires_at
        self.auth: list[str | None] = []

    def __call__(self, req, timeout=None):
        import urllib.parse

        header = req.get_header("Authorization") or req.get_header("X-api-key")
        self.auth.append(header)
        if header in ("Bearer old-token", "test-key") and len(self.calls) >= self.expires_at:
            self.calls.append((req.get_method(), urllib.parse.urlsplit(req.full_url).path, None))
            raise _http_error(401)
        return super().__call__(req, timeout)


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch):
    """A stored sign-in for https://cloud.example.com whose token the server stops taking mid-upload.

    Its clock says the token is still good, so only a forced refresh (after a
    401) swaps in ``new-token``; ``forced`` counts those refreshes.
    """
    from comfy_cli.target import Target

    for var in ("COMFY_CLOUD_AUTH_TOKEN", "COMFY_CLOUD_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    state = SimpleNamespace(
        session=SimpleNamespace(access_token="old-token", base_url="https://cloud.example.com"), forced=0
    )

    def get_session(*, refresh=True, force=False, allow_clear=True, expect=None, rejected_access_token=None):
        if force:
            state.forced += 1
            state.session = SimpleNamespace(access_token="new-token", base_url="https://cloud.example.com")
        return state.session

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)
    target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        auth_token="old-token",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: target)
    return state


_COMPLETE_1 = "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete"


class TestUploadOutlastsTheToken:
    """A stored sign-in's token can expire during a long PUT.

    Completion and polling refresh it and carry on, never resending the file.
    """

    def test_a_401_on_complete_refreshes_the_sign_in_and_completes(
        self, signed_in, monkeypatch, capsys, model_file, no_wait
    ):
        cloud = _ExpiringCloud(dict(_UPLOAD_ROUTES), expires_at=3)  # features, lookup, start, then expired
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert signed_in.forced == 1
        assert len(storage.puts) == 1
        assert cloud.paths().count(_COMPLETE_1) == 2
        assert cloud.auth[3:] == ["Bearer old-token"] + ["Bearer new-token"] * 4

    def test_a_401_on_a_task_poll_refreshes_the_sign_in_and_keeps_waiting(
        self, signed_in, monkeypatch, capsys, model_file, no_wait
    ):
        cloud = _ExpiringCloud(dict(_UPLOAD_ROUTES), expires_at=4)  # expires after complete
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert signed_in.forced == 1
        assert len(storage.puts) == 1 and cloud.paths().count(_COMPLETE_1) == 1
        assert cloud.paths().count("/api/tasks/task-1") == 3

    def test_an_api_key_is_not_refreshed_and_its_401_surfaces(
        self, cloud_target, signed_in, monkeypatch, capsys, model_file, no_wait
    ):
        monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: cloud_target)
        cloud = _ExpiringCloud(dict(_UPLOAD_ROUTES), expires_at=3)
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 1 and env["error"]["code"] == "cloud_unauthorized"
        assert signed_in.forced == 0
        assert len(storage.puts) == 1 and cloud.paths().count(_COMPLETE_1) == 1

    def test_a_token_handed_in_by_env_is_not_refreshed_even_beside_a_matching_sign_in(
        self, signed_in, monkeypatch, capsys, model_file, no_wait
    ):
        monkeypatch.setenv("COMFY_CLOUD_AUTH_TOKEN", "old-token")
        cloud = _ExpiringCloud(dict(_UPLOAD_ROUTES), expires_at=3)
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 1 and env["error"]["code"] == "cloud_unauthorized"
        assert signed_in.forced == 0
        assert cloud.paths().count(_COMPLETE_1) == 1


def _jwt(account: str, n: int = 1) -> str:
    """A Comfy Cloud access token for ``account``'s user and workspace; ``n`` tells its refreshes apart."""

    def part(claims: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()

    claims = {"sub": f"user-{account}", "workspace_id": f"ws-{account}", "jti": f"{account}-{n}"}
    return f"{part({'alg': 'ES256', 'typ': 'JWT'})}.{part(claims)}.signature"


def _account_of(header: str | None) -> str:
    if header == "test-key":
        return "a"
    payload = (header or "").removeprefix("Bearer ").split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["sub"].removeprefix("user-")


@pytest.fixture
def stored_login(monkeypatch: pytest.MonkeyPatch):
    """A stored sign-in as account A for https://cloud.example.com.

    ``login`` is the ``(account, n)`` the store holds: another shell signing in
    replaces it. A forced refresh (after a 401) rotates whichever sign-in is
    stored then, keeping its account; ``forced`` counts those refreshes.
    """
    from comfy_cli.target import Target

    for var in ("COMFY_CLOUD_AUTH_TOKEN", "COMFY_CLOUD_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    state = SimpleNamespace(login=("a", 1), forced=0)

    def get_session(*, refresh=True, force=False, allow_clear=True, expect=None, rejected_access_token=None):
        if force:
            state.forced += 1
            state.login = (state.login[0], state.login[1] + 1)
        return SimpleNamespace(access_token=_jwt(*state.login), base_url="https://cloud.example.com")

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)
    target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        auth_token=_jwt("a"),
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: target)
    return state


class _AccountCloud(_Cloud):
    """``_Cloud`` that files the upload under account A, as Comfy Cloud does.

    During call number ``lands_during`` another shell's sign-in lands: the store
    then holds ``lands``. A's first token is refused with a 401 from call number
    ``a_expires_from`` on. A request as any other account gets a 404, which is
    what Cloud answers B for A's staged upload or A's task.
    """

    def __init__(self, routes: dict, login, *, lands=("b", 1), lands_during=None, a_expires_from=None):
        super().__init__(routes)
        self.login, self.lands, self.lands_during, self.a_expires_from = login, lands, lands_during, a_expires_from
        self.auth: list[str | None] = []

    def __call__(self, req, timeout=None):
        import urllib.parse

        index = len(self.calls)
        header = req.get_header("Authorization") or req.get_header("X-api-key")
        self.auth.append(header)
        if index == self.lands_during:
            self.login.login = self.lands
        refusal = None
        if self.a_expires_from is not None and index >= self.a_expires_from and header == f"Bearer {_jwt('a')}":
            refusal = _http_error(401)
        elif _account_of(header) != "a":
            refusal = _cloud_error(404, "NOT_FOUND", "nothing by that id for this account")
        if refusal is not None:
            self.calls.append((req.get_method(), urllib.parse.urlsplit(req.full_url).path, None))
            raise refusal
        return super().__call__(req, timeout)


class TestUploadKeepsItsAccount:
    """Another shell signing in to another account mid-upload does not take the upload over.

    Comfy Cloud files the staged bytes and the task under the user and workspace
    that started them, so completing or reading them as anyone else gets a 404.
    """

    @pytest.mark.parametrize(
        "lands_during", [pytest.param(2, id="before-complete"), pytest.param(4, id="during-polling")]
    )
    def test_another_login_does_not_change_the_upload_owner(
        self, stored_login, monkeypatch, capsys, model_file, no_wait, lands_during
    ):
        cloud = _AccountCloud(dict(_UPLOAD_ROUTES), stored_login, lands_during=lands_during)
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.auth == [f"Bearer {_jwt('a')}"] * 7
        assert stored_login.login == ("b", 1) and stored_login.forced == 0
        assert len(storage.puts) == 1

    @pytest.mark.parametrize(
        ("lands_during", "a_expires_from", "operation"),
        [pytest.param(2, 3, "complete", id="at-complete"), pytest.param(4, 5, "task", id="during-polling")],
    )
    def test_an_expired_token_with_another_login_stored_ends_with_a_clear_error(
        self, stored_login, monkeypatch, capsys, model_file, no_wait, lands_during, a_expires_from, operation
    ):
        cloud = _AccountCloud(
            dict(_UPLOAD_ROUTES), stored_login, lands_during=lands_during, a_expires_from=a_expires_from
        )
        env, rc, _, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud
        )

        assert rc == 1 and env["error"]["code"] == "cloud_unauthorized"
        assert env["error"]["details"] == {"status": 401, "operation": operation, "reason": "account_replaced"}
        assert "another account" in env["error"]["message"]
        # B's token was never sent, and B's sign-in was never refreshed on A's behalf.
        assert set(cloud.auth) == {f"Bearer {_jwt('a')}"}
        assert stored_login.login == ("b", 1) and stored_login.forced == 0
        assert len(storage.puts) == 1

    def test_the_same_login_refreshed_in_another_shell_is_picked_up(
        self, stored_login, monkeypatch, capsys, model_file, no_wait
    ):
        cloud = _AccountCloud(dict(_UPLOAD_ROUTES), stored_login, lands=("a", 2), lands_during=2, a_expires_from=3)
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud)

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.auth[3:] == [f"Bearer {_jwt('a', 2)}"] * 4
        assert stored_login.forced == 0

    def test_the_same_login_is_refreshed_after_a_401(self, stored_login, monkeypatch, capsys, model_file, no_wait):
        cloud = _AccountCloud(dict(_UPLOAD_ROUTES), stored_login, a_expires_from=3)
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud)

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.auth[3:] == [f"Bearer {_jwt('a')}"] + [f"Bearer {_jwt('a', 2)}"] * 4
        assert stored_login.forced == 1

    @pytest.mark.parametrize("explicit", ["env-token", "api-key"])
    def test_an_explicit_credential_is_sent_unchanged_when_another_login_lands(
        self, cloud_target, stored_login, monkeypatch, capsys, model_file, no_wait, explicit
    ):
        # `cloud_target` first, so the sign-in's target is the one resolved unless the key case swaps it back.
        if explicit == "env-token":
            monkeypatch.setenv("COMFY_CLOUD_AUTH_TOKEN", _jwt("a"))
            sent = f"Bearer {_jwt('a')}"
        else:
            monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: cloud_target)
            sent = "test-key"
        cloud = _AccountCloud(dict(_UPLOAD_ROUTES), stored_login, lands_during=2)
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [200], cloud=cloud)

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.auth == [sent] * 7
        assert stored_login.forced == 0


class TestTruncatedResponses:
    def test_a_truncated_task_poll_is_retried(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [http.client.IncompleteRead(b'{"id": "ta', 40), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.paths().count("/api/tasks/task-1") == 2

    def test_a_truncated_complete_answer_is_an_error_envelope_not_a_crash(
        self, cloud_target, monkeypatch, capsys, model_file
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("POST", _COMPLETE_1)] = http.client.IncompleteRead(b'{"task_', 30)
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "cloud_http_error"
        assert env["error"]["details"] == {"operation": "complete"}

    def test_a_task_poll_cut_short_of_its_content_length_is_retried(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [lambda: _wire(b'{"id": "task-1", "sta', declared=100), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.paths().count("/api/tasks/task-1") == 2

    def test_a_chunked_task_poll_cut_short_is_retried(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [lambda: _wire(b'{"id": "task-1", "sta', chunked=100), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.paths().count("/api/tasks/task-1") == 2

    def test_a_whole_malformed_task_answer_still_ends_the_wait_at_once(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [lambda: _wire(b'{"id": "task-1", "sta'), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "cloud_http_error"
        assert "not valid JSON" in env["error"]["message"]
        assert cloud.paths().count("/api/tasks/task-1") == 1


class _WireSocket:
    def __init__(self, wire: bytes):
        self._wire = wire

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self._wire)


def _wire(body: bytes, *, declared: int | None = None, chunked: int | None = None) -> http.client.HTTPResponse:
    """A real ``http.client`` response to ``body`` as sent on the wire, the connection closing after it.

    ``declared`` sets the Content-Length (default: the body's length). ``chunked``
    sends the body as one chunk announced as that many bytes.
    """
    if chunked is not None:
        head, payload = "Transfer-Encoding: chunked", f"{chunked:x}\r\n".encode() + body
    else:
        head, payload = f"Content-Length: {len(body) if declared is None else declared}", body
    resp = http.client.HTTPResponse(_WireSocket(f"HTTP/1.1 200 OK\r\n{head}\r\n\r\n".encode() + payload))
    resp.begin()
    return resp


class TestBodiesCutShort:
    """A body the connection cut short is a transport failure, never a malformed answer.

    A chunked body raises ``IncompleteRead`` from ``read``; a Content-Length body
    read with a size returns the bytes that arrived, so ``http_request`` checks
    them against the declared length.
    """

    def test_a_body_cut_short_of_its_content_length_raises_incomplete_read(self, cloud_target, monkeypatch):
        from comfy_cli.command import cloud_http

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(b'{"assets": [', declared=40))
        with pytest.raises(http.client.IncompleteRead) as caught:
            cloud_http.http_request(cloud_target.url("assets"), cloud_target, strict_json=True)
        assert caught.value.partial == b'{"assets": [' and caught.value.expected == 28

    def test_a_body_the_size_cap_stops_is_not_called_cut_short(self, cloud_target, monkeypatch):
        from comfy_cli.command import cloud_http

        monkeypatch.setattr(cloud_http, "_MAX_BODY_BYTES", 8)
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(b'{"assets": []}'))
        with pytest.raises(cloud_http.ResponseUnparseable):
            cloud_http.http_request(cloud_target.url("assets"), cloud_target, strict_json=True)

    @pytest.mark.parametrize("strict", [True, False])
    def test_a_body_past_the_size_cap_is_never_parsed(self, cloud_target, monkeypatch, strict):
        from comfy_cli.command import cloud_http

        # The first 8 bytes are a valid JSON object plus whitespace; the rest is not.
        monkeypatch.setattr(cloud_http, "_MAX_BODY_BYTES", 8)
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(b'{}      , "assets": [1]}'))
        if strict:
            with pytest.raises(cloud_http.ResponseUnparseable):
                cloud_http.http_request(cloud_target.url("assets"), cloud_target, strict_json=True)
        else:
            assert cloud_http.http_request(cloud_target.url("assets"), cloud_target) == (200, None)

    def test_a_chunked_body_past_the_size_cap_is_never_parsed(self, cloud_target, monkeypatch):
        from comfy_cli.command import cloud_http

        # No Content-Length: only reading past the cap shows the body is longer.
        body = b'{}      , "assets": [1]}'
        monkeypatch.setattr(cloud_http, "_MAX_BODY_BYTES", 8)
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(body, chunked=len(body)))
        with pytest.raises(cloud_http.ResponseUnparseable):
            cloud_http.http_request(cloud_target.url("assets"), cloud_target, strict_json=True)

    @pytest.mark.parametrize(
        "cut", [pytest.param({"declared": 40}, id="content-length"), pytest.param({"chunked": 40}, id="chunked")]
    )
    def test_a_listing_cut_short_is_cloud_http_error_not_a_crash(self, cloud_target, monkeypatch, capsys, cut):
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(b'{"assets": [', **cut))
        env, rc = _run_with_rc(["ls", "--where", "cloud"], capsys)

        assert rc == 1 and env["error"]["code"] == "cloud_http_error"

    @pytest.mark.parametrize(
        "cut", [pytest.param({"declared": 40}, id="content-length"), pytest.param({"chunked": 40}, id="chunked")]
    )
    def test_an_ensure_answer_cut_short_is_cloud_http_error_not_a_crash(self, cloud_target, monkeypatch, capsys, cut):
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _wire(b'{"id": "asse', **cut))
        env, rc = _run_with_rc(["ensure", "--hash", "a" * 64, "--where", "cloud"], capsys)

        assert rc == 1 and env["error"]["code"] == "cloud_http_error"


_NOT_ROOT = pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: -1)() == 0,
    reason="needs a file the user cannot read (Windows and root ignore the mode)",
)


def _then(action, answer):
    """A route outcome that runs ``action`` before ``_Cloud`` answers with ``answer``."""

    def outcome():
        action()
        return _Cloud({("GET", "/x"): answer})(SimpleNamespace(get_method=lambda: "GET", full_url="/x", data=None))

    return outcome


class TestUploadFileErrors:
    """A file that cannot be read is an upload error envelope, never a traceback."""

    @_NOT_ROOT
    def test_an_unreadable_file_is_refused_locally(self, model_file):
        # From the command line, click's `Path` check refuses a file the user
        # cannot read before the command runs; this is the command's own check.
        model_file.chmod(0)
        try:
            refusal = assets_library._local_refusal(model_file)
        finally:
            model_file.chmod(0o600)

        assert refusal is not None and refusal[0] == "file_unreadable"
        assert "Permission denied" in refusal[1]

    def test_a_file_whose_stat_fails_is_refused_before_any_request(self, cloud_target, monkeypatch, capsys, model_file):
        def stat(self, *a, **kw):
            raise OSError(5, "Input/output error", str(self))

        monkeypatch.setattr(Path, "is_file", lambda self: True)
        monkeypatch.setattr(Path, "stat", stat)
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], {}, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == "file_unreadable"
        assert "Input/output error" in env["error"]["message"]
        assert cloud.calls == [] and storage.puts == []

    def test_a_file_removed_before_its_size_is_read_is_refused_before_the_upload(
        self, cloud_target, monkeypatch, capsys, model_file
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/features")] = _then(model_file.unlink, _FOLDERS)
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == "file_not_found"
        assert cloud.paths() == ["/api/features"] and storage.puts == []

    def test_a_file_removed_before_the_put_fails_without_a_retry(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("POST", "/api/assets/uploads")] = [_then(model_file.unlink, _started(1)), _started(2)]
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200, 200]
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert env["error"]["details"]["reason"] == "file_unreadable"
        assert env["error"]["details"]["attempts"] == 1
        assert storage.puts == [] and cloud.paths("POST").count("/api/assets/uploads") == 1
        assert "SECRETSIG" not in everything

    @_NOT_ROOT
    def test_a_file_made_unreadable_before_the_put_fails_without_a_retry(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("POST", "/api/assets/uploads")] = [_then(lambda: model_file.chmod(0), _started(1)), _started(2)]
        try:
            env, rc, cloud, storage, _ = _upload(
                monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200, 200]
            )
        finally:
            model_file.chmod(0o600)

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert env["error"]["details"]["reason"] == "file_unreadable"
        assert storage.puts == [] and cloud.paths("POST").count("/api/assets/uploads") == 1

    @pytest.mark.parametrize("wrapped", [True, False], ids=["transport-wraps-it", "raised-as-is"])
    def test_a_read_error_mid_put_is_not_retried_as_a_transport_failure(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, wrapped
    ):
        import requests

        from comfy_cli import builder_api

        def failing_read(self, size=-1):
            raise OSError(5, "Input/output error")

        class _Wrapping(_Storage):
            def __call__(self, url, data=None, **kw):
                try:
                    return super().__call__(url, data=data, **kw)
                except OSError as e:  # urllib3 reports a body that fails to read as a dropped connection
                    raise requests.ConnectionError(("Connection aborted.", e)) from e

        monkeypatch.setattr(builder_api._CountingReader, "read", failing_read)
        storage = _Wrapping([200, 200]) if wrapped else None
        env, rc, cloud, storage, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [200, 200], storage=storage
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert env["error"]["details"]["reason"] == "file_unreadable"
        assert "Input/output error" in env["error"]["message"]
        assert cloud.paths("POST").count("/api/assets/uploads") == 1
        assert "SECRETSIG" not in everything

    def test_requests_sends_the_body_with_its_content_length_not_chunked(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        # A presigned storage PUT refuses a chunked body, so the length Requests
        # takes from the reader must be the file's.
        import requests

        class _Preparing(_Storage):
            prepared: dict | None = None

            def __call__(self, url, data=None, headers=None, **kw):
                self.prepared = dict(requests.Request("PUT", url, data=data, headers=headers).prepare().headers)
                return super().__call__(url, data=data, headers=headers, **kw)

        storage = _Preparing([200])
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], dict(_UPLOAD_ROUTES), [200], storage=storage
        )

        assert rc == 0 and env["ok"] is True
        assert storage.prepared["Content-Length"] == str(model_file.stat().st_size)
        assert "Transfer-Encoding" not in storage.prepared
        assert storage.puts[0]["body"] == model_file.read_bytes()

    def test_a_missing_ca_bundle_ends_the_upload_with_a_tls_setup_error(
        self, cloud_target, monkeypatch, model_file, tmp_path, no_wait
    ):
        # The real Requests adapter: with REQUESTS_CA_BUNDLE naming a removed
        # file, it raises a plain OSError while setting up TLS, before any
        # connection or read of the body. Run from the root CLI, that is a
        # handled upload failure carrying the certificate message: neither a
        # crash (`internal_error`) nor a fault in the model (`file_unreadable`).
        import requests

        from comfy_cli.cmdline import app as root_app

        real_put = requests.put

        class _RealPut(_Storage):
            attempts = 0

            def __call__(self, url, **kw):
                self.attempts += 1
                return real_put(url, **kw)

        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "removed-ca-bundle.pem"))
        cloud, storage = _Cloud(dict(_UPLOAD_ROUTES)), _RealPut([])
        monkeypatch.setattr("urllib.request.urlopen", cloud)
        monkeypatch.setattr("requests.put", storage)
        result = CliRunner().invoke(
            root_app,
            ["--json", "assets", "library", "upload", str(model_file), "--type", "loras", "--where", "cloud"],
            env={"NO_COLOR": "1", "COLUMNS": "400"},
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert lines, f"no envelope on stdout (rc={result.exit_code}, exc={result.exception!r})"
        env = json.loads(lines[-1])

        assert result.exit_code == 1 and env["ok"] is False
        assert env["error"]["code"] == "asset_upload_failed", env["error"]
        assert env["error"]["details"]["reason"] == "tls_setup"
        assert env["error"]["details"]["attempts"] == 1
        assert "CA certificate bundle" in env["error"]["message"]
        assert storage.attempts == 1
        assert cloud.paths("POST").count("/api/assets/uploads") == 1
        assert "SECRETSIG" not in result.output


def _throttled(retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("https://cloud.example.com/api/tasks/task-1", 429, "slow down", headers, io.BytesIO())


@pytest.fixture
def sleeps(monkeypatch, no_wait):
    taken: list[float] = []

    def sleep(s):
        taken.append(s)
        no_wait[0] += s

    monkeypatch.setattr(assets_library, "_sleep", sleep)
    return taken


class TestTaskPollThrottled:
    def test_a_429_is_retried(self, cloud_target, monkeypatch, capsys, model_file, sleeps):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [_throttled(), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert cloud.paths().count("/api/tasks/task-1") == 2
        assert sleeps == [assets_library._TASK_POLL_SECONDS]

    def test_a_429_waits_its_retry_after(self, cloud_target, monkeypatch, capsys, model_file, sleeps):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [_throttled("7"), _DONE]
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert sleeps == [7.0]

    def test_a_retry_after_past_the_window_is_cut_to_the_window(
        self, cloud_target, monkeypatch, capsys, model_file, sleeps
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = [_throttled("3600"), _DONE]
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["id"] == "asset-9"
        assert sleeps == [assets_library._UNANSWERED_WINDOW_SECONDS]

    @pytest.mark.parametrize("retry_after", [None, "3600"])
    def test_429s_past_the_window_end_the_wait(
        self, cloud_target, monkeypatch, capsys, model_file, sleeps, no_wait, retry_after
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/tasks/task-1")] = _throttled(retry_after)
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "cloud_rate_limited"
        assert env["error"]["details"]["status"] == 429
        assert no_wait[0] >= assets_library._UNANSWERED_WINDOW_SECONDS
        assert no_wait[0] < assets_library._UNANSWERED_WINDOW_SECONDS + 2 * assets_library._TASK_POLL_SECONDS


class _BodyCutOff(io.RawIOBase):
    """An error response whose body the server stopped sending partway."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def readable(self):
        return True

    def read(self, n=-1):
        raise self.exc


class TestErrorBodyCutOff:
    @pytest.mark.parametrize(
        ("status", "exc", "code"),
        [
            (401, http.client.IncompleteRead(b'{"err', 40), "cloud_unauthorized"),
            (403, ConnectionResetError(54, "Connection reset by peer"), "cloud_unauthorized"),
            (500, http.client.IncompleteRead(b"<ht", 40), "cloud_http_error"),
        ],
    )
    def test_the_status_is_classified_when_the_error_body_is_cut_off(
        self, cloud_target, monkeypatch, capsys, status, exc, code
    ):
        err = urllib.error.HTTPError("https://cloud.example.com/api/assets", status, "err", {}, _BodyCutOff(exc))
        _patch_urlopen(monkeypatch, err)
        env, rc = _run_with_rc(["ls"], capsys)

        assert rc == 1 and env["error"]["code"] == code
        assert env["error"]["details"]["status"] == status


# --- the upload checks the bytes, the name and the workspace ------------------


def _row(name: str = "style.safetensors", hash_: str = _MODEL_HASH, asset_id: str = "asset-3") -> dict:
    return {"id": asset_id, "name": name, "hash": hash_, "tags": ["models", "loras"], "loader_path": name}


class _UrlCloud(_Cloud):
    """``_Cloud`` that also keeps each request's full URL."""

    def __init__(self, routes: dict):
        super().__init__(routes)
        self.urls: list[str] = []

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        return super().__call__(req, timeout)


class TestUploadVerifies:
    def test_a_dry_run_reports_the_hashes_and_sends_nothing(self, cloud_target, monkeypatch, capsys, model_file):
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras", "--dry-run"], dict(_UPLOAD_ROUTES), []
        )

        assert rc == 0 and env["ok"] is True and env["changed"] is False
        assert env["data"] == {
            "id": None,
            "name": "style.safetensors",
            "folder": "loras",
            # No record exists yet, so Comfy Cloud has no loader_path to give.
            "loader_path": None,
            "hash": _MODEL_HASH,
            "sha256": _MODEL_SHA256,
            "size": 16,
            "workspace_id": None,
            "task_id": None,
            "uploaded": False,
            "already_present": False,
            "dry_run": True,
        }
        assert cloud.paths() == ["/api/features", "/api/assets"] and storage.puts == []

    def test_the_lookup_asks_for_the_models_of_that_name_in_that_folder(
        self, cloud_target, monkeypatch, capsys, model_file
    ):
        import urllib.parse

        cloud = _UrlCloud(dict(_UPLOAD_ROUTES))
        _upload(monkeypatch, capsys, [str(model_file), "--type", "loras", "--dry-run"], {}, [], cloud=cloud)

        query = urllib.parse.parse_qs(urllib.parse.urlsplit(cloud.urls[1]).query)
        assert query == {
            "include_tags": ["models,loras"],
            "name_contains": ["style.safetensors"],
            # The workspace's library, every member's models, is what a run
            # loads from; the public catalog is not.
            "workspace_library": ["true"],
            "include_public": ["false"],
            "limit": ["500"],
        }

    @pytest.mark.parametrize("dry_run", [False, True])
    def test_the_same_bytes_under_the_name_are_not_uploaded_again(
        self, cloud_target, monkeypatch, capsys, model_file, dry_run
    ):
        routes = dict(_UPLOAD_ROUTES)
        stored_form = _MODEL_HASH.removeprefix("blake3:").upper() + ".safetensors"
        routes[("GET", "/api/assets")] = {
            # A longer name that only contains the file's name is not the file's.
            "assets": [_row(hash_=stored_form), _row("old-style.safetensors", "blake3:" + "ab" * 32, "asset-4")],
            "has_more": False,
        }
        args = [str(model_file), "--type", "loras"] + (["--dry-run"] if dry_run else [])
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, args, routes, [])

        assert rc == 0 and env["changed"] is False
        assert env["data"]["id"] == "asset-3" and env["data"]["loader_path"] == "style.safetensors"
        assert env["data"]["already_present"] is True and env["data"]["uploaded"] is False
        assert env["data"]["dry_run"] is dry_run
        assert cloud.paths() == ["/api/features", "/api/assets"] and storage.puts == []

    def test_a_name_already_holding_other_bytes_is_refused(self, cloud_target, monkeypatch, capsys, model_file):
        routes = dict(_UPLOAD_ROUTES)
        other = "blake3:" + "ab" * 32
        routes[("GET", "/api/assets")] = {"assets": [_row(hash_=other), _row()], "has_more": False}
        args = [str(model_file), "--type", "loras", "--yes"]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == "name_in_use"
        assert env["error"]["details"]["hash"] == _MODEL_HASH
        assert env["error"]["details"]["existing"] == [{"id": "asset-3", "hash": other}]
        assert cloud.paths() == ["/api/features", "/api/assets"] and storage.puts == []

    def test_a_row_with_no_hash_counts_as_other_bytes(self, cloud_target, monkeypatch, capsys, model_file):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets")] = {"assets": [_row(hash_=None)], "has_more": False}
        args = [str(model_file), "--type", "loras", "--yes"]
        env, rc, _, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 1 and env["error"]["details"]["reason"] == "name_in_use" and storage.puts == []

    def test_assets_library_upload_still_uploads_a_new_version_of_a_name(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        # A script that replaces a model by uploading new bytes under its name keeps working.
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets")] = {"assets": [_row(hash_="blake3:" + "ab" * 32)], "has_more": False}
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["ok"] is True and env["changed"] is True
        assert env["data"]["id"] == "asset-9" and env["data"]["uploaded"] is True
        assert env["data"]["already_present"] is False
        (put,) = storage.puts
        assert put["body"] == model_file.read_bytes()
        assert "/api/assets/uploads/00000000-0000-0000-0000-000000000001/complete" in cloud.paths()

    def test_assets_library_upload_sends_the_same_bytes_again_over_a_newer_version(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        # The name also holds other bytes, so only a new upload makes this
        # file the one the name loads; it is not "already present".
        routes = dict(_UPLOAD_ROUTES)
        other = "blake3:" + "ab" * 32
        routes[("GET", "/api/assets")] = {"assets": [_row(hash_=other, asset_id="asset-4"), _row()], "has_more": False}
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["uploaded"] is True and env["data"]["already_present"] is False
        (put,) = storage.puts
        assert put["body"] == model_file.read_bytes()

    def test_a_lookup_too_long_to_settle_the_name_is_refused_for_model_upload(
        self, cloud_target, monkeypatch, capsys, model_file
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets")] = {"assets": [], "has_more": True}
        args = [str(model_file), "--type", "loras", "--yes"]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == "name_unverified"
        assert "/api/assets/uploads" not in cloud.paths() and storage.puts == []

    def test_assets_library_upload_uploads_when_the_lookup_is_too_long_to_settle_the_name(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets")] = {"assets": [_row()], "has_more": True}
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["uploaded"] is True and env["data"]["already_present"] is False
        (put,) = storage.puts
        assert put["body"] == model_file.read_bytes()

    def test_an_unusable_lookup_is_an_error_not_a_go_ahead(self, cloud_target, monkeypatch, capsys, model_file):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets")] = {"assets": "none"}
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [])

        assert rc == 1 and env["error"]["code"] == "cloud_http_error" and storage.puts == []

    def test_a_file_changed_after_it_was_hashed_sends_none_of_the_change(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        # Same size, other bytes: written after the hash and before the PUT reads it.
        rewrite = _then(lambda: model_file.write_bytes(b"fedcba9876543210"), _started(1))
        routes[("POST", "/api/assets/uploads")] = [rewrite, _started(2)]
        storage = _SentSoFar([200, 200])
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=storage
        )

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        details = env["error"]["details"]
        assert details["reason"] == "file_changed" and details["hash"] == _MODEL_HASH
        assert details["attempts"] == 1
        # The one block is checked before any of it is sent.
        assert [p["body"] for p in storage.puts] == [b""]
        assert cloud.paths("POST").count("/api/assets/uploads") == 1
        assert not any(p.endswith("/complete") for p in cloud.paths())

    def test_only_the_blocks_before_a_changed_one_are_sent(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        from comfy_cli import hashing

        monkeypatch.setattr(hashing, "_CHUNK_SIZE", 4)  # blocks 0123 4567 89ab cdef
        routes = dict(_UPLOAD_ROUTES)
        rewrite = _then(lambda: model_file.write_bytes(b"01234567XXXXcdef"), _started(1))
        routes[("POST", "/api/assets/uploads")] = [rewrite, _started(2)]
        storage = _SentSoFar([200, 200])
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=storage
        )

        assert rc == 1 and env["error"]["details"]["reason"] == "file_changed"
        assert [p["body"] for p in storage.puts] == [b"01234567"]
        assert not any(p.endswith("/complete") for p in cloud.paths())

    def test_a_file_cut_short_after_it_was_hashed_is_not_completed(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        from comfy_cli import hashing

        monkeypatch.setattr(hashing, "_CHUNK_SIZE", 4)
        routes = dict(_UPLOAD_ROUTES)
        routes[("POST", "/api/assets/uploads")] = [_then(lambda: model_file.write_bytes(b"0123456789"), _started(1))]
        storage = _SentSoFar([200])
        env, rc, cloud, storage, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=storage
        )

        assert rc == 1 and env["error"]["details"]["reason"] == "file_changed"
        assert [p["body"] for p in storage.puts] == [b"01234567"]
        assert not any(p.endswith("/complete") for p in cloud.paths())

    def test_bytes_added_after_the_hash_are_not_sent(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = dict(_UPLOAD_ROUTES)
        grow = _then(lambda: model_file.write_bytes(_MODEL_BYTES + b"tail"), _started(1))
        routes[("POST", "/api/assets/uploads")] = [grow, _started(2)]
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        # The hashed bytes are whole, so they are the upload; the added tail stays here.
        assert rc == 0 and env["data"]["hash"] == _MODEL_HASH and env["data"]["size"] == len(_MODEL_BYTES)
        assert [p["body"] for p in storage.puts] == [_MODEL_BYTES]

    def test_a_stored_hash_that_is_not_the_files_fails_the_upload(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        wrong = "blake3:" + "cd" * 32
        routes[("GET", "/api/tasks/task-1")] = {**_DONE, "result": {**_DONE["result"], "hash": wrong}}
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "asset_upload_unverified"
        assert env["error"]["details"] == {
            "reason": "hash_mismatch",
            "asset_id": "asset-9",
            "task_id": "task-1",
            "hash": _MODEL_HASH,
            "stored_hash": wrong,
        }

    def test_a_task_with_no_hash_is_checked_against_the_stored_asset(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = dict(_UPLOAD_ROUTES)
        result = {k: v for k, v in _DONE["result"].items() if k != "hash"}
        routes[("GET", "/api/tasks/task-1")] = {**_DONE, "result": result}
        routes[("GET", "/api/assets/asset-9")] = {
            "id": "asset-9",
            "hash": _MODEL_HASH.removeprefix("blake3:"),
            "loader_path": "style.safetensors",
        }
        env, rc, cloud, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["uploaded"] is True and env["data"]["loader_path"] == "style.safetensors"
        assert cloud.paths()[-1] == "/api/assets/asset-9"
        assert cloud.paths().count("/api/assets/asset-9") == 1

    @pytest.mark.parametrize("answer", [_http_error(500), b"not json"], ids=["asset-unreadable", "asset-unparseable"])
    def test_a_record_that_cannot_be_read_after_a_verified_upload_leaves_loader_path_null(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, answer
    ):
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets/asset-9")] = answer
        env, rc, cloud, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        # The task's hash already matched, so the bytes are verified; only the loader_path is missing.
        assert rc == 0 and env["ok"] is True and env["data"]["uploaded"] is True
        assert env["data"]["id"] == "asset-9" and env["data"]["loader_path"] is None
        assert cloud.paths()[-1] == "/api/assets/asset-9" and len(storage.puts) == 1

    @pytest.mark.parametrize("present", [False, True], ids=["uploaded", "already-present"])
    def test_a_record_without_loader_path_reports_null_not_the_file_name(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, present
    ):
        import jsonschema

        # Comfy Cloud gives no loader_path when the name does not form a valid model path.
        routes = dict(_UPLOAD_ROUTES)
        routes[("GET", "/api/assets/asset-9")] = {"id": "asset-9", "hash": _MODEL_HASH, "loader_path": None}
        if present:
            row = {k: v for k, v in _row().items() if k != "loader_path"}
            routes[("GET", "/api/assets")] = {"assets": [row], "has_more": False}
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 0 and env["data"]["loader_path"] is None
        assert env["data"]["already_present"] is present and env["data"]["uploaded"] is not present
        jsonschema.Draft202012Validator(_model_schema()).validate(env["data"])

    @pytest.mark.parametrize(
        "answer", [_http_error(500), {"id": "asset-9"}], ids=["asset-unreadable", "asset-without-hash"]
    )
    def test_a_stored_hash_that_cannot_be_read_fails_the_upload(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, answer
    ):
        routes = dict(_UPLOAD_ROUTES)
        result = {k: v for k, v in _DONE["result"].items() if k != "hash"}
        routes[("GET", "/api/tasks/task-1")] = {**_DONE, "result": result}
        routes[("GET", "/api/assets/asset-9")] = answer
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [200])

        assert rc == 1 and env["error"]["code"] == "asset_upload_unverified"
        assert env["error"]["details"]["reason"] == "hash_unavailable"
        assert env["error"]["details"]["stored_hash"] is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (_MODEL_HASH, _MODEL_HASH),
            (_MODEL_HASH.removeprefix("blake3:").upper(), _MODEL_HASH),
            (_MODEL_HASH.removeprefix("blake3:") + ".safetensors", _MODEL_HASH),
            ("blake3:abc", None),
            (None, None),
            (42, None),
        ],
    )
    def test_content_hash_reads_every_form_comfy_cloud_reports(self, value, expected):
        assert assets_library._content_hash(value) == expected


# --- comfy model upload -------------------------------------------------------

_WORKSPACE = {"id": "ws-1", "name": "Studio", "type": "team", "auth_method": "api_key"}


def _model_upload(monkeypatch, capsys, args: list[str], routes: dict, puts: list, **kw):
    from comfy_cli.command.models import models

    return _upload(monkeypatch, capsys, args, routes, puts, app=models.app, **kw)


@pytest.fixture
def agent_caller(monkeypatch):
    """The caller is an agent: nothing can answer a prompt."""
    monkeypatch.setattr(
        "comfy_cli.interaction.detect_caller", lambda: Caller(kind="agent", agentic=True, source_env="CLAUDECODE")
    )
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: True)


def _model_schema() -> dict:
    path = Path(assets_library.__file__).resolve().parents[1] / "schemas" / "model_upload.json"
    return json.loads(path.read_text())


class TestModelUpload:
    def test_the_dry_run_names_the_workspace_and_the_sha256_to_approve(
        self, cloud_target, monkeypatch, capsys, model_file, agent_caller
    ):
        import jsonschema

        routes = {**_UPLOAD_ROUTES, ("GET", "/api/workspaces/current"): _WORKSPACE}
        args = [str(model_file), "--type", "loras", "--workspace", "ws-1", "--dry-run"]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 0 and env["command"] == "model upload"
        data = env["data"]
        assert data["sha256"] == _MODEL_SHA256 and data["hash"] == _MODEL_HASH
        assert data["workspace_id"] == "ws-1" and data["loader_path"] is None
        assert data["dry_run"] is True and data["uploaded"] is False and data["already_present"] is False
        jsonschema.Draft202012Validator(_model_schema()).validate(data)
        assert cloud.paths() == ["/api/workspaces/current", "/api/features", "/api/assets"]
        assert storage.puts == []

    def test_the_approved_upload_runs_and_reports_the_loader_path(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, agent_caller
    ):
        import jsonschema

        routes = {**_UPLOAD_ROUTES, ("GET", "/api/workspaces/current"): _WORKSPACE}
        args = [str(model_file), "--type", "loras", "--workspace", "ws-1", "--yes", "--expected-sha256", _MODEL_SHA256]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [200])

        assert rc == 0 and env["changed"] is True
        data = env["data"]
        assert data["id"] == "asset-9" and data["uploaded"] is True and data["task_id"] == "task-1"
        assert data["loader_path"] == "style.safetensors" and data["workspace_id"] == "ws-1"
        jsonschema.Draft202012Validator(_model_schema()).validate(data)
        (put,) = storage.puts
        assert put["body"] == model_file.read_bytes()

    def test_the_sha256_may_carry_a_prefix_and_any_case(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, agent_caller
    ):
        args = [str(model_file), "--type", "loras", "--yes", "--expected-sha256", "sha256:" + _MODEL_SHA256.upper()]
        env, rc, *_ = _model_upload(monkeypatch, capsys, args, dict(_UPLOAD_ROUTES), [200])

        assert rc == 0 and env["data"]["uploaded"] is True

    def test_without_yes_an_agent_is_refused_before_anything_is_sent(
        self, cloud_target, monkeypatch, capsys, model_file, agent_caller
    ):
        args = [str(model_file), "--type", "loras", "--expected-sha256", _MODEL_SHA256]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, dict(_UPLOAD_ROUTES), [])

        assert rc == 1 and env["error"]["code"] == "model_upload_needs_confirm"
        details = env["error"]["details"]
        assert details["missing"] == ["--yes"] and details["sha256"] == _MODEL_SHA256
        assert details["type"] == "loras" and details["file"] == str(model_file)
        assert "/api/assets/uploads" not in cloud.paths() and storage.puts == []

    def test_bytes_other_than_the_approved_ones_are_not_uploaded(
        self, cloud_target, monkeypatch, capsys, model_file, agent_caller
    ):
        args = [str(model_file), "--type", "loras", "--yes", "--expected-sha256", "0" * 64]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, dict(_UPLOAD_ROUTES), [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_invalid"
        assert env["error"]["details"]["reason"] == "file_changed"
        assert env["error"]["details"]["sha256"] == _MODEL_SHA256
        assert env["error"]["details"]["expected_sha256"] == "0" * 64
        assert cloud.paths() == ["/api/features"] and storage.puts == []

    def test_a_file_replaced_after_its_approval_was_checked_sends_none_of_the_new_bytes(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, agent_caller
    ):
        # The sha256 matches when it is checked; another process then replaces
        # the file with same-sized bytes nobody approved, before the PUT.
        routes = dict(_UPLOAD_ROUTES)
        swap = _then(lambda: model_file.write_bytes(b"fedcba9876543210"), _started(1))
        routes[("POST", "/api/assets/uploads")] = [swap, _started(2)]
        storage = _SentSoFar([200, 200])
        args = [str(model_file), "--type", "loras", "--yes", "--expected-sha256", _MODEL_SHA256]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [], storage=storage)

        assert rc == 1 and env["error"]["code"] == "asset_upload_failed"
        assert env["error"]["details"]["reason"] == "file_changed"
        assert [p["body"] for p in storage.puts] == [b""]
        assert not any(p.endswith("/complete") for p in cloud.paths())

    def test_a_malformed_sha256_is_refused_before_any_request(
        self, cloud_target, monkeypatch, capsys, model_file, agent_caller
    ):
        args = [str(model_file), "--type", "loras", "--yes", "--expected-sha256", "abc"]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, dict(_UPLOAD_ROUTES), [])

        assert rc == 1 and env["error"]["details"]["reason"] == "bad_expected_sha256"
        assert cloud.calls == [] and storage.puts == []

    def test_another_workspace_gets_nothing(self, cloud_target, monkeypatch, capsys, model_file, agent_caller):
        routes = {**_UPLOAD_ROUTES, ("GET", "/api/workspaces/current"): {**_WORKSPACE, "id": "ws-2"}}
        args = [str(model_file), "--type", "loras", "--workspace", "ws-1", "--yes", "--expected-sha256", _MODEL_SHA256]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_wrong_workspace"
        assert env["error"]["details"] == {
            "reason": "workspace_mismatch",
            "workspace": "ws-1",
            "current_workspace": "ws-2",
        }
        assert cloud.paths() == ["/api/workspaces/current"] and storage.puts == []

    @pytest.mark.parametrize("answer", [_http_error(404), {"name": "Studio"}], ids=["older-server", "no-id"])
    def test_a_workspace_comfy_cloud_does_not_name_gets_nothing(
        self, cloud_target, monkeypatch, capsys, model_file, agent_caller, answer
    ):
        routes = {**_UPLOAD_ROUTES, ("GET", "/api/workspaces/current"): answer}
        args = [str(model_file), "--type", "loras", "--workspace", "ws-1", "--dry-run"]
        env, rc, cloud, storage, _ = _model_upload(monkeypatch, capsys, args, routes, [])

        assert rc == 1 and env["error"]["code"] == "asset_upload_wrong_workspace"
        assert env["error"]["details"]["reason"] == "workspace_unknown"
        assert cloud.paths() == ["/api/workspaces/current"] and storage.puts == []

    def test_it_is_registered_under_the_model_noun_with_a_schema(self):
        from comfy_cli.command.models import models
        from comfy_cli.discovery import COMMAND_SCHEMAS, STREAM_EVENT_SCHEMAS

        assert "upload" in {c.name for c in models.app.registered_commands}
        assert COMMAND_SCHEMAS["comfy model upload"] == "model_upload"
        assert STREAM_EVENT_SCHEMAS["comfy model upload"] == "build_push_event"


# --- Resumable uploads through both public commands ----------------------------


def _resumable_started(n=1):
    import time
    from datetime import datetime, timezone

    return {
        **_started(n),
        "upload_mode": "resumable",
        "chunk_size_bytes": 64 * 1024**2,
        "expires_at": datetime.fromtimestamp(time.time() + 7 * 24 * 3600, timezone.utc).isoformat(),
    }


def _resume_routes():
    return {
        **_UPLOAD_ROUTES,
        ("POST", "/api/assets/uploads"): [_resumable_started(1), _resumable_started(2)],
        ("GET", "/api/user"): {"id": "user-1"},
        ("GET", "/api/workspaces/current"): {"id": "ws-1"},
    }


class _ResumableBucket:
    def __init__(self, *, crash=False, held=b"", status=None):
        self.crash, self.held, self.status = crash, held, status
        self.puts = []

    def __call__(self, url, *, data, headers, **kw):
        self.puts.append({"url": url, "headers": headers, "body": data, **kw})
        assert kw["allow_redirects"] is False
        if self.crash:
            self.held = data[:5]
            self.crash = False
            raise KeyboardInterrupt()
        if not headers["Content-Range"].startswith("bytes */"):
            offset = int(headers["Content-Range"].split()[1].split("-")[0])
            assert offset == len(self.held)
            self.held += data
        code = self.status or (200 if len(self.held) == 16 else 308)
        return SimpleNamespace(
            status_code=code,
            headers={"Range": f"bytes=0-{len(self.held) - 1}"} if self.held else {},
            close=lambda: None,
        )


class TestResumableCommands:
    @pytest.mark.parametrize("model", [False, True], ids=["assets-library", "model"])
    def test_killed_upload_resumes_the_same_session_on_rerun(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, model
    ):
        run = _model_upload if model else _upload
        args = [str(model_file), "--type", "loras"] + (
            ["--yes", "--workspace", "ws-1", "--expected-sha256", _MODEL_SHA256] if model else []
        )
        bucket = _ResumableBucket(crash=True)
        # CliRunner converts KeyboardInterrupt to an exit; the session has already been saved.
        monkeypatch.setattr("requests.put", bucket)
        cloud = _Cloud(_resume_routes())
        monkeypatch.setattr("urllib.request.urlopen", cloud)
        from comfy_cli.command.models import models

        result = CliRunner().invoke(models.app if model else assets_library.app, ["upload", *args, "--where", "cloud"])
        assert result.exit_code != 0
        assert bucket.held == _MODEL_BYTES[:5]
        capsys.readouterr()
        bucket.puts.clear()
        env, rc, cloud, _, everything = run(monkeypatch, capsys, args, _resume_routes(), [], storage=bucket)
        assert rc == 0 and env["data"]["uploaded"] is True
        assert "/api/assets/uploads" not in cloud.paths()
        assert bucket.puts[0]["headers"]["Content-Range"] == "bytes */16"
        assert bucket.puts[1]["headers"]["Content-Range"] == "bytes 5-15/16"
        assert bucket.held == _MODEL_BYTES
        assert "SECRETSIG" not in everything
        from comfy_cli.config_manager import ConfigManager

        assert list((Path(ConfigManager().get_config_path()) / "model-uploads").glob("*.json")) == []

    def test_crash_after_final_piece_completes_without_resending(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = _resume_routes()
        routes[("POST", _COMPLETE_1)] = _cloud_error(403, "NO", "try again")
        bucket = _ResumableBucket()
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=bucket)
        assert rc == 1 and bucket.held == _MODEL_BYTES
        bucket.puts.clear()
        env, rc, cloud, _, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], _resume_routes(), [], storage=bucket
        )
        assert rc == 0 and "/api/assets/uploads" not in cloud.paths()
        assert [p["headers"]["Content-Range"] for p in bucket.puts] == ["bytes */16"]

    @pytest.mark.parametrize("expired_at", ["bucket", "complete"])
    def test_expired_session_starts_fresh(self, cloud_target, monkeypatch, capsys, model_file, no_wait, expired_at):
        routes = _resume_routes()
        if expired_at == "complete":
            routes[("POST", _COMPLETE_1)] = _http_error(404)
        bucket = _ResumableBucket(status=404 if expired_at == "bucket" else None)
        original = bucket.__call__

        def put(url, **kw):
            if url.endswith("2"):
                bucket.status = None
                bucket.held = b""
            return original(url, **kw)

        monkeypatch.setattr("requests.put", put)
        # _upload accepts a callable in place of its storage fake.
        env, rc, cloud, _, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=put
        )
        assert rc == 0 and cloud.paths("POST").count("/api/assets/uploads") == 2

    @pytest.mark.parametrize("change", ["workspace", "account", "same-size-bytes", "folder", "mtime", "size", "server"])
    def test_rerun_with_a_changed_file_or_identity_starts_fresh(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, change
    ):
        bucket = _ResumableBucket(status=403)
        args = [str(model_file), "--type", "loras"]
        env, rc, *_ = _upload(monkeypatch, capsys, args, _resume_routes(), [], storage=bucket)
        assert rc == 1
        routes = _resume_routes()
        if change == "workspace":
            routes[("GET", "/api/workspaces/current")] = {"id": "ws-2"}
        elif change == "account":
            routes[("GET", "/api/user")] = {"id": "user-2"}
        elif change == "same-size-bytes":
            info = model_file.stat()
            model_file.write_bytes(b"X" + _MODEL_BYTES[1:])
            os.utime(model_file, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif change == "mtime":
            info = model_file.stat()
            os.utime(model_file, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        elif change == "size":
            model_file.write_bytes(_MODEL_BYTES + b"x")
        elif change == "folder":
            args[-1] = "checkpoints"
        else:
            from dataclasses import replace

            monkeypatch.setattr(
                "comfy_cli.target.resolve_target", lambda **kw: replace(cloud_target, base_url="https://other.example")
            )
        # A refusal at storage is enough to see whether start was called, regardless of stored hash.
        bucket.held = b""
        env, rc, cloud, _, _ = _upload(monkeypatch, capsys, args, routes, [], storage=bucket)
        assert rc == 1 and "/api/assets/uploads" in cloud.paths()
        assert not bucket.puts[-1]["headers"]["Content-Range"].startswith("bytes */")

    def test_resumable_still_checks_changed_blocks_and_stored_hash(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        routes = _resume_routes()
        routes[("POST", "/api/assets/uploads")] = _then(
            lambda: model_file.write_bytes(b"X" + _MODEL_BYTES[1:]), _resumable_started()
        )
        bucket = _ResumableBucket()
        env, rc, cloud, _, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=bucket
        )
        assert rc == 1 and env["error"]["details"]["reason"] == "file_changed"
        assert bucket.puts == [] and _COMPLETE_1 not in cloud.paths()
        model_file.write_bytes(_MODEL_BYTES)
        routes = _resume_routes()
        routes[("GET", "/api/tasks/task-1")] = {**_DONE, "result": {**_DONE["result"], "hash": "b" * 64}}
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=bucket)
        assert rc == 1 and env["error"]["code"] == "asset_upload_unverified"

    def test_complete_retries_transient_errors(self, cloud_target, monkeypatch, capsys, model_file, no_wait):
        routes = _resume_routes()
        routes[("POST", _COMPLETE_1)] = [_http_error(429), _http_error(503), {"task_id": "task-1"}]
        env, rc, cloud, _, _ = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=_ResumableBucket()
        )
        assert rc == 0 and cloud.paths().count(_COMPLETE_1) == 3


class TestResumableRefusals:
    def test_server_size_refusal_prints_its_message_before_sending(self, cloud_target, monkeypatch, capsys, model_file):
        routes = _resume_routes()
        routes[("POST", "/api/assets/uploads")] = _cloud_error(400, "FILE_TOO_LARGE", "Model files can be up to 25 GB.")
        env, rc, _, storage, _ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [])
        assert rc == 1 and env["error"]["message"] == "Model files can be up to 25 GB."
        assert storage.puts == []

    @pytest.mark.parametrize("endpoint", ["user", "workspaces/current"])
    def test_unknown_identity_does_not_save_a_session(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait, endpoint
    ):
        routes = _resume_routes()
        routes[("GET", "/api/" + endpoint)] = {}
        bucket = _ResumableBucket()
        env, rc, *_ = _upload(monkeypatch, capsys, [str(model_file), "--type", "loras"], routes, [], storage=bucket)
        assert rc == 0
        from comfy_cli.config_manager import ConfigManager

        assert list((Path(ConfigManager().get_config_path()) / "model-uploads").glob("*.json")) == []

    def test_paused_upload_redacts_the_credential_and_keeps_the_state(
        self, cloud_target, monkeypatch, capsys, model_file, no_wait
    ):
        import requests

        def lost(*a, **kw):
            raise requests.ConnectionError("SECRETSIG")

        env, rc, _, _, everything = _upload(
            monkeypatch, capsys, [str(model_file), "--type", "loras"], _resume_routes(), [], storage=lost
        )
        assert rc == 1 and env["error"]["details"]["reason"] == "upload_paused"
        assert "SECRETSIG" not in everything
        assert "Run the same command again" in env["error"]["message"]
        from comfy_cli.config_manager import ConfigManager

        assert len(list((Path(ConfigManager().get_config_path()) / "model-uploads").glob("*.json"))) == 1
