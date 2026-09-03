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

The 404 mapping was found in prod (Langfuse
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
