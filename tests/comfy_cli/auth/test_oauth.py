"""OAuth flow unit tests — PKCE, callback server, refresh, store round-trip.

The live round-trip against testcloud is exercised manually; these tests
cover everything that doesn't require a browser.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
import urllib.parse
import urllib.request
from unittest.mock import patch

import pytest

from comfy_cli.auth import store as auth_store
from comfy_cli.cloud import oauth


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


def test_pkce_pair_satisfies_server_format():
    verifier, challenge = oauth.generate_pkce_pair()
    # Server requires 43-char base64url challenge with S256.
    assert len(challenge) == 43
    # Challenge is base64url(sha256(verifier)) — verify the math.
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    assert challenge == expected


def test_pkce_pairs_are_unique_per_call():
    pairs = {oauth.generate_pkce_pair() for _ in range(20)}
    assert len(pairs) == 20  # zero collisions across 20 draws


def test_state_is_high_entropy():
    states = {oauth.generate_state() for _ in range(100)}
    assert len(states) == 100


# ---------------------------------------------------------------------------
# Authorize URL construction
# ---------------------------------------------------------------------------


def test_build_authorize_url_includes_every_required_param():
    url = oauth._build_authorize_url(
        base_url="https://testcloud.comfy.org",
        client_id="mcp-dyn-abc",
        redirect_uri="http://127.0.0.1:51234/callback",
        scopes=("mcp:tools:read", "mcp:tools:call"),
        state="STATE",
        challenge="C" * 43,
        resource="https://testcloud.comfy.org/mcp",
    )
    parsed = urllib.parse.urlsplit(url)
    qs = urllib.parse.parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "testcloud.comfy.org"
    assert parsed.path == "/oauth/authorize"
    assert qs["response_type"] == ["code"]
    assert qs["client_id"] == ["mcp-dyn-abc"]
    assert qs["redirect_uri"] == ["http://127.0.0.1:51234/callback"]
    assert qs["scope"] == ["mcp:tools:read mcp:tools:call"]
    assert qs["state"] == ["STATE"]
    assert qs["code_challenge_method"] == ["S256"]
    assert qs["resource"] == ["https://testcloud.comfy.org/mcp"]


# ---------------------------------------------------------------------------
# Callback server — happy path + bad state + error param
# ---------------------------------------------------------------------------


def _drive_callback(*, expected_state: str, query: str) -> oauth._CallbackCapture:
    capture = oauth._CallbackCapture()
    handler_cls = oauth._build_handler(
        expected_state=expected_state,
        capture=capture,
        success_html="OK",
        failure_html="FAIL",
    )
    port = oauth._pick_free_port()
    server = oauth.http.server.HTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}{oauth.CALLBACK_PATH}?{query}", timeout=2
        ).read()
    except Exception:  # noqa: BLE001 — failure HTML still returns a body
        pass
    t.join(timeout=2)
    server.server_close()
    return capture


def test_callback_happy_path_captures_code():
    cap = _drive_callback(expected_state="STATE", query="code=THECODE&state=STATE")
    assert cap.code == "THECODE"
    assert cap.state == "STATE"
    assert cap.error is None


def test_callback_rejects_state_mismatch():
    cap = _drive_callback(expected_state="EXPECTED", query="code=THECODE&state=WRONG")
    assert cap.code is None
    assert cap.error is not None
    assert "missing_code_or_state_mismatch" in cap.error


def test_success_html_interpolates_base_url_host():
    """The success page must show the real base URL host, not a hardcoded one."""
    rendered = oauth._SUCCESS_HTML.replace("__HOST__", "fe-pr-12159.testenvs.comfy.org")
    assert "fe-pr-12159.testenvs.comfy.org" in rendered
    assert "__HOST__" not in rendered
    # The template should not bake any specific deployment into the source.
    assert "testcloud.comfy.org" not in oauth._SUCCESS_HTML


def test_callback_surfaces_oauth_error_param():
    cap = _drive_callback(
        expected_state="STATE",
        query="error=access_denied&error_description=user+cancelled&state=STATE",
    )
    assert cap.code is None
    assert cap.error == "access_denied"
    assert cap.error_description == "user cancelled"


# ---------------------------------------------------------------------------
# DCR + token exchange + refresh (mocked HTTP)
# ---------------------------------------------------------------------------


def test_register_client_returns_parsed_record(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        oauth,
        "_post_json",
        lambda url, body: {
            "client_id": "mcp-dyn-NEW",
            "client_name": body["client_name"],
            "redirect_uris": body["redirect_uris"],
            "client_id_issued_at": 1700000000,
        },
    )
    result = oauth.register_client(
        base_url="https://testcloud.comfy.org",
        client_name="comfy-cli",
        redirect_uris=("http://127.0.0.1:0/callback",),
    )
    assert result.client_id == "mcp-dyn-NEW"
    assert result.client_name == "comfy-cli"
    assert result.issued_at == 1700000000


def test_exchange_code_returns_token_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        oauth,
        "_post_form",
        lambda url, body: {
            "access_token": "comfy_at_AAA",
            "refresh_token": "comfy_rt_BBB",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "mcp:tools:read mcp:tools:call",
        },
    )
    before = int(time.time())
    tokens = oauth.exchange_code(
        base_url="https://testcloud.comfy.org",
        client_id="cid",
        code="THECODE",
        redirect_uri="http://127.0.0.1:5/callback",
        code_verifier="V" * 43,
    )
    after = int(time.time())
    assert tokens.access_token == "comfy_at_AAA"
    assert tokens.refresh_token == "comfy_rt_BBB"
    assert tokens.token_type == "Bearer"
    assert tokens.expires_in == 3600
    # expires_at is now + expires_in, allow 1s drift between before/after.
    assert before + 3600 - 1 <= tokens.expires_at <= after + 3600 + 1


def test_refresh_tokens_calls_token_endpoint(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake_post_form(url, body):
        seen["url"] = url
        seen["body"] = body
        return {"access_token": "NEW_AT", "refresh_token": "NEW_RT", "token_type": "Bearer", "expires_in": 60}

    monkeypatch.setattr(oauth, "_post_form", fake_post_form)
    tokens = oauth.refresh_tokens(
        base_url="https://testcloud.comfy.org",
        client_id="cid",
        refresh_token="OLD_RT",
    )
    assert seen["url"].endswith("/oauth/token")
    assert seen["body"]["grant_type"] == "refresh_token"
    assert seen["body"]["refresh_token"] == "OLD_RT"
    assert seen["body"]["client_id"] == "cid"
    assert tokens.access_token == "NEW_AT"


def test_exchange_code_sends_resource_indicator(monkeypatch: pytest.MonkeyPatch):
    # RFC 8707: resource= must travel on the token POST, not just authorize.
    seen = {}

    def fake(url, body):
        seen["body"] = body
        return {"access_token": "AT", "refresh_token": "RT", "token_type": "Bearer", "expires_in": 60}

    monkeypatch.setattr(oauth, "_post_form", fake)
    oauth.exchange_code(
        base_url="https://testcloud.comfy.org",
        client_id="cid",
        code="THECODE",
        redirect_uri="http://127.0.0.1:5/callback",
        code_verifier="V" * 43,
        resource="https://testcloud.comfy.org/api",
    )
    assert seen["body"]["resource"] == "https://testcloud.comfy.org/api"


def test_refresh_tokens_sends_resource_indicator(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake(url, body):
        seen["body"] = body
        return {"access_token": "AT", "refresh_token": "RT", "token_type": "Bearer", "expires_in": 60}

    monkeypatch.setattr(oauth, "_post_form", fake)
    oauth.refresh_tokens(
        base_url="https://testcloud.comfy.org",
        client_id="cid",
        refresh_token="OLD_RT",
        resource="https://testcloud.comfy.org/api",
        scopes=("comfy-cloud:workflows:read", "comfy-cloud:jobs:read"),
    )
    assert seen["body"]["resource"] == "https://testcloud.comfy.org/api"
    assert seen["body"]["scope"] == "comfy-cloud:workflows:read comfy-cloud:jobs:read"


def test_exchange_code_maps_http_failure_to_token_error(monkeypatch: pytest.MonkeyPatch):
    def boom(url, body):
        raise oauth._HTTPFail(400, '{"error":"invalid_grant"}')

    monkeypatch.setattr(oauth, "_post_form", boom)
    with pytest.raises(oauth.OAuthTokenError) as exc:
        oauth.exchange_code(
            base_url="https://testcloud.comfy.org",
            client_id="cid",
            code="x",
            redirect_uri="http://127.0.0.1:5/callback",
            code_verifier="V" * 43,
        )
    assert exc.value.code == "oauth_token_failed"
    assert "invalid_grant" in exc.value.details.get("body", "")


def test_token_response_redacts_access_and_refresh():
    redacted = oauth._redact_token_response(
        {"access_token": "verysecrettoken", "refresh_token": "anotherverysecret", "expires_in": 60}
    )
    assert redacted["access_token"] != "verysecrettoken"
    assert redacted["refresh_token"] != "anotherverysecret"
    assert "verysecrettoken" not in str(redacted)
    assert redacted["expires_in"] == 60  # non-secret fields untouched


# ---------------------------------------------------------------------------
# run_login end-to-end (mocked network + faked browser callback)
# ---------------------------------------------------------------------------


def test_run_login_orchestrates_full_flow(monkeypatch: pytest.MonkeyPatch):
    # 1) Mock DCR.
    monkeypatch.setattr(
        oauth,
        "_post_json",
        lambda url, body: {"client_id": "mcp-dyn-LIVE", "redirect_uris": body["redirect_uris"]},
    )
    # 2) Mock token exchange.
    monkeypatch.setattr(
        oauth,
        "_post_form",
        lambda url, body: {"access_token": "AT", "refresh_token": "RT", "token_type": "Bearer", "expires_in": 600},
    )

    captured_url = {}

    # 3) Replace browser-open with an immediate HTTP GET to the callback so the
    # localhost server completes and run_login returns.
    def fake_open(url, **kwargs):
        captured_url["url"] = url
        parsed = urllib.parse.urlsplit(url)
        qs = urllib.parse.parse_qs(parsed.query)
        redirect = qs["redirect_uri"][0]
        state = qs["state"][0]
        # Hit our own loopback server in a thread so urlopen doesn't deadlock.
        def hit():
            try:
                urllib.request.urlopen(
                    f"{redirect}?code=THECODE&state={urllib.parse.quote(state)}",
                    timeout=2,
                ).read()
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=hit, daemon=True).start()
        return True

    with patch.object(oauth.webbrowser, "open", side_effect=fake_open):
        result = oauth.run_login(
            base_url="https://testcloud.comfy.org",
            resource="https://cloud.comfy.org/mcp",
            scopes=("mcp:tools:read", "mcp:tools:call"),
            client_id=None,
            register_if_missing=True,  # force DCR for this orchestration test
            timeout_s=5,
        )

    assert result.client_id == "mcp-dyn-LIVE"
    assert result.tokens.access_token == "AT"
    assert result.tokens.refresh_token == "RT"
    assert result.scope == "mcp:tools:read mcp:tools:call"
    # The redirect URI we used must be loopback.
    assert result.redirect_uri.startswith("http://127.0.0.1:")
    # The authorize URL we built must include resource + S256.
    assert "code_challenge_method=S256" in captured_url["url"]
    assert "resource=https" in captured_url["url"]


def test_run_login_times_out_when_browser_never_returns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        oauth,
        "_post_json",
        lambda url, body: {"client_id": "mcp-dyn-X", "redirect_uris": body["redirect_uris"]},
    )
    with patch.object(oauth.webbrowser, "open", return_value=True):
        with pytest.raises(oauth.OAuthTimeout) as exc:
            oauth.run_login(
                base_url="https://testcloud.comfy.org",
                resource="https://testcloud.comfy.org/mcp",
                timeout_s=0.5,
            )
    assert exc.value.code == "oauth_timeout"


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------


def test_save_and_get_cloud_session_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    session = auth_store.save_cloud_session(
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        client_id="mcp-dyn-ABC",
        scope="mcp:tools:read mcp:tools:call",
        access_token="AT",
        refresh_token="RT",
        token_type="Bearer",
        expires_at=int(time.time()) + 3600,
    )
    loaded = auth_store.get_cloud_session()
    assert loaded is not None
    assert loaded.client_id == session.client_id
    assert loaded.access_token == "AT"
    assert loaded.refresh_token == "RT"
    assert loaded.token_type == "Bearer"


def test_clear_cloud_session_removes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    auth_store.save_cloud_session(
        base_url="x", resource="y", client_id="c", scope="s",
        access_token="AT", refresh_token=None, token_type="Bearer", expires_at=None,
    )
    assert auth_store.clear_cloud_session() is True
    assert auth_store.get_cloud_session() is None
    # Idempotent.
    assert auth_store.clear_cloud_session() is False


def test_session_to_dict_redacts_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    session = auth_store.save_cloud_session(
        base_url="x", resource="y", client_id="c", scope="s",
        access_token="verysecretaccesstokenAAA",
        refresh_token="verysecretrefreshtokenBBB",
        token_type="Bearer",
        expires_at=int(time.time()) + 3600,
    )
    d = session.to_dict(redact=True)
    assert d["tokens_redacted"] is True
    assert "verysecret" not in str(d)
    assert d["access_token"] != "verysecretaccesstokenAAA"


def test_session_is_expired_after_window():
    session = auth_store.CloudSession(
        base_url="x", resource="y", client_id="c", scope="s",
        access_token="AT", refresh_token="RT", token_type="Bearer",
        expires_at=int(time.time()) - 1,  # already past
        saved_at="2026-01-01T00:00:00+00:00",
    )
    assert session.is_expired() is True


def test_session_not_expired_when_future():
    session = auth_store.CloudSession(
        base_url="x", resource="y", client_id="c", scope="s",
        access_token="AT", refresh_token="RT", token_type="Bearer",
        expires_at=int(time.time()) + 3600,
        saved_at="2026-01-01T00:00:00+00:00",
    )
    assert session.is_expired() is False


class TestEnsureFreshSession:
    """Proactive refresh: keep an expired-but-refreshable session alive without
    forcing the user to re-run `cloud login`."""

    def _expired(self, refresh: str | None = "RT") -> auth_store.CloudSession:
        return auth_store.CloudSession(
            base_url="https://c", resource="https://c/api", client_id="cid", scope="s",
            access_token="OLD", refresh_token=refresh, token_type="Bearer",
            expires_at=int(time.time()) - 1, saved_at="2026-01-01T00:00:00+00:00",
        )

    def test_refreshes_expired_session_with_refresh_token(self, monkeypatch):
        saved: dict = {}
        fresh = auth_store.CloudSession(
            base_url="https://c", resource="https://c/api", client_id="cid", scope="s",
            access_token="NEW", refresh_token="RT2", token_type="Bearer",
            expires_at=int(time.time()) + 3600, saved_at="2026-01-01T00:00:01+00:00",
        )
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: self._expired())
        monkeypatch.setattr(auth_store, "save_cloud_session", lambda **kw: saved.update(kw) or fresh)
        monkeypatch.setattr(oauth, "refresh_tokens", lambda **kw: oauth.TokenSet(
            access_token="NEW", refresh_token="RT2", token_type="Bearer",
            expires_in=3600, expires_at=int(time.time()) + 3600, scope="s"))
        result = oauth.ensure_fresh_session()
        assert result.access_token == "NEW"
        assert result.is_expired() is False
        assert saved["access_token"] == "NEW"

    def test_no_refresh_token_returns_stale_without_calling_refresh(self, monkeypatch):
        called = []
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: self._expired(refresh=None))
        monkeypatch.setattr(oauth, "refresh_tokens", lambda **kw: called.append(1))
        result = oauth.ensure_fresh_session()
        assert result.is_expired() is True
        assert called == []

    def test_valid_session_not_refreshed(self, monkeypatch):
        valid = auth_store.CloudSession(
            base_url="x", resource="y", client_id="c", scope="s",
            access_token="AT", refresh_token="RT", token_type="Bearer",
            expires_at=int(time.time()) + 3600, saved_at="2026-01-01T00:00:00+00:00",
        )
        called = []
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: valid)
        monkeypatch.setattr(oauth, "refresh_tokens", lambda **kw: called.append(1))
        assert oauth.ensure_fresh_session() is valid
        assert called == []

    def test_refresh_failure_falls_back_to_stale_session(self, monkeypatch):
        def boom(**kw):
            raise oauth.OAuthRefreshError("dead", hint="re-login", details={})
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: self._expired())
        monkeypatch.setattr(oauth, "refresh_tokens", boom)
        result = oauth.ensure_fresh_session()
        assert result.is_expired() is True  # no crash; caller's expiry check still fires

    def test_none_when_no_session(self, monkeypatch):
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)
        assert oauth.ensure_fresh_session() is None
