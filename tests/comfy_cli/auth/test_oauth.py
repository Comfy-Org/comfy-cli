"""OAuth flow unit tests — PKCE, callback server, refresh, store round-trip.

The live round-trip against testcloud is exercised manually; these tests
cover everything that doesn't require a browser.
"""

from __future__ import annotations

import base64
import hashlib
import os
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
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
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
        urllib.request.urlopen(f"http://127.0.0.1:{port}{oauth.CALLBACK_PATH}?{query}", timeout=2).read()
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
        base_url="x",
        resource="y",
        client_id="c",
        scope="s",
        access_token="AT",
        refresh_token=None,
        token_type="Bearer",
        expires_at=None,
    )
    assert auth_store.clear_cloud_session() is True
    assert auth_store.get_cloud_session() is None
    # Idempotent.
    assert auth_store.clear_cloud_session() is False


def test_session_to_dict_redacts_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    session = auth_store.save_cloud_session(
        base_url="x",
        resource="y",
        client_id="c",
        scope="s",
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
        base_url="x",
        resource="y",
        client_id="c",
        scope="s",
        access_token="AT",
        refresh_token="RT",
        token_type="Bearer",
        expires_at=int(time.time()) - 1,  # already past
        saved_at="2026-01-01T00:00:00+00:00",
    )
    assert session.is_expired() is True


def test_session_not_expired_when_future():
    session = auth_store.CloudSession(
        base_url="x",
        resource="y",
        client_id="c",
        scope="s",
        access_token="AT",
        refresh_token="RT",
        token_type="Bearer",
        expires_at=int(time.time()) + 3600,
        saved_at="2026-01-01T00:00:00+00:00",
    )
    assert session.is_expired() is False


class TestEnsureFreshSession:
    """Proactive refresh: keep an expired-but-refreshable session alive without
    forcing the user to re-run `cloud login`."""

    def _expired(self, refresh: str | None = "RT") -> auth_store.CloudSession:
        return auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="OLD",
            refresh_token=refresh,
            token_type="Bearer",
            expires_at=int(time.time()) - 1,
            saved_at="2026-01-01T00:00:00+00:00",
        )

    def test_refreshes_expired_session_with_refresh_token(self, monkeypatch):
        saved: dict = {}
        fresh = auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="NEW",
            refresh_token="RT2",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
            saved_at="2026-01-01T00:00:01+00:00",
        )
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: self._expired())
        monkeypatch.setattr(auth_store, "save_cloud_session", lambda **kw: saved.update(kw) or fresh)
        monkeypatch.setattr(
            oauth,
            "refresh_tokens",
            lambda **kw: oauth.TokenSet(
                access_token="NEW",
                refresh_token="RT2",
                token_type="Bearer",
                expires_in=3600,
                expires_at=int(time.time()) + 3600,
                scope="s",
            ),
        )
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
            base_url="x",
            resource="y",
            client_id="c",
            scope="s",
            access_token="AT",
            refresh_token="RT",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
            saved_at="2026-01-01T00:00:00+00:00",
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

    def _valid(self) -> auth_store.CloudSession:
        return auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="OLD",
            refresh_token="RT",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
            saved_at="2026-01-01T00:00:00+00:00",
        )

    def test_force_refreshes_even_when_locally_valid(self, monkeypatch):
        """Reactive path: a server 401 means refresh even if our clock thinks
        the access token is still good (skew / no recorded expiry)."""
        saved: dict = {}
        called = []
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: self._valid())
        monkeypatch.setattr(auth_store, "save_cloud_session", lambda **kw: saved.update(kw) or self._valid())

        def _refresh(**kw):
            called.append(1)
            return oauth.TokenSet(
                access_token="NEW",
                refresh_token="RT2",
                token_type="Bearer",
                expires_in=3600,
                expires_at=int(time.time()) + 3600,
                scope="s",
            )

        monkeypatch.setattr(oauth, "refresh_tokens", _refresh)
        result = oauth.ensure_fresh_session(force=True)
        assert called == [1]  # refreshed despite local validity
        assert saved["access_token"] == "NEW"
        assert result is not None

    def test_force_without_refresh_token_is_noop(self, monkeypatch):
        called = []
        valid_no_rt = auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="OLD",
            refresh_token=None,
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
            saved_at="2026-01-01T00:00:00+00:00",
        )
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: valid_no_rt)
        monkeypatch.setattr(oauth, "refresh_tokens", lambda **kw: called.append(1))
        result = oauth.ensure_fresh_session(force=True)
        assert result is valid_no_rt
        assert called == []  # no refresh token → nothing to spend


class TestConcurrentRefresh:
    """Rotation-correct, concurrency-safe refresh.

    The auth server rotates the refresh token on every refresh and trips
    reuse-detection (invalidating the whole family) if a consumed token is ever
    replayed. These tests pin the cross-process lock + double-check + atomic
    persist + reuse handling that guarantee a refresh token is never sent twice.
    All network is mocked — ``oauth.refresh_tokens`` is the only seam patched.
    """

    @pytest.fixture
    def persisted(self, tmp_path, monkeypatch):
        """A real, on-disk secret store so the file lock and the
        read-modify-write actually exercise the store (not mocked reads)."""
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
        return path

    def _persist_expired(self, *, refresh_token: str = "RT0") -> None:
        auth_store.save_cloud_session(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="OLD",
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_at=int(time.time()) - 1,  # already expired → refresh path
        )

    @staticmethod
    def _token_set(*, access: str, refresh: str) -> oauth.TokenSet:
        return oauth.TokenSet(
            access_token=access,
            refresh_token=refresh,
            token_type="Bearer",
            expires_in=3600,
            expires_at=int(time.time()) + 3600,
            scope="s",
        )

    # (a) + (c): two concurrent refreshes coalesce to ONE network call, and the
    # rotated refresh token is what ends up persisted.
    def test_concurrent_refresh_coalesces_to_single_network_call(self, persisted, monkeypatch):
        self._persist_expired(refresh_token="RT0")

        calls: list[str] = []
        start_barrier = threading.Barrier(2)

        def fake_refresh(**kw):
            calls.append(kw["refresh_token"])
            time.sleep(0.15)  # widen the window so the second thread must wait
            return self._token_set(access="NEW", refresh="RT1")

        monkeypatch.setattr(oauth, "refresh_tokens", fake_refresh)

        results: list = []

        def worker():
            start_barrier.wait()  # both threads read the stale token first
            results.append(oauth.ensure_fresh_session())

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one network refresh — the second waiter saw the rotated token
        # under the lock (double-check) and never replayed the consumed RT0.
        assert calls == ["RT0"]
        # The consumed token was never sent twice.
        assert "RT0" not in calls[1:]
        # Both callers end up with the fresh access token.
        assert all(r is not None and r.access_token == "NEW" for r in results)
        # (c) Rotation persisted: the store holds the NEW refresh token.
        stored = auth_store.get_cloud_session()
        assert stored.refresh_token == "RT1"
        assert stored.access_token == "NEW"

    # (b) Double-check: if a peer already rotated the token between our pre-lock
    # read and the post-lock re-read, we adopt it and do NOT refresh again.
    def test_double_check_skips_refresh_when_peer_already_rotated(self, persisted, monkeypatch):
        self._persist_expired(refresh_token="RT0")

        # Simulate a peer that rotated RT0 → RT1 while we waited for the lock:
        # the pre-lock read sees RT0; the post-lock re-read sees a fresh RT1.
        fresh_peer = auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="PEER_NEW",
            refresh_token="RT1",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
            saved_at="2026-01-01T00:00:02+00:00",
        )
        reads = [self._stale(), fresh_peer]
        monkeypatch.setattr(auth_store, "get_cloud_session", lambda: reads.pop(0) if reads else fresh_peer)
        monkeypatch.setattr(oauth, "refresh_tokens", lambda **kw: pytest.fail("must not refresh after peer rotation"))

        result = oauth.ensure_fresh_session()
        assert result.refresh_token == "RT1"
        assert result.access_token == "PEER_NEW"

    def _stale(self) -> auth_store.CloudSession:
        return auth_store.CloudSession(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="OLD",
            refresh_token="RT0",
            token_type="Bearer",
            expires_at=int(time.time()) - 1,
            saved_at="2026-01-01T00:00:00+00:00",
        )

    # (d) Reuse-detected / invalid_grant is fatal: clear the session, return
    # None, surface login guidance, and NEVER loop.
    def test_reuse_detected_clears_session_and_does_not_loop(self, persisted, monkeypatch):
        self._persist_expired(refresh_token="RT0")

        calls = []

        def boom(**kw):
            calls.append(1)
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                hint="run `comfy cloud login`",
                details={
                    "status": 400,
                    "body": '{"error":"invalid_grant","error_description":"refresh token reuse detected"}',
                },
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        result = oauth.ensure_fresh_session()
        assert result is None  # family is dead → caller surfaces cloud_unauthorized
        assert calls == [1]  # exactly one attempt, no retry loop
        # Stored session was cleared so the dead token is never replayed again.
        assert auth_store.get_cloud_session() is None

    def test_is_fatal_token_error_classification(self):
        def err(status, body=""):
            return oauth.OAuthRefreshError("refresh failed", details={"status": status, "body": body})

        # Fatal: the refresh-token family is genuinely dead → re-login required.
        assert oauth._is_fatal_token_error(err(400, "invalid_grant")) is True
        assert oauth._is_fatal_token_error(err(400, '{"error":"invalid_token"}')) is True
        # Gated on the body, NOT the bare status: recoverable 4xx must NOT wipe
        # the session (re-register / fix config instead of forcing re-login).
        assert oauth._is_fatal_token_error(err(401, '{"error":"invalid_client"}')) is False
        assert oauth._is_fatal_token_error(err(400, '{"error":"invalid_request"}')) is False
        assert oauth._is_fatal_token_error(err(400, '{"error":"invalid_scope"}')) is False
        assert oauth._is_fatal_token_error(err(401)) is False  # bare 401, no error code
        assert oauth._is_fatal_token_error(err(0, "Connection refused")) is False  # transient network
        assert oauth._is_fatal_token_error(err(503)) is False  # server hiccup, token may be fine
        # Message-based detection when status is absent.
        assert oauth._is_fatal_token_error(oauth.OAuthRefreshError("reuse detected", details={})) is True

    def test_describe_token_error_extracts_server_reason(self):
        def err(body):
            return oauth.OAuthRefreshError("refresh failed", details={"status": 400, "body": body})

        # error + error_description → "code: description".
        assert (
            oauth._describe_token_error(
                err('{"error":"invalid_grant","error_description":"workspace membership lost"}')
            )
            == "invalid_grant: workspace membership lost"
        )
        # error only.
        assert oauth._describe_token_error(err('{"error":"invalid_grant"}')) == "invalid_grant"
        # Nothing useful (network error, empty body) → None so the caller falls
        # back to its generic phrasing.
        assert oauth._describe_token_error(err("")) is None
        assert oauth._describe_token_error(err("not json")) is None

    def test_take_last_fatal_refresh_reason_is_one_shot(self):
        oauth._last_fatal.reason = "invalid_grant: resource state changed"
        assert oauth.take_last_fatal_refresh_reason() == "invalid_grant: resource state changed"
        # Cleared after read — a later unrelated failure can't inherit it.
        assert oauth.take_last_fatal_refresh_reason() is None

    # (d) end-to-end surface: after a reuse clears the session, building a cloud
    # client with no other credentials raises Unauthenticated (mapped to
    # cloud_unauthorized by the command layer) — exactly once.
    def test_reactive_client_raises_unauthenticated_on_reuse(self, persisted, monkeypatch):
        import comfy_cli.comfy_client as comfy_client

        self._persist_expired(refresh_token="RT0")

        calls = []

        def boom(**kw):
            calls.append(1)
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                details={"status": 400, "body": "refresh token reuse detected"},
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        target = comfy_client.Target(
            kind="cloud",
            base_url="https://c",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="OLD",
        )
        client = comfy_client.Client(target)
        with pytest.raises(comfy_client.Unauthenticated):
            client._try_refresh_token()
        assert calls == [1]  # one attempt, no loop
        assert auth_store.get_cloud_session() is None  # cleared

    def test_unauthenticated_message_surfaces_server_reason(self, persisted, monkeypatch):
        """The user-facing failure carries the auth server's real
        error_description (not the canned "reuse detected" guess), so a logout
        caused by e.g. lost workspace membership is diagnosable."""
        import comfy_cli.comfy_client as comfy_client

        self._persist_expired(refresh_token="RT0")

        def boom(**kw):
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                details={
                    "status": 400,
                    "body": '{"error":"invalid_grant","error_description":"workspace membership lost"}',
                },
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        target = comfy_client.Target(
            kind="cloud",
            base_url="https://c",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="OLD",
        )
        client = comfy_client.Client(target)
        with pytest.raises(comfy_client.Unauthenticated, match="workspace membership lost"):
            client._try_refresh_token()

    def test_transient_network_failure_keeps_session(self, persisted, monkeypatch):
        """A URLError (status 0) must NOT clear the family — the token is
        probably still good; keep the stale session for the caller to retry."""
        self._persist_expired(refresh_token="RT0")

        def boom(**kw):
            raise oauth.OAuthRefreshError("refresh failed: <urlopen error>", details={"status": 0, "body": ""})

        monkeypatch.setattr(oauth, "refresh_tokens", boom)
        result = oauth.ensure_fresh_session()
        assert result is not None and result.refresh_token == "RT0"  # session preserved
        assert auth_store.get_cloud_session() is not None

    def test_successful_refresh_never_repersists_spent_token(self, persisted, monkeypatch):
        """Rotating server: once a refresh SUCCEEDS the token we sent is spent.
        If the response omits a new refresh token we must persist ``None`` — never
        fall back to re-saving the consumed one (that would guarantee an
        invalid_grant / forced logout on the very next use)."""
        self._persist_expired(refresh_token="RT0")

        def no_rotation(**kw):
            # 200 OK with a fresh access token but NO new refresh token.
            return oauth.TokenSet(
                access_token="NEW_AT",
                refresh_token=None,
                token_type="Bearer",
                expires_in=3600,
                expires_at=int(time.time()) + 3600,
                scope="s",
            )

        monkeypatch.setattr(oauth, "refresh_tokens", no_rotation)
        result = oauth.ensure_fresh_session()
        assert result is not None and result.access_token == "NEW_AT"
        # The spent RT0 is gone — not re-persisted.
        assert result.refresh_token is None
        assert auth_store.get_cloud_session().refresh_token is None


class TestAtomicPersist:
    """save_cloud_session must write atomically (temp + rename) so a crash mid
    write can never leave a half-written / corrupt session file."""

    @pytest.fixture
    def persisted(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
        return path

    def test_uses_temp_then_rename(self, persisted, monkeypatch):
        seen = {}
        real_replace = os.replace

        def spy_replace(src, dst):
            seen["src"] = str(src)
            seen["dst"] = str(dst)
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy_replace)
        auth_store.save_cloud_session(
            base_url="b",
            resource="r",
            client_id="c",
            scope="s",
            access_token="AT",
            refresh_token="RT",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
        )
        # Wrote to a distinct temp path, then renamed onto the real file.
        assert seen["src"] != seen["dst"]
        assert seen["src"].endswith(".tmp")
        assert seen["dst"] == str(persisted)

    def test_mid_write_failure_leaves_original_intact(self, persisted, monkeypatch):
        # Seed a known-good session.
        good = auth_store.save_cloud_session(
            base_url="b",
            resource="r",
            client_id="c",
            scope="s",
            access_token="GOOD",
            refresh_token="RTgood",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
        )
        before = persisted.read_text(encoding="utf-8")

        # Simulate a crash at the rename step of the next write.
        def boom_replace(src, dst):
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(os, "replace", boom_replace)
        with pytest.raises(OSError):
            auth_store.save_cloud_session(
                base_url="b",
                resource="r",
                client_id="c",
                scope="s",
                access_token="HALF",
                refresh_token="RThalf",
                token_type="Bearer",
                expires_at=int(time.time()) + 3600,
            )

        # The original file is byte-for-byte intact (no partial/corrupt write).
        assert persisted.read_text(encoding="utf-8") == before
        # And the half-written temp file was cleaned up.
        leftovers = list(persisted.parent.glob("secrets.json*.tmp"))
        assert leftovers == []
        # The store still reads the good session.
        reread = auth_store.get_cloud_session()
        assert reread.access_token == good.access_token == "GOOD"


class TestApi401DoesNotClearSession:
    """Regression: a data/API 401 (a partner-node ``cloud_unauthorized`` blip,
    or any normal REST endpoint 401) must NOT, by itself, wipe the login.

    The reactive 401 path triggers at most ONE locked refresh-and-retry. The
    stored session is only ever cleared when the *token endpoint itself* returns
    a fatal ``invalid_grant`` during that refresh — never because a data request
    came back 401 after a successful token refresh.
    """

    @pytest.fixture
    def persisted(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
        return path

    def _persist_valid(self, *, refresh_token: str = "RT0", access: str = "AT0") -> None:
        auth_store.save_cloud_session(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token=access,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,  # locally valid — force=True drives the refresh
        )

    def _cloud_target(self, token: str = "AT0"):
        import comfy_cli.comfy_client as comfy_client

        return comfy_client.Target(
            kind="cloud",
            base_url="https://c",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token=token,
        )

    # (1) Refresh SUCCEEDS, but the API request still 401s → retry happened once,
    # the session is preserved, and the caller surfaces the unauthorized error.
    def test_refresh_succeeds_but_api_still_401_does_not_clear(self, persisted, monkeypatch):
        import comfy_cli.comfy_client as comfy_client

        self._persist_valid(refresh_token="RT0", access="AT0")

        refresh_calls: list[str] = []

        def fake_refresh(**kw):
            refresh_calls.append(kw["refresh_token"])
            return oauth.TokenSet(
                access_token="AT1",  # a genuinely fresh access token
                refresh_token="RT1",
                token_type="Bearer",
                expires_in=3600,
                expires_at=int(time.time()) + 3600,
                scope="s",
            )

        monkeypatch.setattr(oauth, "refresh_tokens", fake_refresh)

        clears: list[int] = []
        monkeypatch.setattr(
            auth_store,
            "clear_cloud_session",
            lambda: clears.append(1) or False,
        )

        # The server keeps returning 401 even after a clean token refresh — i.e.
        # an API/entitlement problem, not a token problem.
        attempts: list[str] = []

        def always_401(req, timeout=None):
            attempts.append(req.get_header("Authorization"))
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(comfy_client._OPENER, "open", always_401)

        client = comfy_client.Client(self._cloud_target("AT0"))
        with pytest.raises(comfy_client.HTTPError) as exc:
            client.get_job_status("p1")

        assert exc.value.status == 401  # surfaced as-is to the caller
        assert refresh_calls == ["RT0"]  # exactly one refresh, with the stored token
        # The retry carried the rotated access token, proving refresh installed it.
        assert attempts == ["Bearer AT0", "Bearer AT1"]
        assert clears == []  # session NEVER cleared on an API-only 401
        assert auth_store.get_cloud_session() is not None  # still signed in
        assert auth_store.get_cloud_session().refresh_token == "RT1"

    # (2) Refresh itself returns invalid_grant / reuse → token-endpoint failure
    # IS fatal: the session is cleared exactly once, with no retry loop.
    def test_token_endpoint_invalid_grant_clears_once(self, persisted, monkeypatch):
        import comfy_cli.comfy_client as comfy_client

        self._persist_valid(refresh_token="RT0", access="AT0")

        refresh_calls: list[int] = []

        def boom(**kw):
            refresh_calls.append(1)
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                details={
                    "status": 400,
                    "body": '{"error":"invalid_grant","error_description":"refresh token reuse detected"}',
                },
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        real_clear = auth_store.clear_cloud_session
        clears: list[int] = []

        def counting_clear():
            clears.append(1)
            return real_clear()

        monkeypatch.setattr(auth_store, "clear_cloud_session", counting_clear)

        def first_401(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(comfy_client._OPENER, "open", first_401)

        client = comfy_client.Client(self._cloud_target("AT0"))
        with pytest.raises(comfy_client.Unauthenticated):
            client.get_job_status("p1")

        assert refresh_calls == [1]  # exactly one refresh attempt, no loop
        assert clears == [1]  # cleared exactly once
        assert auth_store.get_cloud_session() is None  # family is dead → wiped


class TestWatcherContextRefresh:
    """Background watcher (``clear_session_on_auth_failure=False``): a fatal
    refresh failure must NEVER clear the shared session. The foreground command
    owns the session lifecycle; a detached poller hitting a transient/spurious
    invalid_grant must not log the user off mid-run."""

    @pytest.fixture
    def persisted(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
        return path

    def _persist_valid(self) -> None:
        auth_store.save_cloud_session(
            base_url="https://c",
            resource="https://c/api",
            client_id="cid",
            scope="s",
            access_token="AT0",
            refresh_token="RT0",
            token_type="Bearer",
            expires_at=int(time.time()) + 3600,
        )

    # (3) allow_clear=False at the oauth layer: invalid_grant returns None but
    # leaves the stored session intact.
    def test_ensure_fresh_session_allow_clear_false_keeps_session(self, persisted, monkeypatch):
        self._persist_valid()

        def boom(**kw):
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                details={"status": 400, "body": "invalid_grant: refresh token reuse detected"},
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        result = oauth.ensure_fresh_session(force=True, allow_clear=False)
        assert result is None  # could not produce a usable token
        # ...but the shared session is preserved for the foreground command.
        assert auth_store.get_cloud_session() is not None
        assert auth_store.get_cloud_session().refresh_token == "RT0"

    # (3) end-to-end: a watcher-context Client surfaces Unauthenticated on a
    # fatal refresh but does NOT clear the session.
    def test_watcher_client_does_not_clear_on_fatal_refresh(self, persisted, monkeypatch):
        import comfy_cli.comfy_client as comfy_client

        self._persist_valid()

        def boom(**kw):
            raise oauth.OAuthRefreshError(
                "refresh failed: HTTP 400",
                details={"status": 400, "body": "refresh token reuse detected"},
            )

        monkeypatch.setattr(oauth, "refresh_tokens", boom)

        def first_401(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(comfy_client._OPENER, "open", first_401)

        target = comfy_client.Target(
            kind="cloud",
            base_url="https://c",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="AT0",
        )
        client = comfy_client.Client(target, clear_session_on_auth_failure=False)
        with pytest.raises(comfy_client.Unauthenticated):
            client.get_job_status("p1")

        # The detached watcher never wiped the shared login.
        assert auth_store.get_cloud_session() is not None
        assert auth_store.get_cloud_session().refresh_token == "RT0"


class TestLockFileStability:
    """DEFECT B root cause: the refresh lock must be a stable sidecar file, NOT
    the data file. ``save_cloud_session`` persists via ``os.replace`` (new
    inode each write); locking the data file directly breaks ``flock`` mutual
    exclusion exactly during a refresh, letting two processes replay the same
    rotated token and trip reuse-detection."""

    def test_lock_path_is_sidecar_not_data_file(self, tmp_path, monkeypatch):
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
        assert auth_store.lock_path() == tmp_path / "secrets.json.lock"
        assert auth_store.lock_path() != auth_store.secrets_path()

    def test_lock_inode_survives_atomic_replace(self, tmp_path, monkeypatch):
        """The lock file's identity (inode) is stable across many session
        persists, so every process serializes on the same lock."""
        path = tmp_path / "secrets.json"
        monkeypatch.setattr(auth_store, "secrets_path", lambda: path)

        def _save(rt: str) -> None:
            auth_store.save_cloud_session(
                base_url="https://c",
                resource="https://c/api",
                client_id="cid",
                scope="s",
                access_token="AT",
                refresh_token=rt,
                token_type="Bearer",
                expires_at=int(time.time()) + 3600,
            )

        _save("RT0")
        lock_file = auth_store.lock_path()
        inode_before = lock_file.stat().st_ino
        # Many atomic replaces of the *data* file...
        for i in range(5):
            _save(f"RT{i + 1}")
        # ...leave the lock file's inode unchanged (data file inode does change).
        assert lock_file.stat().st_ino == inode_before
