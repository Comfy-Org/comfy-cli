"""The renewable partner-node token local runs send."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import threading
import time
import urllib.error

import pytest

from comfy_cli import credentials
from comfy_cli.auth import store as auth_store
from comfy_cli.cloud import command, oauth, partner_token
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode

STAGING = "https://stagingcloud.comfy.org"
STAGING_API = "https://stagingapi.comfy.org"


def _jwt(**claims) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJSUzI1NiJ9.{payload}.signature"


def _access(*, user="user-1", workspace="ws-1", family="fam-1", client="comfy-cli", ttl=900, iat=None) -> str:
    now = int(time.time()) if iat is None else iat
    return _jwt(
        sub=user,
        workspace_id=workspace,
        fam=family,
        client_id=client,
        aud=["comfy-cloud"],
        iat=now,
        exp=now + ttl,
        jti=f"a-{time.monotonic_ns()}",
    )


def _partner(*, sid="sid-1", left=5400, lifetime=5400, aud=("comfy-partner-node",)) -> str:
    exp = int(time.time()) + left
    return _jwt(aud=list(aud), sid=sid, sub="user-1", iat=exp - lifetime, exp=exp, jti=f"p-{time.monotonic_ns()}")


def _sign_in(*, base_url=STAGING, left=900, keep=False, **claims) -> str:
    token = _access(**claims)
    auth_store.save_cloud_session(
        base_url=base_url,
        resource=f"{base_url}/api",
        client_id="comfy-cli",
        scope="comfy-cloud:jobs:write",
        access_token=token,
        refresh_token="refresh-1",
        token_type="Bearer",
        expires_at=int(time.time()) + left,
        keep_partner_node_token=keep,
    )
    return token


def _cache(token: str, **changes) -> None:
    key = dict(partner_token.login_key(auth_store.get_cloud_session()) or {}, **changes)
    auth_store.save_partner_node_token({"login": key, "token": token})


class FakeServer:
    """Stands in for ingest and comfy-api at ``partner_token._post``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.mint: list = []
        self.renew: list = []
        self.revoke: list = []
        self.delay = 0.0
        self._lock = threading.Lock()

    def __call__(self, url, bearer, body):
        with self._lock:
            self.calls.append((url, bearer, body))
        time.sleep(self.delay)
        if url.endswith("/api/auth/token"):
            answer = self.mint.pop(0) if self.mint else {"token": _partner(sid=f"sid-{len(self.calls)}")}
        elif url.endswith("/auth/partner-node/renew"):
            answer = self.renew.pop(0)
        elif url.endswith("/api/auth/token/revoke"):
            answer = self.revoke.pop(0) if self.revoke else None
        else:
            raise AssertionError(f"unexpected POST {url}")
        if callable(answer):
            answer = answer()
        if isinstance(answer, partner_token.PartnerTokenError):
            raise answer
        return answer

    def paths(self) -> list[str]:
        return [url for url, _bearer, _body in self.calls]


@pytest.fixture(autouse=True)
def _no_network_refresh(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("a sign-in refresh reached the network")

    monkeypatch.setattr(oauth, "_post_form", refuse)


def _signed_out(**kwargs):
    raise oauth.OAuthRefreshError("refresh failed", details={"status": 400, "body": '{"error":"invalid_grant"}'})


@pytest.fixture
def server(monkeypatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(partner_token, "_post", fake)
    return fake


@pytest.fixture
def refreshes(monkeypatch) -> list[str]:
    """Answer sign-in refreshes with a new access token for the same sign-in."""
    issued: list[str] = []

    def refresh_tokens(**kwargs):
        token = _access()
        issued.append(token)
        return oauth.TokenSet(
            access_token=token,
            refresh_token=f"refresh-{len(issued) + 1}",
            token_type="Bearer",
            expires_in=900,
            expires_at=int(time.time()) + 900,
            scope="comfy-cloud:jobs:write",
        )

    monkeypatch.setattr(oauth, "refresh_tokens", refresh_tokens)
    return issued


def _local_run() -> tuple[str, str] | None:
    return credentials.resolve_partner_credential(partner_node_token=True)


def _refusal(status: int, code: str | None = None) -> partner_token.PartnerTokenError:
    return partner_token.PartnerTokenError(status, code)


class TestMintAndReuse:
    def test_a_signed_in_local_run_sends_a_minted_partner_node_token(self, server, refreshes):
        access = _sign_in()
        field, token = _local_run()
        assert field == "auth_token_comfy_org"
        assert partner_token._parse(token) is not None
        assert server.calls == [(f"{STAGING}/api/auth/token", access, {"resource": "partner-node"})]
        assert refreshes == []

    def test_later_runs_reuse_the_cached_token_without_a_request(self, server):
        _sign_in()
        first = _local_run()
        second = _local_run()
        assert first == second
        assert server.paths() == [f"{STAGING}/api/auth/token"]

    def test_a_refreshed_sign_in_keeps_its_cached_token(self, server, refreshes):
        _sign_in()
        first = _local_run()
        credentials.get_session(refresh=True, force=True)
        assert auth_store.get_cloud_session().access_token == refreshes[-1]
        assert _local_run() == first
        assert len(server.calls) == 1

    def test_the_token_is_stored_in_the_secret_store_with_its_sign_in(self, server):
        _sign_in()
        _field, token = _local_run()
        record = json.loads(auth_store.secrets_path().read_text())["partner_node_token"]
        assert record == {
            "login": {
                "base_url": STAGING,
                "client": "comfy-cli",
                "user": "user-1",
                "workspace": "ws-1",
                "family": "fam-1",
            },
            "token": token,
        }

    def test_concurrent_runs_share_one_mint(self, server):
        _sign_in()
        server.delay = 0.2
        outcomes: list = [None] * 4

        def worker(index: int) -> None:
            try:
                outcomes[index] = _local_run()
            except BaseException as error:  # noqa: BLE001 — reported by the assertion below
                outcomes[index] = error

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(isinstance(outcome, tuple) for outcome in outcomes), outcomes
        assert len(set(outcomes)) == 1
        assert partner_token._parse(outcomes[0][1]) is not None
        assert server.paths() == [f"{STAGING}/api/auth/token"]


class TestCacheKey:
    @pytest.mark.parametrize(
        "changed",
        [{"user": "user-2"}, {"workspace": "ws-2"}, {"family": "fam-2"}, {"client": "comfy-desktop"}],
        ids=["another-account", "another-workspace", "a-new-login", "another-client"],
    )
    def test_a_token_cached_for_another_sign_in_is_never_sent(self, server, changed):
        _sign_in()
        stale = _partner(sid="other")
        _cache(stale, **changed)
        _field, token = _local_run()
        assert token != stale
        assert server.paths() == [f"{STAGING}/api/auth/token"]

    def test_a_token_cached_for_another_environment_is_never_sent(self, server):
        _sign_in(base_url="https://cloud.comfy.org")
        stale = _partner(sid="other")
        _cache(stale, base_url=STAGING)
        _field, token = _local_run()
        assert token != stale
        assert server.paths() == ["https://cloud.comfy.org/api/auth/token"]

    def test_logging_in_again_drops_the_cached_token(self, server):
        _sign_in()
        _local_run()
        _sign_in(user="user-2")
        assert auth_store.get_partner_node_token() is None

    def test_the_cached_token_survives_other_secret_store_writes(self, server):
        _sign_in()
        _local_run()
        auth_store.set("civitai", "key")
        assert auth_store.get_partner_node_token() is not None

    @pytest.mark.parametrize("base_url", ["http://localhost:8080", "https://pr-1.testenvs.comfy.org"])
    def test_an_environment_without_a_known_comfy_api_gets_the_sign_in_token(self, server, base_url):
        access = _sign_in(base_url=base_url)
        assert _local_run() == ("auth_token_comfy_org", access)
        assert server.calls == []

    def test_a_sign_in_token_naming_no_family_gets_the_sign_in_token(self, server):
        access = _sign_in(family="")
        assert _local_run() == ("auth_token_comfy_org", access)
        assert server.calls == []


class TestRenewal:
    def test_a_token_with_time_left_is_sent_without_renewing(self, server):
        _sign_in()
        held = _partner(left=partner_token.RENEW_WITHIN_S + 60)
        _cache(held)
        assert _local_run() == ("auth_token_comfy_org", held)
        assert server.calls == []

    @pytest.mark.parametrize(("left_min", "renews"), [(46, False), (44, True)])
    def test_the_renewal_window_is_45_minutes(self, server, left_min, renews):
        _sign_in()
        held = _partner(left=left_min * 60)
        _cache(held)
        renewed = _partner()
        server.renew.append({"token": renewed})
        assert _local_run() == ("auth_token_comfy_org", renewed if renews else held)

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_refused_renewal_mints_even_while_the_token_outlives_a_sign_in_token(self, server, status):
        _sign_in()
        _cache(_partner(left=1200))
        server.renew.append(_refusal(status, "partner_node_session_revoked"))
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)

    def test_a_renewal_a_few_seconds_shorter_is_not_a_session_end(self, server):
        _sign_in()
        _cache(_partner(left=600))
        renewed = _partner(left=5397, lifetime=5397)
        server.renew.append({"token": renewed})
        assert _local_run() == ("auth_token_comfy_org", renewed)
        assert auth_store.get_partner_node_token()["token"] == renewed
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    def test_a_capped_renewal_is_sent_when_there_is_no_live_sign_in_to_mint_with(self, server):
        _sign_in()
        session = auth_store.get_cloud_session()
        _cache(_partner(left=600))
        capped = _partner(left=1200, lifetime=1200)
        server.renew.append({"token": capped})
        assert partner_token.current(session, lambda: None) == capped
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    def test_a_token_that_expires_soon_is_renewed_through_comfy_api(self, server):
        _sign_in()
        held = _partner(left=partner_token.RENEW_WITHIN_S - 60)
        _cache(held)
        renewed = _partner()
        server.renew.append({"token": renewed})
        assert _local_run() == ("auth_token_comfy_org", renewed)
        assert server.calls == [(f"{STAGING_API}/auth/partner-node/renew", held, None)]
        assert auth_store.get_partner_node_token()["token"] == renewed

    def test_an_expired_token_is_renewed_too(self, server):
        _sign_in()
        held = _partner(left=-600)
        _cache(held)
        renewed = _partner()
        server.renew.append({"token": renewed})
        assert _local_run() == ("auth_token_comfy_org", renewed)

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_refused_renewal_mints_a_new_token(self, server, status):
        access = _sign_in()
        _cache(_partner(left=60))
        server.renew.append(_refusal(status, "partner_node_session_revoked"))
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)
        assert server.calls[1] == (f"{STAGING}/api/auth/token", access, {"resource": "partner-node"})
        assert auth_store.get_partner_node_token()["token"] == minted

    def test_a_refused_renewal_and_a_refused_mint_drop_the_dead_token(self, server, refreshes):
        _sign_in(left=300)
        _cache(_partner(left=60))
        server.renew.append(_refusal(403))
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])
        assert auth_store.get_partner_node_token() == {"login": partner_token.login_key(auth_store.get_cloud_session())}

    def test_a_dropped_token_still_has_logout_revoke_the_sign_ins_sessions(self, server, refreshes, capsys):
        _sign_in(left=300)
        _cache(_partner(left=60))
        server.renew.append(_refusal(403))
        server.mint.append(_refusal(503, "partner_node_token_disabled"))
        _local_run()
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        command.logout_cmd()
        envelope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert server.calls[-1] == (f"{STAGING}/api/auth/token/revoke", refreshes[-1], None)

    def test_a_renewal_stopped_at_the_session_end_mints_a_new_one(self, server):
        _sign_in()
        _cache(_partner(left=600))
        server.renew.append({"token": _partner(left=900, lifetime=900)})
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)
        assert auth_store.get_partner_node_token()["token"] == minted

    def test_a_renewal_stopped_at_the_session_end_is_sent_when_the_mint_fails(self, server):
        _sign_in()
        _cache(_partner(left=600))
        capped = _partner(left=1200, lifetime=1200)
        server.renew.append({"token": capped})
        server.mint.append(_refusal(503, "partner_node_token_disabled"))
        assert _local_run() == ("auth_token_comfy_org", capped)

    def test_a_renewal_that_falls_inside_a_sign_in_tokens_life_during_the_mint_is_not_sent(
        self, server, refreshes, monkeypatch
    ):
        _sign_in(left=600)
        _cache(_partner(left=600))
        server.renew.append({"token": _partner(left=905, lifetime=905)})
        offset = [0.0]
        real_time = time.time
        monkeypatch.setattr(
            partner_token, "time", type("Clock", (), {"time": staticmethod(lambda: real_time() + offset[0])})
        )

        def slow_refusal():
            offset[0] += 10
            return _refusal(503, "partner_node_token_disabled")

        server.mint.append(slow_refusal)
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])

    def test_a_renewal_stopped_within_a_sign_in_tokens_life_is_not_sent(self, server, refreshes):
        _sign_in(left=600)
        _cache(_partner(left=600))
        server.renew.append({"token": _partner(left=840, lifetime=840)})
        server.mint.append(_refusal(503, "partner_node_token_disabled"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])

    def test_a_renewal_stopped_at_the_session_end_is_not_cached_so_the_next_run_tries_a_mint_again(
        self, server, refreshes
    ):
        _sign_in(left=600)
        held = _partner(left=600)
        _cache(held)
        server.renew.append({"token": _partner(left=840, lifetime=840)})
        server.mint.append(_refusal(503, "partner_node_token_disabled"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])
        assert auth_store.get_partner_node_token()["token"] == held
        server.renew.append({"token": _partner(left=810, lifetime=810)})
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)

    def test_tokens_shorter_than_the_renewal_window_renew_without_minting(self, server):
        _sign_in()
        _cache(_partner(left=600, lifetime=1800))
        renewed = _partner(left=1800, lifetime=1800)
        server.renew.append({"token": renewed})
        assert _local_run() == ("auth_token_comfy_org", renewed)
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    def test_a_renewal_for_another_session_is_not_used(self, server):
        _sign_in()
        held = _partner(left=1200)
        _cache(held)
        server.renew.append({"token": _partner(sid="someone-else")})
        assert _local_run() == ("auth_token_comfy_org", held)
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    def test_429_keeps_a_cached_token_that_outlives_a_sign_in_token_and_mints_nothing(self, server):
        _sign_in()
        held = _partner(left=1200)
        _cache(held)
        server.renew.append(_refusal(429, "rate_limited"))
        assert _local_run() == ("auth_token_comfy_org", held)
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    @pytest.mark.parametrize("left", [600, 30])
    def test_429_for_a_token_inside_a_sign_in_tokens_life_sends_the_sign_in_token(self, server, refreshes, left):
        _sign_in(left=300)
        _cache(_partner(left=left))
        server.renew.append(_refusal(429, "rate_limited"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])
        assert server.paths() == [f"{STAGING_API}/auth/partner-node/renew"]

    @pytest.mark.parametrize(("status", "code"), [(502, None), (0, "URLError")])
    def test_a_failed_renewal_of_a_token_inside_a_sign_in_tokens_life_mints(self, server, status, code):
        _sign_in()
        _cache(_partner(left=600))
        server.renew.append(_refusal(status, code))
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)

    @pytest.mark.parametrize(
        ("status", "code"), [(404, None), (502, None), (503, None), (0, "URLError"), (200, "unusable_token")]
    )
    def test_a_failed_renewal_of_an_expired_token_mints_a_new_one(self, server, status, code):
        _sign_in()
        _cache(_partner(left=30))
        server.renew.append(_refusal(status, code))
        minted = _partner(sid="sid-new")
        server.mint.append({"token": minted})
        assert _local_run() == ("auth_token_comfy_org", minted)
        assert auth_store.get_partner_node_token()["token"] == minted

    def test_a_failed_renewal_and_a_failed_mint_keep_the_expired_token_for_the_next_renewal(self, server, refreshes):
        _sign_in(left=300)
        held = _partner(left=-60)
        _cache(held)
        server.renew.append(_refusal(502))
        server.mint.append(_refusal(0, "URLError"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])
        assert auth_store.get_partner_node_token()["token"] == held


class TestFallback:
    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (403, "oauth_token_mint_refused"),
            (403, "insufficient_scope"),
            (404, None),
            (401, "oauth_grant_ended"),
            (503, "partner_node_token_disabled"),
            (500, None),
            (0, "URLError"),
            (200, "unusable_token"),
        ],
    )
    def test_a_refused_or_failed_mint_sends_a_refreshed_sign_in_token(self, server, refreshes, status, code):
        _sign_in(left=600)
        server.mint.append(_refusal(status, code))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])
        assert len(refreshes) == 1
        assert auth_store.get_cloud_session().access_token == refreshes[-1]
        assert auth_store.get_partner_node_token() is None

    def test_a_sign_in_token_under_a_minute_old_is_not_refreshed_again(self, server, refreshes):
        access = _sign_in(left=890)
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() == ("auth_token_comfy_org", access)
        assert refreshes == []

    def test_a_sign_in_token_with_13_minutes_left_is_refreshed_for_the_run(self, server, refreshes):
        _sign_in(left=780)
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() == ("auth_token_comfy_org", refreshes[-1])

    def test_a_signed_out_sign_in_still_sends_its_valid_access_token(self, server, monkeypatch):
        access = _sign_in(left=600)
        server.mint.append(_refusal(401, "oauth_grant_ended"))

        def signed_out(**kwargs):
            raise oauth.OAuthRefreshError(
                "refresh failed", details={"status": 400, "body": '{"error":"invalid_grant"}'}
            )

        monkeypatch.setattr(oauth, "refresh_tokens", signed_out)
        assert _local_run() == ("auth_token_comfy_org", access)
        assert auth_store.get_cloud_session() is not None

    @pytest.mark.parametrize(
        "change",
        [{"access_token": _access(user="user-2")}, {"expires_at": 1}, None],
        ids=["another-account", "expired", "signed-out"],
    )
    def test_a_mint_needs_the_same_live_sign_in(self, server, change):
        _sign_in()
        session = auth_store.get_cloud_session()
        live = None if change is None else dataclasses.replace(session, **change)
        assert partner_token.current(session, lambda: live) is None
        assert server.calls == []

    @pytest.mark.parametrize("record", [["not", "a", "dict"], "token", {"login": "x", "token": 1}])
    def test_a_malformed_cached_record_is_ignored(self, server, record):
        _sign_in()
        auth_store.save_partner_node_token(
            {"login": partner_token.login_key(auth_store.get_cloud_session()), "token": "bad"}
        )
        data = json.loads(auth_store.secrets_path().read_text())
        data["partner_node_token"] = record
        auth_store.secrets_path().write_text(json.dumps(data))
        _field, token = _local_run()
        assert partner_token._parse(token) is not None
        assert server.paths() == [f"{STAGING}/api/auth/token"]

    def test_an_unexpected_error_sends_the_sign_in_token(self, server, monkeypatch):
        access = _sign_in()
        monkeypatch.setattr(partner_token, "current", lambda *a: 1 / 0)
        assert _local_run() == ("auth_token_comfy_org", access)

    def test_a_held_secrets_lock_gives_up_on_the_partner_node_token(self, server, monkeypatch):
        access = _sign_in()
        monkeypatch.setattr(credentials, "_PARTNER_NODE_LOCK_TIMEOUT_S", 0.1)
        acquired, release = threading.Event(), threading.Event()

        def hold():
            from comfy_cli import locking

            with locking.file_lock(auth_store.lock_path()):
                acquired.set()
                release.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        acquired.wait(5)
        try:
            assert credentials._partner_node_token(credentials._SignIn()) is None
        finally:
            release.set()
            holder.join()
        assert server.calls == []
        assert access


class TestUnchangedCredentials:
    def test_an_api_key_run_is_unchanged(self, server, monkeypatch):
        _sign_in()
        monkeypatch.setenv("COMFY_API_KEY", "partner-key")
        assert _local_run() == ("api_key_comfy_org", "partner-key")
        assert server.calls == []

    def test_a_stored_key_without_a_sign_in_is_unchanged(self, server):
        auth_store.set(credentials.CLOUD_API_KEY_PROVIDER, "stored-key")
        assert _local_run() == ("api_key_comfy_org", "stored-key")
        assert server.calls == []

    def test_a_handed_in_token_is_sent_unchanged(self, server, monkeypatch):
        _sign_in()
        _cache(_partner())
        monkeypatch.setenv(credentials.CLOUD_BEARER_ENV_VAR, "handed-in-token")
        assert _local_run() == ("auth_token_comfy_org", "handed-in-token")
        assert server.calls == []

    def test_deploy_run_still_sends_the_sign_in_token(self, server):
        access = _sign_in()
        _cache(_partner())
        assert credentials.resolve_partner_credential() == ("auth_token_comfy_org", access)
        assert server.calls == []


class TestPost:
    def _http_error(self, status, body: bytes):
        return urllib.error.HTTPError("https://x", status, "refused", {}, io.BytesIO(body))

    @pytest.mark.parametrize(
        ("body", "code"),
        [
            (b'{"code":"oauth_grant_ended","message":"sign in again"}', "oauth_grant_ended"),
            (b'{"error":{"type":"auth_type_not_allowed"}}', "auth_type_not_allowed"),
            (b'{"message":"Partner-node session cannot be renewed"}', None),
            (b"<html>", None),
        ],
    )
    def test_an_http_refusal_carries_its_status_and_code(self, monkeypatch, body, code):
        def refuse(*args, **kwargs):
            raise self._http_error(403, body)

        monkeypatch.setattr("comfy_cli.http.request_json", refuse)
        with pytest.raises(partner_token.PartnerTokenError) as raised:
            partner_token._post(f"{STAGING}/api/auth/token", "t", {"resource": "partner-node"})
        assert (raised.value.status, raised.value.code) == (403, code)

    @pytest.mark.parametrize("error", [urllib.error.URLError("down"), TimeoutError("slow"), ConnectionResetError()])
    def test_a_network_failure_is_status_0(self, monkeypatch, error):
        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr("comfy_cli.http.request_json", fail)
        with pytest.raises(partner_token.PartnerTokenError) as raised:
            partner_token._post(f"{STAGING}/api/auth/token", "t", None)
        assert raised.value.status == 0

    def test_the_request_carries_the_bearer_and_body(self, monkeypatch):
        seen = {}

        def answer(url, target, **kwargs):
            seen.update(url=url, bearer=target.auth_token, body=kwargs["body"], method=kwargs["method"])
            return 200, {"token": "t"}

        monkeypatch.setattr("comfy_cli.http.request_json", answer)
        assert partner_token._post(f"{STAGING}/api/auth/token", "bearer", {"resource": "partner-node"}) == {
            "token": "t"
        }
        assert seen == {
            "url": f"{STAGING}/api/auth/token",
            "bearer": "bearer",
            "body": {"resource": "partner-node"},
            "method": "POST",
        }


class TestParse:
    @pytest.mark.parametrize("aud", [["comfy-partner-node"], "comfy-partner-node"])
    def test_a_partner_node_audience_is_accepted_as_a_list_or_a_string(self, aud):
        assert partner_token._parse(_jwt(aud=aud, sid="s", exp=1)) is not None

    @pytest.mark.parametrize(
        "claims",
        [
            {"aud": ["comfy-cloud"], "sid": "s", "exp": 1},
            {"aud": "comfy-partner-node", "exp": 1},
            {"aud": "comfy-partner-node", "sid": "s"},
            {"aud": "comfy-partner-node", "sid": "s", "exp": True},
        ],
        ids=["sign-in-audience", "no-sid", "no-exp", "bool-exp"],
    )
    def test_anything_else_is_not_a_partner_node_token(self, claims):
        assert partner_token._parse(_jwt(**claims)) is None

    @pytest.mark.parametrize(
        "exp",
        [float("inf"), float("nan"), 10**400, -5, 0, 10**12],
        ids=["inf", "nan", "huge", "negative", "zero", "far"],
    )
    def test_an_absurd_expiry_is_not_a_partner_node_token(self, exp):
        assert partner_token._parse(_jwt(aud="comfy-partner-node", sid="s", exp=exp)) is None

    def test_an_absurd_issue_time_only_drops_the_lifetime(self):
        token = partner_token._parse(_jwt(aud="comfy-partner-node", sid="s", exp=2_000_000_000, iat=10**400))
        assert token is not None and token.lifetime is None

    @pytest.mark.parametrize("token", [None, "", "a.b", "a.!!!.c", "a.bnVsbA.c"])
    def test_a_malformed_token_is_not_a_partner_node_token(self, token):
        assert partner_token._parse(token) is None


class TestLogout:
    @pytest.fixture(autouse=True)
    def _no_tracking(self, monkeypatch):
        monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **k: None)

    def _logout(self, capsys) -> dict:
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        command.logout_cmd()
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    def test_logout_revokes_with_the_sign_in_token_and_drops_the_cached_token(self, server, capsys):
        access = _sign_in()
        _local_run()
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert server.calls[-1] == (f"{STAGING}/api/auth/token/revoke", access, None)
        assert auth_store.get_partner_node_token() is None
        assert auth_store.get_cloud_session() is None

    def test_an_expired_sign_in_is_refreshed_so_the_revoke_reaches_every_session(self, server, refreshes, capsys):
        _sign_in(left=-60)
        _cache(_partner(left=-600))
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", refreshes[-1], None)]
        assert auth_store.get_cloud_session() is None

    def test_an_invalid_grant_refresh_is_not_taken_as_an_ended_sign_in(self, server, monkeypatch, capsys):
        monkeypatch.setattr(oauth, "refresh_tokens", _signed_out)
        _sign_in(left=-60)
        held = _partner(left=-600)
        _cache(held)
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        command.logout_cmd()
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip().splitlines()[-1])["data"]["partner_node_revocation"] == "token"
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", held, None)]
        assert "Revoked only the cached partner-node token" in captured.err

    def test_an_invalid_grant_refresh_without_a_cached_token_reports_failed(self, server, monkeypatch, capsys):
        monkeypatch.setattr(oauth, "refresh_tokens", _signed_out)
        _sign_in(left=-60)
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "failed"
        assert server.calls == []

    def test_an_unreachable_refresh_revokes_with_the_cached_token(self, server, monkeypatch, capsys):
        def unreachable(**kwargs):
            raise oauth.OAuthRefreshError("down", details={"status": 0})

        monkeypatch.setattr(oauth, "refresh_tokens", unreachable)
        _sign_in(left=-60)
        held = _partner(left=-600)
        _cache(held)
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "token"
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", held, None)]
        assert auth_store.get_cloud_session() is None

    def test_an_expired_sign_in_without_a_cached_token_is_still_revoked(self, server, refreshes, capsys):
        _sign_in(left=-60)
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert len(refreshes) == 1
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", refreshes[-1], None)]

    def test_a_refused_sign_in_revoke_falls_back_to_the_cached_token(self, server, capsys):
        access = _sign_in()
        held = _partner()
        _cache(held)
        server.revoke.append(_refusal(500))
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "token"
        assert [bearer for _url, bearer, _body in server.calls] == [access, held]

    def test_a_failed_revoke_still_signs_out_and_says_so(self, server, capsys):
        _sign_in()
        _cache(_partner())
        server.revoke.extend([_refusal(0, "URLError"), _refusal(0, "URLError")])
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        command.logout_cmd()
        captured = capsys.readouterr()
        envelope = json.loads(captured.out.strip().splitlines()[-1])
        assert envelope["ok"] is True
        assert envelope["data"]["partner_node_revocation"] == "failed"
        assert "Could not revoke this sign-in's partner-node tokens" in captured.err
        assert auth_store.get_cloud_session() is None
        assert auth_store.get_partner_node_token() is None

    def test_logout_reads_and_clears_the_store_under_one_lock(self, server, monkeypatch, capsys):
        from comfy_cli import locking

        _sign_in()
        held_lock = []
        real_clear = auth_store.clear_cloud_session

        def clear():
            held_lock.append(any(depth >= 1 for depth in locking._held_depths().values()))
            return real_clear()

        monkeypatch.setattr(auth_store, "clear_cloud_session", clear)
        self._logout(capsys)
        assert held_lock == [True]

    def test_logout_without_a_cached_token_still_revokes_the_sign_ins_sessions(self, server, capsys):
        access = _sign_in()
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", access, None)]

    def test_logout_of_an_environment_without_partner_node_tokens_makes_no_request(self, server, capsys):
        _sign_in(base_url="http://localhost:8080")
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] is None
        assert server.calls == []

    def test_logout_clears_the_store_before_any_revoke_request(self, server, monkeypatch):
        _sign_in()
        _local_run()
        seen = []

        def revoke(url, bearer, body):
            seen.append((auth_store.get_cloud_session(), auth_store.get_partner_node_token()))
            raise KeyboardInterrupt

        monkeypatch.setattr(partner_token, "_post", revoke)
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        with pytest.raises(KeyboardInterrupt):
            command.logout_cmd()
        assert seen == [(None, None)]

    def test_a_cached_token_for_another_environment_is_revoked_at_its_own(self, server, monkeypatch, capsys):
        monkeypatch.setattr(oauth, "refresh_tokens", _signed_out)
        _sign_in(left=-60)
        held = _partner()
        _cache(held, base_url="https://cloud.comfy.org")
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "token"
        assert server.calls == [("https://cloud.comfy.org/api/auth/token/revoke", held, None)]

    def test_a_cached_token_of_another_login_is_revoked_with_itself_too(self, server, capsys):
        access = _sign_in()
        held = _partner()
        _cache(held, family="fam-other")
        envelope = self._logout(capsys)
        assert envelope["data"]["partner_node_revocation"] == "token"
        assert server.calls == [
            (f"{STAGING}/api/auth/token/revoke", access, None),
            (f"{STAGING}/api/auth/token/revoke", held, None),
        ]


class TestReplacedSignIn:
    """A new login, the setup sign-in and a stale-session clear revoke only the cached token's own session."""

    @pytest.fixture(autouse=True)
    def _no_side_effects(self, monkeypatch):
        monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **k: None)
        monkeypatch.setattr(command.knowledge, "refresh_if_stale", lambda: None)

    def _login_result(self, user: str) -> oauth.LoginResult:
        tokens = oauth.TokenSet(
            access_token=_access(user=user, family=f"fam-{user}"),
            refresh_token="refresh-new",
            token_type="Bearer",
            expires_in=900,
            expires_at=int(time.time()) + 900,
            scope="comfy-cloud:jobs:write",
        )
        return oauth.LoginResult(
            tokens=tokens,
            client_id="comfy-cli",
            base_url=STAGING,
            resource=f"{STAGING}/api",
            scope="comfy-cloud:jobs:write",
            redirect_uri="http://127.0.0.1:1/callback",
        )

    def _login(self, monkeypatch, capsys, user="user-2"):
        monkeypatch.setattr(command, "run_login", lambda **kw: self._login_result(user))
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud login", version="test"))
        command.login_cmd(no_browser=True, timeout=1)
        return capsys.readouterr()

    def _setup(self, monkeypatch, capsys):
        from comfy_cli.command import setup

        monkeypatch.setattr(oauth, "run_login", lambda **kw: self._login_result("user-2"))
        assert setup._auth_browser(None) is True
        return capsys.readouterr()

    def _set_base_url(self, capsys, url="https://testcloud.comfy.org"):
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud set-base-url", version="test"))
        command.set_base_url_cmd(url=url, clear=False)
        return capsys.readouterr()

    def test_logging_in_again_revokes_only_the_cached_token(self, server, monkeypatch, capsys):
        _sign_in()
        _local_run()
        held = auth_store.get_partner_node_token()["token"]
        captured = self._login(monkeypatch, capsys)
        assert server.calls[1:] == [(f"{STAGING}/api/auth/token/revoke", held, None)]
        assert auth_store.get_partner_node_token() is None
        assert partner_token.login_key(auth_store.get_cloud_session())["user"] == "user-2"
        assert "Could not revoke" not in captured.err

    def test_setup_sign_in_revokes_only_the_cached_token(self, server, monkeypatch, capsys):
        _sign_in()
        _local_run()
        held = auth_store.get_partner_node_token()["token"]
        self._setup(monkeypatch, capsys)
        assert server.calls[1:] == [(f"{STAGING}/api/auth/token/revoke", held, None)]
        assert auth_store.get_partner_node_token() is None

    def test_clearing_a_stale_sign_in_revokes_only_its_cached_token(self, server, capsys):
        _sign_in(left=-60)
        held = _partner()
        _cache(held)
        self._set_base_url(capsys)
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", held, None)]
        assert auth_store.get_cloud_session() is None

    def test_the_replaced_token_is_revoked_at_its_own_environment(self, server, capsys):
        _sign_in(base_url="https://cloud.comfy.org", left=-60)
        held = _partner()
        _cache(held)
        self._set_base_url(capsys, url="https://stagingcloud.comfy.org")
        assert server.calls == [("https://cloud.comfy.org/api/auth/token/revoke", held, None)]

    @pytest.mark.parametrize("flow", ["login", "setup", "set-base-url"])
    def test_a_replaced_sign_in_without_a_cached_token_makes_no_request(self, server, monkeypatch, capsys, flow):
        _sign_in(left=-60 if flow == "set-base-url" else 900)
        if flow == "login":
            self._login(monkeypatch, capsys)
        elif flow == "setup":
            self._setup(monkeypatch, capsys)
        else:
            self._set_base_url(capsys)
        assert server.calls == []

    def test_a_dropped_token_marker_makes_no_request(self, server, monkeypatch, capsys):
        _sign_in()
        auth_store.save_partner_node_token({"login": partner_token.login_key(auth_store.get_cloud_session())})
        self._login(monkeypatch, capsys)
        assert server.calls == []

    @pytest.mark.parametrize("flow", ["login", "setup", "set-base-url"])
    def test_a_failed_revoke_of_the_replaced_token_warns(self, server, monkeypatch, capsys, flow):
        _sign_in(left=-60 if flow == "set-base-url" else 900)
        _cache(_partner())
        server.revoke.append(_refusal(0, "URLError"))
        if flow == "login":
            captured = self._login(monkeypatch, capsys)
            text = captured.err
            assert json.loads(captured.out.strip().splitlines()[-1])["ok"] is True
        elif flow == "setup":
            text = self._setup(monkeypatch, capsys).out
        else:
            captured = self._set_base_url(capsys)
            text = captured.err
            assert json.loads(captured.out.strip().splitlines()[-1])["data"]["cleared_stale_session"] is True
        assert "Could not revoke the partner-node token of the replaced sign-in" in text

    @pytest.mark.parametrize("flow", ["login", "setup", "set-base-url"])
    def test_the_replaced_sign_in_is_dropped_before_its_token_is_revoked(self, server, monkeypatch, capsys, flow):
        _sign_in(left=-60 if flow == "set-base-url" else 900)
        _cache(_partner())
        seen = []

        def interrupted(url, bearer, body):
            seen.append(auth_store.get_partner_node_token())
            raise KeyboardInterrupt

        monkeypatch.setattr(partner_token, "_post", interrupted)
        with pytest.raises(KeyboardInterrupt):
            if flow == "login":
                self._login(monkeypatch, capsys)
            elif flow == "setup":
                self._setup(monkeypatch, capsys)
            else:
                self._set_base_url(capsys)
        assert seen == [None]
        session = auth_store.get_cloud_session()
        if flow == "set-base-url":
            assert session is None
        else:
            assert partner_token.login_key(session)["user"] == "user-2"

    @pytest.mark.parametrize("left", [900, None], ids=["live-session-for-another-url", "no-session"])
    def test_set_base_url_that_clears_nothing_makes_no_request(self, server, capsys, left):
        if left is not None:
            _sign_in(left=left)
            _cache(_partner())
        captured = self._set_base_url(capsys)
        assert json.loads(captured.out.strip().splitlines()[-1])["data"]["cleared_stale_session"] is False
        assert server.calls == []

    def test_set_base_url_holds_the_secrets_lock_from_the_read_to_the_clear(self, server, monkeypatch, capsys):
        from comfy_cli import locking

        _sign_in(left=-60)
        held_lock = []
        real_clear = auth_store.clear_cloud_session

        def clear():
            held_lock.append(any(depth >= 1 for depth in locking._held_depths().values()))
            return real_clear()

        monkeypatch.setattr(auth_store, "clear_cloud_session", clear)
        self._set_base_url(capsys)
        assert held_lock == [True]


class TestUnwritableStore:
    def test_no_token_is_minted_into_a_store_that_cannot_keep_it(self, server, monkeypatch):
        access = _sign_in()

        def unwritable(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(auth_store, "_write_all", unwritable)
        assert _local_run() == ("auth_token_comfy_org", access)
        assert _local_run() == ("auth_token_comfy_org", access)
        assert server.calls == []

    def test_an_absurd_cached_token_does_not_break_logout(self, server, capsys):
        access = _sign_in()
        login = partner_token.login_key(auth_store.get_cloud_session())
        auth_store.save_partner_node_token(
            {"login": login, "token": _jwt(aud="comfy-partner-node", sid="s", exp=10**400)}
        )
        set_renderer(Renderer(mode=OutputMode.JSON, command="cloud logout", version="test"))
        command.logout_cmd()
        envelope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert envelope["ok"] is True
        assert envelope["data"]["partner_node_revocation"] == "sign_in"
        assert server.calls == [(f"{STAGING}/api/auth/token/revoke", access, None)]


class TestOneRefreshPerResolution:
    @pytest.fixture
    def failing_refresh(self, monkeypatch) -> list[str]:
        monkeypatch.setattr(oauth, "_REFRESH_RETRY_DELAYS_S", ())
        calls: list[str] = []

        def refresh_tokens(**kwargs):
            calls.append(kwargs["refresh_token"])
            raise oauth.OAuthRefreshError(
                calls[-1] and "HTTP 503", details={"status": 503, "body": "", "retryable": True}
            )

        monkeypatch.setattr(oauth, "refresh_tokens", refresh_tokens)
        return calls

    def test_an_expired_sign_in_whose_refresh_fails_is_refreshed_once(self, server, failing_refresh):
        _sign_in(left=20)
        assert _local_run() is None
        assert len(failing_refresh) == 1
        assert server.calls == []

    def test_a_failed_refresh_mints_with_the_access_token_still_valid(self, server, failing_refresh):
        access = _sign_in(left=45)
        _field, token = _local_run()
        assert partner_token._parse(token) is not None
        assert server.calls == [(f"{STAGING}/api/auth/token", access, {"resource": "partner-node"})]
        assert len(failing_refresh) == 1

    def test_a_signed_out_sign_in_is_refreshed_once_and_sends_its_valid_token(self, server, monkeypatch):
        calls: list[str] = []

        def signed_out(**kwargs):
            calls.append(kwargs["refresh_token"])
            _signed_out()

        monkeypatch.setattr(oauth, "refresh_tokens", signed_out)
        access = _sign_in(left=45)
        assert _local_run() == ("auth_token_comfy_org", access)
        assert len(calls) == 1

    def test_a_refresh_that_returns_an_almost_expired_token_is_not_repeated(self, server, monkeypatch):
        calls: list[str] = []

        def short_lived(**kwargs):
            calls.append(kwargs["refresh_token"])
            return oauth.TokenSet(
                access_token=_access(ttl=20),
                refresh_token=f"refresh-{len(calls) + 1}",
                token_type="Bearer",
                expires_in=20,
                expires_at=int(time.time()) + 20,
                scope="comfy-cloud:jobs:write",
            )

        monkeypatch.setattr(oauth, "refresh_tokens", short_lived)
        _sign_in(left=600)
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() is None
        assert len(calls) == 1

    def test_a_refused_mint_still_gets_one_proactive_refresh(self, server, failing_refresh):
        access = _sign_in(left=600)
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() == ("auth_token_comfy_org", access)
        assert len(failing_refresh) == 1


class TestRotatedTokensSave:
    def test_a_failed_save_is_retried_so_the_next_run_does_not_send_the_spent_refresh_token(self, server, monkeypatch):
        monkeypatch.setattr(oauth, "_SAVE_RETRY_DELAY_S", 0)
        sent: list[str] = []

        def refresh_tokens(**kwargs):
            sent.append(kwargs["refresh_token"])
            return oauth.TokenSet(
                access_token=_access(),
                refresh_token=f"refresh-{len(sent) + 1}",
                token_type="Bearer",
                expires_in=20,
                expires_at=int(time.time()) + 20,
                scope="comfy-cloud:jobs:write",
            )

        monkeypatch.setattr(oauth, "refresh_tokens", refresh_tokens)
        real_save = auth_store.save_cloud_session
        failures = [OSError(13, "Permission denied")]

        def save(**kwargs):
            if kwargs.get("keep_partner_node_token") and failures:
                raise failures.pop()
            return real_save(**kwargs)

        monkeypatch.setattr(auth_store, "save_cloud_session", save)
        _sign_in(left=20)
        credentials.get_session(refresh=True)
        assert auth_store.get_cloud_session().refresh_token == "refresh-2"
        credentials.get_session(refresh=True)
        assert sent == ["refresh-1", "refresh-2"]

    def test_a_save_that_keeps_failing_is_tried_twice(self, monkeypatch, refreshes):
        monkeypatch.setattr(oauth, "_SAVE_RETRY_DELAY_S", 0)
        attempts = []

        def save(**kwargs):
            attempts.append(kwargs["refresh_token"])
            raise OSError(28, "No space left on device")

        _sign_in(left=20)
        monkeypatch.setattr(auth_store, "save_cloud_session", save)
        with pytest.raises(oauth.SessionNotSaved):
            credentials.get_session(refresh=True)
        assert len(attempts) == 2
        assert len(refreshes) == 1


class TestUnsavedRefresh:
    """A refresh whose tokens could not be stored spent the stored refresh token; no later refresh may send it."""

    @pytest.fixture
    def unsaved(self, monkeypatch, refreshes):
        monkeypatch.setattr(oauth, "_SAVE_RETRY_DELAY_S", 0)
        real_save = auth_store.save_cloud_session

        def save(**kwargs):
            if kwargs.get("keep_partner_node_token"):
                raise OSError(28, "No space left on device")
            return real_save(**kwargs)

        monkeypatch.setattr(auth_store, "save_cloud_session", save)
        return refreshes

    def test_the_refresh_before_a_mint_is_sent_once(self, server, unsaved):
        stored = _sign_in(left=45)
        assert _local_run() == ("auth_token_comfy_org", stored)
        assert len(unsaved) == 1
        assert server.calls == []

    def test_the_fallback_refresh_is_sent_once(self, server, unsaved):
        stored = _sign_in(left=600)
        server.mint.append(_refusal(403, "oauth_token_mint_refused"))
        assert _local_run() == ("auth_token_comfy_org", stored)
        assert len(unsaved) == 1

    def test_the_refresh_raises_an_os_error_naming_the_unsaved_session(self, unsaved):
        _sign_in(left=10)
        with pytest.raises(oauth.SessionNotSaved) as raised:
            credentials.get_session(refresh=True)
        assert isinstance(raised.value, OSError)
        assert len(unsaved) == 1

    def test_the_fallback_refresh_is_not_repeated_by_the_plain_path(self, server, unsaved):
        stored = _sign_in(base_url="http://localhost:8080", left=45)
        assert _local_run() == ("auth_token_comfy_org", stored)
        assert len(unsaved) == 1
