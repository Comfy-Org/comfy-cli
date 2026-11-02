"""A long data-plane command outlives the sign-in token it started with."""

from __future__ import annotations

import http.client
import io
import urllib.error
from types import SimpleNamespace

import pytest

import comfy_cli.deploy_events as deploy_events
from comfy_cli.credentials import DataPlaneToken
from comfy_cli.deploy_events import JobWatchRequest, watch_job
from comfy_cli.target import Target

_TARGET = Target(kind="cloud", base_url="https://dep.run.comfy.app", path_prefix="/api/v2", auth_token="t0")
_JOB_URL = "https://dep.run.comfy.app/api/v2/jobs/job-1"
_EVENTS_URL = f"{_JOB_URL}/events"


class _SignIn:
    """The stored sign-in: a plain read returns ``proactive``; a forced refresh stores and returns ``forced``."""

    def __init__(self, proactive: str, forced: str | None) -> None:
        self.proactive = proactive
        self.forced = forced
        self.forced_calls = 0

    def __call__(self, *, refresh=True, force=False, allow_clear=True, expect=None, rejected_access_token=None):
        if force:
            self.forced_calls += 1
            if not self.forced:
                return None
            self.proactive = self.forced
            return SimpleNamespace(access_token=self.forced)
        return SimpleNamespace(access_token=self.proactive)


def _unauthorized() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_JOB_URL, 401, "expired", http.client.HTTPMessage(), io.BytesIO(b"{}"))


def _succeeded() -> dict:
    return {
        "id": "job-1",
        "status": "succeeded",
        "outputs": [{"id": "o-1", "node_id": "9", "name": "a.png", "type": "image", "url": "https://x"}],
    }


def _install_gateway(monkeypatch, valid: set[str]) -> list[str | None]:
    """A gateway whose event stream ends at once, and whose job read refuses any token outside ``valid``."""
    sent: list[str | None] = []

    def open_stream(request, *, timeout):
        sent.append(request.get_header("Authorization"))
        return io.BytesIO(b"")

    def request_json(url, target, **_kwargs):
        sent.append(target.auth_token)
        if target.auth_token not in valid:
            raise _unauthorized()
        return 200, _succeeded()

    monkeypatch.setattr(deploy_events, "no_redirect_urlopen", open_stream)
    monkeypatch.setattr(deploy_events, "request_json", request_json)
    return sent


def test_a_sign_in_token_is_reread_before_each_request(monkeypatch) -> None:
    # Given
    sign_in = _SignIn(proactive="t1", forced=None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    token = DataPlaneToken("t0", refreshes=True)

    # When
    first = token.current()
    sign_in.proactive = "t2"
    second = token.current()

    # Then
    assert (first, second) == ("t1", "t2")


def test_a_key_is_never_reread_or_refreshed(monkeypatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda **_: pytest.fail("sign-in consulted"))
    token = DataPlaneToken("comfyui-key", refreshes=False)

    # When / Then
    assert token.current() == "comfyui-key"
    assert token.after_rejection() is False


def test_the_watch_sends_the_freshly_read_token(monkeypatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.credentials.get_session", _SignIn(proactive="t1", forced=None))
    sent = _install_gateway(monkeypatch, valid={"t1"})
    request = JobWatchRequest(_TARGET, _JOB_URL, _EVENTS_URL, DataPlaneToken("t0", refreshes=True))

    # When
    result = watch_job(request, sleep_fn=lambda _s: None)

    # Then
    assert result.job["status"] == "succeeded"
    assert sent == ["Bearer t1", "t1"]


def test_a_401_on_the_final_read_refreshes_once_and_reads_again(monkeypatch) -> None:
    # Given: the stored token still looks valid to our clock, but the gateway refuses it.
    sign_in = _SignIn(proactive="t0", forced="t1")
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    sent = _install_gateway(monkeypatch, valid={"t1"})
    sleeps: list[float] = []
    request = JobWatchRequest(_TARGET, _JOB_URL, _EVENTS_URL, DataPlaneToken("t0", refreshes=True))

    # When
    result = watch_job(request, sleep_fn=sleeps.append)

    # Then
    assert result.job["status"] == "succeeded"
    assert sign_in.forced_calls == 1
    assert sleeps == []
    assert sent[1:] == ["t0", "t1"]


def test_a_401_the_refresh_cannot_cure_surfaces_rather_than_looping(monkeypatch) -> None:
    # Given
    sign_in = _SignIn(proactive="t0", forced="t1")
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    _install_gateway(monkeypatch, valid=set())
    request = JobWatchRequest(_TARGET, _JOB_URL, _EVENTS_URL, DataPlaneToken("t0", refreshes=True))

    # When / Then
    with pytest.raises(urllib.error.HTTPError) as caught:
        watch_job(request, sleep_fn=lambda _s: None)
    assert caught.value.code == 401
    assert sign_in.forced_calls == 1


def test_a_watch_without_a_token_holder_surfaces_the_401(monkeypatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda **_: pytest.fail("sign-in consulted"))
    _install_gateway(monkeypatch, valid=set())

    # When / Then
    with pytest.raises(urllib.error.HTTPError):
        watch_job(JobWatchRequest(_TARGET, _JOB_URL, _EVENTS_URL), sleep_fn=lambda _s: None)


def test_each_asset_request_carries_the_sign_in_as_of_that_request(monkeypatch, tmp_path) -> None:
    # Given: the sign-in rotates between the hash probe and the mint.
    from comfy_cli.deploy_assets import AssetResolveRequest, DeployAssetClient

    sign_in = _SignIn(proactive="t1", forced=None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    sent: list[str | None] = []

    def request_json(url, target, *, method="GET", **_kwargs):
        sent.append(target.auth_token)
        sign_in.proactive = "t2"
        return (200, None) if method == "HEAD" else (201, {"id": "asset-1"})

    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", request_json)
    path = tmp_path / "input.png"
    path.write_bytes(b"abc")
    client = DeployAssetClient("https://dep.run.comfy.app", DataPlaneToken("t0", refreshes=True))

    # When
    result = client.resolve_asset(AssetResolveRequest(local_path=path, file_path="inputs/input.png"))

    # Then
    assert result.asset == {"id": "asset-1"}
    assert sent == ["t1", "t2"]


def test_a_sign_in_to_another_environment_mid_run_is_never_sent_or_refreshed(monkeypatch) -> None:
    # Given: the run started on one environment's sign-in; another shell then signs in elsewhere.
    stored = SimpleNamespace(access_token="t0", base_url="https://cloud.comfy.org")

    refreshing: list[bool] = []

    def get_session(*, refresh=True, force=False, allow_clear=True, rejected_access_token=None):
        if refresh or force:
            refreshing.append(allow_clear)
        return stored

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)
    token = DataPlaneToken("t0", refreshes=True, base_url="https://cloud.comfy.org")
    stored = SimpleNamespace(access_token="other-env", base_url="https://stg.cloud.comfy.org")

    # When / Then: neither a proactive nor a forced refresh touches the other sign-in.
    assert token.current() == "t0"
    assert token.after_rejection() is False
    assert refreshing == []


def test_an_unusable_sign_in_store_keeps_the_watch_on_the_token_it_holds(monkeypatch) -> None:
    # Given: the store's lock cannot be taken once the watch is under way.
    def get_session(**_kwargs):
        raise PermissionError("auth store lock unavailable")

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)
    sent = _install_gateway(monkeypatch, valid={"t0"})
    request = JobWatchRequest(_TARGET, _JOB_URL, _EVENTS_URL, DataPlaneToken("t0", refreshes=True))

    # When
    result = watch_job(request, sleep_fn=lambda _s: None)

    # Then
    assert result.job["status"] == "succeeded"
    assert sent == ["Bearer t0", "t0"]
    assert request.token is not None and request.token.after_rejection() is False


@pytest.mark.parametrize(
    "token",
    ["opaque-token", "", "a.b", "a.!!!.c", "a.bm90IGpzb24.c", "a.WzFd.c", "a.e30.c", "a.eyJzdWIiOiAxfQ.c"],
    ids=["opaque", "empty", "two-parts", "bad-base64", "not-json", "a-list", "no-claims", "non-string-sub"],
)
def test_a_token_that_names_no_account_reads_as_none(token) -> None:
    from comfy_cli.credentials import _token_account

    assert _token_account(token) is None


def test_a_held_token_is_kept_when_the_store_holds_another_account(monkeypatch) -> None:
    # Given: tokens for user u1 then u2 (payloads {"sub": "u1"} / {"sub": "u2"})
    held, other = "h.eyJzdWIiOiAidTEifQ.s", "h.eyJzdWIiOiAidTIifQ.s"
    sign_in = _SignIn(proactive=other, forced="h.eyJzdWIiOiAidTIiLCAibiI6IDJ9.s")
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    token = DataPlaneToken(held, refreshes=True, same_account=True)

    # When / Then: neither read nor a 401 picks up the other account, or spends its refresh
    assert token.current() == held
    assert token.after_rejection() is False
    assert token.account_replaced is True and sign_in.forced_calls == 0


# --- Another login stored between the account check and the refresh (real store) ---

_CLOUD = "https://cloud.example.com"
_EXPIRED, _LATER = 1, 4102444800


def _jwt(account: str, generation: int = 1) -> str:
    import base64
    import json

    payload = json.dumps({"sub": account, "workspace_id": "ws-" + account, "jti": generation}).encode()
    return "h." + base64.urlsafe_b64encode(payload).rstrip(b"=").decode() + ".s"


def _save_login(account: str, *, expires_at: int):
    from comfy_cli.auth import store

    return store.save_cloud_session(
        base_url=_CLOUD,
        resource=_CLOUD + "/api",
        client_id="cid",
        scope="s",
        access_token=_jwt(account),
        refresh_token="refresh-" + account,
        token_type="Bearer",
        expires_at=expires_at,
    )


@pytest.fixture
def login_race(tmp_path, monkeypatch):
    """A real secret store and token endpoint; ``login_lands_after_check`` stores B right after the holder's check.

    Only the token endpoint is replaced. ``state.refreshed`` lists the refresh
    tokens it was sent; ``state.fatal`` makes it answer ``invalid_grant``.
    """
    from comfy_cli import credentials
    from comfy_cli.cloud import oauth

    monkeypatch.setenv("COMFY_SECRETS_PATH", str(tmp_path / "secrets.json"))
    state = SimpleNamespace(refreshed=[], fatal=False, other_expires_at=_LATER, pending=False)
    real_get_session = credentials.get_session

    def get_session(**kwargs):
        observed = real_get_session(**kwargs)
        if kwargs.get("refresh") is False and state.pending:
            state.pending = False
            _save_login("b", expires_at=state.other_expires_at)
        return observed

    def token_endpoint(**kwargs):
        state.refreshed.append(kwargs["refresh_token"])
        if state.fatal:
            raise oauth.OAuthRefreshError(
                "refresh failed", details={"status": 400, "body": '{"error":"invalid_grant"}'}
            )
        account = kwargs["refresh_token"].split("-")[1]
        return oauth.TokenSet(
            access_token=_jwt(account, len(state.refreshed) + 1),
            refresh_token=kwargs["refresh_token"] + "-rotated",
            token_type="Bearer",
            expires_in=900,
            expires_at=_LATER,
            scope="s",
        )

    monkeypatch.setattr(credentials, "get_session", get_session)
    monkeypatch.setattr(oauth, "refresh_tokens", token_endpoint)
    return state


def _assert_login_b_untouched(expires_at: int) -> None:
    from comfy_cli.auth import store

    stored = store.get_cloud_session()
    assert stored is not None
    assert (stored.access_token, stored.refresh_token, stored.expires_at) == (_jwt("b"), "refresh-b", expires_at)


@pytest.mark.parametrize("fatal", [False, True], ids=["rotates", "invalid-grant"])
def test_a_401_never_refreshes_or_clears_a_login_stored_after_the_account_check(login_race, fatal) -> None:
    # Given: an upload holds A's token; B signs in right after the holder checked the store
    held = _save_login("a", expires_at=_LATER).access_token
    token = DataPlaneToken(held, refreshes=True, base_url=_CLOUD, same_account=True, allow_clear=True)
    login_race.fatal, login_race.pending = fatal, True

    # When
    retried = token.after_rejection()

    # Then: B's refresh token is not spent, B stays stored, and the 401 reads as account replaced
    assert retried is False
    assert login_race.refreshed == []
    _assert_login_b_untouched(_LATER)
    assert token.account_replaced is True


@pytest.mark.parametrize("fatal", [False, True], ids=["rotates", "invalid-grant"])
def test_a_proactive_refresh_never_touches_a_login_stored_after_the_account_check(login_race, fatal) -> None:
    # Given: A's token is due for refresh; B, also due, signs in right after the holder checked the store
    held = _save_login("a", expires_at=_EXPIRED).access_token
    token = DataPlaneToken(held, refreshes=True, base_url=_CLOUD, same_account=True, allow_clear=True)
    login_race.fatal, login_race.pending, login_race.other_expires_at = fatal, True, _EXPIRED

    # When
    sent = token.current()

    # Then: the held token is kept and B is left as stored
    assert sent == held
    assert login_race.refreshed == []
    _assert_login_b_untouched(_EXPIRED)


def test_the_upload_account_is_still_rotated_by_either_refresh(login_race) -> None:
    from comfy_cli.auth import store

    # Given: the store holds the upload's own login, due for refresh
    held = _save_login("a", expires_at=_EXPIRED).access_token
    token = DataPlaneToken(held, refreshes=True, base_url=_CLOUD, same_account=True, allow_clear=True)

    # When / Then: the proactive refresh rotates it, and so does the one after a 401
    assert token.current() == _jwt("a", 2)
    assert token.after_rejection() is True
    assert login_race.refreshed == ["refresh-a", "refresh-a-rotated"]
    assert store.get_cloud_session().refresh_token == "refresh-a-rotated-rotated"
    assert token.account_replaced is False


def test_a_holder_not_bound_to_an_account_refreshes_without_the_account_check(monkeypatch) -> None:
    # Given: a deploy-style holder (no same_account) over a stub store
    calls: list[dict] = []

    def get_session(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(access_token="t1", base_url=_CLOUD)

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)
    token = DataPlaneToken("t0", refreshes=True, base_url=_CLOUD)

    # When
    token.current()
    token.after_rejection()

    # Then: the refresh is asked exactly as before, with no expectation attached
    assert calls and all("expect" not in kwargs for kwargs in calls)


def test_an_account_bound_holder_adopts_the_same_accounts_peer_refresh(login_race) -> None:
    # A peer finishes its refresh while the held token's rejected request is in flight.
    held = _save_login("a", expires_at=_LATER).access_token
    token = DataPlaneToken(held, refreshes=True, base_url=_CLOUD, same_account=True)
    peer = _jwt("a", 2)
    from comfy_cli.auth import store

    stored = store.get_cloud_session()
    store.save_cloud_session(
        base_url=stored.base_url,
        resource=stored.resource,
        client_id=stored.client_id,
        scope=stored.scope,
        access_token=peer,
        refresh_token="refresh-a-peer",
        token_type="Bearer",
        expires_at=_LATER,
    )

    assert token.after_rejection() is True
    assert token.current() == peer
    assert login_race.refreshed == []
    assert store.get_cloud_session().refresh_token == "refresh-a-peer"
    assert token.account_replaced is False
