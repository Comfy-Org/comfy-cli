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

    def __call__(self, *, refresh=True, force=False, allow_clear=True):
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

    def get_session(*, refresh=True, force=False, allow_clear=True):
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
