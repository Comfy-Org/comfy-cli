"""A deploy or builder client built from the stored sign-in survives its token expiring.

The access token lasts fifteen minutes, so a wait that polls past that gets a 401
on its next request. A client built from the stored session forces the shared
refresh once and retries with the new token; one handed a token directly never
swaps it.
"""

from __future__ import annotations

import http.client
import io
import urllib.error
from types import SimpleNamespace

import pytest

from comfy_cli.builder_api import BuilderClient
from comfy_cli.deploy_api import DeployAPIError, DeployClient

_DEPLOY_URL = "https://deploy.test/deploy"
_BUILDER_URL = "https://builder.test"
_ANSWER = {"id": "x", "status": "ready", "releaseId": "r1", "statusUrl": "https://s.test/r1"}


class _Server:
    """Refuses any token but the ones it was told are valid, and records each request."""

    def __init__(self, valid: set[str]) -> None:
        self.valid = valid
        self.tokens: list[str | None] = []

    def __call__(self, url, target, **kwargs):
        self.tokens.append(target.auth_token)
        if target.auth_token not in self.valid:
            raise urllib.error.HTTPError(url, 401, "expired", http.client.HTTPMessage(), io.BytesIO(b"{}"))
        return 200, dict(_ANSWER)


class _Sessions:
    """Stands in for the stored sign-in: hands out ``first`` until forced, then ``after``."""

    def __init__(self, first: str, after: str | None) -> None:
        self.first = first
        self.after = after
        self.forced = 0

    def __call__(self, *, refresh=True, force=False, allow_clear=True):
        if force:
            self.forced += 1
            return SimpleNamespace(access_token=self.after) if self.after else None
        return SimpleNamespace(access_token=self.first)


def _install(monkeypatch, server: _Server, sessions: _Sessions) -> None:
    monkeypatch.setattr("comfy_cli.credentials.get_session", sessions)
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", server)
    monkeypatch.setattr("comfy_cli.builder_api.request_json", server)


_DEPLOY_CALLS = {
    "get": lambda c: c.get_deployment("dep-1"),
    "create": lambda c: c.create_deployment("r1", {"gpuClass": "b200", "region": "us"}),
    "stop": lambda c: c.stop_deployment("dep-1"),
}

_BUILDER_CALLS = {
    "get_release": lambda c: c.get_release("r1"),
    "create_build": lambda c: c.create_build("n", {"models": []}),
    "list_builds": lambda c: c.list_builds(),
    "delete_build": lambda c: c.delete_build("b1"),
    "delete_release": lambda c: c.delete_release("r1"),
    "validate_build": lambda c: c.validate_build("b1"),
    "update_build": lambda c: c.update_build("b1", {"models": []}, "2026-09-18T00:00:00Z"),
}


@pytest.mark.parametrize("call", _DEPLOY_CALLS.values(), ids=_DEPLOY_CALLS.keys())
def test_deploy_client_retries_once_with_a_refreshed_token_after_a_401(monkeypatch, call) -> None:
    # Given
    server = _Server(valid={"fresh"})
    sessions = _Sessions(first="expired", after="fresh")
    _install(monkeypatch, server, sessions)
    client = DeployClient.from_session(_DEPLOY_URL)

    # When
    call(client)

    # Then
    assert server.tokens == ["expired", "fresh"]
    assert sessions.forced == 1


@pytest.mark.parametrize("call", _BUILDER_CALLS.values(), ids=_BUILDER_CALLS.keys())
def test_builder_client_retries_once_with_a_refreshed_token_after_a_401(monkeypatch, call) -> None:
    # Given
    server = _Server(valid={"fresh"})
    sessions = _Sessions(first="expired", after="fresh")
    _install(monkeypatch, server, sessions)
    client = BuilderClient.from_session(_BUILDER_URL)

    # When
    call(client)

    # Then
    assert server.tokens == ["expired", "fresh"]
    assert sessions.forced == 1


def test_a_wait_keeps_the_refreshed_token_for_later_polls(monkeypatch) -> None:
    # Given
    server = _Server(valid={"fresh"})
    sessions = _Sessions(first="expired", after="fresh")
    _install(monkeypatch, server, sessions)
    client = DeployClient.from_session(_DEPLOY_URL)

    # When
    client.get_deployment("dep-1")
    client.get_deployment("dep-1")

    # Then
    assert server.tokens == ["expired", "fresh", "fresh"]
    assert sessions.forced == 1


def test_a_refresh_that_returns_no_new_token_surfaces_the_401(monkeypatch) -> None:
    # Given
    server = _Server(valid=set())
    sessions = _Sessions(first="expired", after="expired")
    _install(monkeypatch, server, sessions)
    client = DeployClient.from_session(_DEPLOY_URL)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        client.get_deployment("dep-1")

    # Then
    assert exc_info.value.code == "deploy_not_signed_in"
    assert server.tokens == ["expired"]


def test_a_second_401_after_the_refresh_surfaces_rather_than_looping(monkeypatch) -> None:
    # Given
    server = _Server(valid=set())
    sessions = _Sessions(first="expired", after="also-refused")
    _install(monkeypatch, server, sessions)
    client = BuilderClient.from_session(_BUILDER_URL)

    # When
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        client.get_release("r1")

    # Then
    assert exc_info.value.code == 401
    assert server.tokens == ["expired", "also-refused"]


@pytest.mark.parametrize(
    "build",
    [lambda: DeployClient(_DEPLOY_URL, "injected"), lambda: BuilderClient(_BUILDER_URL, "injected")],
    ids=["deploy", "builder"],
)
def test_a_client_handed_a_token_directly_never_swaps_it(monkeypatch, build) -> None:
    # Given
    server = _Server(valid=set())
    sessions = _Sessions(first="stored", after="stored-refreshed")
    _install(monkeypatch, server, sessions)
    client = build()

    # When
    with pytest.raises((DeployAPIError, urllib.error.HTTPError)):
        client.get_release("r1") if isinstance(client, BuilderClient) else client.get_deployment("dep-1")

    # Then
    assert server.tokens == ["injected"]
    assert sessions.forced == 0
