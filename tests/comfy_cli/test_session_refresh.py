"""A deploy or builder client built from the stored sign-in survives its token expiring.

The access token lasts fifteen minutes, so a wait that polls past that gets a 401
on its next request. A client built from the stored session forces the shared
refresh once and retries with the new token; one handed a token directly never
swaps it.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.error
from types import SimpleNamespace

import pytest

from comfy_cli.builder_api import BuilderClient
from comfy_cli.credentials import Credential
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
    """Stands in for the stored sign-in: hands out ``first`` until forced, then ``after``.

    A forced refresh that yields no token clears the store, as a fatal refresh does.
    """

    def __init__(self, first: str, after: str | None) -> None:
        self.first = first
        self.after = after
        self.forced = 0
        self.cleared = False

    def __call__(self, *, refresh=True, force=False, allow_clear=True, rejected_access_token=None):
        if force:
            self.forced += 1
            if self.after is None:
                self.cleared = allow_clear
                return None
            return SimpleNamespace(access_token=self.after, refresh_token="rt", base_url="https://cloud.comfy.org")
        if self.cleared:
            return None
        return SimpleNamespace(access_token=self.first, refresh_token="rt", base_url="https://cloud.comfy.org")


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
    "create_build": lambda c: c.create_build_response("n", {"models": []}),
    "list_builds": lambda c: c.list_builds(),
    "delete_build": lambda c: c.delete_build("b1"),
    "delete_release": lambda c: c.delete_release("r1"),
    "validate_build": lambda c: c.validate_build("b1"),
    "update_build": lambda c: c.update_build("b1", {"models": []}, "2026-09-18T00:00:00Z"),
}


@pytest.mark.parametrize("lagging", ["builder", "deploy"])
def test_overlapping_clients_reuse_the_peer_refresh_before_retrying(monkeypatch, tmp_path, lagging) -> None:
    """A late 401 for OLD must not rotate the NEW token a peer just obtained."""
    from comfy_cli.auth import store
    from comfy_cli.cloud import oauth

    monkeypatch.setattr(store, "secrets_path", lambda: tmp_path / "secrets.json")
    store.save_cloud_session(
        base_url="https://cloud.comfy.org",
        resource="https://cloud.comfy.org/api",
        client_id="cli",
        scope="s",
        access_token="OLD",
        refresh_token="RT0",
        token_type="Bearer",
        expires_at=int(time.time()) + 3600,
    )
    refreshes = []

    def refresh(**kwargs):
        refreshes.append(kwargs["refresh_token"])
        generation = len(refreshes)
        return oauth.TokenSet(
            access_token=f"NEW{generation}",
            refresh_token=f"RT{generation}",
            token_type="Bearer",
            expires_in=3600,
            expires_at=int(time.time()) + 3600,
            scope="s",
        )

    monkeypatch.setattr(oauth, "refresh_tokens", refresh)
    clients = {"builder": BuilderClient.from_session(_BUILDER_URL), "deploy": DeployClient.from_session(_DEPLOY_URL)}
    peer = "deploy" if lagging == "builder" else "builder"
    calls = {
        "builder": lambda: clients["builder"].get_release("r1"),
        "deploy": lambda: clients["deploy"].get_deployment("dep-1"),
    }
    requests = []

    def send(url, target, **kwargs):
        name = "builder" if url.startswith(_BUILDER_URL) else "deploy"
        requests.append((name, target.auth_token))
        if target.auth_token == "OLD":
            if name == lagging:
                calls[peer]()  # Peer finishes its refresh while this stale request is in flight.
            raise urllib.error.HTTPError(url, 401, "expired", http.client.HTTPMessage(), io.BytesIO(b"{}"))
        return 200, dict(_ANSWER)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", send)
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", send)

    assert calls[lagging]()["status"] == "ready"
    assert refreshes == ["RT0"]
    assert requests == [(lagging, "OLD"), (peer, "OLD"), (peer, "NEW1"), (lagging, "NEW1")]


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


class _HeaderServer(_Server):
    def __init__(self, valid: set[str]) -> None:
        super().__init__(valid)
        self.headers: list[dict | None] = []

    def __call__(self, url, target, **kwargs):
        self.headers.append(kwargs.get("headers"))
        return super().__call__(url, target, **kwargs)


def test_the_retried_create_carries_the_same_idempotency_key(monkeypatch) -> None:
    # Given
    server = _HeaderServer(valid={"fresh"})
    _install(monkeypatch, server, _Sessions(first="expired", after="fresh"))
    client = DeployClient.from_session(_DEPLOY_URL)

    # When
    client.create_deployment("r1", {"gpuClass": "b200", "region": "us"}, idempotency_key="key-1")

    # Then
    assert server.headers == [{"Idempotency-Key": "key-1"}, {"Idempotency-Key": "key-1"}]


def test_a_refresh_that_signs_the_user_out_surfaces_the_401_as_not_signed_in(monkeypatch) -> None:
    # Given
    server = _Server(valid=set())
    _install(monkeypatch, server, _Sessions(first="expired", after=None))
    client = DeployClient.from_session(_DEPLOY_URL)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        client.get_deployment("dep-1")

    # Then
    assert exc_info.value.code == "deploy_not_signed_in"
    assert server.tokens == ["expired"]


def test_a_build_command_refused_after_the_refresh_says_to_sign_in_again(monkeypatch) -> None:
    # Given
    import json

    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    server = _Server(valid=set())
    _install(monkeypatch, server, _Sessions(first="expired", after=None))
    monkeypatch.delenv("COMFY_BUILDER_TOKEN", raising=False)

    # When
    result = CliRunner().invoke(app, ["--json", "build", "ls"], env={"COMFY_BUILDER_URL": _BUILDER_URL})

    # Then
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert result.exit_code == 1
    assert envelope["error"]["code"] == "build_not_signed_in"
    assert "comfy cloud login" in (envelope["error"]["hint"] or "")


def test_a_refused_injected_token_says_to_replace_it_not_to_sign_in(monkeypatch) -> None:
    # Given
    import json

    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    server = _Server(valid=set())
    _install(monkeypatch, server, _Sessions(first="stored", after="stored-refreshed"))

    # When
    result = CliRunner().invoke(
        app,
        ["--json", "build", "ls"],
        env={"COMFY_BUILDER_URL": _BUILDER_URL, "COMFY_BUILDER_TOKEN": "injected"},
    )

    # Then
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert envelope["error"]["code"] == "build_not_signed_in"
    assert "COMFY_BUILDER_TOKEN" in envelope["error"]["hint"]
    assert "comfy cloud login" not in envelope["error"]["hint"]
    assert server.tokens == ["injected"]


# --- A 401 that outlives the refresh during a watch -----------------------------
#
# Several commands polling at once can each be refused after the shared refresh:
# a peer can hold the refresh lock past this process's timeout, and then this one
# retries nothing. The build or deployment was accepted with the same sign-in a
# moment earlier and runs on, so while that sign-in is still stored, signing in
# again is not the fix. A 401 before anything was submitted keeps its sign-in code.


class _AnswersTheCut(_Server):
    """Answers the cut whatever token it carries, then refuses every read the watch makes."""

    def __call__(self, url, target, **kwargs):
        if kwargs.get("method") == "POST":
            self.tokens.append(target.auth_token)
            return 200, dict(_ANSWER)
        return super().__call__(url, target, **kwargs)


def _watched_cut(monkeypatch, server: _Server, sessions: _Sessions):
    import json

    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    _install(monkeypatch, server, sessions)
    monkeypatch.delenv("COMFY_BUILDER_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
    result = CliRunner().invoke(
        app,
        ["--json", "build", "release", "create", "--id", "b1", "--target", "linux/nvidia", "--watch"],
        env={"COMFY_BUILDER_URL": _BUILDER_URL},
    )
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    return result, envelope["error"]


def test_a_release_watch_refused_after_the_refresh_says_the_release_keeps_building(monkeypatch) -> None:
    # Given a cut the builder accepts, then a watch it refuses even after the refresh
    server = _AnswersTheCut(valid=set())
    sessions = _Sessions(first="signed-in", after="refreshed")

    # When
    result, error = _watched_cut(monkeypatch, server, sessions)

    # Then only the watch ended: its own code and exit status, and no sign-in prompt
    assert result.exit_code == 75
    assert error["code"] == "watch_auth_lost"
    assert "may still be building" in error["message"]
    assert "comfy build release show r1" in error["hint"]
    assert "comfy cloud login" not in error["hint"]
    assert error["details"] == {"releaseId": "r1"}
    assert server.tokens == ["signed-in", "signed-in", "refreshed"]
    assert sessions.forced == 1


def test_a_release_watch_whose_refresh_signed_the_user_out_still_says_to_sign_in(monkeypatch) -> None:
    # Given a watch refused, and a refresh that cleared the sign-in
    server = _AnswersTheCut(valid=set())
    sessions = _Sessions(first="signed-in", after=None)

    # When
    result, error = _watched_cut(monkeypatch, server, sessions)

    # Then nobody is signed in any more, so signing in is the move
    assert result.exit_code == 1
    assert error["code"] == "build_not_signed_in"
    assert "comfy cloud login" in error["hint"]


def test_a_cut_refused_before_anything_was_made_keeps_the_sign_in_code(monkeypatch) -> None:
    # Given a builder that refuses the cut itself, with the sign-in still stored
    server = _Server(valid=set())
    sessions = _Sessions(first="signed-in", after="refreshed")

    # When
    result, error = _watched_cut(monkeypatch, server, sessions)

    # Then nothing is running, so the refusal is the sign-in's
    assert result.exit_code == 1
    assert error["code"] == "build_not_signed_in"
    assert server.tokens == ["signed-in", "refreshed"]


def test_a_deploy_watch_refused_after_the_refresh_ends_the_watch_not_the_deployment(monkeypatch) -> None:
    # Given
    from comfy_cli.command.deploy_runtime import DeployWatchAuthLostError, poll_deployment

    server = _Server(valid=set())
    sessions = _Sessions(first="signed-in", after="refreshed")
    _install(monkeypatch, server, sessions)
    sleeps: list[float] = []

    # When
    with pytest.raises(DeployWatchAuthLostError) as exc_info:
        poll_deployment(DeployClient.from_session(_DEPLOY_URL), "dep-1", sleeps.append)

    # Then the read was refused after one refresh and one retry, and never retried again
    assert exc_info.value.deployment_id == "dep-1"
    assert exc_info.value.code == "watch_auth_lost"
    assert server.tokens == ["signed-in", "refreshed"]
    assert sleeps == []


def test_a_watch_with_an_unrefreshable_stored_token_requires_sign_in(monkeypatch) -> None:
    from comfy_cli.command.deploy_runtime import poll_deployment

    # A token without a refresh token can remain on disk after its lifetime ends.
    session = SimpleNamespace(access_token="expired", refresh_token=None, base_url="https://cloud.comfy.org")
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda **_: session)
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", _Server(valid=set()))

    with pytest.raises(DeployAPIError) as exc_info:
        poll_deployment(DeployClient.from_session(_DEPLOY_URL), "dep-1", lambda _: None)

    assert exc_info.value.code == "deploy_not_signed_in"


class _SignedInElsewhere(_Sessions):
    """A sign-in to another environment, stored while the watch ran."""

    def __call__(self, *, refresh=True, force=False, allow_clear=True, rejected_access_token=None):
        session = super().__call__(refresh=refresh, force=force, allow_clear=allow_clear)
        if self.forced and not force:
            return SimpleNamespace(access_token=self.after, refresh_token="rt", base_url="https://staging.comfy.test")
        return session


@pytest.mark.parametrize(
    "sessions",
    [_Sessions(first="signed-in", after=None), _SignedInElsewhere(first="signed-in", after="staging")],
    ids=["refresh_signed_the_user_out", "signed_in_to_another_environment"],
)
def test_a_deploy_watch_with_no_sign_in_left_to_keep_says_to_sign_in(monkeypatch, sessions) -> None:
    # Given
    from comfy_cli.command.deploy_runtime import poll_deployment

    _install(monkeypatch, _Server(valid=set()), sessions)

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        poll_deployment(DeployClient.from_session(_DEPLOY_URL), "dep-1", lambda _: None)

    # Then
    assert exc_info.value.code == "deploy_not_signed_in"


@pytest.mark.parametrize(
    "build",
    [
        lambda: DeployClient(_DEPLOY_URL, "injected"),
        lambda: DeployClient(
            _DEPLOY_URL, api_key=Credential(kind="api_key", value="k", source="env:COMFY_CLOUD_API_KEY")
        ),
    ],
    ids=["handed_token", "workspace_key"],
)
def test_a_deploy_watch_on_a_credential_it_cannot_refresh_keeps_the_sign_in_code(monkeypatch, build) -> None:
    # Given a sign-in stored beside a client that was never built from it
    from comfy_cli.command.deploy_runtime import poll_deployment

    server = _Server(valid=set())
    _install(monkeypatch, server, _Sessions(first="signed-in", after="refreshed"))

    # When
    with pytest.raises(DeployAPIError) as exc_info:
        poll_deployment(build(), "dep-1", lambda _: None)

    # Then the credential the client sent is the one to replace
    assert exc_info.value.code == "deploy_not_signed_in"
