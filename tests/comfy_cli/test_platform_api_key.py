"""comfy build and comfy deploy authenticate with a workspace API key when one is set.

comfy-builder and comfy-deploy take the key in ``X-API-Key`` and swap it at
ingest on every request, so a client holding a key sends it in place of any
stored sign-in, and never tries the sign-in refresh when the key is refused.
"""

from __future__ import annotations

import http.client
import io
import urllib.error
from types import SimpleNamespace

import pytest

from comfy_cli.builder_api import BuilderAuthError, BuilderClient, BuilderCredentialRefused
from comfy_cli.command.deploy_runtime import command_clients
from comfy_cli.credentials import Credential
from comfy_cli.deploy_api import DeployAPIError, DeployAuthError, DeployClient

_DEPLOY_URL = "https://deploy.test/deploy"
_BUILDER_URL = "https://builder.test"
_KEY = Credential(kind="api_key", value="comfyui-team-key", source="env:COMFY_CLOUD_API_KEY")


class _Server:
    """Answers with ``status`` and records the credentials each request carried."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.seen: list[tuple[str | None, str | None]] = []

    def __call__(self, url, target, **kwargs):
        self.seen.append((target.api_key, target.auth_token))
        if self.status != 200:
            raise urllib.error.HTTPError(url, self.status, "refused", http.client.HTTPMessage(), io.BytesIO(b"{}"))
        return 200, {"id": "x", "status": "ready"}


def _install(monkeypatch, server: _Server, key: Credential | None, sign_in: str | None = "signed-in-token") -> list:
    refreshes: list = []

    def sessions(*, refresh=True, force=False, allow_clear=True):
        if force:
            refreshes.append(force)
        return SimpleNamespace(access_token=sign_in) if sign_in else None

    monkeypatch.setattr("comfy_cli.credentials.platform_api_key", lambda: key)
    monkeypatch.setattr("comfy_cli.credentials.get_session", sessions)
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", server)
    monkeypatch.setattr("comfy_cli.builder_api.request_json", server)
    return refreshes


_CLIENTS = {
    "builder": (lambda: BuilderClient.from_credentials(_BUILDER_URL), lambda c: c.get_build("b1")),
    "deploy": (lambda: DeployClient.from_credentials(_DEPLOY_URL), lambda c: c.get_deployment("dep-1")),
}


@pytest.mark.parametrize(("make", "call"), _CLIENTS.values(), ids=_CLIENTS.keys())
def test_a_key_is_sent_in_place_of_the_stored_sign_in(monkeypatch, make, call) -> None:
    server = _Server()
    _install(monkeypatch, server, _KEY)

    call(make())

    assert server.seen == [("comfyui-team-key", None)]


@pytest.mark.parametrize(("make", "call"), _CLIENTS.values(), ids=_CLIENTS.keys())
def test_without_a_key_the_stored_sign_in_is_sent(monkeypatch, make, call) -> None:
    server = _Server()
    _install(monkeypatch, server, None)

    call(make())

    assert server.seen == [(None, "signed-in-token")]


@pytest.mark.parametrize(("make", "call"), _CLIENTS.values(), ids=_CLIENTS.keys())
def test_a_refused_key_is_sent_once_and_never_refreshes_a_sign_in(monkeypatch, make, call) -> None:
    server = _Server(status=401)
    refreshes = _install(monkeypatch, server, _KEY)

    with pytest.raises((urllib.error.HTTPError, DeployAPIError)):
        call(make())

    assert (server.seen, refreshes) == ([("comfyui-team-key", None)], [])


def test_a_refused_key_names_its_source_on_deploy(monkeypatch) -> None:
    _install(monkeypatch, _Server(status=401), _KEY)

    with pytest.raises(DeployAPIError) as refused:
        DeployClient.from_credentials(_DEPLOY_URL).get_deployment("dep-1")

    assert (refused.value.code, refused.value.hint) == (
        "deploy_not_signed_in",
        "the workspace API key in COMFY_CLOUD_API_KEY was refused; replace it with a valid key",
    )


@pytest.mark.parametrize(
    "make",
    [lambda: BuilderClient.from_credentials(_BUILDER_URL), lambda: DeployClient.from_credentials(_DEPLOY_URL)],
    ids=["builder", "deploy"],
)
def test_no_key_and_no_sign_in_names_both_ways_in(monkeypatch, make) -> None:
    _install(monkeypatch, _Server(), None, sign_in=None)

    with pytest.raises((BuilderAuthError, DeployAuthError)) as missing:
        make()

    assert "COMFY_CLOUD_API_KEY" in str(missing.value) and "comfy cloud login" in str(missing.value)


def test_a_build_token_still_comes_before_a_key(monkeypatch) -> None:
    from comfy_cli.command import build

    _install(monkeypatch, _Server(), _KEY)
    monkeypatch.setenv("COMFY_BUILDER_TOKEN", "forwarded-cloud-jwt")

    client = build._builder_client(renderer=None, builder_url=_BUILDER_URL)

    assert (client.target.auth_token, client.target.api_key) == ("forwarded-cloud-jwt", None)


_SAVED = Credential(kind="api_key", value="comfyui-saved", source="stored:comfy-cloud-api-key")


@pytest.mark.parametrize(
    ("make", "hint"),
    [
        (
            lambda: BuilderClient(_BUILDER_URL, api_key=_KEY),
            "the workspace API key in COMFY_CLOUD_API_KEY was refused; replace it with a valid key",
        ),
        (
            lambda: BuilderClient(_BUILDER_URL, api_key=_SAVED),
            "the workspace API key saved by `comfy cloud set-key` was refused; replace it with a valid key",
        ),
        (
            lambda: BuilderClient(_BUILDER_URL, "forwarded-cloud-jwt"),
            "replace COMFY_BUILDER_TOKEN with a fresh Cloud JWT",
        ),
        (lambda: BuilderClient.from_session(_BUILDER_URL), "run `comfy cloud login` first"),
    ],
    ids=["env-key", "saved-key", "build-token", "sign-in"],
)
def test_a_refused_build_names_the_credential_the_client_sent(monkeypatch, make, hint) -> None:
    _install(monkeypatch, _Server(status=401), None)

    with pytest.raises(BuilderCredentialRefused) as refused:
        make().get_build("b1")

    assert (refused.value.code, refused.value.hint) == (401, hint)


def test_a_saved_key_stands_in_once_a_dead_sign_in_is_cleared(monkeypatch) -> None:
    server = _Server()
    _install(monkeypatch, server, None, sign_in=None)
    # The first look sees a stale sign-in on disk; its failed refresh clears it.
    answers = iter([None, _SAVED])
    monkeypatch.setattr("comfy_cli.credentials.platform_api_key", lambda: next(answers))

    BuilderClient.from_credentials(_BUILDER_URL).get_build("b1")

    assert server.seen == [("comfyui-saved", None)]


def test_a_deploy_command_reports_a_key_the_builder_refused(monkeypatch) -> None:
    _install(monkeypatch, _Server(status=401), _KEY)
    builder, _deploy = command_clients()

    with pytest.raises(DeployAPIError) as refused:
        builder.list_releases("b1")

    assert (refused.value.code, refused.value.hint) == (
        "deploy_not_signed_in",
        "the workspace API key in COMFY_CLOUD_API_KEY was refused; replace it with a valid key",
    )


def test_a_refreshed_sign_in_the_builder_still_refuses_names_the_login(monkeypatch) -> None:
    server = _Server(status=401)
    refreshed = iter(["refreshed-token"])

    def sessions(*, refresh=True, force=False, allow_clear=True):
        return SimpleNamespace(access_token=next(refreshed) if force else "signed-in-token")

    monkeypatch.setattr("comfy_cli.credentials.get_session", sessions)
    monkeypatch.setattr("comfy_cli.builder_api.request_json", server)

    with pytest.raises(BuilderCredentialRefused) as refused:
        BuilderClient.from_session(_BUILDER_URL).get_build("b1")

    assert (server.seen, refused.value.hint) == (
        [(None, "signed-in-token"), (None, "refreshed-token")],
        "run `comfy cloud login` first",
    )
