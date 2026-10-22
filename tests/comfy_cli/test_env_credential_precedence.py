"""A credential handed in through the environment outranks a stored sign-in.

A caller that runs comfy-cli on someone's behalf hands it the credential to
use in ``COMFY_CLOUD_AUTH_TOKEN`` (a Cloud JWT) or ``COMFY_CLOUD_API_KEY`` (a
workspace API key). A ``comfy cloud login`` left on the same machine, perhaps
for another account or workspace, must not take its place, and must not be
refreshed or rewritten while the handed-in credential is in effect. With no
credential in the environment (unset, empty or whitespace) the stored sign-in
is used exactly as before, refresh included.

Every consumer is covered against the real secret store (a tmp
``secrets.json``) with only the token endpoint faked:

- cloud requests (``resolve_target``: ``comfy run --where cloud``, jobs, status);
- partner-node auth injected into ``extra_data`` on a local ``comfy run``;
- the partner proxy (``comfy generate``);
- the builder and deploy clients (``comfy build`` / ``comfy deploy``);
- the reactive refresh after a 401 (``Client``, ``DataPlaneToken``);
- ``comfy cloud whoami``.
"""

from __future__ import annotations

import json
import time

import pytest

from comfy_cli import credentials
from comfy_cli.auth import store as auth_store
from comfy_cli.cloud import oauth

BASE_URL = "https://cloud.comfy.org"
STORED_TOKEN = "AT-stored"
FRESH_TOKEN = "AT-refreshed"
ENV_TOKEN = "env-jwt"
ENV_KEY = "comfyui-env-key"
STORED_KEY = "comfyui-stored-key"


@pytest.fixture
def store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A real secret store in tmp, no ambient credentials, and a counting token endpoint."""
    path = tmp_path / "secrets.json"
    monkeypatch.setattr(auth_store, "secrets_path", lambda: path)
    for var in ("COMFY_CLOUD_AUTH_TOKEN", "COMFY_CLOUD_API_KEY", "COMFY_API_KEY", "COMFY_BUILDER_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("COMFY_CLOUD_BASE_URL", raising=False)
    refreshes: list[dict] = []

    def _refresh_tokens(**kw):
        refreshes.append(kw)
        return oauth.TokenSet(
            access_token=FRESH_TOKEN,
            refresh_token="RT-next",
            token_type="Bearer",
            expires_in=3600,
            expires_at=int(time.time()) + 3600,
            scope="s",
        )

    monkeypatch.setattr(oauth, "refresh_tokens", _refresh_tokens)
    return path, refreshes


def _save_session(*, expired: bool) -> None:
    auth_store.save_cloud_session(
        base_url=BASE_URL,
        resource=f"{BASE_URL}/api",
        client_id="cid",
        scope="s",
        access_token=STORED_TOKEN,
        refresh_token="RT0",
        token_type="Bearer",
        expires_at=int(time.time()) + (-10 if expired else 3600),
    )


def _apply(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for var, value in env.items():
        monkeypatch.setenv(var, value)


# Each row: (environment, stored sign-in?, expected credential value, expected source)
# The stored sign-in is persisted EXPIRED, so using it means refreshing it and
# writing the refreshed tokens back.
CLOUD_MATRIX = [
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN}, True, ENV_TOKEN, "env:COMFY_CLOUD_AUTH_TOKEN", id="env-token+stored->env"
    ),
    pytest.param({"COMFY_CLOUD_API_KEY": ENV_KEY}, True, ENV_KEY, "env:COMFY_CLOUD_API_KEY", id="env-key+stored->env"),
    pytest.param({}, True, FRESH_TOKEN, "session", id="stored-only->stored-refreshed"),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN}, False, ENV_TOKEN, "env:COMFY_CLOUD_AUTH_TOKEN", id="env-token-only->env"
    ),
    pytest.param({"COMFY_CLOUD_API_KEY": ENV_KEY}, False, ENV_KEY, "env:COMFY_CLOUD_API_KEY", id="env-key-only->env"),
    pytest.param({"COMFY_CLOUD_AUTH_TOKEN": ""}, True, FRESH_TOKEN, "session", id="empty-env-token+stored->stored"),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": "  \n\t", "COMFY_CLOUD_API_KEY": "   "},
        True,
        FRESH_TOKEN,
        "session",
        id="whitespace-env+stored->stored",
    ),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN, "COMFY_CLOUD_API_KEY": ENV_KEY},
        True,
        ENV_TOKEN,
        "env:COMFY_CLOUD_AUTH_TOKEN",
        id="env-token-outranks-env-key",
    ),
]


def _assert_session_effect(path, refreshes, *, env_wins: bool, had_session: bool, before: bytes | None) -> None:
    if env_wins:
        assert refreshes == [], "the stored sign-in was refreshed although an env credential was in effect"
        if had_session:
            assert path.read_bytes() == before, (
                "the stored sign-in was rewritten although an env credential was in effect"
            )
    elif had_session:
        assert len(refreshes) == 1
        assert auth_store.get_cloud_session().access_token == FRESH_TOKEN, "the refreshed sign-in was not written back"


def _snapshot(path):
    return path.read_bytes() if path.exists() else None


# ---------------------------------------------------------------------------
# cloud requests: resolve_target -> Target (comfy run --where cloud, jobs, status)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("env", "had_session", "expected", "source"), CLOUD_MATRIX)
def test_cloud_credential(store, monkeypatch, env, had_session, expected, source):
    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    cred = credentials.resolve_cloud_credential(purpose="cloud", base_url=BASE_URL)

    assert cred is not None and cred.value == expected and cred.source == source
    _assert_session_effect(path, refreshes, env_wins=source.startswith("env:"), had_session=had_session, before=before)


@pytest.mark.parametrize(("env", "had_session", "expected", "source"), CLOUD_MATRIX)
def test_cloud_target(store, monkeypatch, env, had_session, expected, source):
    from comfy_cli.target import resolve_target

    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    target = resolve_target(where="cloud")

    sent = target.api_key if source == "env:COMFY_CLOUD_API_KEY" else target.auth_token
    assert sent == expected
    _assert_session_effect(path, refreshes, env_wins=source.startswith("env:"), had_session=had_session, before=before)


def test_stored_key_still_last_when_signed_in(store, monkeypatch):
    """A `comfy cloud set-key` key stays behind a sign-in; only the environment moved up."""
    _save_session(expired=False)
    auth_store.set(credentials.CLOUD_API_KEY_PROVIDER, STORED_KEY)
    cred = credentials.resolve_cloud_credential(purpose="cloud", base_url=BASE_URL)
    assert cred is not None and cred.source == "session" and cred.value == STORED_TOKEN


# ---------------------------------------------------------------------------
# partner-node auth injected into extra_data on a local `comfy run`
# ---------------------------------------------------------------------------

PARTNER_MATRIX = [
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN},
        True,
        ("auth_token_comfy_org", ENV_TOKEN),
        True,
        id="env-token+stored->env",
    ),
    pytest.param({"COMFY_API_KEY": ENV_KEY}, True, ("api_key_comfy_org", ENV_KEY), True, id="partner-key+stored->env"),
    pytest.param(
        {"COMFY_CLOUD_API_KEY": ENV_KEY}, True, ("api_key_comfy_org", ENV_KEY), True, id="cloud-key+stored->env"
    ),
    pytest.param({}, True, ("auth_token_comfy_org", FRESH_TOKEN), False, id="stored-only->stored-refreshed"),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN},
        False,
        ("auth_token_comfy_org", ENV_TOKEN),
        True,
        id="env-token-only->env",
    ),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": " ", "COMFY_API_KEY": "", "COMFY_CLOUD_API_KEY": "\t"},
        True,
        ("auth_token_comfy_org", FRESH_TOKEN),
        False,
        id="blank-env+stored->stored",
    ),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN, "COMFY_API_KEY": ENV_KEY},
        True,
        ("auth_token_comfy_org", ENV_TOKEN),
        True,
        id="env-token-outranks-env-key",
    ),
]


@pytest.mark.parametrize(("env", "had_session", "expected", "env_wins"), PARTNER_MATRIX)
def test_partner_extra_data_credential(store, monkeypatch, env, had_session, expected, env_wins):
    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    assert credentials.resolve_partner_credential() == expected
    _assert_session_effect(path, refreshes, env_wins=env_wins, had_session=had_session, before=before)


# ---------------------------------------------------------------------------
# the partner proxy (comfy generate)
# ---------------------------------------------------------------------------

GENERATE_MATRIX = [
    pytest.param({"COMFY_API_KEY": ENV_KEY}, True, ENV_KEY, True, id="env-key+stored->env"),
    pytest.param({}, True, FRESH_TOKEN, False, id="stored-only->stored-refreshed"),
    pytest.param({"COMFY_API_KEY": ENV_KEY}, False, ENV_KEY, True, id="env-key-only->env"),
    pytest.param({"COMFY_API_KEY": "   "}, True, FRESH_TOKEN, False, id="whitespace-env+stored->stored"),
]


@pytest.mark.parametrize(("env", "had_session", "expected", "env_wins"), GENERATE_MATRIX)
def test_generate_credential(store, monkeypatch, env, had_session, expected, env_wins):
    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    cred = credentials.resolve_cloud_credential(purpose="partner")

    assert cred is not None and cred.value == expected
    _assert_session_effect(path, refreshes, env_wins=env_wins, had_session=had_session, before=before)


def test_generate_flag_still_outranks_env(store, monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", ENV_KEY)
    cred = credentials.resolve_cloud_credential(purpose="partner", explicit="flag-key")
    assert cred is not None and cred.source == "flag"


# ---------------------------------------------------------------------------
# builder and deploy clients (comfy build / comfy deploy)
# ---------------------------------------------------------------------------

PLATFORM_MATRIX = [
    pytest.param({"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN}, True, ("token", ENV_TOKEN), True, id="env-token+stored->env"),
    pytest.param({"COMFY_CLOUD_API_KEY": ENV_KEY}, True, ("key", ENV_KEY), True, id="env-key+stored->env"),
    pytest.param({}, True, ("token", FRESH_TOKEN), False, id="stored-only->stored-refreshed"),
    pytest.param({"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN}, False, ("token", ENV_TOKEN), True, id="env-token-only->env"),
    pytest.param(
        {"COMFY_CLOUD_AUTH_TOKEN": "  "}, True, ("token", FRESH_TOKEN), False, id="whitespace-env+stored->stored"
    ),
]


def _sent(target) -> tuple[str, str]:
    return ("key", target.api_key) if target.api_key else ("token", target.auth_token)


@pytest.mark.parametrize(("env", "had_session", "expected", "env_wins"), PLATFORM_MATRIX)
def test_builder_client(store, monkeypatch, env, had_session, expected, env_wins):
    from comfy_cli.builder_api import BuilderClient

    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    client = BuilderClient.from_credentials("https://builder.example")

    assert _sent(client.target) == expected
    assert client._refreshes_on_401 is not env_wins
    _assert_session_effect(path, refreshes, env_wins=env_wins, had_session=had_session, before=before)


@pytest.mark.parametrize(("env", "had_session", "expected", "env_wins"), PLATFORM_MATRIX)
def test_deploy_client(store, monkeypatch, env, had_session, expected, env_wins):
    from comfy_cli.deploy_api import DeployClient

    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    client = DeployClient.from_credentials("https://deploy.example")

    assert _sent(client.target) == expected
    assert (client.sign_in_base_url is None) is env_wins
    _assert_session_effect(path, refreshes, env_wins=env_wins, had_session=had_session, before=before)


def test_builder_token_still_outranks_everything(store, monkeypatch):
    """COMFY_BUILDER_TOKEN is the builder's own handed-in token and keeps first place."""
    from comfy_cli.command.build import _builder_client

    _save_session(expired=False)
    monkeypatch.setenv("COMFY_BUILDER_TOKEN", "builder-jwt")
    monkeypatch.setenv("COMFY_CLOUD_AUTH_TOKEN", ENV_TOKEN)
    client = _builder_client(renderer=None, builder_url="https://builder.example")
    assert client.target.auth_token == "builder-jwt"


# ---------------------------------------------------------------------------
# the reactive refresh after a 401
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "refreshed"),
    [
        pytest.param({"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN}, False, id="env-token-401-leaves-stored-alone"),
        pytest.param({}, True, id="stored-token-401-refreshes"),
    ],
)
def test_client_401_refresh(store, monkeypatch, env, refreshed):
    from comfy_cli.comfy_client import Client
    from comfy_cli.target import resolve_target

    path, refreshes = store
    _save_session(expired=False)
    _apply(monkeypatch, env)
    client = Client(resolve_target(where="cloud"))
    before = _snapshot(path)

    assert client._try_refresh_token() is refreshed
    if refreshed:
        assert client.target.auth_token == FRESH_TOKEN
        assert auth_store.get_cloud_session().access_token == FRESH_TOKEN
    else:
        assert client.target.auth_token == ENV_TOKEN
        assert refreshes == []
        assert path.read_bytes() == before


def test_data_plane_token_never_refreshes_an_env_token_equal_to_the_stored_one(store, monkeypatch):
    _save_session(expired=False)
    monkeypatch.setenv("COMFY_CLOUD_AUTH_TOKEN", STORED_TOKEN)
    token = credentials.DataPlaneToken.for_target_token(STORED_TOKEN, base_url=BASE_URL, allow_clear=False)
    assert token.after_rejection() is False
    assert store[1] == []


# ---------------------------------------------------------------------------
# comfy cloud whoami
# ---------------------------------------------------------------------------


def _whoami(monkeypatch) -> dict:
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    result = CliRunner().invoke(app, ["--json", "cloud", "whoami"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])["data"]


@pytest.mark.parametrize(
    ("env", "had_session", "auth_method", "source", "env_wins"),
    [
        pytest.param(
            {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN},
            True,
            "oauth",
            "env:COMFY_CLOUD_AUTH_TOKEN",
            True,
            id="env-token+stored",
        ),
        pytest.param(
            {"COMFY_CLOUD_API_KEY": ENV_KEY}, True, "api_key", "env:COMFY_CLOUD_API_KEY", True, id="env-key+stored"
        ),
        pytest.param({}, True, "oauth", "session", False, id="stored-only"),
        pytest.param(
            {"COMFY_CLOUD_AUTH_TOKEN": ENV_TOKEN},
            False,
            "oauth",
            "env:COMFY_CLOUD_AUTH_TOKEN",
            True,
            id="env-token-only",
        ),
    ],
)
def test_whoami_names_the_credential_in_effect(store, monkeypatch, env, had_session, auth_method, source, env_wins):
    path, refreshes = store
    if had_session:
        _save_session(expired=True)
    _apply(monkeypatch, env)
    before = _snapshot(path)

    data = _whoami(monkeypatch)

    assert data["signed_in"] is True
    assert data["auth_method"] == auth_method
    assert data["credential_source"] == source
    _assert_session_effect(path, refreshes, env_wins=env_wins, had_session=had_session, before=before)


@pytest.mark.parametrize(
    ("stored_key", "signed_in", "auth_method", "source"),
    [
        pytest.param(True, True, "api_key", f"stored:{credentials.CLOUD_API_KEY_PROVIDER}", id="falls-to-stored-key"),
        pytest.param(False, False, None, None, id="nothing-usable"),
    ],
)
def test_whoami_does_not_report_a_session_for_another_base_url(
    store, monkeypatch, stored_key, signed_in, auth_method, source
):
    """whoami reports what cloud calls send: resolve_target never sends a live
    session minted for another base URL, so neither does whoami's report."""
    from comfy_cli.target import resolve_target

    _save_session(expired=False)
    if stored_key:
        auth_store.set(credentials.CLOUD_API_KEY_PROVIDER, STORED_KEY)
    monkeypatch.setenv("COMFY_CLOUD_BASE_URL", "https://testcloud.comfy.org")

    data = _whoami(monkeypatch)

    assert data["stale_base_url"] is True
    assert data["expired"] is False
    assert data["signed_in"] is signed_in
    assert data["auth_method"] == auth_method
    assert data["credential_source"] == source
    target = resolve_target(where="cloud")
    assert target.auth_token is None
    assert target.api_key == (STORED_KEY if stored_key else None)
