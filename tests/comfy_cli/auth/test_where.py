"""Phase 4: --where resolution and the local-only cloud preflight."""

from __future__ import annotations

import pytest

from comfy_cli import where as where_module
from comfy_cli.auth import store as auth_store


@pytest.fixture
def isolated_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    yield tmp_path / "secrets.json"


def test_resolve_flag_wins():
    r = where_module.resolve(flag="cloud", env={"COMFY_WHERE": "local"}, config_value="local")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "flag"


def test_resolve_env_wins_over_config():
    r = where_module.resolve(flag=None, env={"COMFY_WHERE": "cloud"}, config_value="local")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "env"


def test_resolve_config_used_when_no_flag_or_env():
    r = where_module.resolve(flag=None, env={}, config_value="cloud")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "config"


def test_resolve_defaults_to_local():
    r = where_module.resolve(flag=None, env={}, config_value=None)
    assert r.target is where_module.WhereTarget.LOCAL
    assert r.source == "default"


def test_resolve_invalid_raises():
    with pytest.raises(ValueError) as exc_info:
        where_module.resolve(flag="hybrid")
    assert "hybrid" in str(exc_info.value)


def test_cloud_preflight_without_session_returns_not_configured(isolated_secrets):
    err = where_module.cloud_preflight()
    assert err is not None
    assert err.code == "cloud_not_configured"
    assert "comfy cloud login" in err.hint


def test_cloud_preflight_with_valid_session_allows_proceeding(isolated_secrets):
    """A live session is enough to clear preflight — the cloud client takes over."""
    auth_store.save_cloud_session(
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        client_id="mcp-dyn-test-id",
        scope="mcp:tools:read mcp:tools:call",
        access_token="fake-access-token",
        refresh_token="fake-refresh-token",
        token_type="Bearer",
        expires_at=9999999999,
    )
    assert where_module.cloud_preflight() is None


def test_cloud_preflight_with_expired_unrefreshable_session_returns_unauthorized(isolated_secrets, monkeypatch):
    # Expired session whose refresh fails (dead refresh token) → unauthorized.
    from comfy_cli.cloud import oauth

    def _refresh_fails(**kw):
        raise oauth.OAuthRefreshError("dead", hint="re-login", details={})

    monkeypatch.setattr(oauth, "refresh_tokens", _refresh_fails)
    auth_store.save_cloud_session(
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        client_id="mcp-dyn-test-id",
        scope="mcp:tools:read mcp:tools:call",
        access_token="fake-access-token",
        refresh_token="fake-refresh-token",
        token_type="Bearer",
        expires_at=1,  # epoch second 1 → long expired
    )
    err = where_module.cloud_preflight()
    assert err is not None
    assert err.code == "cloud_unauthorized"
    assert "comfy cloud login" in err.hint


def test_cloud_preflight_refreshes_expired_session(isolated_secrets, monkeypatch):
    # Expired session + working refresh token → preflight refreshes and passes.
    import time

    from comfy_cli.cloud import oauth

    monkeypatch.setattr(
        oauth,
        "refresh_tokens",
        lambda **kw: oauth.TokenSet(
            access_token="fresh-access-token",
            refresh_token="fresh-refresh-token",
            token_type="Bearer",
            expires_in=3600,
            expires_at=int(time.time()) + 3600,
            scope="s",
        ),
    )
    auth_store.save_cloud_session(
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        client_id="mcp-dyn-test-id",
        scope="mcp:tools:read mcp:tools:call",
        access_token="stale-access-token",
        refresh_token="good-refresh-token",
        token_type="Bearer",
        expires_at=1,  # expired
    )
    assert where_module.cloud_preflight() is None
    # The refreshed token was persisted, so subsequent commands use it.
    assert auth_store.get_cloud_session().access_token == "fresh-access-token"
