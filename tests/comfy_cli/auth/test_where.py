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


# ---------------------------------------------------------------------------
# resolve_default: reads the persisted where_default config key for the caller
# ---------------------------------------------------------------------------


class _FakeConfigManager:
    """Stand-in for ConfigManager whose ``get`` behaviour is programmable."""

    _value: str | None = None
    _raise: bool = False

    def get(self, key):
        assert key == where_module.CONFIG_KEY_WHERE_DEFAULT
        if self._raise:
            raise RuntimeError("corrupt config")
        return self._value


@pytest.fixture
def fake_config(monkeypatch):
    import comfy_cli.config_manager as cm

    monkeypatch.setattr(cm, "ConfigManager", _FakeConfigManager)
    return _FakeConfigManager


def test_resolve_default_uses_config_value(fake_config, monkeypatch):
    monkeypatch.setattr(fake_config, "_value", "cloud")
    r = where_module.resolve_default(env={}, project_value=None)
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "config"


def test_resolve_default_flag_beats_config(fake_config, monkeypatch):
    monkeypatch.setattr(fake_config, "_value", "cloud")
    r = where_module.resolve_default(flag="local", env={}, project_value=None)
    assert r.target is where_module.WhereTarget.LOCAL
    assert r.source == "flag"


def test_resolve_default_broken_config_falls_through(fake_config, monkeypatch):
    """A config read that raises never breaks routing — it drops to None."""
    monkeypatch.setattr(fake_config, "_raise", True)
    r = where_module.resolve_default(env={}, project_value=None)
    assert r.target is where_module.WhereTarget.LOCAL
    assert r.source == "default"


def test_resolve_default_forwards_project_value(fake_config, monkeypatch):
    monkeypatch.setattr(fake_config, "_value", "local")
    r = where_module.resolve_default(env={}, project_value="cloud")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "project"


# ---------------------------------------------------------------------------
# project/1 precedence: flag → env → project → config → auto
# ---------------------------------------------------------------------------


def test_resolve_flag_beats_project():
    r = where_module.resolve(flag="local", env={}, project_value="cloud")
    assert r.target is where_module.WhereTarget.LOCAL
    assert r.source == "flag"


def test_resolve_env_beats_project():
    r = where_module.resolve(flag=None, env={"COMFY_WHERE": "local"}, project_value="cloud")
    assert r.target is where_module.WhereTarget.LOCAL
    assert r.source == "env"


def test_resolve_project_beats_config():
    r = where_module.resolve(flag=None, env={}, project_value="cloud", config_value="local")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "project"


def test_resolve_explicit_none_project_value_disables_lookup(monkeypatch):
    """``project_value=None`` opts out entirely — no filesystem discovery."""
    import comfy_cli.project as project_mod

    def _boom(*a, **kw):  # pragma: no cover — must not be called
        raise AssertionError("find_project must not be called when project_value=None")

    monkeypatch.setattr(project_mod, "find_project", _boom)
    r = where_module.resolve(flag=None, env={}, project_value=None, config_value="cloud")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "config"


def test_resolve_default_sentinel_discovers_project(tmp_path, monkeypatch):
    """Left unset, resolve() looks up the governing project's defaults.where —
    the ~10 existing call sites get project routing without changes."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "comfy.yaml").write_text("schema: project/1\ndefaults:\n  where: cloud\n")
    monkeypatch.chdir(proj)

    r = where_module.resolve(flag=None, env={}, config_value="local")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "project"


def test_resolve_no_project_falls_through_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no comfy.yaml anywhere up a tmp tree
    r = where_module.resolve(flag=None, env={}, config_value="cloud")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "config"


def test_resolve_project_without_where_default_falls_through(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "comfy.yaml").write_text("schema: project/1\n")
    monkeypatch.chdir(proj)
    r = where_module.resolve(flag=None, env={}, config_value="cloud")
    assert r.target is where_module.WhereTarget.CLOUD
    assert r.source == "config"


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
