"""Tests for the single shared credential resolver (`comfy_cli.credentials`).

Two test groups:

1. A behavior matrix over (explicit flag, OAuth session state, env var,
   stored key) × purpose, pinning the precedence order and the per-purpose
   differences (env var name, value stripping) that previously lived in four
   hand-rolled chains.

2. A "no direct reads" ratchet: an AST scan asserting that no module under
   ``comfy_cli/`` (outside the resolver itself and the auth/oauth internals)
   reads ``COMFY_API_KEY``/``COMFY_CLOUD_API_KEY`` from the environment or
   calls ``get_cloud_session``/``ensure_fresh_session`` directly — so the
   chains can never drift apart again.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from comfy_cli import credentials
from comfy_cli.auth import store as auth_store
from comfy_cli.cloud import oauth
from comfy_cli.credentials import (
    CLOUD_API_KEY_PROVIDER,
    Credential,
    find_api_key,
    get_session,
    resolve_cloud_credential,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _session(*, expired: bool = False, base_url: str = "https://cloud.comfy.org", token: str = "oauth-token-123"):
    s = MagicMock()
    s.is_expired.return_value = expired
    s.access_token = token
    s.base_url = base_url
    return s


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """No ambient credentials: no env vars, no stored key, no session."""
    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "get", lambda _provider: None)
    monkeypatch.setattr(auth_store, "get_cloud_session", lambda: None)
    monkeypatch.setattr(oauth, "ensure_fresh_session", lambda **kw: None)
    return monkeypatch


def _stored(key: str):
    record = MagicMock()
    record.key = key
    return lambda name: record if name == CLOUD_API_KEY_PROVIDER else None


ENV_VAR = {"cloud": "COMFY_CLOUD_API_KEY", "partner": "COMFY_API_KEY"}
OTHER_ENV_VAR = {"cloud": "COMFY_API_KEY", "partner": "COMFY_CLOUD_API_KEY"}


# ---------------------------------------------------------------------------
# 1. behavior matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("purpose", ["cloud", "partner"])
class TestPrecedence:
    def test_nothing_configured_returns_none(self, purpose, clean_env):
        assert resolve_cloud_credential(purpose=purpose) is None

    def test_explicit_flag_wins_over_everything(self, purpose, clean_env):
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session())
        cred = resolve_cloud_credential(purpose=purpose, explicit="flag-key")
        assert cred == Credential(kind="api_key", value="flag-key", source="flag")

    def test_explicit_flag_is_stripped(self, purpose, clean_env):
        cred = resolve_cloud_credential(purpose=purpose, explicit="  flag-key \n")
        assert cred is not None
        assert cred.value == "flag-key"

    def test_whitespace_only_explicit_is_ignored(self, purpose, clean_env):
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        cred = resolve_cloud_credential(purpose=purpose, explicit="   \n\t")
        assert cred is not None
        assert cred.source == f"env:{ENV_VAR[purpose]}"

    def test_live_session_outranks_env_and_stored(self, purpose, clean_env):
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(token="live-token"))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred == Credential(kind="oauth", value="live-token", source="session")

    def test_expired_session_falls_through_to_env(self, purpose, clean_env):
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(expired=True))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred == Credential(kind="api_key", value="env-key", source=f"env:{ENV_VAR[purpose]}")

    def test_env_outranks_stored(self, purpose, clean_env):
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred is not None
        assert cred.value == "env-key"
        assert cred.kind == "api_key"

    def test_stored_key_is_last_resort(self, purpose, clean_env):
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred == Credential(kind="api_key", value="stored-key", source=f"stored:{CLOUD_API_KEY_PROVIDER}")

    def test_other_purposes_env_var_is_ignored(self, purpose, clean_env):
        clean_env.setenv(OTHER_ENV_VAR[purpose], "wrong-key")
        assert resolve_cloud_credential(purpose=purpose) is None

    def test_refresh_true_uses_ensure_fresh_session(self, purpose, clean_env):
        calls = []

        def fake_refresh(**kw):
            calls.append(1)
            return _session(token="refreshed-token")

        clean_env.setattr(oauth, "ensure_fresh_session", fake_refresh)
        clean_env.setattr(
            auth_store, "get_cloud_session", lambda: pytest.fail("refresh=True must not use get_cloud_session")
        )
        cred = resolve_cloud_credential(purpose=purpose, refresh=True)
        assert calls == [1]
        assert cred is not None
        assert cred.value == "refreshed-token"

    def test_refresh_false_uses_get_cloud_session_without_refresh(self, purpose, clean_env):
        clean_env.setattr(auth_store, "get_cloud_session", lambda: _session(token="stored-session-token"))
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: pytest.fail("refresh=False must not refresh"))
        cred = resolve_cloud_credential(purpose=purpose, refresh=False)
        assert cred == Credential(kind="oauth", value="stored-session-token", source="session")

    def test_session_with_empty_token_falls_through(self, purpose, clean_env):
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(token=""))
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred is not None
        assert cred.kind == "api_key"

    def test_base_url_mismatch_skips_session(self, purpose, clean_env):
        """The replay-guard: a session minted for another host is never sent."""
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(base_url="https://other.comfy.org"))
        clean_env.setenv(ENV_VAR[purpose], "env-key")
        cred = resolve_cloud_credential(purpose=purpose, base_url="https://cloud.comfy.org")
        assert cred is not None
        assert cred.kind == "api_key"
        assert cred.value == "env-key"

    def test_base_url_match_uses_session(self, purpose, clean_env):
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(base_url="https://cloud.comfy.org"))
        cred = resolve_cloud_credential(purpose=purpose, base_url="https://cloud.comfy.org")
        assert cred is not None
        assert cred.kind == "oauth"

    def test_no_base_url_means_no_replay_guard(self, purpose, clean_env):
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session(base_url="https://other.comfy.org"))
        cred = resolve_cloud_credential(purpose=purpose)
        assert cred is not None
        assert cred.kind == "oauth"


class TestPurposeSpecificStripping:
    """`partner` (the generate proxy) historically strips ambient keys and
    treats whitespace-only values as absent; `cloud` passes them verbatim.
    Encoded per-purpose to preserve each site's exact behavior."""

    def test_partner_strips_env_value(self, clean_env):
        clean_env.setenv("COMFY_API_KEY", "  sk-abc  ")
        cred = resolve_cloud_credential(purpose="partner")
        assert cred is not None
        assert cred.value == "sk-abc"

    def test_partner_whitespace_only_env_is_absent(self, clean_env):
        clean_env.setenv("COMFY_API_KEY", "   \n\t")
        assert resolve_cloud_credential(purpose="partner") is None

    def test_partner_strips_stored_value(self, clean_env):
        clean_env.setattr(auth_store, "get", _stored("  stored-key \n"))
        cred = resolve_cloud_credential(purpose="partner")
        assert cred is not None
        assert cred.value == "stored-key"

    def test_cloud_env_value_passed_verbatim(self, clean_env):
        clean_env.setenv("COMFY_CLOUD_API_KEY", "  sk-abc  ")
        cred = resolve_cloud_credential(purpose="cloud")
        assert cred is not None
        assert cred.value == "  sk-abc  "


class TestFindApiKey:
    """`find_api_key` checks only the ambient key sources (env → stored),
    ignoring any OAuth session — whoami uses it to report an API key that is
    present-but-outranked."""

    def test_ignores_live_session(self, clean_env):
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: _session())
        clean_env.setattr(auth_store, "get_cloud_session", lambda: _session())
        clean_env.setenv("COMFY_CLOUD_API_KEY", "env-key")
        cred = find_api_key(purpose="cloud")
        assert cred == Credential(kind="api_key", value="env-key", source="env:COMFY_CLOUD_API_KEY")

    def test_stored_fallback(self, clean_env):
        clean_env.setattr(auth_store, "get", _stored("stored-key"))
        cred = find_api_key(purpose="cloud")
        assert cred is not None
        assert cred.source == f"stored:{CLOUD_API_KEY_PROVIDER}"

    def test_none_when_absent(self, clean_env):
        assert find_api_key(purpose="cloud") is None


class TestGetSession:
    """`get_session` is the single read gateway to the stored OAuth session."""

    def test_refresh_true_delegates_to_ensure_fresh_session(self, clean_env):
        marker = _session()
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: marker)
        assert get_session(refresh=True) is marker

    def test_refresh_false_delegates_to_get_cloud_session(self, clean_env):
        marker = _session()
        clean_env.setattr(auth_store, "get_cloud_session", lambda: marker)
        clean_env.setattr(oauth, "ensure_fresh_session", lambda **kw: pytest.fail("refresh=False must not refresh"))
        assert get_session(refresh=False) is marker

    def test_force_threads_through_to_ensure_fresh_session(self, clean_env):
        """The reactive 401 path passes force=True; it must reach the refresher
        even when refresh defaults are otherwise off."""
        seen = {}

        def _refresh(**kw):
            seen.update(kw)
            return _session(token="forced")

        clean_env.setattr(oauth, "ensure_fresh_session", _refresh)
        cred = get_session(refresh=False, force=True)
        assert seen.get("force") is True
        assert cred is not None and cred.access_token == "forced"


# ---------------------------------------------------------------------------
# 2. no-direct-reads ratchet
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(credentials.__file__).resolve().parent

# Files allowed to touch the raw credential sources:
#   - credentials.py is the resolver itself.
#   - auth/ holds the secret store (defines get_cloud_session).
#   - cloud/oauth.py defines ensure_fresh_session and the refresh flow.
ALLOWED = {
    "credentials.py",
    "cloud/oauth.py",
}
ALLOWED_DIRS = {"auth"}

# FROZEN allowlist for call sites that could not be migrated yet. Adding to
# this list is a ratchet violation — migrate the new code to
# `comfy_cli.credentials` instead. (Currently empty: all known readers were
# migrated when the resolver was introduced.)
FROZEN_ALLOWLIST: set[str] = set()

ENV_VARS = {"COMFY_API_KEY", "COMFY_CLOUD_API_KEY"}
SESSION_FUNCS = {"get_cloud_session", "ensure_fresh_session"}


def _is_environ_node(node: ast.expr) -> bool:
    """True for `os.environ` / bare `environ` references."""
    if isinstance(node, ast.Attribute) and node.attr == "environ":
        return True
    return isinstance(node, ast.Name) and node.id == "environ"


def _literal(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _violations_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        # os.environ.get("COMFY_..."), os.getenv("COMFY_...")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr == "get" and _is_environ_node(func.value) and node.args:
                if _literal(node.args[0]) in ENV_VARS:
                    found.append(f"{path}:{node.lineno}: os.environ.get({_literal(node.args[0])!r})")
            if func.attr == "getenv" and node.args and _literal(node.args[0]) in ENV_VARS:
                found.append(f"{path}:{node.lineno}: os.getenv({_literal(node.args[0])!r})")
            if func.attr in SESSION_FUNCS:
                found.append(f"{path}:{node.lineno}: call to {func.attr}()")
        # bare get_cloud_session(...) / ensure_fresh_session(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SESSION_FUNCS:
            found.append(f"{path}:{node.lineno}: call to {node.func.id}()")
        # os.environ["COMFY_..."]
        if isinstance(node, ast.Subscript) and _is_environ_node(node.value):
            if _literal(node.slice) in ENV_VARS:
                found.append(f"{path}:{node.lineno}: os.environ[{_literal(node.slice)!r}]")
    return found


def test_no_direct_credential_reads_outside_resolver():
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in ALLOWED or rel in FROZEN_ALLOWLIST:
            continue
        if rel.split("/", 1)[0] in ALLOWED_DIRS:
            continue
        violations.extend(_violations_in(path))
    assert not violations, (
        "Direct credential reads found outside comfy_cli/credentials.py — "
        "use resolve_cloud_credential / find_api_key / get_session instead:\n" + "\n".join(violations)
    )
