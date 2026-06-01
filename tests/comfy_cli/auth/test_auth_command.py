"""``comfy auth`` integration: set / list / remove via subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"


def _validator_for(name: str) -> jsonschema.Validator:
    schema = json.loads((SCHEMAS_DIR / name).read_text())
    store: dict[str, dict] = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        s = json.loads(path.read_text())
        if s.get("$id"):
            store[s["$id"]] = s
        store[path.name] = s
    base = SCHEMAS_DIR.absolute().as_uri() + "/"
    resolver = jsonschema.RefResolver(base_uri=base, referrer=schema, store=store)
    return jsonschema.Draft202012Validator(schema, resolver=resolver)


@pytest.fixture
def cli_env(tmp_path):
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["COMFY_SECRETS_PATH"] = str(tmp_path / "secrets.json")
    return env


def _run(args, env):
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _last_json(stdout: str) -> dict:
    last = [line for line in stdout.splitlines() if line.strip()][-1]
    return json.loads(last)


def test_auth_list_empty_envelope(cli_env):
    res = _run(["--json", "auth", "list"], cli_env)
    assert res.returncode == 0, res.stderr
    env = _last_json(res.stdout)
    _validator_for("envelope.json").validate(env)
    _validator_for("auth.json").validate(env["data"])
    assert env["data"]["providers"] == []
    # `auth` is now scoped to third-party model-host tokens only.
    # Comfy Cloud sign-in lives under `comfy cloud`.
    assert "comfy-cloud" not in env["data"]["supported"]
    assert "civitai" in env["data"]["supported"]
    assert "huggingface" in env["data"]["supported"]


def test_auth_set_then_list_shows_redacted(cli_env):
    res = _run(["--json", "auth", "set", "civitai", "--key", "abcd1234efgh5678abcd1234"], cli_env)
    assert res.returncode == 0, res.stderr
    env = _last_json(res.stdout)
    assert env["changed"] is True
    providers = {p["provider"]: p for p in env["data"]["providers"]}
    assert providers["civitai"]["key"] == "abcd…1234"
    assert providers["civitai"]["key_redacted"] is True
    # Plaintext key must not appear in the envelope.
    assert "abcd1234efgh5678abcd1234" not in json.dumps(env)


def test_auth_remove_idempotent(cli_env):
    _run(["auth", "set", "civitai", "--key", "abcd1234efgh5678"], cli_env)
    res = _run(["--json", "auth", "remove", "civitai"], cli_env)
    assert res.returncode == 0
    res2 = _run(["--json", "auth", "remove", "civitai"], cli_env)
    assert res2.returncode != 0
    env = _last_json(res2.stdout)
    assert env["error"]["code"] == "auth_not_found"


def test_run_where_cloud_without_key_returns_cloud_not_configured(cli_env, tmp_path):
    wf = tmp_path / "workflow.json"
    wf.write_text(json.dumps({"1": {"class_type": "Anything", "inputs": {}}}))
    res = _run(["--json", "run", "--workflow", str(wf), "--where", "cloud"], cli_env)
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["ok"] is False
    assert env["error"]["code"] == "cloud_not_configured"
    assert env["error"]["hint"] is not None


def test_run_where_cloud_with_expired_session_returns_unauthorized(cli_env, tmp_path):
    # Plant an *expired* OAuth session with NO refresh token (so proactive
    # refresh is skipped — no network) → preflight surfaces unauthorized.
    secrets_path = Path(cli_env["COMFY_SECRETS_PATH"])
    secrets_path.write_text(
        json.dumps(
            {
                "providers": {},
                "cloud_session": {
                    "base_url": "https://testcloud.comfy.org",
                    "resource": "https://testcloud.comfy.org/mcp",
                    "client_id": "mcp-dyn-fake-id",
                    "scope": "mcp:tools:read mcp:tools:call",
                    "saved_at": "2026-05-15T00:00:00+00:00",
                    "tokens": {
                        "access_token": "fake-access-token-aaaaaaaa",
                        "refresh_token": None,  # unrefreshable
                        "token_type": "Bearer",
                        "expires_at": 1,  # long expired
                    },
                },
            }
        )
    )
    wf = tmp_path / "workflow.json"
    wf.write_text(json.dumps({"1": {"class_type": "Anything", "inputs": {}}}))
    res = _run(["--json", "run", "--workflow", str(wf), "--where", "cloud"], cli_env)
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["error"]["code"] == "cloud_unauthorized"
    assert "comfy cloud login" in env["error"]["hint"]


def test_cloud_whoami_expired_unrefreshable_session_is_not_signed_in(cli_env):
    # An expired OAuth session with no refresh token must report signed_in=False
    # (and expired=True) — not a misleading signed_in=True.
    secrets_path = Path(cli_env["COMFY_SECRETS_PATH"])
    secrets_path.write_text(
        json.dumps(
            {
                "providers": {},
                "cloud_session": {
                    "base_url": "https://testcloud.comfy.org",
                    "resource": "https://testcloud.comfy.org/mcp",
                    "client_id": "mcp-dyn-fake-id",
                    "scope": "mcp:tools:read",
                    "saved_at": "2026-05-15T00:00:00+00:00",
                    "tokens": {
                        "access_token": "fake-access-token-aaaaaaaa",
                        "refresh_token": None,  # unrefreshable
                        "token_type": "Bearer",
                        "expires_at": 1,  # long expired
                    },
                },
            }
        )
    )
    res = _run(["--json", "cloud", "whoami"], cli_env)
    assert res.returncode == 0
    env = _last_json(res.stdout)
    assert env["data"]["signed_in"] is False
    assert env["data"]["expired"] is True


def test_auth_set_comfy_cloud_rejected(cli_env):
    res = _run(
        ["--json", "auth", "set", "comfy-cloud", "--key", "sk-test-1234567890"],
        cli_env,
    )
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["error"]["code"] == "auth_use_login_for_cloud"
    assert "comfy cloud login" in (env["error"]["hint"] or "")


def test_cloud_whoami_not_signed_in(cli_env):
    """`whoami` lives under the `cloud` namespace now, not `auth`."""
    res = _run(["--json", "cloud", "whoami"], cli_env)
    assert res.returncode == 0
    env = _last_json(res.stdout)
    assert env["command"] == "cloud whoami"
    assert env["data"]["signed_in"] is False
    assert env["data"]["auth_method"] is None
    assert env["data"]["api_key_source"] is None
    assert env["data"]["base_url"].startswith("https://")


def test_cloud_whoami_signed_in_via_api_key_env(cli_env):
    """API key in env should make whoami report signed_in=true, auth_method=api_key."""
    env = dict(cli_env)
    env["COMFY_CLOUD_API_KEY"] = "sk-test-1234567890"
    res = _run(["--json", "cloud", "whoami"], env)
    assert res.returncode == 0
    payload = _last_json(res.stdout)
    assert payload["data"]["signed_in"] is True
    assert payload["data"]["auth_method"] == "api_key"
    assert payload["data"]["api_key_source"] == "env"
    # Plaintext key must never appear in the envelope.
    assert "sk-test-1234567890" not in json.dumps(payload)


def test_cloud_whoami_signed_in_via_api_key_store(cli_env):
    """API key in the on-disk store should also flip signed_in=true."""
    secrets_path = Path(cli_env["COMFY_SECRETS_PATH"])
    secrets_path.write_text(
        json.dumps(
            {
                "providers": {
                    "comfy-cloud-api-key": {"key": "sk-store-1234567890", "updated_at": "2026-05-19T00:00:00+00:00"}
                },
                "cloud_session": None,
            }
        )
    )
    res = _run(["--json", "cloud", "whoami"], cli_env)
    assert res.returncode == 0
    payload = _last_json(res.stdout)
    assert payload["data"]["signed_in"] is True
    assert payload["data"]["auth_method"] == "api_key"
    assert payload["data"]["api_key_source"] == "store"
    assert "sk-store-1234567890" not in json.dumps(payload)


def test_legacy_auth_whoami_is_gone(cli_env):
    """We removed the alias deliberately — `auth whoami` is no longer a command."""
    res = _run(["auth", "whoami"], cli_env)
    assert res.returncode != 0


def test_run_where_invalid_value_errors(cli_env, tmp_path):
    wf = tmp_path / "workflow.json"
    wf.write_text("{}")
    res = _run(["--json", "run", "--workflow", str(wf), "--where", "spaceship"], cli_env)
    assert res.returncode != 0
    env = _last_json(res.stdout)
    assert env["error"]["code"] == "where_invalid"
