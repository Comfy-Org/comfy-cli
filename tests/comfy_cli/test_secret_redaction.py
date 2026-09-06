"""Regression guard: no command may echo an *unredacted* secret.

A 2026-07-16 dogfooding transcript reported ``comfy cloud login`` printing an
API key in plaintext. On current ``main`` that is no longer reproducible —
every echo path funnels its secret through :func:`store._redact` (``auth set``
and ``cloud set-key`` print ``record.to_dict()['key']``; ``cloud login`` /
``cloud whoami`` emit ``session.to_dict(redact=True)``). But that redaction is
only ever a *default* argument at each call site — nothing keeps a future
command (or a refactor that flips a ``redact=`` default) from leaking the raw
value again.

This module is that keep-it-that-way guard. For every secret-bearing command,
in BOTH pretty and ``--json`` output modes, it feeds a long sentinel secret and
asserts the FULL sentinel never appears in captured stdout **or** stderr. The
redacted ``sk-S…cdef`` form is expected and fine.

Adding coverage for a NEW secret-bearing command is a single ``_Case`` entry in
``CASES`` below (or, if the command needs a mocked network handshake like
``cloud login``, a sibling of ``test_login_success_never_echoes_secret``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from comfy_cli.auth import store

# Sentinels are all > 16 chars so they take the ``sk-S…cdef`` redaction branch,
# not the ``<=16 → ***`` one (`store._redact`). Distinct values per secret slot
# so a leak of *any* of them is caught individually.
SENTINEL_KEY = "sk-SENTINEL-KEY-0123456789abcdef"
SENTINEL_ACCESS = "sk-SENTINEL-ACCESS-0123456789abcdef"
SENTINEL_REFRESH = "sk-SENTINEL-REFRESH-0123456789abcdef"

# A far-future expiry keeps ``cloud whoami`` from spending the (fake) refresh
# token — no network, the seeded access token is reported as-is (redacted).
_FAR_FUTURE_EPOCH = 4102444800  # 2100-01-01T00:00:00Z


def _assert_redacts(text: str) -> None:
    assert len(text) > 16, "sentinel must exceed the `<=16 → ***` redaction cap"


for _s in (SENTINEL_KEY, SENTINEL_ACCESS, SENTINEL_REFRESH):
    _assert_redacts(_s)


@dataclass(frozen=True)
class _Case:
    """One secret-bearing invocation to guard.

    ``args`` are the CLI args *after* the global output-mode flag. ``secrets``
    lists every full sentinel that must never appear in the output. ``seed``
    (optional) plants on-disk state (e.g. a stored session) before the command
    runs; it is called with ``COMFY_SECRETS_PATH`` already pointing at the temp
    store.
    """

    id: str
    args: tuple[str, ...]
    secrets: tuple[str, ...]
    seed: Callable[[], object] | None = field(default=None)


def _seed_cloud_session() -> None:
    store.save_cloud_session(
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        client_id="mcp-dyn-fake-id",
        scope="mcp:tools:read mcp:tools:call",
        access_token=SENTINEL_ACCESS,
        refresh_token=SENTINEL_REFRESH,
        token_type="Bearer",
        expires_at=_FAR_FUTURE_EPOCH,
    )


CASES: tuple[_Case, ...] = (
    _Case(
        id="auth-set-civitai",
        args=("auth", "set", "civitai", "--key", SENTINEL_KEY),
        secrets=(SENTINEL_KEY,),
    ),
    _Case(
        id="cloud-set-key",
        args=("cloud", "set-key", "--key", SENTINEL_KEY),
        secrets=(SENTINEL_KEY,),
    ),
    _Case(
        id="cloud-whoami",
        args=("cloud", "whoami"),
        secrets=(SENTINEL_ACCESS, SENTINEL_REFRESH),
        seed=_seed_cloud_session,
    ),
)

# `--json` → JSON envelope mode; `--no-json` forces pretty even under a
# non-tty test stdout (see comfy_cli/output/renderer.py mode resolution).
MODES = ("--json", "--no-json")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["DO_NOT_TRACK"] = "1"  # never touch telemetry from a test
    env["COMFY_SECRETS_PATH"] = str(tmp_path / "secrets.json")
    # Mirror the secrets path into this process too, so a case's `seed`
    # (which calls store.save_cloud_session) writes the same file the
    # subprocess reads.
    monkeypatch.setenv("COMFY_SECRETS_PATH", env["COMFY_SECRETS_PATH"])
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    return env


def _run(args, env):
    return subprocess.run(
        [sys.executable, "-m", "comfy_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
@pytest.mark.parametrize("mode", MODES)
def test_command_never_echoes_unredacted_secret(case: _Case, mode: str, cli_env):
    if case.seed is not None:
        case.seed()

    res = _run([mode, *case.args], cli_env)

    # A silent early exit (nonzero, empty output) would pass the "secret
    # absent" check for the wrong reason — require the command to succeed so
    # the redaction path is genuinely exercised.
    assert res.returncode == 0, f"{case.id}/{mode} failed:\nstdout={res.stdout!r}\nstderr={res.stderr!r}"

    combined = res.stdout + "\n" + res.stderr
    for secret in case.secrets:
        assert secret not in combined, f"{case.id}/{mode} leaked an unredacted secret in output:\n{combined!r}"


def test_auth_set_json_carries_only_the_redacted_key(cli_env):
    """Positive control: prove the guard actually observes the redaction, not
    an empty/errored output that would trivially satisfy 'secret absent'."""
    res = _run(["--json", "auth", "set", "civitai", "--key", SENTINEL_KEY], cli_env)
    assert res.returncode == 0, res.stderr
    last = [line for line in res.stdout.splitlines() if line.strip()][-1]
    payload = json.loads(last)
    providers = {p["provider"]: p for p in payload["data"]["providers"]}
    assert providers["civitai"]["key"] == store._redact(SENTINEL_KEY)
    assert providers["civitai"]["key_redacted"] is True
    assert SENTINEL_KEY not in json.dumps(payload)


# ---------------------------------------------------------------------------
# cloud login — needs the OAuth handshake mocked, so it runs in-process via
# Typer's CliRunner (a subprocess can't patch `run_login`). The success path
# emits `session.to_dict(redact=True)`, same as whoami, but we cover it
# explicitly per the ticket.
# ---------------------------------------------------------------------------


def _fake_login_result():
    from comfy_cli.cloud.oauth import LoginResult, TokenSet

    tokens = TokenSet(
        access_token=SENTINEL_ACCESS,
        refresh_token=SENTINEL_REFRESH,
        token_type="Bearer",
        expires_in=None,
        expires_at=_FAR_FUTURE_EPOCH,
        scope="mcp:tools:read mcp:tools:call",
    )
    return LoginResult(
        tokens=tokens,
        client_id="mcp-dyn-fake-id",
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        scope="mcp:tools:read mcp:tools:call",
        redirect_uri="http://127.0.0.1:0/callback",
    )


@pytest.mark.parametrize("mode", MODES)
def test_login_success_never_echoes_secret(mode: str, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    secrets_path = tmp_path / "secrets.json"
    invoke_env = {
        "NO_COLOR": "1",
        "DO_NOT_TRACK": "1",
        "COMFY_SECRETS_PATH": str(secrets_path),
    }

    runner = CliRunner()
    with patch("comfy_cli.cloud.command.run_login", return_value=_fake_login_result()):
        result = runner.invoke(app, [mode, "cloud", "login", "--no-browser"], env=invoke_env)

    assert result.exit_code == 0, f"login/{mode} failed: output={result.output!r} exc={result.exception!r}"
    # Honor the module's 'stdout OR stderr' invariant like the subprocess cases:
    # result.output is stdout-only, so a secret leaked to stderr would slip past.
    combined = result.output + result.stderr
    for secret in (SENTINEL_ACCESS, SENTINEL_REFRESH):
        assert secret not in combined, f"login/{mode} leaked a secret:\n{combined!r}"

    # Positive control: a real session was persisted, so the "secret absent
    # from output" assertion above passed because redaction fired — not because
    # the command no-op'd and emitted nothing. The on-disk store intentionally
    # keeps the RAW token (only command *output* is redacted), so finding the
    # sentinel in the file is expected and is what proves the write happened.
    assert secrets_path.exists()
    assert SENTINEL_ACCESS in secrets_path.read_text()
