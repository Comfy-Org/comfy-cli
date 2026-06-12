"""Observer commands must not hold session-clearing rights.

June 2026 field incident: a batch production ran dozens of concurrent
``jobs status`` / ``download`` retry loops; one of them hit a fatal
``invalid_grant`` on token refresh and — holding the default
``clear_session_on_auth_failure=True`` — wiped the shared OAuth session
mid-run, turning a transient server-side hiccup into a hard logout that
needed a human to re-run ``comfy cloud login``.

Policy pinned here: commands that *observe* jobs (status/ls/watch snapshots,
download) construct their cloud Client with
``clear_session_on_auth_failure=False`` — same as the detached run watcher.
Only session-lifecycle owners (login/logout, foreground ``run``) may clear.
"""

from __future__ import annotations

import pytest


class _CapturingClient:
    """Stands in for comfy_client.Client; records constructor kwargs."""

    captured: list[dict] = []

    def __init__(self, target, *, timeout: float = 30.0, clear_session_on_auth_failure: bool = True):
        type(self).captured.append({"clear_session_on_auth_failure": clear_session_on_auth_failure})
        self.target = target


@pytest.fixture(autouse=True)
def _reset_captures():
    _CapturingClient.captured = []
    yield
    _CapturingClient.captured = []


@pytest.fixture
def _fake_target(monkeypatch):
    import comfy_cli.target as target_mod

    class _T:
        base_url = "https://cloud.example"

    monkeypatch.setattr(target_mod, "resolve_target", lambda **kw: _T())
    return _T


def test_jobs_cloud_client_cannot_clear_session(monkeypatch, _fake_target):
    import comfy_cli.comfy_client as comfy_client_mod
    from comfy_cli.command import jobs

    monkeypatch.setattr(comfy_client_mod, "Client", _CapturingClient)
    jobs._cloud_client()
    assert _CapturingClient.captured, "jobs._cloud_client() did not construct a Client"
    assert _CapturingClient.captured[-1]["clear_session_on_auth_failure"] is False


def test_download_fallback_client_cannot_clear_session():
    """The download command's API-fallback Client (transfer.py) must pass
    clear_session_on_auth_failure=False. Pinned at source level: a download
    retry loop is exactly the workload that wiped the session in the field."""
    import ast
    import inspect

    from comfy_cli.command import transfer

    tree = ast.parse(inspect.getsource(transfer))
    client_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Client"
    ]
    assert client_calls, "expected at least one Client(...) construction in transfer.py"
    for call in client_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "clear_session_on_auth_failure" in kwargs, (
            f"transfer.py line {call.lineno}: Client(...) without explicit "
            "clear_session_on_auth_failure=False — observer commands must not "
            "be able to wipe the shared OAuth session"
        )
        value = kwargs["clear_session_on_auth_failure"]
        assert isinstance(value, ast.Constant) and value.value is False, (
            f"transfer.py line {call.lineno}: clear_session_on_auth_failure must be the literal False"
        )
