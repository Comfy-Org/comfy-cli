"""``comfy cloud login`` machine-output tests.

The pretty renderer surfaces the OAuth authorize URL for headless users; these
tests pin the machine-mode contract added for agents/MCP: under ``--json`` the
URL is emitted as a ``login_url`` ``event/1`` line and flushed *before*
``run_login`` blocks on the loopback callback, so a pipe-reading parent can open
it. The final envelope (success or ``oauth_timeout`` error) still renders, with
the session redacted.

Pattern: install a renderer in the mode under test, mock
``comfy_cli.cloud.command.run_login`` (so no browser / network / callback wait),
and capture stdout.
"""

from __future__ import annotations

import json

import pytest
import typer

from comfy_cli.cloud import command, oauth
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode

_AUTHORIZE_URL = "https://example/oauth/authorize?response_type=code&code_challenge=abc123"
_ACCESS_TOKEN = "super-secret-access-token-do-not-leak"
_REFRESH_TOKEN = "super-secret-refresh-token-do-not-leak"


def _login_result() -> oauth.LoginResult:
    tokens = oauth.TokenSet(
        access_token=_ACCESS_TOKEN,
        refresh_token=_REFRESH_TOKEN,
        token_type="Bearer",
        expires_in=3600,
        expires_at=9999999999,
        scope="mcp:tools:read mcp:tools:call",
    )
    return oauth.LoginResult(
        tokens=tokens,
        client_id="comfy-cli",
        base_url="https://testcloud.comfy.org",
        resource="https://testcloud.comfy.org/mcp",
        scope="mcp:tools:read mcp:tools:call",
        redirect_uri="http://127.0.0.1:51234/callback",
    )


def _parse_lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _no_tracking(monkeypatch):
    """Keep the ``@track_command`` decorator inert (no network / consent I/O)."""
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _knowledge_warm(monkeypatch):
    """Record the post-login knowledge warm instead of fetching a bundle."""
    calls: list[None] = []
    monkeypatch.setattr(command.knowledge, "refresh_if_stale", lambda: calls.append(None))
    return calls


def test_login_warms_the_knowledge_cache_after_saving_the_session(monkeypatch, capsys, _knowledge_warm):
    order: list[str] = []

    def fake_run_login(**kwargs):
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        return _login_result()

    real_save = command.store.save_cloud_session

    def save_then_record(**kwargs):
        order.append("save")
        return real_save(**kwargs)

    monkeypatch.setattr(command, "run_login", fake_run_login)
    monkeypatch.setattr(command.store, "save_cloud_session", save_then_record)
    monkeypatch.setattr(command.knowledge, "refresh_if_stale", lambda: order.append("warm"))
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud login", version="test"))

    command.login_cmd(no_browser=True, timeout=300)

    assert order == ["save", "warm"]
    assert _parse_lines(capsys.readouterr().out)[-1]["ok"] is True


def test_login_timeout_does_not_warm_the_knowledge_cache(monkeypatch, capsys, _knowledge_warm):
    def fake_run_login(**kwargs):
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        raise oauth.OAuthError("timed out")

    monkeypatch.setattr(command, "run_login", fake_run_login)
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud login", version="test"))

    with pytest.raises(typer.Exit):
        command.login_cmd(no_browser=True, timeout=1)

    assert _knowledge_warm == []


def test_json_login_emits_login_url_event_before_envelope(monkeypatch, capsys):
    """`comfy --json cloud login --no-browser` streams a `login_url` event
    (carrying the authorize URL) ahead of the final, session-redacted envelope."""

    def fake_run_login(**kwargs):
        # run_login fires on_url_ready once the authorize URL is built, before
        # it blocks on the loopback callback. Emulate exactly that ordering.
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        return _login_result()

    monkeypatch.setattr(command, "run_login", fake_run_login)
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud login", version="test"))

    command.login_cmd(no_browser=True, timeout=300)

    out = capsys.readouterr().out
    lines = _parse_lines(out)

    # First the event, then the envelope — the parent must see the URL first.
    assert lines[0]["schema"] == "event/1"
    assert lines[0]["type"] == "login_url"
    assert lines[0]["url"] == _AUTHORIZE_URL
    assert lines[0]["timeout_s"] == 300

    envelope = lines[-1]
    assert envelope["type"] == "envelope"
    assert envelope["ok"] is True
    assert envelope["data"]["action"] == "login"

    # Ordering is explicit, not incidental.
    types = [ln.get("type") for ln in lines]
    assert types.index("login_url") < types.index("envelope")

    # Session is redacted: the raw tokens never reach stdout.
    session = envelope["data"]["session"]
    assert session["tokens_redacted"] is True
    assert _ACCESS_TOKEN not in out
    assert _REFRESH_TOKEN not in out


def test_pretty_login_prints_url_without_event_line(monkeypatch, capsys):
    """Pretty mode is unchanged: the URL is printed for the human, and no
    machine `login_url` event line is emitted."""

    def fake_run_login(**kwargs):
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        return _login_result()

    monkeypatch.setattr(command, "run_login", fake_run_login)
    # Default renderer is pretty (fixture in conftest resets the singleton).
    set_renderer(Renderer(mode=OutputMode.PRETTY, command="cloud login", version="test"))

    command.login_cmd(no_browser=True, timeout=300)

    out = capsys.readouterr().out
    assert _AUTHORIZE_URL in out
    # No machine event/envelope leaked onto stdout in pretty mode.
    assert '"type": "login_url"' not in out
    assert "event/1" not in out


def test_json_login_timeout_still_emits_login_url_before_error_envelope(monkeypatch, capsys):
    """On timeout, the real `run_login` has already fired `on_url_ready` (before
    it blocks on the callback wait), so a `--json` timeout emits the `login_url`
    line *then* the `oauth_timeout` error envelope — the feature's headline
    guarantee (URL surfaced before the block) holds even when the block times
    out. Emulate that ordering, not an impossible URL-less timeout."""

    def fake_run_login(**kwargs):
        # Real run_login surfaces the URL, then the wait times out.
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        raise oauth.OAuthTimeout(
            "timed out waiting for browser callback after 300s",
            hint="re-run `comfy cloud login` and complete the sign-in in your browser",
        )

    monkeypatch.setattr(command, "run_login", fake_run_login)
    set_renderer(Renderer(mode=OutputMode.JSON, command="cloud login", version="test"))

    with pytest.raises(typer.Exit) as exc:
        command.login_cmd(no_browser=True, timeout=300)
    assert exc.value.exit_code == 1

    out = capsys.readouterr().out
    lines = _parse_lines(out)

    # The URL is surfaced first, then the error envelope closes the stream.
    assert lines[0]["type"] == "login_url"
    assert lines[0]["url"] == _AUTHORIZE_URL
    envelope = lines[-1]
    assert envelope["type"] == "envelope"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "oauth_timeout"
    types = [ln.get("type") for ln in lines]
    assert types.index("login_url") < types.index("envelope")


def test_json_login_fails_fast_when_login_url_write_breaks(monkeypatch, capsys):
    """If the machine stream breaks while emitting `login_url` (a piped parent
    hung up), `run_login` re-raises the `OSError` from the callback instead of
    blocking the full timeout, and `login_cmd` fails fast with exit code 1
    rather than a silent 300s hang."""

    class _BrokenStream:
        def write(self, _s):
            raise BrokenPipeError("parent hung up")

        def flush(self):
            raise BrokenPipeError("parent hung up")

    blocked = {"waited": False}

    def real_ish_run_login(**kwargs):
        # Mirror run_login's contract: fire the URL callback (which now writes to
        # the broken stream) *before* the callback wait. A correct implementation
        # never reaches the wait because the OSError propagates.
        kwargs["on_url_ready"](_AUTHORIZE_URL)
        blocked["waited"] = True  # only reached if the OSError was swallowed
        raise oauth.OAuthTimeout("timed out", hint="")

    monkeypatch.setattr(command, "run_login", real_ish_run_login)
    renderer = Renderer(mode=OutputMode.JSON, command="cloud login", version="test")
    renderer.machine_stream = _BrokenStream()
    set_renderer(renderer)

    with pytest.raises(typer.Exit) as exc:
        command.login_cmd(no_browser=True, timeout=300)
    assert exc.value.exit_code == 1
    assert blocked["waited"] is False  # never blocked on the wait
