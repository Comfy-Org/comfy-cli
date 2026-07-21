"""Tests for ``comfy_cli.local_address`` — COMFY_LOCAL_URL parsing + the
local host/port precedence resolver, plus its use at the real resolution sites
(``resolve_target``, the ``jobs`` resolver, and ``comfy env``)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.local_address import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ENV_LOCAL_URL,
    parse_local_url,
    resolve_local_host_port,
)

# ---------------------------------------------------------------------------
# parse_local_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://127.0.0.1:8189", ("127.0.0.1", 8189)),
        ("127.0.0.1:8189", ("127.0.0.1", 8189)),
        ("http://localhost", ("localhost", None)),  # scheme, no port -> None (falls through)
        ("localhost", ("localhost", None)),  # bare host -> no port
        ("http://example.test:9000/", ("example.test", 9000)),  # trailing path dropped
        ("[::1]:8189", ("::1", 8189)),  # bracketed IPv6 + port
        ("http://[::1]:8189", ("::1", 8189)),  # scheme + bracketed IPv6
        ("[::1]", ("::1", None)),  # bracketed IPv6, no port
        ("::1", ("::1", None)),  # bare IPv6 literal (2+ colons) -> no port
        ("  http://127.0.0.1:8189  ", ("127.0.0.1", 8189)),  # whitespace stripped
        ("HTTP://127.0.0.1:8189", ("127.0.0.1", 8189)),  # scheme case-insensitive
    ],
)
def test_parse_local_url_valid(value, expected):
    assert parse_local_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "   ",  # whitespace only
        "https://127.0.0.1:8189",  # non-http scheme
        "ftp://host:21",  # non-http scheme
        "http://",  # missing host
        ":8189",  # empty host
        "host:notaport",  # non-numeric port
        "host:0",  # port out of range (low)
        "host:70000",  # port out of range (high)
        "[::1:8189",  # unterminated bracket
        "[::1]x8189",  # junk after bracket
        "user@evil.com:8189",  # URL-special char (userinfo '@') in the authority
        "a[xyz]",  # stray brackets in a non-IPv6 authority (markup-injection vector)
        "a[xyz]:8189",  # same, with a port
    ],
)
def test_parse_local_url_invalid_raises(value):
    with pytest.raises(ValueError):
        parse_local_url(value)


# ---------------------------------------------------------------------------
# resolve_local_host_port — precedence
# ---------------------------------------------------------------------------


def test_resolve_defaults_when_nothing_set():
    assert resolve_local_host_port(None, None, env={}) == (DEFAULT_HOST, DEFAULT_PORT)


def test_resolve_flag_wins_over_everything():
    env = {ENV_LOCAL_URL: "http://envhost:9000"}
    bg = ("bghost", 9001, 4242)
    assert resolve_local_host_port("flaghost", 7000, background=bg, env=env) == ("flaghost", 7000)


def test_resolve_env_wins_over_background_and_default():
    env = {ENV_LOCAL_URL: "http://envhost:9000"}
    bg = ("bghost", 9001, 4242)
    assert resolve_local_host_port(None, None, background=bg, env=env) == ("envhost", 9000)


def test_resolve_background_wins_over_default():
    assert resolve_local_host_port(None, None, background=("bghost", 9001, 4242), env={}) == ("bghost", 9001)


def test_resolve_host_and_port_are_independent():
    # Explicit --port with no --host still takes the env var's host.
    env = {ENV_LOCAL_URL: "http://envhost:9000"}
    assert resolve_local_host_port(None, 7000, env=env) == ("envhost", 7000)
    # Explicit --host with no --port still takes the env var's port.
    assert resolve_local_host_port("flaghost", None, env=env) == ("flaghost", 9000)


def test_resolve_env_port_only_keeps_default_host():
    # COMFY_LOCAL_URL carrying only a host still contributes just its host;
    # the port falls through to the default (no port in the URL -> 8188).
    env = {ENV_LOCAL_URL: "http://envhost"}
    assert resolve_local_host_port(None, None, env=env) == ("envhost", DEFAULT_PORT)


def test_resolve_host_only_env_falls_through_to_background_port():
    # A host-only COMFY_LOCAL_URL must NOT shadow a recorded background port
    # with a defaulted 8188: host comes from the env, port from background.
    env = {ENV_LOCAL_URL: "http://envhost"}
    assert resolve_local_host_port(None, None, background=("bghost", 9001, 4242), env=env) == ("envhost", 9001)


def test_resolve_env_brackets_in_host_are_ignored_with_warning(capsys):
    # A stray-bracket authority is malformed: ignored (not a live target), and
    # never flows into a URL / Rich markup where it would corrupt output.
    import comfy_cli.local_address as la

    la._warned.clear()
    env = {ENV_LOCAL_URL: "http://a[xyz]:8189"}
    assert resolve_local_host_port(None, None, background=("bghost", 9001), env=env) == ("bghost", 9001)
    assert ENV_LOCAL_URL in capsys.readouterr().err


def test_resolve_invalid_env_warning_redacts_userinfo(capsys):
    # Credentials in a mistyped value must not be echoed to stderr / CI logs.
    import comfy_cli.local_address as la

    la._warned.clear()
    env = {ENV_LOCAL_URL: "http://user:s3cret@host/path"}  # '@' fails validation
    resolve_local_host_port(None, None, env=env)
    err = capsys.readouterr().err
    assert "s3cret" not in err and "user" not in err
    assert "***@host" in err


def test_resolve_invalid_env_is_ignored_with_warning(capsys):
    import comfy_cli.local_address as la

    la._warned.clear()
    env = {ENV_LOCAL_URL: "https://not-http:9000"}
    # Falls through to background/default; the bad value is ignored, not raised.
    assert resolve_local_host_port(None, None, background=("bghost", 9001), env=env) == ("bghost", 9001)
    err = capsys.readouterr().err
    assert ENV_LOCAL_URL in err and "https://not-http:9000" in err


def test_resolve_invalid_env_warning_is_deduplicated(capsys):
    import comfy_cli.local_address as la

    la._warned.clear()
    env = {ENV_LOCAL_URL: "garbage://x"}
    resolve_local_host_port(None, None, env=env)
    resolve_local_host_port(None, None, env=env)
    # Only one warning line for the same bad value across the process.
    assert capsys.readouterr().err.count(ENV_LOCAL_URL) == 1


# ---------------------------------------------------------------------------
# Integration — the env var reaches the real resolution sites
# ---------------------------------------------------------------------------


def test_resolve_target_local_honors_env(monkeypatch):
    from comfy_cli.target import resolve_target

    monkeypatch.setenv(ENV_LOCAL_URL, "http://127.0.0.1:8189")
    target = resolve_target(where="local")
    assert target.kind == "local"
    assert target.base_url == "http://127.0.0.1:8189"
    assert target.host == "127.0.0.1"
    assert target.port == 8189


def test_resolve_target_local_flag_beats_env(monkeypatch):
    from comfy_cli.target import resolve_target

    monkeypatch.setenv(ENV_LOCAL_URL, "http://127.0.0.1:8189")
    target = resolve_target(where="local", port=8188)
    # Explicit --port wins; host still comes from the env var.
    assert target.base_url == "http://127.0.0.1:8188"
    assert target.port == 8188


def test_jobs_resolver_honors_env(monkeypatch):
    from comfy_cli.command.jobs import _resolve_host_port

    monkeypatch.setenv(ENV_LOCAL_URL, "http://127.0.0.1:8189")
    with patch("comfy_cli.host_port.ConfigManager") as cm:
        cm.return_value.background = None
        assert _resolve_host_port(None, None) == ("127.0.0.1", 8189)
        # Explicit flag still wins over the env var.
        assert _resolve_host_port(None, 8188) == ("127.0.0.1", 8188)


def test_run_resolver_precedence(monkeypatch):
    from comfy_cli.host_port import resolve_host_port

    monkeypatch.setenv(ENV_LOCAL_URL, "http://envhost:8189")
    with patch("comfy_cli.host_port.ConfigManager") as cm:
        # env beats background beats default.
        cm.return_value.background = ("bghost", 9001, 4242)
        assert resolve_host_port(None, None) == ("envhost", 8189)
        # explicit flag beats env.
        assert resolve_host_port("flaghost", 7000) == ("flaghost", 7000)


def test_env_json_reports_url_from_env_var(monkeypatch):
    """`comfy env --json` server.url honors COMFY_LOCAL_URL (mocked probe)."""
    from comfy_cli.env_checker import EnvChecker

    monkeypatch.setenv(ENV_LOCAL_URL, "http://127.0.0.1:8189")
    checker = EnvChecker()
    with (
        patch("comfy_cli.env_checker.check_comfy_server_running", return_value=True),
        patch("comfy_cli.env_checker.ConfigManager") as cm,
    ):
        cm.return_value.background = None
        cm.return_value.get_data.return_value = {}
        data = checker.fill_data()
    assert data["server"]["running"] is True
    assert data["server"]["url"] == "http://127.0.0.1:8189"


def test_env_probe_uses_resolved_address(monkeypatch):
    """The probe must hit the env-resolved address, not hardcoded :8188."""
    from comfy_cli import env_checker

    monkeypatch.setenv(ENV_LOCAL_URL, "http://127.0.0.1:8189")
    checker = env_checker.EnvChecker()
    with (
        patch("comfy_cli.env_checker.check_comfy_server_running", return_value=False) as probe,
        patch("comfy_cli.env_checker.ConfigManager") as cm,
    ):
        cm.return_value.background = None
        cm.return_value.get_data.return_value = {}
        checker.fill_data()
    assert probe.call_args.kwargs == {"port": 8189, "host": "127.0.0.1"}


def test_env_json_reports_bracketed_ipv6(monkeypatch):
    from comfy_cli.env_checker import EnvChecker

    monkeypatch.setenv(ENV_LOCAL_URL, "http://[::1]:8189")
    checker = EnvChecker()
    with (
        patch("comfy_cli.env_checker.check_comfy_server_running", return_value=True),
        patch("comfy_cli.env_checker.ConfigManager") as cm,
    ):
        cm.return_value.background = None
        cm.return_value.get_data.return_value = {}
        data = checker.fill_data()
    assert data["server"]["url"] == "http://[::1]:8189"


def test_job_watcher_state_beats_env(monkeypatch):
    """Per-job recorded state wins over the env var: flag > state > env > default."""
    from comfy_cli.command import job_watcher

    monkeypatch.setenv(ENV_LOCAL_URL, "http://envhost:8189")
    state = MagicMock()
    state.host = "recorded-host"
    state.port = 9999
    state.prompt_id = "pid"
    captured = {}

    def fake_snapshot(h, p, pid):
        captured["h"], captured["p"] = h, p
        return None  # no record yet -> watcher keeps polling, returns False

    with patch("comfy_cli.command.jobs._snapshot", side_effect=fake_snapshot):
        job_watcher._poll_local_once(state, host=None, port=None)
    assert (captured["h"], captured["p"]) == ("recorded-host", 9999)
