"""Tests for the shared host/port resolver (`comfy_cli.host_port`) and its use
by the `comfy run` command path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli.host_port import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    parse_host_port_arg,
    resolve_host_port,
    validate_host,
)

# ---------------------------------------------------------------------------
# parse_host_port_arg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("localhost", ("localhost", None)),
        ("127.0.0.1:9000", ("127.0.0.1", 9000)),
        ("[::1]:8188", ("::1", 8188)),
        ("[::1]", ("::1", None)),
        ("::1", ("::1", None)),  # bare IPv6 literal (2+ colons) -> no port
        ("127.0.0.1", ("127.0.0.1", None)),
        ("  localhost:8080  ", ("localhost", 8080)),  # whitespace is stripped
        ("localhost:", ("localhost", None)),  # trailing colon, empty port
    ],
)
def test_parse_host_port_arg(value, expected):
    assert parse_host_port_arg(value) == expected


def test_parse_host_port_arg_non_numeric_port_raises():
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg("localhost:notaport")


def test_parse_host_port_arg_non_numeric_bracketed_port_raises():
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg("[::1]:notaport")


def test_parse_host_port_arg_unterminated_bracket_raises():
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg("[::1")


@pytest.mark.parametrize("value", ["[::1]8188", "[::1]junk"])
def test_parse_host_port_arg_trailing_text_after_bracket_raises(value):
    # A missing/garbled ':' after ']' must error, not silently drop the port.
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg(value)


@pytest.mark.parametrize("value", ["host:0", "host:-1", "host:99999", "[::1]:0", "[::1]:70000"])
def test_parse_host_port_arg_out_of_range_port_raises(value):
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg(value)


@pytest.mark.parametrize("value", [":8188", "[]:8188"])
def test_parse_host_port_arg_empty_host_raises(value):
    with pytest.raises(typer.BadParameter):
        parse_host_port_arg(value)


# ---------------------------------------------------------------------------
# validate_host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "example.com"])
def test_validate_host_allows_safe(host):
    assert validate_host(host) == host


@pytest.mark.parametrize("host", ["evil.com/@x", "a/b", "h@ost", "h?ost", "h#ost"])
def test_validate_host_rejects_url_special_chars(host):
    with pytest.raises(typer.BadParameter):
        validate_host(host)


@pytest.mark.parametrize("host", ["host\r\nX", "ho st", "host\t", "host\x00"])
def test_validate_host_rejects_whitespace_and_control_chars(host):
    with pytest.raises(typer.BadParameter):
        validate_host(host)


@pytest.mark.parametrize("host", ["", "   ", "\t"])
def test_validate_host_rejects_empty(host):
    # An empty host is resolved as *absent* downstream (`host or env or
    # DEFAULT_HOST`), so accepting it silently retargets the request at a
    # server the caller never named — e.g. `--host "$UNSET_VAR"`.
    with pytest.raises(typer.BadParameter):
        validate_host(host)


@pytest.mark.parametrize("host", ["a%0d%0aX-Injected:%201", "host%2fpath", "h%40ost", "host%23x"])
def test_validate_host_rejects_percent_encoded_specials(host):
    # urllib.request.Request._parse unquotes the host it splits out of the
    # URL, so a percent-encoded payload decodes downstream of this guard.
    with pytest.raises(typer.BadParameter):
        validate_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1:8188", "localhost:8188", "example.com:80"])
def test_validate_host_rejects_embedded_port(host):
    # Colon-bearing hosts get bracketed as IPv6 literals by callers, so a
    # combined host:port would become `http://[127.0.0.1:8188]:8188` with the
    # embedded port silently dropped. Only `comfy run` takes the combined
    # form, and it splits it via parse_host_port_arg first.
    with pytest.raises(typer.BadParameter):
        validate_host(host)


@pytest.mark.parametrize("host", ["::1", "[::1]", "fe80::1", "2001:db8::8a2e:370:7334"])
def test_validate_host_allows_ipv6_literals(host):
    assert validate_host(host) == host


# ---------------------------------------------------------------------------
# resolve_host_port
# ---------------------------------------------------------------------------


def _patch_background(bg):
    """Patch ConfigManager as seen by resolve_host_port so `.background` == bg."""
    cfg = MagicMock()
    cfg.background = bg
    return patch("comfy_cli.host_port.ConfigManager", return_value=cfg)


def test_resolve_host_port_brackets_ipv6():
    with _patch_background(None):
        assert resolve_host_port("::1", None) == ("[::1]", DEFAULT_PORT)


def test_resolve_host_port_does_not_double_bracket():
    with _patch_background(None):
        assert resolve_host_port("[::1]", 8188) == ("[::1]", 8188)


def test_resolve_host_port_rejects_unsafe_host():
    with _patch_background(None), pytest.raises(typer.BadParameter):
        resolve_host_port("evil.com/@x", None)


def test_resolve_host_port_applies_defaults():
    with _patch_background(None):
        assert resolve_host_port(None, None) == (DEFAULT_HOST, DEFAULT_PORT)


def test_resolve_host_port_falls_back_to_background():
    # config.background is a (host, port, pid) tuple.
    with _patch_background(("10.0.0.5", 9001, 4242)):
        assert resolve_host_port(None, None) == ("10.0.0.5", 9001)


def test_resolve_host_port_explicit_overrides_background():
    with _patch_background(("10.0.0.5", 9001, 4242)):
        assert resolve_host_port("localhost", 7000) == ("localhost", 7000)


# ---------------------------------------------------------------------------
# `comfy run` integration — host flows through the resolver
# ---------------------------------------------------------------------------


def _write_workflow(tmp_path):
    wf = tmp_path / "wf.json"
    wf.write_text('{"1": {"class_type": "X", "inputs": {}}}')
    return str(wf)


def test_run_ipv6_host_port_reaches_execute(tmp_path):
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    with patch("comfy_cli.command.run.execute") as mock_execute:
        result = CliRunner().invoke(
            app,
            ["run", "--workflow", _write_workflow(tmp_path), "--host", "[::1]:8188"],
            env={"COMFY_WHERE": "local"},
        )
    assert result.exit_code == 0, result.output
    args, _ = mock_execute.call_args
    # execute(workflow, host, port, ...)
    assert args[1] == "[::1]"
    assert args[2] == 8188


def test_run_unsafe_host_exits_before_execute(tmp_path):
    from typer.testing import CliRunner

    from comfy_cli.cmdline import app

    with patch("comfy_cli.command.run.execute") as mock_execute:
        result = CliRunner().invoke(
            app,
            ["run", "--workflow", _write_workflow(tmp_path), "--host", "x/@y"],
            env={"COMFY_WHERE": "local"},
        )
    assert result.exit_code != 0
    mock_execute.assert_not_called()


# ---------------------------------------------------------------------------
# explicit-flag validation at the shared choke point (BE-6306 review)
# ---------------------------------------------------------------------------
#
# `resolve_local_host_port` resolves each half as `value or env or bg or
# DEFAULT`, so every FALSY explicit flag reads as "not passed" and silently
# retargets the request at a different server. `resolve_host_port` therefore
# validates an explicitly-passed host/port BEFORE that chain runs.


def test_empty_host_flag_is_rejected():
    """`--host ""` (a wrapper interpolating an unset variable) must error, not
    fall through to the env/background/default server."""
    with pytest.raises(typer.BadParameter, match="empty host"):
        resolve_host_port("", None)


def test_blank_host_flag_is_rejected():
    with pytest.raises(typer.BadParameter, match="empty host"):
        resolve_host_port("   ", None)


@pytest.mark.parametrize("bad_port", [0, -1, 65536, 99999])
def test_out_of_range_port_flag_is_rejected(bad_port):
    with pytest.raises(typer.BadParameter, match="out of range"):
        resolve_host_port(None, bad_port)


def test_none_host_and_port_still_resolve(monkeypatch):
    """The guards fire only on an EXPLICIT value — `None` still means absent."""
    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)
    with patch("comfy_cli.host_port.ConfigManager") as cm:
        cm.return_value.background = None
        assert resolve_host_port(None, None) == (DEFAULT_HOST, DEFAULT_PORT)


def test_boundary_ports_are_accepted(monkeypatch):
    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)
    with patch("comfy_cli.host_port.ConfigManager") as cm:
        cm.return_value.background = None
        assert resolve_host_port(None, 1)[1] == 1
        assert resolve_host_port(None, 65535)[1] == 65535
