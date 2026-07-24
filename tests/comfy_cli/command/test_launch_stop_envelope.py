"""Envelope emission for the ``launch`` / ``stop`` lifecycle commands (BE-4273).

Unlike other subcommands, ``comfy launch`` / ``comfy stop`` historically emitted
no ``envelope/1`` on stdout under ``--json``, so programmatic callers (e.g.
comfy-local-mcp, which parses the versioned envelope) got nothing machine-readable
back. These tests pin the envelope on the reachable success and error paths and
assert pretty (human) output is untouched.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import constants
from comfy_cli.cmdline import stop
from comfy_cli.command import launch as launch_mod
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing


@pytest.fixture(autouse=True)
def reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _json_renderer(command: str) -> None:
    set_renderer(Renderer(mode=OutputMode.JSON, command=command))


def _pretty_renderer(command: str) -> None:
    set_renderer(Renderer(mode=OutputMode.PRETTY, command=command))


def _envelopes(captured_out: str) -> list[dict]:
    lines = [ln for ln in captured_out.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


# --------------------------------------------------------------------------- #
# stop
# --------------------------------------------------------------------------- #


class TestStopEnvelope:
    def _config(self, bg_info):
        cfg = MagicMock()
        cfg.config = {"DEFAULT": {constants.CONFIG_KEY_BACKGROUND: "x"} if bg_info else {}}
        cfg.background = bg_info
        return cfg

    def test_success_emits_envelope(self, capsys):
        _json_renderer("stop")
        cfg = self._config(("127.0.0.1", 8188, 4242))
        with (
            patch("comfy_cli.cmdline.ConfigManager", return_value=cfg),
            patch("comfy_cli.cmdline.utils.kill_all", return_value=True),
        ):
            stop()

        cfg.remove_background.assert_called_once()
        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["schema"] == "envelope/1"
        assert env["type"] == "envelope"
        assert env["ok"] is True
        assert env["command"] == "stop"
        assert env["data"] == {"stopped": True, "host": "127.0.0.1", "port": 8188, "pid": 4242}

    def test_no_background_emits_error_envelope(self, capsys):
        _json_renderer("stop")
        cfg = self._config(None)
        with patch("comfy_cli.cmdline.ConfigManager", return_value=cfg):
            with pytest.raises(typer.Exit) as exc:
                stop()
        assert exc.value.exit_code == 1

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["error"]["code"] == "no_background_server"

    def test_kill_failure_emits_error_envelope(self, capsys):
        _json_renderer("stop")
        cfg = self._config(("127.0.0.1", 8188, 4242))
        with (
            patch("comfy_cli.cmdline.ConfigManager", return_value=cfg),
            patch("comfy_cli.cmdline.utils.kill_all", return_value=False),
        ):
            with pytest.raises(typer.Exit) as exc:
                stop()
        assert exc.value.exit_code == 1

        # A failed kill must NOT drop the record — else a still-alive server is
        # orphaned and a retried `comfy stop` can no longer target it.
        cfg.remove_background.assert_not_called()

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["error"]["code"] == "stop_failed"
        assert env["error"]["details"]["pid"] == 4242

    def test_kill_failure_pretty_mode_exits_nonzero(self, capsys):
        # In pretty mode the failure previously printed but returned 0, so
        # `$?`-checking scripts saw success. It must exit non-zero.
        _pretty_renderer("stop")
        cfg = self._config(("127.0.0.1", 8188, 4242))
        with (
            patch("comfy_cli.cmdline.ConfigManager", return_value=cfg),
            patch("comfy_cli.cmdline.utils.kill_all", return_value=False),
        ):
            with pytest.raises(typer.Exit) as exc:
                stop()
        assert exc.value.exit_code == 1
        cfg.remove_background.assert_not_called()
        assert "Failed to stop" in capsys.readouterr().out

    def test_pretty_mode_stdout_has_no_json(self, capsys):
        _pretty_renderer("stop")
        cfg = self._config(("127.0.0.1", 8188, 4242))
        with (
            patch("comfy_cli.cmdline.ConfigManager", return_value=cfg),
            patch("comfy_cli.cmdline.utils.kill_all", return_value=True),
        ):
            stop()
        out = capsys.readouterr().out
        assert "envelope" not in out
        assert "Background ComfyUI is stopped" in out


# --------------------------------------------------------------------------- #
# launch
# --------------------------------------------------------------------------- #


class TestLaunchEnvelope:
    def test_background_success_helper_emits_envelope(self, capsys):
        _json_renderer("launch")
        launch_mod._emit_launch_success("127.0.0.1", 8188, 9999)

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["schema"] == "envelope/1"
        assert env["ok"] is True
        assert env["command"] == "launch"
        assert env["changed"] is True
        assert env["data"] == {
            "background": True,
            "listen": "127.0.0.1",
            "port": 8188,
            "url": "http://127.0.0.1:8188",
            "pid": 9999,
        }

    def test_background_success_helper_pretty_is_noop(self, capsys):
        _pretty_renderer("launch")
        launch_mod._emit_launch_success("127.0.0.1", 8188, 9999)
        assert capsys.readouterr().out == ""

    def test_workspace_unavailable_emits_error_envelope(self, capsys):
        _json_renderer("launch")
        with patch.object(launch_mod.workspace_manager, "workspace_path", None):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.launch(background=False)
        assert exc.value.exit_code == 1

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["command"] == "launch"
        assert env["error"]["code"] == "not_in_workspace"

    def test_background_already_running_emits_error_envelope(self, capsys):
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = ("127.0.0.1", 8188, 4242)
        with (
            patch("comfy_cli.command.launch.ConfigManager", return_value=cfg),
            patch("comfy_cli.command.launch.utils.is_running", return_value=True),
        ):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.background_launch(extra=[])
        assert exc.value.exit_code == 1

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["error"]["code"] == "server_already_running"

    def test_background_port_in_use_emits_error_envelope(self, capsys):
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = None
        with (
            patch("comfy_cli.command.launch.ConfigManager", return_value=cfg),
            patch("comfy_cli.command.launch.check_comfy_server_running", return_value=True),
        ):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.background_launch(extra=["--port", "1234"])
        assert exc.value.exit_code == 1

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["error"]["code"] == "port_in_use"
        assert env["error"]["details"]["port"] == 1234

    def test_background_invalid_port_emits_error_envelope(self, capsys):
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = None
        with patch("comfy_cli.command.launch.ConfigManager", return_value=cfg):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.background_launch(extra=["--port", "notaport"])
        assert exc.value.exit_code == 1

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["error"]["code"] == "port_invalid"
        assert env["error"]["details"]["port"] == "notaport"

    def test_background_launch_failed_emits_error_envelope(self, capsys):
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = None
        with (
            patch("comfy_cli.command.launch.ConfigManager", return_value=cfg),
            patch("comfy_cli.command.launch.check_comfy_server_running", return_value=False),
            patch("comfy_cli.command.launch.launch_and_monitor", return_value=["boom\n"]),
            patch("comfy_cli.command.launch.os._exit", side_effect=SystemExit(1)),
        ):
            with pytest.raises(SystemExit):
                launch_mod.background_launch(extra=[])

        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["ok"] is False
        assert env["command"] == "launch"
        assert env["error"]["code"] == "launch_failed"
        assert env["error"]["details"]["log"] == "boom\n"

    def test_format_url_brackets_ipv6(self):
        assert launch_mod._format_url("::1", 8188) == "http://[::1]:8188"
        assert launch_mod._format_url("::", 8188) == "http://[::]:8188"
        # Already-bracketed and plain IPv4/hostnames are left as-is.
        assert launch_mod._format_url("[::1]", 8188) == "http://[::1]:8188"
        assert launch_mod._format_url("127.0.0.1", 8188) == "http://127.0.0.1:8188"

    def test_background_success_helper_ipv6_url(self, capsys):
        _json_renderer("launch")
        launch_mod._emit_launch_success("::1", 8188, 9999)
        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["data"]["url"] == "http://[::1]:8188"

    def test_background_port_eq_form_parsed(self, capsys):
        # The ``--port=9000`` form must be honored (was previously ignored).
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = None
        with (
            patch("comfy_cli.command.launch.ConfigManager", return_value=cfg),
            patch("comfy_cli.command.launch.check_comfy_server_running", return_value=True),
        ):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.background_launch(extra=["--port=9000"])
        assert exc.value.exit_code == 1
        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["error"]["code"] == "port_in_use"
        assert env["error"]["details"]["port"] == 9000

    def test_background_out_of_range_port_is_invalid(self, capsys):
        _json_renderer("launch")
        cfg = MagicMock()
        cfg.background = None
        with patch("comfy_cli.command.launch.ConfigManager", return_value=cfg):
            with pytest.raises(typer.Exit) as exc:
                launch_mod.background_launch(extra=["--port", "70000"])
        assert exc.value.exit_code == 1
        env = _envelopes(capsys.readouterr().out)[-1]
        assert env["error"]["code"] == "port_invalid"
        assert env["error"]["details"]["port"] == "70000"

    def test_bounded_log_caps_bytes(self):
        # A verbose failing launch must not embed an unbounded log payload.
        huge = [f"line {i}\n" for i in range(100)]
        out = launch_mod._bounded_log(huge, max_lines=1000, max_bytes=20)
        assert len(out.encode("utf-8")) <= 20
        # Keeps the tail (most recent lines), trimming from the top.
        assert out.endswith("line 99\n")

    def test_bounded_log_caps_lines(self):
        huge = [f"line {i}\n" for i in range(100)]
        out = launch_mod._bounded_log(huge, max_lines=3, max_bytes=1_000_000)
        assert out.splitlines() == ["line 97", "line 98", "line 99"]
