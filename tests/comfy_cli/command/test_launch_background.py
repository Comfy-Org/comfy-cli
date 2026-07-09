"""Regression tests for `comfy launch --background`.

Guards against the Python 3.14 crash where `asyncio.get_event_loop()` raised
`RuntimeError: There is no current event loop in thread 'MainThread'` because
implicit main-thread loop creation was removed. `background_launch` now uses
`asyncio.run(...)`, which works identically on 3.10-3.14+.
"""

from unittest.mock import AsyncMock, patch

import pytest

from comfy_cli.command import launch


@patch("comfy_cli.command.launch.os._exit")
@patch("comfy_cli.command.launch.launch_and_monitor", new_callable=AsyncMock)
@patch("comfy_cli.command.launch.check_comfy_server_running", return_value=False)
@patch("comfy_cli.command.launch.ConfigManager")
def test_background_launch_drives_coroutine_without_event_loop_error(
    mock_config_manager, mock_check_running, mock_monitor, mock_exit
):
    """The background monitor path must run the coroutine without touching
    the removed `asyncio.get_event_loop()` implicit-loop behavior."""
    mock_config_manager.return_value.background = None
    mock_monitor.return_value = None

    # No RuntimeError should escape here on Python 3.14 (or any 3.10+ version),
    # and the monitor coroutine must actually be awaited.
    launch.background_launch(extra=[])

    mock_monitor.assert_awaited_once()
    mock_exit.assert_called_once_with(1)


def test_background_launch_does_not_use_deprecated_get_event_loop():
    """`asyncio.run` is the forward-compatible idiom; `asyncio.get_event_loop`
    (removed-behavior on 3.14) must not reappear in the launch module."""
    import inspect

    source = inspect.getsource(launch)
    assert "asyncio.get_event_loop()" not in source
    assert "asyncio.run(" in source


@patch("comfy_cli.command.launch.os._exit")
@patch("comfy_cli.command.launch.launch_and_monitor", new_callable=AsyncMock)
@patch("comfy_cli.command.launch.check_comfy_server_running", return_value=False)
@patch("comfy_cli.command.launch.ConfigManager")
def test_background_launch_surfaces_error_log(mock_config_manager, mock_check_running, mock_monitor, mock_exit):
    """When the monitor returns a failure log, it is rendered and the process
    exits non-zero."""
    mock_config_manager.return_value.background = None
    mock_monitor.return_value = ["boom\n"]

    with patch("comfy_cli.command.launch.print") as mock_print:
        launch.background_launch(extra=[])

    mock_monitor.assert_awaited_once()
    assert mock_print.called
    mock_exit.assert_called_once_with(1)


@patch("comfy_cli.command.launch.launch_and_monitor", new_callable=AsyncMock)
@patch("comfy_cli.command.launch.check_comfy_server_running", return_value=False)
@patch("comfy_cli.command.launch.ConfigManager")
def test_background_launch_rejects_non_integer_port(mock_config_manager, mock_check_running, mock_monitor):
    """A non-integer --port is rejected before it can be interpolated into the
    log path (`comfyui_<port>.log`), where a value like `../../x` would escape
    the workspace."""
    import typer

    mock_config_manager.return_value.background = None

    with pytest.raises(typer.Exit):
        launch.background_launch(extra=["--port", "../../etc/pwn"])

    # Never reached the launch path.
    mock_monitor.assert_not_awaited()


@patch("comfy_cli.command.launch.utils.is_running", return_value=True)
@patch("comfy_cli.command.launch.ConfigManager")
def test_background_launch_refuses_when_already_running(mock_config_manager, mock_is_running):
    """A second background launch is rejected before reaching the asyncio path."""
    import typer

    mock_config_manager.return_value.background = ("127.0.0.1", 8188, 4242)

    with pytest.raises(typer.Exit):
        launch.background_launch(extra=[])
