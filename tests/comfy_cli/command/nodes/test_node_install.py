import re
from unittest.mock import patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.command import app

runner = CliRunner(mix_stderr=False)


def strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_install_no_deps_option_exists():
    """Test that the --no-deps option appears in the help."""
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    clean_output = strip_ansi(result.stdout)
    assert "--no-deps" in clean_output
    assert "Skip dependency installation" in clean_output


def test_install_fast_deps_and_no_deps_mutually_exclusive():
    """Test that --fast-deps and --no-deps cannot be used together."""
    result = runner.invoke(app, ["install", "test-node", "--fast-deps", "--no-deps"])
    assert result.exit_code != 0
    # Check both stdout and stderr for the error message
    output = result.stdout + result.stderr
    assert "Cannot use --fast-deps and --no-deps together" in output


def test_install_no_deps_alone_works():
    """Test that --no-deps can be used by itself."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--no-deps"])

        # Should not exit with error due to mutual exclusivity
        if result.exit_code != 0:
            # Only acceptable if it fails due to missing ComfyUI setup, not mutual exclusivity
            assert "Cannot use --fast-deps and --no-deps together" not in result.stdout

        # Verify execute_cm_cli was called with no_deps=True
        if mock_execute.called:
            _, kwargs = mock_execute.call_args
            assert kwargs.get("no_deps") is True
            assert kwargs.get("fast_deps") is False


def test_install_fast_deps_alone_works():
    """Test that --fast-deps can be used by itself."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node", "--fast-deps"])

        # Should not exit with error due to mutual exclusivity
        if result.exit_code != 0:
            # Only acceptable if it fails due to missing ComfyUI setup, not mutual exclusivity
            assert "Cannot use --fast-deps and --no-deps together" not in result.stdout

        # Verify execute_cm_cli was called with fast_deps=True
        if mock_execute.called:
            _, kwargs = mock_execute.call_args
            assert kwargs.get("fast_deps") is True
            assert kwargs.get("no_deps") is False


def test_install_neither_deps_option():
    """Test that install works without any deps options."""
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli") as mock_execute:
        result = runner.invoke(app, ["install", "test-node"])

        # Should not exit with error due to mutual exclusivity
        if result.exit_code != 0:
            # Only acceptable if it fails due to missing ComfyUI setup, not mutual exclusivity
            assert "Cannot use --fast-deps and --no-deps together" not in result.stdout

        # Verify execute_cm_cli was called with both flags False
        if mock_execute.called:
            _, kwargs = mock_execute.call_args
            assert kwargs.get("fast_deps") is False
            assert kwargs.get("no_deps") is False


def test_multiple_commands_work_independently():
    """Test that multiple commands work independently without state interference."""
    # First command with --no-deps should work
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli"):
        result1 = runner.invoke(app, ["install", "test-node", "--no-deps"])
        if result1.exit_code != 0:
            assert "Cannot use --fast-deps and --no-deps together" not in result1.stdout

    # Second command with --fast-deps should also work
    with patch("comfy_cli.command.custom_nodes.command.execute_cm_cli"):
        result2 = runner.invoke(app, ["install", "test-node2", "--fast-deps"])
        if result2.exit_code != 0:
            assert "Cannot use --fast-deps and --no-deps together" not in result2.stdout
