"""Tests for install command functionality"""

from io import StringIO
from unittest.mock import patch

import pytest

from comfy_cli.command.install import _print_npm_not_found_help


class TestPrintNpmNotFoundHelp:
    """Tests for _print_npm_not_found_help function"""

    @pytest.fixture
    def capture_output(self):
        """Fixture to capture rich console output"""
        output = StringIO()
        with patch(
            "comfy_cli.command.install.rprint",
            side_effect=lambda *args: output.write(str(args[0]) + "\n" if args else "\n"),
        ):
            yield output

    def test_npm_not_found_help_shows_common_message(self, capture_output):
        """Test that common npm not found message is shown regardless of OS"""
        with patch("platform.system", return_value="Linux"):
            _print_npm_not_found_help("v20.0.0")

        output_text = capture_output.getvalue()
        assert "npm is not installed or not found in PATH" in output_text
        assert "npm is a package manager that usually comes bundled with Node.js" in output_text
        assert "v20.0.0" in output_text
        assert "After fixing npm, run your comfy command again" in output_text

    def test_npm_not_found_help_windows(self, capture_output):
        """Test Windows-specific instructions"""
        with patch("platform.system", return_value="Windows"):
            _print_npm_not_found_help("v18.17.0")

        output_text = capture_output.getvalue()
        assert "How to fix this on Windows" in output_text
        assert "Add or remove programs" in output_text
        assert "Command Prompt or PowerShell" in output_text

    def test_npm_not_found_help_macos(self, capture_output):
        """Test macOS-specific instructions"""
        with patch("platform.system", return_value="Darwin"):
            _print_npm_not_found_help("v18.17.0")

        output_text = capture_output.getvalue()
        assert "How to fix this on macOS" in output_text
        assert "Homebrew" in output_text
        assert "brew install node" in output_text
        assert ".pkg file" in output_text
        assert "Cmd+Q" in output_text

    def test_npm_not_found_help_linux(self, capture_output):
        """Test Linux-specific instructions"""
        with patch("platform.system", return_value="Linux"):
            _print_npm_not_found_help("v18.17.0")

        output_text = capture_output.getvalue()
        assert "How to fix this on Linux" in output_text
        assert "sudo apt" in output_text
        assert "Ubuntu/Debian" in output_text
        assert "Fedora" in output_text
        assert "NodeSource" in output_text

    def test_npm_not_found_help_unknown_os_falls_back_to_linux(self, capture_output):
        """Test that unknown OS falls back to Linux instructions"""
        with patch("platform.system", return_value="FreeBSD"):
            _print_npm_not_found_help("v18.17.0")

        output_text = capture_output.getvalue()
        # Should show Linux instructions as fallback
        assert "How to fix this on Linux" in output_text
