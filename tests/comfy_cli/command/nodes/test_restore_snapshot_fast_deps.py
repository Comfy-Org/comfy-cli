"""Tests for `comfy node restore-snapshot --fast-deps` (issue #217, BE-4276).

cm_cli's `restore-snapshot` accepts `--uv-compile` but NOT `--no-deps`, so the
DependencyCompiler-based fast path used by `install`/`reinstall` is unavailable
there. `--fast-deps` therefore forwards the uv-compile fast path (which cm_cli's
restore-snapshot does support). These tests pin that wiring: the flag forwards
`uv_compile=True`, default behavior is unchanged when omitted, and the flag is
never silently ignored.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes import command as cn_command

runner = CliRunner()


def _invoke(args):
    """Invoke restore-snapshot with execute_cm_cli mocked; return (result, call)."""
    with patch.object(cn_command, "execute_cm_cli") as mock_exec:
        result = runner.invoke(cn_command.app, ["restore-snapshot", *args])
    return result, mock_exec


def test_fast_deps_forwards_uv_compile(tmp_path):
    """`--fast-deps` forwards the uv-compile fast path to cm_cli."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}")
    result, mock_exec = _invoke([str(snap), "--fast-deps"])
    assert result.exit_code == 0
    mock_exec.assert_called_once()
    assert mock_exec.call_args.kwargs["uv_compile"] is True


def test_default_omitted_does_not_force_uv_compile(tmp_path):
    """Default (no flag) is unchanged: uv_compile stays False."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}")
    result, mock_exec = _invoke([str(snap)])
    assert result.exit_code == 0
    mock_exec.assert_called_once()
    assert mock_exec.call_args.kwargs["uv_compile"] is False


def test_explicit_uv_compile_still_works(tmp_path):
    """The pre-existing `--uv-compile` flag is unaffected."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}")
    result, mock_exec = _invoke([str(snap), "--uv-compile"])
    assert result.exit_code == 0
    assert mock_exec.call_args.kwargs["uv_compile"] is True


def test_fast_deps_with_no_uv_compile_is_rejected(tmp_path):
    """`--fast-deps --no-uv-compile` is contradictory and errors clearly (not ignored)."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}")
    result, mock_exec = _invoke([str(snap), "--fast-deps", "--no-uv-compile"])
    assert result.exit_code != 0
    mock_exec.assert_not_called()


def test_fast_deps_forwards_pip_extras(tmp_path):
    """`--fast-deps` composes with the pip-restore extras (they still pass through)."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}")
    result, mock_exec = _invoke([str(snap), "--fast-deps", "--pip-non-url"])
    assert result.exit_code == 0
    args = mock_exec.call_args.args[0]
    assert args[0] == "restore-snapshot"
    assert "--pip-non-url" in args
    assert mock_exec.call_args.kwargs["uv_compile"] is True
