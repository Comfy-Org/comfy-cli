import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.command.custom_nodes import cm_cli_util


def _setup_cm_cli(tmp_path, script_body):
    """Create a stub cm-cli.py with the given body and patch workspace to tmp_path."""
    cm_cli_path = tmp_path / "custom_nodes" / "ComfyUI-Manager" / "cm-cli.py"
    cm_cli_path.parent.mkdir(parents=True)
    cm_cli_path.write_text(textwrap.dedent(script_body))
    (tmp_path / "config").mkdir(exist_ok=True)
    return tmp_path


def _run(tmp_path, args, *, fast_deps=False, raise_on_error=False):
    """Call execute_cm_cli with standard patches for workspace/config."""
    with (
        patch(
            "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
            return_value=sys.executable,
        ),
        patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
        patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
        patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as MockConfig,
    ):
        MockConfig.return_value.get_config_path.return_value = str(tmp_path / "config")
        return cm_cli_util.execute_cm_cli(args, fast_deps=fast_deps, raise_on_error=raise_on_error)


class TestExecuteCmCli:
    def test_uses_resolved_python(self, tmp_path):
        _setup_cm_cli(tmp_path, 'print("ok")')
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["ok\n"])
        mock_proc.wait.return_value = 0

        with (
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/resolved/python",
            ) as mock_resolve,
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as MockConfig,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            MockConfig.return_value.get_config_path.return_value = str(tmp_path / "config")
            cm_cli_util.execute_cm_cli(["show", "installed"])

        mock_resolve.assert_called_once_with(str(tmp_path))
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "/resolved/python"

    def test_fast_deps_passes_python_to_compiler(self, tmp_path):
        _setup_cm_cli(tmp_path, 'print("ok")')
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["ok\n"])
        mock_proc.wait.return_value = 0

        with (
            patch(
                "comfy_cli.command.custom_nodes.cm_cli_util.resolve_workspace_python",
                return_value="/resolved/python",
            ),
            patch.object(cm_cli_util.workspace_manager, "workspace_path", str(tmp_path)),
            patch.object(cm_cli_util.workspace_manager, "set_recent_workspace"),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.ConfigManager") as MockConfig,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.subprocess.Popen", return_value=mock_proc),
            patch("comfy_cli.command.custom_nodes.cm_cli_util.DependencyCompiler") as MockCompiler,
        ):
            MockConfig.return_value.get_config_path.return_value = str(tmp_path / "config")
            mock_instance = MagicMock()
            MockCompiler.return_value = mock_instance

            cm_cli_util.execute_cm_cli(["install", "some-node"], fast_deps=True)

        MockCompiler.assert_called_once()
        assert MockCompiler.call_args[1]["executable"] == "/resolved/python"

    def test_stdout_returned_and_streamed(self, tmp_path, capsys):
        _setup_cm_cli(
            tmp_path,
            """\
            print("line 1")
            print("line 2")
            print("line 3")
        """,
        )
        result = _run(tmp_path, ["test"])

        assert result == "line 1\nline 2\nline 3\n"
        captured = capsys.readouterr()
        assert "line 1\nline 2\nline 3\n" in captured.out

    @pytest.mark.parametrize("returncode", [1, 2])
    def test_expected_error_codes_return_none(self, tmp_path, returncode):
        _setup_cm_cli(
            tmp_path,
            f"""\
            import sys
            sys.exit({returncode})
        """,
        )
        result = _run(tmp_path, ["test"])
        assert result is None

    def test_unexpected_error_code_raises(self, tmp_path):
        _setup_cm_cli(
            tmp_path,
            """\
            import sys
            sys.exit(42)
        """,
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run(tmp_path, ["test"])
        assert exc_info.value.returncode == 42

    def test_raise_on_error_overrides_silent_return(self, tmp_path):
        _setup_cm_cli(
            tmp_path,
            """\
            import sys
            print("output before fail")
            sys.exit(1)
        """,
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run(tmp_path, ["test"], raise_on_error=True)
        assert exc_info.value.returncode == 1
        assert "output before fail" in exc_info.value.output
