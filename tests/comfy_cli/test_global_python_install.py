"""Integration tests for the global-Python (Docker / bare-metal) install path.

Covers the scenario where comfy-cli is installed via ``pip install`` or
``uv pip install`` into the system Python (no virtualenv).  In this case
``sys.prefix == sys.base_prefix``, and comfy-cli must install ComfyUI
dependencies into the same global environment instead of creating a
workspace ``.venv``.

See https://github.com/Comfy-Org/comfy-cli/issues/393
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli.command import install
from comfy_cli.resolve_python import ensure_workspace_python


def _clean_env():
    """Context manager that removes VIRTUAL_ENV / CONDA_PREFIX for the block."""
    keys = ("VIRTUAL_ENV", "CONDA_PREFIX")
    saved = {k: os.environ.pop(k, None) for k in keys}

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    return _Ctx()


class TestGlobalPythonDetection:
    """ensure_workspace_python must return sys.executable when running from the
    system Python and must NOT create a .venv."""

    def test_global_python_skips_venv_creation(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with (
            _clean_env(),
            patch("comfy_cli.resolve_python.sys") as mock_sys,
            patch("comfy_cli.resolve_python._is_externally_managed", return_value=False),
        ):
            mock_sys.executable = "/usr/bin/python3"
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            result = ensure_workspace_python(str(workspace))

        assert result == "/usr/bin/python3"
        assert not (workspace / ".venv").exists()
        assert not (workspace / "venv").exists()

    def test_isolated_env_creates_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with _clean_env(), patch("comfy_cli.resolve_python.sys") as mock_sys:
            mock_sys.executable = sys.executable
            mock_sys.prefix = "/home/user/.local/pipx/venvs/comfy-cli"
            mock_sys.base_prefix = "/usr"
            result = ensure_workspace_python(str(workspace))

        assert (workspace / ".venv").is_dir()
        assert ".venv" in result


class TestGlobalPythonInstallExecute:
    """install.execute with --fast-deps must pass through the global Python to
    DependencyCompiler (no .venv indirection)."""

    def _run_execute(self, tmp_path, *, fast_deps, python="/usr/bin/python3"):
        repo_dir = str(tmp_path)

        with (
            patch("comfy_cli.command.install.ensure_workspace_python", return_value=python) as mock_ensure,
            patch("comfy_cli.command.install.clone_comfyui"),
            patch("comfy_cli.command.install.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.install.pip_install_comfyui_dependencies") as mock_pip,
            patch("comfy_cli.command.install.DependencyCompiler") as MockCompiler,
            patch("comfy_cli.command.install.WorkspaceManager"),
            patch.object(install.workspace_manager, "skip_prompting", True),
            patch.object(install.workspace_manager, "setup_workspace_manager"),
        ):
            MockCompiler.Install_Build_Deps = MagicMock()
            MockCompiler.return_value = MagicMock()

            install.execute(
                url="https://github.com/comfyanonymous/ComfyUI.git",
                comfy_path=repo_dir,
                restore=False,
                skip_manager=True,
                version="nightly",
                fast_deps=fast_deps,
            )

        return mock_ensure, mock_pip, MockCompiler

    def test_fast_deps_uses_global_python(self, tmp_path):
        mock_ensure, mock_pip, MockCompiler = self._run_execute(tmp_path, fast_deps=True)

        mock_ensure.assert_called_once_with(str(tmp_path))
        mock_pip.assert_not_called()
        MockCompiler.Install_Build_Deps.assert_called_once_with(executable="/usr/bin/python3")
        assert MockCompiler.call_args[1]["executable"] == "/usr/bin/python3"

    def test_non_fast_deps_uses_global_python(self, tmp_path):
        mock_ensure, mock_pip, MockCompiler = self._run_execute(tmp_path, fast_deps=False)

        mock_ensure.assert_called_once_with(str(tmp_path))
        mock_pip.assert_called_once()
        assert mock_pip.call_args[1]["python"] == "/usr/bin/python3"
        MockCompiler.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("TEST_TORCH_BACKEND") != "true",
    reason="Set TEST_TORCH_BACKEND=true to run integration tests that call uv pip compile",
)
class TestDependencyCompilerGlobalPython:
    """Integration tests: run the real DependencyCompiler compile step (no
    mocks, requires network) using the current Python as if it were a global
    install.  Verifies the compiled output contains expected packages and
    correct index URLs."""

    @pytest.fixture()
    def workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "custom_nodes").mkdir()
        (ws / "requirements.txt").write_text("pyyaml\nrequests\n")
        return ws

    def test_compile_produces_complete_output(self, workspace):
        from comfy_cli.uv import DependencyCompiler

        dep = DependencyCompiler(
            cwd=str(workspace),
            executable=sys.executable,
            gpu=None,
            outDir=str(workspace),
        )

        dep.compile_deps()

        compiled = Path(dep.out).read_text()
        pkg_lines = [
            ln.split("==")[0].strip().lower()
            for ln in compiled.splitlines()
            if "==" in ln and not ln.strip().startswith("#")
        ]
        assert "pyyaml" in pkg_lines
        assert "requests" in pkg_lines
        assert "--index-url" in compiled

    def test_compile_nvidia_resolves_torch(self, workspace):
        (workspace / "requirements.txt").write_text("torch\npyyaml\n")

        from comfy_cli.constants import GPU_OPTION
        from comfy_cli.uv import DependencyCompiler

        dep = DependencyCompiler(
            cwd=str(workspace),
            executable=sys.executable,
            gpu=GPU_OPTION.NVIDIA,
            cuda_version="12.6",
            outDir=str(workspace),
        )

        dep.compile_deps()

        compiled = Path(dep.out).read_text().lower()
        assert "torch==" in compiled
        assert "pyyaml==" in compiled
        assert "https://pypi.org/simple" in compiled

    def test_install_targets_correct_python(self, workspace):
        from comfy_cli.uv import DependencyCompiler

        dep = DependencyCompiler(
            cwd=str(workspace),
            executable=sys.executable,
            gpu=None,
            outDir=str(workspace),
        )
        dep.compile_deps()

        with patch("comfy_cli.uv._check_call") as mock_call:
            dep.install_deps()

        cmd = mock_call.call_args[1].get("cmd") or mock_call.call_args[0][0]
        assert cmd[0] == str(Path(sys.executable).expanduser().absolute())
        assert "--requirement" in cmd
