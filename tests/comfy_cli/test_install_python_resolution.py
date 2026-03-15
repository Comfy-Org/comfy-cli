import sys
from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.command import install
from comfy_cli.constants import GPU_OPTION


class TestPipInstallComfyuiDependencies:
    def test_uses_python_param_cpu(self, tmp_path):
        repo_dir = str(tmp_path)
        (tmp_path / "requirements.txt").write_text("some-package\n")

        with patch("comfy_cli.command.install.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            install.pip_install_comfyui_dependencies(
                repo_dir,
                gpu=None,
                plat=None,
                cuda_version=None,
                skip_torch_or_directml=False,
                skip_requirement=False,
                python="/resolved/python",
            )

        for c in mock_run.call_args_list:
            cmd = c[0][0]
            assert cmd[0] == "/resolved/python", f"Expected /resolved/python but got {cmd[0]} in {cmd}"
            assert cmd[0] != sys.executable


class TestPipInstallManager:
    def test_uses_python_param(self, tmp_path):
        (tmp_path / "manager_requirements.txt").write_text("comfyui-manager\n")

        with (
            patch("comfy_cli.command.install.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
            patch("comfy_cli.command.custom_nodes.cm_cli_util.find_cm_cli") as mock_find,
        ):
            mock_find.cache_clear = MagicMock()
            install.pip_install_manager(str(tmp_path), python="/resolved/python")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/resolved/python"


class TestExecute:
    def test_calls_ensure_and_passes_resolved_python(self, tmp_path):
        repo_dir = str(tmp_path)

        with (
            patch("comfy_cli.command.install.ensure_workspace_python", return_value="/resolved/python") as mock_ensure,
            patch("comfy_cli.command.install.clone_comfyui"),
            patch("comfy_cli.command.install.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.install.pip_install_comfyui_dependencies") as mock_pip_deps,
            patch("comfy_cli.command.install.WorkspaceManager"),
            patch.object(install.workspace_manager, "skip_prompting", True),
            patch.object(install.workspace_manager, "setup_workspace_manager"),
        ):
            install.execute(
                url="https://github.com/test/test.git",
                comfy_path=repo_dir,
                restore=False,
                skip_manager=True,
                version="nightly",
            )

        mock_ensure.assert_called_once_with(repo_dir)
        mock_pip_deps.assert_called_once()
        assert mock_pip_deps.call_args[1]["python"] == "/resolved/python"

    def test_fast_deps_passes_python_to_dependency_compiler(self, tmp_path):
        repo_dir = str(tmp_path)

        with (
            patch("comfy_cli.command.install.ensure_workspace_python", return_value="/resolved/python"),
            patch("comfy_cli.command.install.clone_comfyui"),
            patch("comfy_cli.command.install.check_comfy_repo", return_value=(True, None)),
            patch("comfy_cli.command.install.DependencyCompiler") as MockCompiler,
            patch("comfy_cli.command.install.WorkspaceManager"),
            patch.object(install.workspace_manager, "skip_prompting", True),
            patch.object(install.workspace_manager, "setup_workspace_manager"),
        ):
            MockCompiler.Install_Build_Deps = MagicMock()
            mock_instance = MagicMock()
            MockCompiler.return_value = mock_instance

            install.execute(
                url="https://github.com/test/test.git",
                comfy_path=repo_dir,
                restore=False,
                skip_manager=True,
                version="nightly",
                fast_deps=True,
            )

        MockCompiler.Install_Build_Deps.assert_called_once_with(executable="/resolved/python")
        MockCompiler.assert_called_once()
        assert MockCompiler.call_args[1]["executable"] == "/resolved/python"


def _get_torch_install_cmd(calls):
    """Find the subprocess.run call that installs torch packages."""
    for c in calls:
        cmd = c[0][0]
        if "torch" in cmd and "requirements.txt" not in cmd:
            return cmd
    return None


class TestTorchInstallCommands:
    @pytest.mark.parametrize(
        "rocm_version,expected_url",
        [
            (constants.ROCmVersion.v7_1, "https://download.pytorch.org/whl/rocm7.1"),
            (constants.ROCmVersion.v7_0, "https://download.pytorch.org/whl/rocm7.0"),
            (constants.ROCmVersion.v6_3, "https://download.pytorch.org/whl/rocm6.3"),
            (constants.ROCmVersion.v6_2, "https://download.pytorch.org/whl/rocm6.2"),
            (constants.ROCmVersion.v6_1, "https://download.pytorch.org/whl/rocm6.1"),
        ],
    )
    def test_amd_uses_index_url_with_rocm_version(self, tmp_path, rocm_version, expected_url):
        repo_dir = str(tmp_path)
        (tmp_path / "requirements.txt").write_text("some-package\n")

        with patch("comfy_cli.command.install.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            install.pip_install_comfyui_dependencies(
                repo_dir,
                gpu=GPU_OPTION.AMD,
                plat=constants.OS.LINUX,
                cuda_version=constants.CUDAVersion.v12_6,
                skip_torch_or_directml=False,
                skip_requirement=False,
                python="/usr/bin/python",
                rocm_version=rocm_version,
            )

        cmd = _get_torch_install_cmd(mock_run.call_args_list)
        assert "--index-url" in cmd
        assert "--extra-index-url" not in cmd
        assert expected_url in cmd

    @pytest.mark.parametrize(
        "cuda_version,expected_url",
        [
            (constants.CUDAVersion.v12_9, "https://download.pytorch.org/whl/cu129"),
            (constants.CUDAVersion.v12_6, "https://download.pytorch.org/whl/cu126"),
            (constants.CUDAVersion.v12_4, "https://download.pytorch.org/whl/cu124"),
            (constants.CUDAVersion.v12_1, "https://download.pytorch.org/whl/cu121"),
            (constants.CUDAVersion.v11_8, "https://download.pytorch.org/whl/cu118"),
        ],
    )
    def test_nvidia_uses_index_url_with_cuda_version(self, tmp_path, cuda_version, expected_url):
        repo_dir = str(tmp_path)
        (tmp_path / "requirements.txt").write_text("some-package\n")

        with patch("comfy_cli.command.install.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            install.pip_install_comfyui_dependencies(
                repo_dir,
                gpu=GPU_OPTION.NVIDIA,
                plat=constants.OS.WINDOWS,
                cuda_version=cuda_version,
                skip_torch_or_directml=False,
                skip_requirement=False,
                python="/usr/bin/python",
            )

        cmd = _get_torch_install_cmd(mock_run.call_args_list)
        assert "--index-url" in cmd
        assert "--extra-index-url" not in cmd
        assert expected_url in cmd

    def test_nvidia_linux_uses_index_url(self, tmp_path):
        repo_dir = str(tmp_path)
        (tmp_path / "requirements.txt").write_text("some-package\n")

        with patch("comfy_cli.command.install.subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            install.pip_install_comfyui_dependencies(
                repo_dir,
                gpu=GPU_OPTION.NVIDIA,
                plat=constants.OS.LINUX,
                cuda_version=constants.CUDAVersion.v12_6,
                skip_torch_or_directml=False,
                skip_requirement=False,
                python="/usr/bin/python",
            )

        cmd = _get_torch_install_cmd(mock_run.call_args_list)
        assert "--index-url" in cmd
        assert "https://download.pytorch.org/whl/cu126" in cmd
