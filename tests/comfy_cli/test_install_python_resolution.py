import sys
from unittest.mock import MagicMock, patch

from comfy_cli.command import install


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


class TestPipInstallManagerDependencies:
    def test_uses_python_param(self, tmp_path):
        manager_dir = tmp_path / "custom_nodes" / "ComfyUI-Manager"
        manager_dir.mkdir(parents=True)
        (manager_dir / "requirements.txt").write_text("some-package\n")

        with patch("comfy_cli.command.install.subprocess.run") as mock_run:
            install.pip_install_manager_dependencies(str(tmp_path), python="/resolved/python")

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
                manager_url="https://github.com/test/manager.git",
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
                manager_url="https://github.com/test/manager.git",
                comfy_path=repo_dir,
                restore=False,
                skip_manager=True,
                version="nightly",
                fast_deps=True,
            )

        MockCompiler.Install_Build_Deps.assert_called_once_with(executable="/resolved/python")
        MockCompiler.assert_called_once()
        assert MockCompiler.call_args[1]["executable"] == "/resolved/python"
