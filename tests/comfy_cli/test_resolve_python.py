import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from comfy_cli.resolve_python import (
    _get_python_binary,
    _is_externally_managed,
    create_workspace_venv,
    ensure_workspace_python,
    resolve_workspace_python,
)


def _clean_env(**overrides):
    """Return a patch.dict that clears VIRTUAL_ENV and CONDA_PREFIX, then applies overrides."""
    removals = {k: overrides.pop(k, None) for k in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env = {k: v for k, v in overrides.items()}
    env.update({k: v for k, v in removals.items() if v is not None})
    keys_to_remove = [k for k, v in removals.items() if v is None and k in os.environ]

    class _Ctx:
        def __enter__(self_ctx):
            self_ctx._old = {k: os.environ.get(k) for k in keys_to_remove}
            for k in keys_to_remove:
                os.environ.pop(k, None)
            self_ctx._patcher = patch.dict(os.environ, env, clear=False)
            self_ctx._patcher.__enter__()
            return self_ctx

        def __exit__(self_ctx, *args):
            self_ctx._patcher.__exit__(*args)
            for k, v in self_ctx._old.items():
                if v is not None:
                    os.environ[k] = v

    return _Ctx()


def _make_fake_python(base_dir, name="bin/python"):
    """Create a fake python binary (empty file) and return its path."""
    p = base_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _make_real_venv(base_dir):
    """Create a real venv and return the python path inside it."""
    subprocess.run([sys.executable, "-m", "venv", str(base_dir)], check=True)
    python = os.path.join(str(base_dir), "bin", "python")
    assert os.path.isfile(python)
    return python


class TestIsExternallyManaged:
    def test_true_when_marker_exists(self, tmp_path):
        marker = tmp_path / "EXTERNALLY-MANAGED"
        marker.touch()
        with patch("comfy_cli.resolve_python.sysconfig.get_path", return_value=str(tmp_path)):
            assert _is_externally_managed() is True

    def test_false_when_no_marker(self, tmp_path):
        with patch("comfy_cli.resolve_python.sysconfig.get_path", return_value=str(tmp_path)):
            assert _is_externally_managed() is False

    def test_false_when_stdlib_is_none(self):
        with patch("comfy_cli.resolve_python.sysconfig.get_path", return_value=None):
            assert _is_externally_managed() is False


class TestGetPythonBinary:
    @patch("comfy_cli.resolve_python.platform.system", return_value="Linux")
    def test_unix(self, _mock):
        assert _get_python_binary("/some/env") == "/some/env/bin/python"

    @patch("comfy_cli.resolve_python.platform.system", return_value="Darwin")
    def test_macos(self, _mock):
        assert _get_python_binary("/some/env") == "/some/env/bin/python"

    @patch("comfy_cli.resolve_python.platform.system", return_value="Windows")
    def test_windows(self, _mock):
        result = _get_python_binary("/some/env")
        assert result.endswith(os.path.join("Scripts", "python.exe"))


class TestResolveWorkspacePython:
    def test_virtual_env_takes_precedence_over_workspace_venv(self, tmp_path):
        venv_dir = tmp_path / "user_venv"
        python = _make_fake_python(venv_dir)
        workspace = tmp_path / "workspace"
        _make_fake_python(workspace / ".venv")

        with _clean_env(VIRTUAL_ENV=str(venv_dir)):
            result = resolve_workspace_python(str(workspace))
        assert result == str(python)

    def test_virtual_env_takes_precedence_over_conda(self, tmp_path):
        venv_dir = tmp_path / "user_venv"
        venv_python = _make_fake_python(venv_dir)
        conda_dir = tmp_path / "conda_env"
        _make_fake_python(conda_dir)

        with _clean_env(VIRTUAL_ENV=str(venv_dir), CONDA_PREFIX=str(conda_dir)):
            result = resolve_workspace_python(None)
        assert result == str(venv_python)

    def test_conda_prefix_when_no_virtual_env(self, tmp_path):
        conda_dir = tmp_path / "conda"
        python = _make_fake_python(conda_dir)

        with _clean_env(CONDA_PREFIX=str(conda_dir)):
            result = resolve_workspace_python(str(tmp_path / "workspace"))
        assert result == str(python)

    def test_conda_prefix_python_missing_falls_through(self, tmp_path):
        conda_dir = tmp_path / "broken_conda"
        conda_dir.mkdir()

        with _clean_env(CONDA_PREFIX=str(conda_dir)):
            result = resolve_workspace_python(None)
        assert result == sys.executable

    def test_virtual_env_missing_falls_to_conda(self, tmp_path):
        broken_venv = tmp_path / "broken_venv"
        broken_venv.mkdir()
        conda_dir = tmp_path / "conda"
        conda_python = _make_fake_python(conda_dir)

        with _clean_env(VIRTUAL_ENV=str(broken_venv), CONDA_PREFIX=str(conda_dir)):
            result = resolve_workspace_python(None)
        assert result == str(conda_python)

    def test_workspace_dot_venv_found(self, tmp_path):
        workspace = tmp_path / "workspace"
        python = _make_fake_python(workspace / ".venv")

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == str(python)

    def test_workspace_venv_found(self, tmp_path):
        workspace = tmp_path / "workspace"
        python = _make_fake_python(workspace / "venv")

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == str(python)

    def test_dot_venv_preferred_over_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        dot_python = _make_fake_python(workspace / ".venv")
        _make_fake_python(workspace / "venv")

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == str(dot_python)

    def test_workspace_dot_venv_dir_exists_but_python_missing(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == sys.executable

    def test_workspace_dot_venv_broken_falls_to_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)
        venv_python = _make_fake_python(workspace / "venv")

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == str(venv_python)

    def test_workspace_venv_dir_exists_but_python_missing(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / "venv").mkdir(parents=True)

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == sys.executable

    def test_fallback_to_sys_executable(self, tmp_path):
        with _clean_env():
            result = resolve_workspace_python(str(tmp_path))
        assert result == sys.executable

    def test_none_workspace_path(self):
        with _clean_env():
            result = resolve_workspace_python(None)
        assert result == sys.executable

    def test_virtual_env_python_missing_falls_through(self, tmp_path):
        venv_dir = tmp_path / "broken_venv"
        venv_dir.mkdir()

        with _clean_env(VIRTUAL_ENV=str(venv_dir)):
            result = resolve_workspace_python(str(tmp_path))
        assert result == sys.executable

    def test_with_real_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        expected = _make_real_venv(workspace / ".venv")

        with _clean_env():
            result = resolve_workspace_python(str(workspace))
        assert result == expected
        r = subprocess.run([result, "-c", "print('ok')"], capture_output=True, text=True)
        assert r.returncode == 0
        assert r.stdout.strip() == "ok"


class TestEnsureWorkspacePython:
    def test_with_virtual_env_does_not_create_venv(self, tmp_path):
        venv_dir = tmp_path / "ext_venv"
        python = _make_fake_python(venv_dir)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with _clean_env(VIRTUAL_ENV=str(venv_dir)):
            result = ensure_workspace_python(str(workspace))
        assert result == str(python)
        assert not (workspace / ".venv").exists()

    def test_with_conda_does_not_create_venv(self, tmp_path):
        conda_dir = tmp_path / "conda"
        python = _make_fake_python(conda_dir)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with _clean_env(CONDA_PREFIX=str(conda_dir)):
            result = ensure_workspace_python(str(workspace))
        assert result == str(python)
        assert not (workspace / ".venv").exists()

    def test_with_both_env_vars_uses_virtual_env(self, tmp_path):
        venv_dir = tmp_path / "venv"
        venv_python = _make_fake_python(venv_dir)
        conda_dir = tmp_path / "conda"
        _make_fake_python(conda_dir)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with _clean_env(VIRTUAL_ENV=str(venv_dir), CONDA_PREFIX=str(conda_dir)):
            result = ensure_workspace_python(str(workspace))
        assert result == str(venv_python)
        assert not (workspace / ".venv").exists()

    def test_global_python_returns_sys_executable(self, tmp_path):
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

    def test_global_python_pep668_creates_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with (
            _clean_env(),
            patch("comfy_cli.resolve_python.sys") as mock_sys,
            patch("comfy_cli.resolve_python._is_externally_managed", return_value=True),
        ):
            mock_sys.executable = sys.executable
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            result = ensure_workspace_python(str(workspace))

        assert (workspace / ".venv").is_dir()
        assert os.path.isfile(result)
        assert ".venv" in result

    def test_creates_venv_when_isolated_env(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with _clean_env():
            # Simulate isolated env (pipx/uv tool): prefix != base_prefix
            with patch("comfy_cli.resolve_python.sys") as mock_sys:
                mock_sys.executable = sys.executable
                mock_sys.prefix = "/home/user/.local/pipx/venvs/comfy-cli"
                mock_sys.base_prefix = "/usr"
                result = ensure_workspace_python(str(workspace))

        assert (workspace / ".venv").is_dir()
        assert os.path.isfile(result)
        assert ".venv" in result
        r = subprocess.run([result, "-c", "print('ok')"], capture_output=True, text=True)
        assert r.returncode == 0

    def test_existing_dot_venv_reused(self, tmp_path):
        workspace = tmp_path / "workspace"
        python = _make_fake_python(workspace / ".venv")

        with _clean_env():
            result = ensure_workspace_python(str(workspace))
        assert result == str(python)

    def test_existing_venv_reused(self, tmp_path):
        workspace = tmp_path / "workspace"
        python = _make_fake_python(workspace / "venv")

        with _clean_env():
            result = ensure_workspace_python(str(workspace))
        assert result == str(python)

    def test_broken_dot_venv_falls_to_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)
        venv_python = _make_fake_python(workspace / "venv")

        with _clean_env():
            result = ensure_workspace_python(str(workspace))
        assert result == str(venv_python)

    def test_broken_dot_venv_global_python_returns_sys_executable(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)

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

    def test_broken_dot_venv_isolated_env_creates_new(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / ".venv").mkdir(parents=True)

        with _clean_env(), patch("comfy_cli.resolve_python.sys") as mock_sys:
            mock_sys.executable = sys.executable
            mock_sys.prefix = "/home/user/.local/pipx/venvs/comfy-cli"
            mock_sys.base_prefix = "/usr"
            result = ensure_workspace_python(str(workspace))
        assert os.path.isfile(result)
        assert ".venv" in result
        r = subprocess.run([result, "-c", "print('ok')"], capture_output=True, text=True)
        assert r.returncode == 0


class TestCreateWorkspaceVenv:
    def test_creates_working_venv(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = create_workspace_venv(str(workspace))
        assert (workspace / ".venv").is_dir()
        assert os.path.isfile(result)
        r = subprocess.run([result, "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True)
        assert r.returncode == 0
        assert str(workspace) in r.stdout.strip()

    def test_created_venv_has_pip(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = create_workspace_venv(str(workspace))
        r = subprocess.run([result, "-m", "pip", "--version"], capture_output=True, text=True)
        assert r.returncode == 0
        assert "pip" in r.stdout

    def test_created_venv_is_isolated(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = create_workspace_venv(str(workspace))
        assert result != sys.executable
        r = subprocess.run([result, "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True)
        assert r.returncode == 0
        assert r.stdout.strip() != sys.prefix

    def test_returns_platform_specific_path(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = create_workspace_venv(str(workspace))
        if sys.platform == "win32":
            assert "Scripts" in result
        else:
            assert "bin" in result

    def test_idempotent(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result1 = create_workspace_venv(str(workspace))
        result2 = create_workspace_venv(str(workspace))
        assert result1 == result2
        assert os.path.isfile(result2)
        r = subprocess.run([result2, "-c", "print('ok')"], capture_output=True, text=True)
        assert r.returncode == 0

    def test_failure_raises(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with patch("comfy_cli.resolve_python.subprocess.run", side_effect=subprocess.CalledProcessError(1, "venv")):
            with pytest.raises(subprocess.CalledProcessError):
                create_workspace_venv(str(workspace))
