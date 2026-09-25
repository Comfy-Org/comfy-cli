"""Unit tests for the e2e ``exec`` helper (run without TEST_E2E)."""

import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "_e2e_uv_compile", os.path.join(os.path.dirname(__file__), "test_e2e_uv_compile.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_exec_list_keeps_path_with_spaces_as_one_argument(tmp_path):
    # An interpreter path containing a space (e.g. a VIRTUAL_ENV under
    # "My Projects") must reach the child process as a single argument.
    script_dir = tmp_path / "dir with spaces"
    script_dir.mkdir()
    script = script_dir / "echo_argv.py"
    script.write_text("import sys; print(sys.argv[1:])")

    proc = _mod.exec([sys.executable, str(script), "a b"])

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "['a b']"


def test_exec_string_still_runs_through_shell():
    proc = _mod.exec(f'"{sys.executable}" -c "print(1 + 1)"')

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2"


def test_pypi_manager_override_bootstraps_pip_before_pip_install(monkeypatch, tmp_path):
    # `--fast-deps --skip-manager` never calls ensure_pip during `comfy install`,
    # so a uv-managed workspace venv may have no pip when the override runs.
    calls = []
    monkeypatch.setattr(_mod, "ensure_pip", lambda python, cwd=None: calls.append(("ensure_pip", python)))

    def fake_exec(cmd, **kwargs):
        calls.append(("exec", cmd))
        stdout = "Name: comfyui-manager\nVersion: 4.3\n" if "show" in cmd else ""
        return _mod.subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(_mod, "exec", fake_exec)

    _mod._apply_manager_override(str(tmp_path), "/venv/bin/python", "4.3")

    assert calls[0] == ("ensure_pip", "/venv/bin/python")
    assert calls[1] == ("exec", ["/venv/bin/python", "-m", "pip", "install", "comfyui-manager==4.3", "--pre"])
