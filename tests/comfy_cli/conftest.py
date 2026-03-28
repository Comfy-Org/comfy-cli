import os

import pytest


@pytest.fixture(autouse=True)
def _preserve_cwd():
    """Restore the working directory after every test.

    Several functions in comfy_cli.command.install (execute,
    pip_install_comfyui_dependencies) call os.chdir() as a side effect.
    Without this fixture the changed CWD leaks into subsequent tests and
    can cause hard-to-debug failures.
    """
    original = os.getcwd()
    yield
    os.chdir(original)
