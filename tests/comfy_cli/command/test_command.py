import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app, g_exclusivity, g_gpu_exclusivity


@pytest.fixture(scope="function")
def runner():
    g_exclusivity.reset_for_testing()
    g_gpu_exclusivity.reset_for_testing()
    return CliRunner()


@pytest.fixture(scope="function")
def mock_execute():
    with patch("comfy_cli.command.install.execute") as mock:
        yield mock


@pytest.fixture(scope="function")
def mock_prompt_select_enum():
    def mocked_prompt_select_enum(question: str, choices: list, force_prompting: bool = False):
        return choices[0]

    with patch(
        "comfy_cli.ui.prompt_select_enum",
        new=mocked_prompt_select_enum,
    ) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_tracking_consent():
    with patch("comfy_cli.tracking.prompt_tracking_consent"):
        yield


@pytest.mark.parametrize(
    "cmd",
    [
        ["--here", "install"],
        ["--workspace", "./ComfyUI", "install"],
    ],
)
def test_install_here(cmd, runner, mock_execute, mock_prompt_select_enum):
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0, result.stdout

    args, _ = mock_execute.call_args
    url, manager_url, comfy_path, *_ = args
    assert url == "https://github.com/comfyanonymous/ComfyUI"
    assert manager_url == "https://github.com/ltdrdata/ComfyUI-Manager"
    assert comfy_path == os.path.join(os.getcwd(), "ComfyUI")


def test_version(runner):
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert "0.0.0" in result.stdout
