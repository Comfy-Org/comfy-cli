import os
import pytest
from unittest.mock import patch
from typer.testing import CliRunner

from comfy_cli.cmdline import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_execute():
    with patch("comfy_cli.command.install.execute") as mock:
        yield mock


@pytest.fixture
def mock_prompt_select_enum():
    def mocked_prompt_select_enum(
        question: str, choices: list, force_prompting: bool = False
    ):
        return choices[0]

    with patch(
        "comfy_cli.ui.prompt_select_enum",
        new=mocked_prompt_select_enum,
    ) as mock:
        yield mock



@pytest.mark.parametrize("cmd", [
    ["--here", "install"],
    ["--workspace", "./", "install"],
])
def test_install_here(runner, mock_execute, mock_prompt_select_enum):
    result = runner.invoke(app, ["--here", "install"])
    assert result.exit_code == 0, result.stdout

    args, kwargs = mock_execute.call_args
    url, manager_url, comfy_path, *_ = args
    assert url == "https://github.com/comfyanonymous/ComfyUI"
    assert manager_url == "https://github.com/ltdrdata/ComfyUI-Manager"
    assert comfy_path == os.getcwd()
