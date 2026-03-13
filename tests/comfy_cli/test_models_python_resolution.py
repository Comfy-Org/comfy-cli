import builtins
from unittest.mock import patch

import typer.testing

from comfy_cli.command.models.models import app

runner = typer.testing.CliRunner()

_real_import = builtins.__import__


def _block_huggingface_hub(name, *args, **kwargs):
    if name == "huggingface_hub":
        raise ImportError("blocked by test")
    return _real_import(name, *args, **kwargs)


class TestDownloadHuggingfacePipInstall:
    def test_uses_resolved_python(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"

        with (
            patch("comfy_cli.command.models.models.get_workspace", return_value=workspace),
            patch("comfy_cli.command.models.models.check_unauthorized", return_value=True),
            patch(
                "comfy_cli.command.models.models.config_manager.get_or_override",
                side_effect=lambda env_key, config_key, set_value=None: "fake-hf-token" if "HF" in env_key else None,
            ),
            patch("builtins.__import__", side_effect=_block_huggingface_hub),
            patch("comfy_cli.resolve_python.resolve_workspace_python", return_value="/resolved/python"),
            patch("subprocess.check_call") as mock_check_call,
        ):
            result = runner.invoke(
                app,
                [
                    "download",
                    "--url",
                    url,
                    "--relative-path",
                    "models",
                    "--filename",
                    "sd-v1-4.ckpt",
                ],
            )

        assert mock_check_call.called, f"check_call not called; output: {result.output}"
        cmd = mock_check_call.call_args[0][0]
        assert cmd[0] == "/resolved/python"
        assert cmd == ["/resolved/python", "-m", "pip", "install", "huggingface_hub"]
