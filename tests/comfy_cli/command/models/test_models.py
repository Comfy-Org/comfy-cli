import pathlib
from unittest.mock import Mock, patch

import typer.testing

from comfy_cli import constants
from comfy_cli.command.models.models import _format_elapsed, app, check_civitai_url, check_huggingface_url, list_models


def _make_model_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a realistic model directory tree and return its root."""
    model_dir = tmp_path / "models"
    (model_dir / "root_model.safetensors").parent.mkdir(parents=True, exist_ok=True)
    (model_dir / "root_model.safetensors").write_bytes(b"x" * 100)
    (model_dir / "checkpoints").mkdir()
    (model_dir / "checkpoints" / "sd15.safetensors").write_bytes(b"x" * 200)
    (model_dir / "loras" / "SD1.5").mkdir(parents=True)
    (model_dir / "loras" / "SD1.5" / "detail.safetensors").write_bytes(b"x" * 300)
    (model_dir / "empty_dir").mkdir()
    return model_dir


def test_list_models_finds_files_in_subdirectories(tmp_path):
    model_dir = _make_model_tree(tmp_path)
    result = list_models(model_dir)
    names = {f.name for f in result}
    assert "sd15.safetensors" in names
    deep = [f for f in result if f.name == "detail.safetensors"]
    assert len(deep) == 1
    assert deep[0].relative_to(model_dir) == pathlib.Path("loras/SD1.5/detail.safetensors")


def test_list_models_finds_root_level_files(tmp_path):
    model_dir = _make_model_tree(tmp_path)
    result = list_models(model_dir)
    names = {f.name for f in result}
    assert "root_model.safetensors" in names


def test_list_models_returns_empty_for_missing_directory(tmp_path):
    assert list_models(tmp_path / "nonexistent") == []


def test_list_models_ignores_directories(tmp_path):
    model_dir = _make_model_tree(tmp_path)
    result = list_models(model_dir)
    assert all(f.is_file() for f in result)
    dir_names = {f.name for f in result}
    assert "empty_dir" not in dir_names
    assert "checkpoints" not in dir_names


runner = typer.testing.CliRunner()


def test_list_command_shows_type_column(tmp_path):
    _make_model_tree(tmp_path)
    with patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path):
        result = runner.invoke(app, ["list", "--relative-path", "models"])
    assert result.exit_code == 0
    assert "Type" in result.output
    assert "checkpoints" in result.output
    assert "loras/SD1.5" in result.output
    assert "root_model.safetensors" in result.output


def test_remove_with_path_traversal_is_rejected(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "legit.bin").write_bytes(b"x")

    secret = tmp_path / "secret.txt"
    secret.write_text("sensitive")

    with patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path):
        result = runner.invoke(
            app,
            ["remove", "--relative-path", "models", "--model-names", "../secret.txt", "--confirm"],
        )
    assert secret.exists()
    assert "Invalid model path" in result.output


def test_remove_deletes_model_in_subdirectory(tmp_path):
    model_dir = _make_model_tree(tmp_path)
    target = model_dir / "checkpoints" / "sd15.safetensors"
    assert target.exists()

    with patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path):
        result = runner.invoke(
            app,
            ["remove", "--relative-path", "models", "--model-names", "checkpoints/sd15.safetensors", "--confirm"],
        )
    assert result.exit_code == 0
    assert not target.exists()


def test_remove_rejects_directory_name(tmp_path):
    _make_model_tree(tmp_path)

    with patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path):
        result = runner.invoke(
            app,
            ["remove", "--relative-path", "models", "--model-names", "checkpoints", "--confirm"],
        )
    assert (tmp_path / "models" / "checkpoints").is_dir()
    assert "not found" in result.output


def test_remove_deletes_root_level_model(tmp_path):
    model_dir = _make_model_tree(tmp_path)
    target = model_dir / "root_model.safetensors"
    assert target.exists()

    with patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path):
        result = runner.invoke(
            app,
            ["remove", "--relative-path", "models", "--model-names", "root_model.safetensors", "--confirm"],
        )
    assert result.exit_code == 0
    assert not target.exists()


def test_remove_interactive_shows_relative_paths(tmp_path):
    _make_model_tree(tmp_path)

    with (
        patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path),
        patch("comfy_cli.command.models.models.ui") as mock_ui,
    ):
        mock_ui.prompt_multi_select.return_value = ["checkpoints/sd15.safetensors"]
        mock_ui.prompt_confirm_action.return_value = True
        runner.invoke(app, ["remove", "--relative-path", "models"])

    choices = mock_ui.prompt_multi_select.call_args[0][1]
    assert "checkpoints/sd15.safetensors" in choices
    assert "loras/SD1.5/detail.safetensors" in choices
    assert "root_model.safetensors" in choices
    assert not (tmp_path / "models" / "checkpoints" / "sd15.safetensors").exists()


def test_valid_model_url():
    url = "https://civitai.com/models/43331"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_with_version():
    url = "https://civitai.com/models/43331/majicmix-realistic"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_with_version_and_additional_segments():
    url = "https://civitai.com/models/43331/majicmix-realistic/extra"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_with_query():
    url = "https://civitai.com/models/43331?version=12345"
    assert check_civitai_url(url) == (True, False, 43331, 12345)


def test_valid_api_url():
    url = "https://civitai.com/api/download/models/67890"
    assert check_civitai_url(url) == (False, True, None, 67890)


def test_invalid_url():
    url = "https://example.com/models/43331"
    assert check_civitai_url(url) == (False, False, None, None)


def test_malformed_url():
    url = "https://civitai.com/models/"
    assert check_civitai_url(url) == (False, False, None, None)


def test_invalid_model_id_url():
    url = "https://civitai.com/models/invalid_id"
    assert check_civitai_url(url) == (False, False, None, None)


def test_malformed_query_url():
    url = "https://civitai.com/models/43331?version="
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_model_url_with_model_version_id_query():
    url = "https://civitai.com/models/43331?modelVersionId=485088"
    assert check_civitai_url(url) == (True, False, 43331, 485088)


def test_model_url_with_model_version_id_invalid():
    url = "https://civitai.com/models/43331?modelVersionId=abc"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_api_v1_model_versions_url():
    url = "https://civitai.com/api/v1/model-versions/1617665"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_valid_api_v1_model_versions_camelcase_segment():
    url = "https://civitai.com/api/v1/modelVersions/1617665"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_valid_api_download_with_query_params():
    url = "https://civitai.com/api/download/models/1617665?type=Model&format=SafeTensor"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_api_download_trailing_slash_is_ok():
    url = "https://civitai.com/api/download/models/1617665/"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_api_download_non_numeric_id_models_version():
    url = "https://civitai.com/api/v1/modelVersions/notanumber"
    assert check_civitai_url(url) == (False, True, None, None)


def test_api_download_non_numeric_id():
    url = "https://civitai.com/api/download/models/notanumber"
    assert check_civitai_url(url) == (False, True, None, None)


def test_model_url_with_slug_and_query():
    url = "https://civitai.com/models/43331/majicmix-realistic?modelVersionId=485088"
    assert check_civitai_url(url) == (True, False, 43331, 485088)


def test_www_subdomain_is_accepted():
    url = "https://www.civitai.com/models/43331?version=12345"
    assert check_civitai_url(url) == (True, False, 43331, 12345)


def test_completly_mailformed_civitai_url():
    url = "https://civitai.com/"
    assert check_civitai_url(url) == (False, False, None, None)


def test_non_evil_civitai_url():
    url = "https://evilcivitai.com/models/43331?version=12345"
    assert check_civitai_url(url) == (False, False, None, None)


def test_valid_model_url_red_domain():
    url = "https://civitai.red/models/43331"
    assert check_civitai_url(url) == (True, False, 43331, None)


def test_valid_model_url_red_with_query():
    url = "https://civitai.red/models/43331?modelVersionId=485088"
    assert check_civitai_url(url) == (True, False, 43331, 485088)


def test_valid_api_download_url_red_domain():
    url = "https://civitai.red/api/download/models/1617665?type=Model&format=SafeTensor"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_valid_api_v1_model_versions_url_red_domain():
    url = "https://civitai.red/api/v1/model-versions/1617665"
    assert check_civitai_url(url) == (False, True, None, 1617665)


def test_www_subdomain_red_is_accepted():
    url = "https://www.civitai.red/models/43331?version=12345"
    assert check_civitai_url(url) == (True, False, 43331, 12345)


def test_non_evil_civitai_red_url():
    url = "https://evilcivitai.red/models/43331?version=12345"
    assert check_civitai_url(url) == (False, False, None, None)


def test_red_as_spoofed_subdomain_of_other_tld():
    url = "https://civitai.red.evil.com/models/43331"
    assert check_civitai_url(url) == (False, False, None, None)


def test_valid_huggingface_url():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4.ckpt", None, "main")


def test_valid_huggingface_url_sd_audio():
    url = "https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.safetensors"
    assert check_huggingface_url(url) == (True, "stabilityai/stable-audio-open-1.0", "model.safetensors", None, "main")


def test_valid_huggingface_url_with_folder():
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    assert check_huggingface_url(url) == (
        True,
        "runwayml/stable-diffusion-v1-5",
        "v1-5-pruned-emaonly.ckpt",
        None,
        "main",
    )


def test_valid_huggingface_url_with_subfolder():
    url = "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt"
    assert check_huggingface_url(url) == (
        True,
        "stabilityai/stable-diffusion-2-1",
        "v2-1_768-ema-pruned.ckpt",
        None,
        "main",
    )


def test_valid_huggingface_url_with_encoded_filename():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4%20(1).ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4 (1).ckpt", None, "main")


def test_invalid_huggingface_url():
    url = "https://example.com/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (False, None, None, None, None)


def test_invalid_huggingface_url_structure():
    url = "https://huggingface.co/CompVis/stable-diffusion-v1-4/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (False, None, None, None, None)


def test_huggingface_url_with_com_domain():
    url = "https://huggingface.com/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"
    assert check_huggingface_url(url) == (True, "CompVis/stable-diffusion-v1-4", "sd-v1-4.ckpt", None, "main")


def test_huggingface_url_with_folder_structure():
    url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    assert check_huggingface_url(url) == (
        True,
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0.safetensors",
        None,
        "main",
    )


class TestFormatElapsed:
    def test_under_one_minute(self):
        assert _format_elapsed(5.3) == "5.3s"

    def test_fractional_seconds(self):
        assert _format_elapsed(0.4) == "0.4s"

    def test_rounds_up_to_minute_boundary(self):
        assert _format_elapsed(59.95) == "1m 0s"

    def test_exactly_sixty_seconds(self):
        assert _format_elapsed(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert _format_elapsed(154) == "2m 34s"

    def test_over_one_hour(self):
        assert _format_elapsed(3661) == "1h 1m 1s"

    def test_large_duration(self):
        assert _format_elapsed(7384) == "2h 3m 4s"


# ---------------------------------------------------------------------------
# --downloader CLI option tests
# ---------------------------------------------------------------------------


class TestDownloadCommandDownloaderOption:
    def test_downloader_flag_forwarded(self, tmp_path):
        """--downloader aria2 flag is forwarded to download_file."""
        with (
            patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path),
            patch("comfy_cli.command.models.models.download_file") as mock_dl,
            patch("comfy_cli.command.models.models.check_civitai_url", return_value=(False, False, None, None)),
            patch(
                "comfy_cli.command.models.models.check_huggingface_url", return_value=(False, None, None, None, None)
            ),
            patch("comfy_cli.command.models.models.ui") as mock_ui,
            patch("comfy_cli.command.models.models.config_manager"),
            patch("comfy_cli.tracking.track_command", lambda _cmd: lambda fn: fn),
        ):
            mock_ui.prompt_input.side_effect = ["mymodel.bin", ""]
            result = runner.invoke(
                app,
                [
                    "download",
                    "--url",
                    "http://example.com/model.bin",
                    "--downloader",
                    "aria2",
                    "--filename",
                    "model.bin",
                ],
            )

            assert mock_dl.called
            _, kwargs = mock_dl.call_args
            assert kwargs.get("downloader") == "aria2"
            assert "Done in " in result.output

    def test_default_from_config(self, tmp_path):
        """Config default_downloader is used when no --downloader flag."""
        mock_cfg = Mock()
        mock_cfg.get_or_override.return_value = None
        mock_cfg.get.side_effect = lambda key: "aria2" if key == constants.CONFIG_KEY_DEFAULT_DOWNLOADER else None

        with (
            patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path),
            patch("comfy_cli.command.models.models.download_file") as mock_dl,
            patch("comfy_cli.command.models.models.check_civitai_url", return_value=(False, False, None, None)),
            patch(
                "comfy_cli.command.models.models.check_huggingface_url", return_value=(False, None, None, None, None)
            ),
            patch("comfy_cli.command.models.models.ui") as mock_ui,
            patch("comfy_cli.command.models.models.config_manager", mock_cfg),
            patch("comfy_cli.tracking.track_command", lambda _cmd: lambda fn: fn),
        ):
            mock_ui.prompt_input.side_effect = ["mymodel.bin", ""]
            runner.invoke(
                app,
                [
                    "download",
                    "--url",
                    "http://example.com/model.bin",
                    "--filename",
                    "model.bin",
                ],
            )

            assert mock_dl.called
            _, kwargs = mock_dl.call_args
            assert kwargs.get("downloader") == "aria2"

    def test_cli_flag_overrides_config(self, tmp_path):
        """CLI --downloader flag takes precedence over config."""
        mock_cfg = Mock()
        mock_cfg.get_or_override.return_value = None
        mock_cfg.get.side_effect = lambda key: "aria2" if key == constants.CONFIG_KEY_DEFAULT_DOWNLOADER else None

        with (
            patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path),
            patch("comfy_cli.command.models.models.download_file") as mock_dl,
            patch("comfy_cli.command.models.models.check_civitai_url", return_value=(False, False, None, None)),
            patch(
                "comfy_cli.command.models.models.check_huggingface_url", return_value=(False, None, None, None, None)
            ),
            patch("comfy_cli.command.models.models.ui") as mock_ui,
            patch("comfy_cli.command.models.models.config_manager", mock_cfg),
            patch("comfy_cli.tracking.track_command", lambda _cmd: lambda fn: fn),
        ):
            mock_ui.prompt_input.side_effect = ["mymodel.bin", ""]
            runner.invoke(
                app,
                [
                    "download",
                    "--url",
                    "http://example.com/model.bin",
                    "--downloader",
                    "httpx",
                    "--filename",
                    "model.bin",
                ],
            )

            assert mock_dl.called
            _, kwargs = mock_dl.call_args
            assert kwargs.get("downloader") == "httpx"


class TestDownloadCommandErrorHandling:
    """Verify DownloadException is rendered as a friendly one-line error (not a traceback)."""

    def _run_with_download_error(self, tmp_path, exc):
        with (
            patch("comfy_cli.command.models.models.get_workspace", return_value=tmp_path),
            patch("comfy_cli.command.models.models.download_file", side_effect=exc),
            patch("comfy_cli.command.models.models.check_civitai_url", return_value=(False, False, None, None)),
            patch(
                "comfy_cli.command.models.models.check_huggingface_url",
                return_value=(False, None, None, None, None),
            ),
            patch("comfy_cli.command.models.models.ui") as mock_ui,
            patch("comfy_cli.command.models.models.config_manager"),
            patch("comfy_cli.tracking.track_command", lambda _cmd: lambda fn: fn),
        ):
            mock_ui.prompt_input.side_effect = ["mymodel.bin", ""]
            return runner.invoke(
                app,
                [
                    "download",
                    "--url",
                    "http://example.com/model.bin",
                    "--filename",
                    "model.bin",
                ],
            )

    def test_download_exception_exits_with_code_1(self, tmp_path):
        from comfy_cli.file_utils import DownloadException

        result = self._run_with_download_error(tmp_path, DownloadException("boom"))

        assert result.exit_code == 1
        assert "boom" in result.output

    def test_download_exception_does_not_show_traceback(self, tmp_path):
        from comfy_cli.file_utils import DownloadException

        result = self._run_with_download_error(tmp_path, DownloadException("boom"))

        assert "Traceback" not in result.output
        assert "DownloadException" not in result.output
        # Rich markup must be rendered as styling, not leak through as literal tags.
        assert "[bold red]" not in result.output
        assert "[/bold red]" not in result.output

    def test_download_exception_skips_done_message(self, tmp_path):
        from comfy_cli.file_utils import DownloadException

        result = self._run_with_download_error(tmp_path, DownloadException("boom"))

        assert "Done in" not in result.output

    def test_download_exception_with_markup_chars_does_not_crash(self, tmp_path):
        """A DownloadException message containing rich-markup metacharacters (e.g. from a
        server JSON body embedded via guess_status_code_reason) must not raise MarkupError
        nor be silently stripped — the error must render literally and exit cleanly."""
        from comfy_cli.file_utils import DownloadException

        # Covers both the closing-tag crash case and the bracketed-style stripping case.
        result = self._run_with_download_error(tmp_path, DownloadException("server said [/] at /path/[id]/resource"))

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "MarkupError" not in result.output
        # Literal markup characters must survive to the output so the user sees the real message.
        assert "[/]" in result.output
        assert "[id]" in result.output
