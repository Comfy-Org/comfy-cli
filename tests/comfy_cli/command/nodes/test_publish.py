from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.command import app

runner = CliRunner()


def test_publish_fails_on_security_violations():
    # Mock subprocess.run to simulate security violations
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "S102 Use of exec() detected"

    with patch("subprocess.run", return_value=mock_result):
        result = runner.invoke(app, ["publish"])

        assert result.exit_code == 1
        assert "Security issues found" in result.stdout


def test_publish_continues_on_no_security_violations():
    # Mock subprocess.run to simulate no violations
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("comfy_cli.command.custom_nodes.command.extract_node_configuration") as mock_extract,
        patch("typer.prompt") as mock_prompt,
        patch("comfy_cli.command.custom_nodes.command.registry_api.publish_node_version") as mock_publish,
        patch("comfy_cli.command.custom_nodes.command.zip_files") as mock_zip,
        patch("comfy_cli.command.custom_nodes.command.upload_file_to_signed_url") as mock_upload,
    ):
        # Setup the mocks
        mock_extract.return_value = {"name": "test-node"}
        mock_prompt.return_value = "test-token"
        mock_publish.return_value = MagicMock(signedUrl="https://test.url")

        # Run the publish command
        _result = runner.invoke(app, ["publish"])

        # Verify the publish flow continued
        assert mock_extract.called
        assert mock_publish.called
        assert mock_zip.called
        assert mock_upload.called


def test_publish_handles_missing_ruff():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = runner.invoke(app, ["publish"])

        assert result.exit_code == 1
        assert "Ruff is not installed" in result.stdout


def test_publish_with_token_option():
    # Mock subprocess.run to simulate no violations
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("comfy_cli.command.custom_nodes.command.extract_node_configuration") as mock_extract,
        patch("comfy_cli.command.custom_nodes.command.registry_api.publish_node_version") as mock_publish,
        patch("comfy_cli.command.custom_nodes.command.zip_files") as mock_zip,
        patch("comfy_cli.command.custom_nodes.command.upload_file_to_signed_url") as mock_upload,
    ):
        # Setup the mocks
        mock_extract.return_value = {"name": "test-node"}
        mock_publish.return_value = MagicMock(signedUrl="https://test.url")

        # Run the publish command with token
        _result = runner.invoke(app, ["publish", "--token", "test-token"])

        # Verify the publish flow worked with provided token
        assert mock_extract.called
        assert mock_publish.called
        assert mock_zip.called
        assert mock_upload.called
