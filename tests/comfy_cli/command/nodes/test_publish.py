from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.command import app
from comfy_cli.registry.types import ComfyConfig, ProjectConfig, PyProjectConfig

runner = CliRunner()


def create_mock_config(includes_list=None):
    if includes_list is None:
        includes_list = []

    mock_pyproject_config = MagicMock()

    mock_tool_comfy_section = MagicMock()
    mock_tool_comfy_section.name = "test-node"
    mock_tool_comfy_section.version = "0.1.0"
    mock_tool_comfy_section.description = "A test node."
    mock_tool_comfy_section.author = "Test Author"
    mock_tool_comfy_section.license = "MIT"
    mock_tool_comfy_section.tags = ["test"]
    mock_tool_comfy_section.repository = "http://example.com/repo"
    mock_tool_comfy_section.homepage = "http://example.com/home"
    mock_tool_comfy_section.documentation = "http://example.com/docs"
    mock_tool_comfy_section.includes = includes_list

    mock_pyproject_config.tool_comfy = mock_tool_comfy_section

    return mock_pyproject_config


def test_publish_fails_on_security_violations():
    # Mock subprocess.run to simulate security violations
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "S102 Use of exec() detected"

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("typer.prompt", return_value="test-token"),
    ):
        result = runner.invoke(app, ["publish"])

        # TODO: re-enable exit when we disable exec and eval
        # assert result.exit_code == 1
        # assert "Security issues found" in result.stdout
        assert "Security warnings found" in result.stdout


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
        mock_extract.return_value = create_mock_config()

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
        mock_extract.return_value = create_mock_config()

        mock_publish.return_value = MagicMock(signedUrl="https://test.url")

        # Run the publish command with token
        _result = runner.invoke(app, ["publish", "--token", "test-token"])

        # Verify the publish flow worked with provided token
        assert mock_extract.called
        assert mock_publish.called
        assert mock_zip.called
        assert mock_upload.called


def test_publish_exits_on_upload_failure():
    # Mock subprocess.run to simulate no violations
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("comfy_cli.command.custom_nodes.command.extract_node_configuration") as mock_extract,
        patch("typer.prompt", return_value="test-token"),
        patch("comfy_cli.command.custom_nodes.command.registry_api.publish_node_version") as mock_publish,
        patch("comfy_cli.command.custom_nodes.command.zip_files") as mock_zip,
        patch("comfy_cli.command.custom_nodes.command.upload_file_to_signed_url") as mock_upload,
    ):
        # Setup the mocks
        mock_extract.return_value = create_mock_config()

        mock_publish.return_value = MagicMock(signedUrl="https://test.url")
        mock_upload.side_effect = Exception("Upload failed with status code: 403")

        # Run the publish command
        result = runner.invoke(app, ["publish"])

        # Verify the command exited with error
        assert result.exit_code == 1
        assert mock_extract.called
        assert mock_publish.called
        assert mock_zip.called
        assert mock_upload.called


def test_publish_fails_when_config_is_none():
    # extract_node_configuration returns None when pyproject.toml is missing;
    # validate_node_for_publishing must exit 1 (not crash on the subsequent
    # `config.project.version` access).
    with patch(
        "comfy_cli.command.custom_nodes.command.extract_node_configuration",
        return_value=None,
    ):
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1


def test_publish_fails_when_version_is_empty():
    # Guards against issue #294: dynamic versions that failed to resolve must
    # not silently POST an empty `version` to the registry. validate_node_for_publishing
    # should exit 1 with a user-facing error pointing at [tool.comfy.version].path.
    empty_version_config = PyProjectConfig(
        project=ProjectConfig(name="x", version=""),
        tool_comfy=ComfyConfig(publisher_id="pub"),
    )
    with patch(
        "comfy_cli.command.custom_nodes.command.extract_node_configuration",
        return_value=empty_version_config,
    ):
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "project version is empty" in result.stdout
        assert "[tool.comfy.version].path" in result.stdout


def test_publish_with_includes_parameter():
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
        includes = ["/js", "/dist"]

        # Setup the mocks
        mock_extract.return_value = create_mock_config(includes)

        mock_publish.return_value = MagicMock(signedUrl="https://test.url")

        # Run the publish command with token
        _result = runner.invoke(app, ["publish", "--token", "test-token"])

        # Verify the publish flow worked with provided token
        assert mock_extract.called
        assert mock_publish.called
        assert mock_zip.called
        assert mock_upload.called
