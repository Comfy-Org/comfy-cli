import contextlib
from types import SimpleNamespace
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


@contextlib.contextmanager
def publish_flow_mocks(echo_changelog=None):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""

    response = MagicMock(signedUrl="https://test.url")
    response.node_version.changelog = echo_changelog

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("comfy_cli.command.custom_nodes.command.extract_node_configuration") as mock_extract,
        patch(
            "comfy_cli.command.custom_nodes.command.registry_api.publish_node_version", return_value=response
        ) as mock_publish,
        patch("comfy_cli.command.custom_nodes.command.zip_files") as mock_zip,
        patch("comfy_cli.command.custom_nodes.command.upload_file_to_signed_url") as mock_upload,
    ):
        mock_extract.return_value = create_mock_config()
        yield SimpleNamespace(extract=mock_extract, publish=mock_publish, zip=mock_zip, upload=mock_upload)


def flatten(output: str) -> str:
    # rich wraps long lines at terminal width; collapse whitespace before matching
    return " ".join(output.split())


def test_publish_changelog_flag_is_stripped_and_sent():
    with publish_flow_mocks(echo_changelog="Fixed a bug") as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", "  Fixed a bug  "])

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "Fixed a bug"


def test_publish_changelog_file_is_read_with_bom_stripped(tmp_path):
    changelog_path = tmp_path / "notes.md"
    changelog_path.write_text("## 1.0.1\n- multi\n- line\n", encoding="utf-8-sig")

    with publish_flow_mocks(echo_changelog="## 1.0.1\n- multi\n- line") as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog-file", str(changelog_path)])

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "## 1.0.1\n- multi\n- line"


def test_publish_changelog_file_dash_reads_stdin():
    with publish_flow_mocks(echo_changelog="from stdin\nline two") as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", "-"],
            input="from stdin\nline two\n",
        )

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "from stdin\nline two"


def test_publish_changelog_stdin_strips_bom():
    with publish_flow_mocks(echo_changelog="piped notes") as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", "-"],
            input="\ufeffpiped notes\n",
        )

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "piped notes"


def test_publish_changelog_stdin_without_token_fails_before_consuming_stdin():
    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--changelog-file", "-"],
            input="notes\n",
        )

    assert result.exit_code == 1
    assert "requires --token" in flatten(result.stdout)
    assert not mocks.extract.called
    assert not mocks.publish.called


def test_publish_changelog_stdin_invalid_utf8_fails():
    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", "-"],
            input=b"\xff\xfe\x00bad",
        )

    assert result.exit_code == 1
    assert "could not read changelog from stdin" in flatten(result.stdout)
    assert not mocks.extract.called
    assert not mocks.publish.called


def test_publish_changelog_env_var_is_used():
    with publish_flow_mocks(echo_changelog="from env") as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token"],
            env={"COMFY_NODE_CHANGELOG": "from env"},
        )

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "from env"


def test_publish_changelog_flag_overrides_env_var():
    with publish_flow_mocks(echo_changelog="from flag") as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog", "from flag"],
            env={"COMFY_NODE_CHANGELOG": "from env"},
        )

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "from flag"


def test_publish_changelog_file_overrides_env_var(tmp_path):
    changelog_path = tmp_path / "notes.md"
    changelog_path.write_text("from file", encoding="utf-8")

    with publish_flow_mocks(echo_changelog="from file") as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", str(changelog_path)],
            env={"COMFY_NODE_CHANGELOG": "from env"},
        )

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == "from file"


def test_publish_changelog_flags_are_mutually_exclusive(tmp_path):
    changelog_path = tmp_path / "notes.md"
    changelog_path.write_text("from file", encoding="utf-8")

    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog", "text", "--changelog-file", str(changelog_path)],
        )

    assert result.exit_code == 1
    assert "mutually exclusive" in flatten(result.stdout)
    assert not mocks.extract.called
    assert not mocks.publish.called


def test_publish_changelog_file_missing_fails_before_validation(tmp_path):
    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", str(tmp_path / "missing.md")],
        )

    assert result.exit_code == 1
    assert "could not read changelog file" in flatten(result.stdout)
    assert not mocks.extract.called
    assert not mocks.publish.called


def test_publish_changelog_file_invalid_utf8_fails(tmp_path):
    changelog_path = tmp_path / "bad.md"
    changelog_path.write_bytes(b"\xff\xfe\x00bad")

    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", str(changelog_path)],
        )

    assert result.exit_code == 1
    assert "could not read changelog file" in flatten(result.stdout)
    assert not mocks.publish.called


def test_publish_changelog_file_directory_fails(tmp_path):
    with publish_flow_mocks() as mocks:
        result = runner.invoke(
            app,
            ["publish", "--token", "test-token", "--changelog-file", str(tmp_path)],
        )

    assert result.exit_code == 1
    assert "could not read changelog file" in flatten(result.stdout)
    assert not mocks.publish.called


def test_publish_changelog_whitespace_only_file_treated_as_absent(tmp_path):
    changelog_path = tmp_path / "blank.md"
    changelog_path.write_text("   \n\t\n", encoding="utf-8")

    with publish_flow_mocks() as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog-file", str(changelog_path)])

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == ""
    assert "did not echo" not in flatten(result.stdout)


def test_publish_warns_when_registry_drops_changelog():
    with publish_flow_mocks(echo_changelog="") as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", "some notes"])

    assert result.exit_code == 0
    assert "Warning: the registry did not echo the changelog back" in flatten(result.stdout)
    assert mocks.zip.called
    assert mocks.upload.called


def test_publish_warns_when_echo_changelog_is_none():
    with publish_flow_mocks(echo_changelog=None) as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", "some notes"])

    assert result.exit_code == 0
    assert "Warning: the registry did not echo the changelog back" in flatten(result.stdout)
    assert mocks.upload.called


def test_publish_no_warning_when_changelog_echoed():
    with publish_flow_mocks(echo_changelog="some notes") as _mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", "some notes"])

    assert result.exit_code == 0
    assert "did not echo" not in flatten(result.stdout)


def test_publish_no_warning_when_echo_differs_only_by_whitespace():
    with publish_flow_mocks(echo_changelog="  some notes \n") as _mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", "some notes"])

    assert result.exit_code == 0
    assert "did not echo" not in flatten(result.stdout)


def test_publish_without_changelog_sends_empty_and_does_not_warn():
    with publish_flow_mocks() as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token"])

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == ""
    assert "did not echo" not in flatten(result.stdout)


def test_publish_empty_changelog_flag_treated_as_absent():
    with publish_flow_mocks() as mocks:
        result = runner.invoke(app, ["publish", "--token", "test-token", "--changelog", ""])

    assert result.exit_code == 0
    assert mocks.publish.call_args.kwargs["changelog"] == ""
    assert "did not echo" not in flatten(result.stdout)
