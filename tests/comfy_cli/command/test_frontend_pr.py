from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from comfy_cli.command.install import (
    PRInfo,
    parse_frontend_pr_reference,
    verify_node_tools,
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_frontend_pr_info():
    return PRInfo(
        number=456,
        head_repo_url="https://github.com/testuser/ComfyUI_frontend.git",
        head_branch="feature-branch",
        base_repo_url="https://github.com/Comfy-Org/ComfyUI_frontend.git",
        base_branch="main",
        title="Add new feature to frontend",
        user="testuser",
        mergeable=True,
    )


class TestFrontendPRReferenceParsing:
    """Test frontend PR reference parsing functionality"""

    def test_parse_frontend_pr_number_format(self):
        """Test parsing #123 format for frontend"""
        repo_owner, repo_name, pr_number = parse_frontend_pr_reference("#456")
        assert repo_owner == "Comfy-Org"
        assert repo_name == "ComfyUI_frontend"
        assert pr_number == 456

    def test_parse_frontend_user_branch_format(self):
        """Test parsing username:branch format for frontend"""
        repo_owner, repo_name, pr_number = parse_frontend_pr_reference("testuser:feature-branch")
        assert repo_owner == "Comfy-Org"
        assert repo_name == "ComfyUI_frontend"
        assert pr_number is None

    def test_parse_frontend_github_url_format(self):
        """Test parsing full GitHub PR URL for frontend"""
        url = "https://github.com/Comfy-Org/ComfyUI_frontend/pull/789"
        repo_owner, repo_name, pr_number = parse_frontend_pr_reference(url)
        assert repo_owner == "Comfy-Org"
        assert repo_name == "ComfyUI_frontend"
        assert pr_number == 789

    def test_parse_frontend_custom_repo_url(self):
        """Test parsing URL from custom repository"""
        url = "https://github.com/customuser/customrepo/pull/123"
        repo_owner, repo_name, pr_number = parse_frontend_pr_reference(url)
        assert repo_owner == "customuser"
        assert repo_name == "customrepo"
        assert pr_number == 123

    def test_parse_frontend_invalid_format(self):
        """Test parsing invalid format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid frontend PR reference format"):
            parse_frontend_pr_reference("invalid-format")

    def test_parse_frontend_empty_string(self):
        """Test parsing empty string raises ValueError"""
        with pytest.raises(ValueError):
            parse_frontend_pr_reference("")


class TestNodeToolsVerification:
    """Test Node.js tools verification"""

    @patch("subprocess.run")
    def test_verify_node_tools_success(self, mock_run):
        """Test successful Node.js, npm, and pnpm verification"""
        # Mock successful node, npm, and pnpm commands
        node_result = Mock()
        node_result.returncode = 0
        node_result.stdout = "v18.0.0"

        npm_result = Mock()
        npm_result.returncode = 0
        npm_result.stdout = "9.0.0"

        pnpm_result = Mock()
        pnpm_result.returncode = 0
        pnpm_result.stdout = "8.0.0"

        mock_run.side_effect = [node_result, npm_result, pnpm_result]

        assert verify_node_tools() is True
        assert mock_run.call_count == 3

    @patch("subprocess.run")
    def test_verify_node_tools_missing_node(self, mock_run):
        """Test when Node.js is not installed"""
        node_result = Mock()
        node_result.returncode = 1

        mock_run.return_value = node_result

        assert verify_node_tools() is False
        mock_run.assert_called_once_with(["node", "--version"], capture_output=True, text=True, check=False)

    @patch("subprocess.run")
    def test_verify_node_tools_missing_npm(self, mock_run):
        """Test when npm is not installed"""
        node_result = Mock()
        node_result.returncode = 0
        node_result.stdout = "v18.0.0"

        npm_result = Mock()
        npm_result.returncode = 1

        mock_run.side_effect = [node_result, npm_result]

        assert verify_node_tools() is False
        assert mock_run.call_count == 2

    @patch("rich.prompt.Confirm.ask")
    @patch("subprocess.run")
    def test_verify_node_tools_auto_install_pnpm(self, mock_run, mock_confirm):
        """Test automatic pnpm installation when user agrees"""
        # Mock successful node and npm
        node_result = Mock()
        node_result.returncode = 0
        node_result.stdout = "v18.0.0"

        npm_result = Mock()
        npm_result.returncode = 0
        npm_result.stdout = "9.0.0"

        # Mock pnpm not found initially
        pnpm_missing = Mock()
        pnpm_missing.returncode = 1

        # Mock successful pnpm installation
        install_result = Mock()
        install_result.returncode = 0

        # Mock pnpm verification after install
        pnpm_verify = Mock()
        pnpm_verify.returncode = 0
        pnpm_verify.stdout = "8.0.0"

        mock_run.side_effect = [node_result, npm_result, pnpm_missing, install_result, pnpm_verify]
        mock_confirm.return_value = True  # User agrees to install

        assert verify_node_tools() is True
        assert mock_run.call_count == 5
        mock_confirm.assert_called_once()

    @patch("rich.prompt.Confirm.ask")
    @patch("subprocess.run")
    def test_verify_node_tools_user_declines_pnpm_install(self, mock_run, mock_confirm):
        """Test when user declines pnpm installation"""
        # Mock successful node and npm
        node_result = Mock()
        node_result.returncode = 0
        node_result.stdout = "v18.0.0"

        npm_result = Mock()
        npm_result.returncode = 0
        npm_result.stdout = "9.0.0"

        # Mock pnpm not found
        pnpm_missing = Mock()
        pnpm_missing.returncode = 1

        mock_run.side_effect = [node_result, npm_result, pnpm_missing]
        mock_confirm.return_value = False  # User declines install

        assert verify_node_tools() is False
        assert mock_run.call_count == 3
        mock_confirm.assert_called_once()

    @patch("subprocess.run")
    def test_verify_node_tools_file_not_found(self, mock_run):
        """Test when commands are not found"""
        mock_run.side_effect = FileNotFoundError("node not found")

        assert verify_node_tools() is False
