import subprocess
from unittest.mock import Mock, patch

import pytest
import requests
from typer.testing import CliRunner

from comfy_cli.cmdline import app, g_exclusivity, g_gpu_exclusivity
from comfy_cli.command.install import PRInfo, fetch_pr_info, find_pr_by_branch, handle_pr_checkout, parse_pr_reference
from comfy_cli.git_utils import checkout_pr


@pytest.fixture(scope="function")
def runner():
    g_exclusivity.reset_for_testing()
    g_gpu_exclusivity.reset_for_testing()
    return CliRunner()


@pytest.fixture
def sample_pr_info():
    return PRInfo(
        number=123,
        head_repo_url="https://github.com/jtydhr88/ComfyUI.git",
        head_branch="load-3d-nodes",
        base_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
        base_branch="master",
        title="Add 3D node loading support",
        user="jtydhr88",
        mergeable=True,
    )


class TestPRReferenceParsing:
    def test_parse_pr_number_format(self):
        """Test parsing #123 format"""
        repo_owner, repo_name, pr_number = parse_pr_reference("#123")
        assert repo_owner == "comfyanonymous"
        assert repo_name == "ComfyUI"
        assert pr_number == 123

    def test_parse_user_branch_format(self):
        """Test parsing username:branch format"""
        repo_owner, repo_name, pr_number = parse_pr_reference("jtydhr88:load-3d-nodes")
        assert repo_owner == "jtydhr88"
        assert repo_name == "ComfyUI"
        assert pr_number is None

    def test_parse_github_url_format(self):
        """Test parsing full GitHub PR URL"""
        url = "https://github.com/comfyanonymous/ComfyUI/pull/456"
        repo_owner, repo_name, pr_number = parse_pr_reference(url)
        assert repo_owner == "comfyanonymous"
        assert repo_name == "ComfyUI"
        assert pr_number == 456

    def test_parse_invalid_format(self):
        """Test parsing invalid format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid PR reference format"):
            parse_pr_reference("invalid-format")

    def test_parse_empty_string(self):
        """Test parsing empty string raises ValueError"""
        with pytest.raises(ValueError):
            parse_pr_reference("")


class TestGitHubAPIIntegration:
    """Test GitHub API integration"""

    @patch("requests.get")
    def test_fetch_pr_info_success(self, mock_get, sample_pr_info):
        """Test successful PR info fetching"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 123,
            "title": "Add 3D node loading support",
            "head": {
                "repo": {"clone_url": "https://github.com/jtydhr88/ComfyUI.git", "owner": {"login": "jtydhr88"}},
                "ref": "load-3d-nodes",
            },
            "base": {"repo": {"clone_url": "https://github.com/comfyanonymous/ComfyUI.git"}, "ref": "master"},
            "mergeable": True,
        }
        mock_get.return_value = mock_response

        result = fetch_pr_info("comfyanonymous", "ComfyUI", 123)

        assert result.number == 123
        assert result.title == "Add 3D node loading support"
        assert result.user == "jtydhr88"
        assert result.head_branch == "load-3d-nodes"
        assert result.mergeable is True

    @patch("requests.get")
    def test_find_pr_by_branch_success(self, mock_get):
        """Test successful PR search by branch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "number": 456,
                "title": "Test PR",
                "head": {
                    "repo": {"clone_url": "https://github.com/testuser/ComfyUI.git", "owner": {"login": "testuser"}},
                    "ref": "test-branch",
                },
                "base": {"repo": {"clone_url": "https://github.com/comfyanonymous/ComfyUI.git"}, "ref": "master"},
                "mergeable": True,
            }
        ]
        mock_get.return_value = mock_response

        result = find_pr_by_branch("comfyanonymous", "ComfyUI", "testuser", "test-branch")

        assert result is not None
        assert result.number == 456
        assert result.title == "Test PR"
        assert result.user == "testuser"
        assert result.head_branch == "test-branch"

    @patch("requests.get")
    def test_find_pr_by_branch_not_found(self, mock_get):
        """Test PR not found by branch"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = find_pr_by_branch("comfyanonymous", "ComfyUI", "testuser", "nonexistent-branch")
        assert result is None

    @patch("requests.get")
    def test_find_pr_by_branch_error(self, mock_get):
        """Test error when searching PR by branch"""
        mock_get.side_effect = requests.RequestException("Network error")

        result = find_pr_by_branch("comfyanonymous", "ComfyUI", "testuser", "test-branch")
        assert result is None

    @patch("requests.get")
    def test_fetch_pr_info_not_found(self, mock_get):
        """Test PR not found (404)"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="Failed to fetch PR"):
            fetch_pr_info("comfyanonymous", "ComfyUI", 999)

    @patch("requests.get")
    def test_fetch_pr_info_rate_limit(self, mock_get):
        """Test GitHub API rate limit handling"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {"x-ratelimit-remaining": "0"}
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="Primary rate limit from Github exceeded!"):
            fetch_pr_info("comfyanonymous", "ComfyUI", 123)


class TestGitOperations:
    """Test Git operations for PR checkout"""

    @patch("subprocess.run")
    @patch("os.chdir")
    @patch("os.getcwd")
    def test_checkout_pr_fork_success(self, mock_getcwd, mock_chdir, mock_subprocess, sample_pr_info):
        """Test successful checkout of PR from fork"""
        mock_getcwd.return_value = "/original/dir"

        mock_subprocess.side_effect = [
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]

        result = checkout_pr("/repo/path", sample_pr_info)

        assert result is True
        assert mock_subprocess.call_count == 4

        calls = mock_subprocess.call_args_list
        assert "git" in calls[0][0][0]
        assert "remote" in calls[1][0][0]
        assert "fetch" in calls[2][0][0]
        assert "checkout" in calls[3][0][0]

    @patch("subprocess.run")
    @patch("os.chdir")
    @patch("os.getcwd")
    def test_checkout_pr_non_fork_success(self, mock_getcwd, mock_chdir, mock_subprocess):
        """Test successful checkout of PR from same repo"""
        mock_getcwd.return_value = "/original/dir"

        pr_info = PRInfo(
            number=123,
            head_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
            head_branch="feature-branch",
            base_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
            base_branch="master",
            title="Feature branch",
            user="comfyanonymous",
            mergeable=True,
        )

        mock_subprocess.side_effect = [
            subprocess.CompletedProcess([], 0),  # fetch succeeds
            subprocess.CompletedProcess([], 0),  # checkout succeeds
        ]

        result = checkout_pr("/repo/path", pr_info)

        assert result is True
        assert mock_subprocess.call_count == 2

    @patch("subprocess.run")
    @patch("os.chdir")
    @patch("os.getcwd")
    def test_checkout_pr_git_failure(self, mock_getcwd, mock_chdir, mock_subprocess, sample_pr_info):
        """Test Git operation failure"""
        mock_getcwd.return_value = "/original/dir"

        error = subprocess.CalledProcessError(1, "git", stderr="Permission denied")
        mock_subprocess.side_effect = error

        result = checkout_pr("/repo/path", sample_pr_info)

        assert result is False


class TestHandlePRCheckout:
    """Test the main PR checkout handler"""

    @patch("comfy_cli.command.install.parse_pr_reference")
    @patch("comfy_cli.command.install.fetch_pr_info")
    @patch("comfy_cli.command.install.checkout_pr")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.ui.prompt_confirm_action")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_handle_pr_checkout_success(
        self,
        mock_makedirs,
        mock_exists,
        mock_confirm,
        mock_clone,
        mock_checkout,
        mock_fetch,
        mock_parse,
        sample_pr_info,
    ):
        """Test successful PR checkout handling"""
        mock_parse.return_value = ("jtydhr88", "ComfyUI", 123)
        mock_fetch.return_value = sample_pr_info
        mock_exists.side_effect = [True, False]  # Parent exists, repo doesn't
        mock_confirm.return_value = True
        mock_checkout.return_value = True

        with patch("comfy_cli.command.install.workspace_manager") as mock_ws:
            mock_ws.skip_prompting = False

            result = handle_pr_checkout("jtydhr88:load-3d-nodes", "/path/to/comfy")

            assert result == "https://github.com/comfyanonymous/ComfyUI.git"
            mock_clone.assert_called_once()
            mock_checkout.assert_called_once()


class TestCommandLineIntegration:
    """Test command line integration"""

    @patch("comfy_cli.command.install.execute")
    def test_install_with_pr_parameter(self, mock_execute, runner):
        """Test install command with --pr parameter"""
        result = runner.invoke(app, ["install", "--pr", "jtydhr88:load-3d-nodes", "--nvidia", "--skip-prompt"])

        assert "Invalid PR reference format" not in result.stdout

        if mock_execute.called:
            call_args = mock_execute.call_args
            assert "pr" in call_args.kwargs or len(call_args.args) > 8

    def test_pr_and_version_conflict(self, runner):
        """Test that --pr conflicts with --version"""
        result = runner.invoke(app, ["install", "--pr", "#123", "--version", "1.0.0"])

        assert result.exit_code != 0

    def test_pr_and_commit_conflict(self, runner):
        """Test that --pr conflicts with --commit"""
        result = runner.invoke(app, ["install", "--pr", "#123", "--version", "nightly", "--commit", "abc123"])

        assert result.exit_code != 0


class TestPRInfoDataClass:
    """Test PRInfo data class"""

    def test_pr_info_is_fork_true(self):
        """Test is_fork property returns True for fork"""
        pr_info = PRInfo(
            number=123,
            head_repo_url="https://github.com/user/ComfyUI.git",
            head_branch="branch",
            base_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
            base_branch="master",
            title="Title",
            user="user",
            mergeable=True,
        )
        assert pr_info.is_fork is True

    def test_pr_info_is_fork_false(self):
        """Test is_fork property returns False for same repo"""
        pr_info = PRInfo(
            number=123,
            head_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
            head_branch="feature",
            base_repo_url="https://github.com/comfyanonymous/ComfyUI.git",
            base_branch="master",
            title="Title",
            user="comfyanonymous",
            mergeable=True,
        )
        assert pr_info.is_fork is False


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_parse_pr_reference_whitespace(self):
        """Test parsing with whitespace"""
        repo_owner, repo_name, pr_number = parse_pr_reference("  #123  ")
        assert repo_owner == "comfyanonymous"
        assert repo_name == "ComfyUI"
        assert pr_number == 123

    @patch("requests.get")
    def test_fetch_pr_info_with_github_token(self, mock_get):
        """Test PR fetching with GitHub token"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 123,
            "title": "Test",
            "head": {"repo": {"clone_url": "url", "owner": {"login": "user"}}, "ref": "branch"},
            "base": {"repo": {"clone_url": "base_url"}, "ref": "master"},
            "mergeable": True,
        }
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"GITHUB_TOKEN": "test-token"}):
            fetch_pr_info("owner", "repo", 123)

            call_args = mock_get.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-token"

    @patch("subprocess.run")
    @patch("os.chdir")
    @patch("os.getcwd")
    def test_checkout_pr_remote_already_exists(self, mock_getcwd, mock_chdir, mock_subprocess, sample_pr_info):
        """Test checkout when remote already exists"""
        mock_getcwd.return_value = "/dir"

        mock_subprocess.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]

        result = checkout_pr("/repo", sample_pr_info)

        assert result is True
        assert mock_subprocess.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
