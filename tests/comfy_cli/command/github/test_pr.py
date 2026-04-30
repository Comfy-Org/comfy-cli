import subprocess
import sys
from unittest.mock import Mock, patch

import pytest
import requests
from typer.testing import CliRunner

from comfy_cli.cmdline import app, g_exclusivity, g_gpu_exclusivity
from comfy_cli.command import install as install_module
from comfy_cli.command.install import (
    GitHubRateLimitError,
    PRInfo,
    _parse_github_owner_repo,
    _resolve_latest_tag_from_local,
    checkout_stable_comfyui,
    fetch_pr_info,
    find_pr_by_branch,
    get_latest_release,
    handle_github_rate_limit,
    handle_pr_checkout,
    parse_pr_reference,
)
from comfy_cli.git_utils import checkout_pr, git_checkout_tag


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


class TestGitCheckoutTag:
    """Cover ``git_checkout_tag``'s skip-fetch-when-tag-is-local behavior.

    The fetch is intentionally avoided when the tag already exists in the
    local clone, both to skip a redundant network round-trip on the happy
    path and to let offline installs succeed when the caller (e.g. the
    `--version latest` resolver) already validated a cached tag.
    """

    @staticmethod
    def _init_repo(path):
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "x@x"], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "x"], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "--allow-empty", "-m", "init", "-q"],
            check=True,
        )

    def test_succeeds_offline_when_tag_already_local(self, tmp_path):
        """The bug: cached-tag offline path must not crash on the redundant fetch.

        Repro: tag exists locally + origin is unreachable. Old code would call
        `git fetch --tags` with check=True and fail; new code skips the fetch
        because the tag is already present.
        """
        self._init_repo(tmp_path)
        subprocess.run(["git", "-C", str(tmp_path), "tag", "v0.20.1"], check=True)
        # Point origin at an unreachable path so any fetch attempt would fail.
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "file:///nonexistent-repo-path-for-test"],
            check=True,
        )

        result = git_checkout_tag(str(tmp_path), "v0.20.1")
        assert result is True

        # HEAD really moved to the tag
        head = subprocess.run(
            ["git", "-C", str(tmp_path), "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert head.stdout.strip() == "v0.20.1"

    def test_fetches_when_tag_missing_locally(self, tmp_path):
        """When the tag isn't local we must still fetch — and an unreachable
        remote is then a real, surfaced error (not silently swallowed)."""
        self._init_repo(tmp_path)
        # Tag is NOT created locally
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "file:///nonexistent-repo-path-for-test"],
            check=True,
        )

        result = git_checkout_tag(str(tmp_path), "v0.20.1")
        assert result is False  # fetch failed, surfaced as a checkout failure


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

    @patch("comfy_cli.command.install.execute")
    @patch("comfy_cli.cmdline.check_comfy_repo", return_value=(False, None))
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.tracking.prompt_tracking_consent")
    def test_commit_without_pr_does_not_conflict(self, mock_track, mock_ws, mock_check, mock_execute, runner):
        """Test that --commit alone does not trigger --pr conflict error (issue #335)"""
        mock_ws.get_workspace_path.return_value = ("/tmp/test", None)
        result = runner.invoke(
            app, ["--skip-prompt", "install", "--version", "nightly", "--commit", "abc123", "--nvidia"]
        )

        assert "--pr cannot be used" not in result.stdout
        assert mock_execute.called

    @patch("comfy_cli.command.install.execute")
    @patch("comfy_cli.cmdline.check_comfy_repo", return_value=(False, None))
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.tracking.prompt_tracking_consent")
    def test_cpu_pr_conflict_with_version(self, mock_track, mock_ws, mock_check, mock_execute, runner):
        """Test that --cpu --pr with --version is rejected"""
        mock_ws.get_workspace_path.return_value = ("/tmp/test", None)
        result = runner.invoke(app, ["--skip-prompt", "install", "--cpu", "--pr", "#123", "--version", "1.0.0"])

        assert result.exit_code != 0
        assert "--pr cannot be used" in result.stdout
        assert not mock_execute.called

    @patch("comfy_cli.command.install.execute")
    @patch("comfy_cli.cmdline.check_comfy_repo", return_value=(False, None))
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.tracking.prompt_tracking_consent")
    def test_cpu_pr_conflict_with_commit(self, mock_track, mock_ws, mock_check, mock_execute, runner):
        """Test that --cpu --pr with --commit is rejected"""
        mock_ws.get_workspace_path.return_value = ("/tmp/test", None)
        result = runner.invoke(
            app, ["--skip-prompt", "install", "--cpu", "--pr", "#123", "--version", "nightly", "--commit", "abc123"]
        )

        assert result.exit_code != 0
        assert "--pr cannot be used" in result.stdout
        assert not mock_execute.called

    @patch("comfy_cli.command.install.execute")
    @patch("comfy_cli.cmdline.check_comfy_repo", return_value=(False, None))
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.tracking.prompt_tracking_consent")
    def test_cpu_pr_passes_pr_to_execute(self, mock_track, mock_ws, mock_check, mock_execute, runner):
        """Test that --cpu --pr passes pr parameter to install_inner.execute"""
        mock_ws.get_workspace_path.return_value = ("/tmp/test", None)
        runner.invoke(app, ["--skip-prompt", "install", "--cpu", "--pr", "#123"])

        assert mock_execute.called
        call_kwargs = mock_execute.call_args.kwargs
        assert call_kwargs.get("pr") == "#123"


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


class TestGetLatestRelease:
    """Test get_latest_release GitHub API calls"""

    @patch("requests.get")
    def test_sends_auth_header_when_token_set(self, mock_get):
        """Ensure GITHUB_TOKEN is sent as Bearer auth to avoid rate limits (issue #425)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v0.18.2",
            "zipball_url": "https://github.com/comfyanonymous/ComfyUI/archive/v0.18.2.zip",
        }
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            result = get_latest_release("comfyanonymous", "ComfyUI")

        headers = mock_get.call_args.kwargs.get("headers", {})
        assert headers["Authorization"] == "Bearer ghp_test123"
        assert result is not None
        assert result["tag"] == "v0.18.2"

    @patch("requests.get")
    def test_no_auth_header_without_token(self, mock_get):
        """Without GITHUB_TOKEN the request has no Authorization header"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v0.18.2",
            "zipball_url": "https://github.com/comfyanonymous/ComfyUI/archive/v0.18.2.zip",
        }
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            get_latest_release("comfyanonymous", "ComfyUI")

        headers = mock_get.call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers

    @patch("requests.get")
    def test_rate_limit_raises_error(self, mock_get):
        """A 403 with exhausted rate limit raises GitHubRateLimitError"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}
        mock_get.return_value = mock_response

        with pytest.raises(GitHubRateLimitError):
            get_latest_release("comfyanonymous", "ComfyUI")

    @patch("requests.get")
    def test_non_semver_tag_returns_release_with_version_none(self, mock_get):
        """Forks may use non-semver tags (e.g. `release-2026-04`); the parser
        must not crash — caller only needs the raw tag string for checkout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "release-2026-04",
            "zipball_url": "https://example/zip",
        }
        mock_get.return_value = mock_response

        result = get_latest_release("some-fork", "ComfyUI")

        assert result is not None
        assert result["tag"] == "release-2026-04"
        assert result["version"] is None


class TestHandleGithubRateLimit:
    def test_primary_rate_limit_message_format(self):
        """Verify the error message does not contain stray characters."""
        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}

        with pytest.raises(GitHubRateLimitError) as exc_info:
            handle_github_rate_limit(mock_response)

        msg = str(exc_info.value)
        assert "1700000000" in msg
        assert msg.endswith("1700000000")  # no stray trailing characters

    def test_retry_after_header(self):
        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-remaining": "5", "retry-after": "30"}

        with pytest.raises(GitHubRateLimitError, match="30 seconds"):
            handle_github_rate_limit(mock_response)

    def test_no_rate_limit_does_not_raise(self):
        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-remaining": "100"}

        handle_github_rate_limit(mock_response)  # should not raise


class TestResolveLatestTagFromLocal:
    """Cover the local-tag resolver added for issue #440 — `--version latest`
    must not require a GitHub API hit when tags are already on disk."""

    @staticmethod
    def _init_repo(path):
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "x@x"], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "x"], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "--allow-empty", "-m", "init", "-q"],
            check=True,
        )

    @classmethod
    def _make_repo(cls, path, tags):
        cls._init_repo(path)
        for tag in tags:
            subprocess.run(["git", "-C", str(path), "tag", tag], check=True)

    def test_picks_highest_stable_semver(self, tmp_path):
        self._make_repo(tmp_path, ["v0.19.5", "v0.20.0", "v0.20.1", "v0.18.2"])
        tag, _fetch_ok = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag == "v0.20.1"

    def test_skips_pre_release_tags(self, tmp_path):
        """GitHub's releases/latest excludes pre-releases; we mirror that."""
        self._make_repo(tmp_path, ["v0.20.0", "v0.20.1", "v0.21.0-rc1", "v0.21.0-beta.1"])
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag == "v0.20.1"

    def test_skips_non_semver_tags(self, tmp_path):
        self._make_repo(tmp_path, ["v0.20.1", "release-foo", "nightly", "weird/slash"])
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag == "v0.20.1"

    def test_returns_none_when_no_tags(self, tmp_path):
        self._init_repo(tmp_path)
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag is None

    def test_returns_none_when_only_prereleases(self, tmp_path):
        self._make_repo(tmp_path, ["v1.0.0-rc1", "v1.0.0-beta"])
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag is None

    def test_returns_none_when_only_non_semver(self, tmp_path):
        self._make_repo(tmp_path, ["main", "release-foo", "nightly"])
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag is None

    def test_returns_none_for_non_git_directory(self, tmp_path):
        tag, fetch_ok = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag is None
        assert fetch_ok is False

    def test_tolerates_fetch_exception(self, tmp_path):
        """Fetch may raise (timeout, OSError) — resolver should still use local tags."""
        self._make_repo(tmp_path, ["v0.20.1"])

        real_run = subprocess.run

        def flaky(args, **kwargs):
            if len(args) >= 4 and args[3] == "fetch":
                raise subprocess.SubprocessError("simulated network failure")
            return real_run(args, **kwargs)

        with patch("comfy_cli.command.install.subprocess.run", side_effect=flaky):
            tag, fetch_ok = _resolve_latest_tag_from_local(str(tmp_path))

        assert tag == "v0.20.1"
        assert fetch_ok is False

    def test_tolerates_fetch_nonzero_exit(self, tmp_path):
        """Fetch may exit non-zero without raising (auth, network, bad remote).

        Without ``check=True`` subprocess.run silently returns a non-zero
        CompletedProcess. The resolver should still produce tags from disk
        and report ``fetch_ok=False`` so the caller can warn the user.
        """
        self._make_repo(tmp_path, ["v0.20.0", "v0.20.1"])
        # Point origin at a path that doesn't exist → fetch exits 128 without raising in Python
        subprocess.run(
            ["git", "-C", str(tmp_path), "remote", "add", "origin", "file:///nonexistent-repo-path-for-test"],
            check=True,
        )

        tag, fetch_ok = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag == "v0.20.1"
        assert fetch_ok is False

    def test_tag_with_v_prefix_normalized(self, tmp_path):
        """Tags may be present with or without the leading 'v'; the higher stable wins."""
        self._make_repo(tmp_path, ["v0.20.0", "0.20.1"])
        tag, _ = _resolve_latest_tag_from_local(str(tmp_path))
        assert tag == "0.20.1"


class TestParseGithubOwnerRepo:
    """Cover the URL parser used by the API fallback to query the same repo
    we cloned from (forks included), instead of always asking upstream."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            # The default URL the install command uses
            ("https://github.com/comfyanonymous/ComfyUI", ("comfyanonymous", "ComfyUI")),
            # With .git suffix
            ("https://github.com/comfyanonymous/ComfyUI.git", ("comfyanonymous", "ComfyUI")),
            # With trailing slash
            ("https://github.com/comfyanonymous/ComfyUI/", ("comfyanonymous", "ComfyUI")),
            # setuptools-style @branch suffix that clone_comfyui supports
            ("https://github.com/comfyanonymous/ComfyUI@master", ("comfyanonymous", "ComfyUI")),
            ("https://github.com/comfyanonymous/ComfyUI.git@release/1.0", ("comfyanonymous", "ComfyUI")),
            # Forks
            ("https://github.com/myfork/ComfyUI", ("myfork", "ComfyUI")),
            ("https://github.com/some-user/some-repo.git", ("some-user", "some-repo")),
            # SSH forms
            ("git@github.com:comfyanonymous/ComfyUI", ("comfyanonymous", "ComfyUI")),
            ("git@github.com:comfyanonymous/ComfyUI.git", ("comfyanonymous", "ComfyUI")),
        ],
    )
    def test_parses_github_urls(self, url, expected):
        assert _parse_github_owner_repo(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "/local/path/to/comfyui",  # local path
            "https://gitlab.com/foo/bar",  # not GitHub
            "https://example.com/owner/repo",  # not GitHub
            "https://github.com/owner/repo/pull/123",  # not a repo URL
            "ftp://github.com/owner/repo",  # exotic scheme — still parses since regex matches `github.com/...`
        ],
    )
    def test_returns_none_for_non_github_urls(self, url):
        # The PR URL form (last case) intentionally doesn't match — `[^/@]+?` excludes `/`
        # so `repo/pull/123` cannot be the second capture; we want this to fall through
        # to the upstream default in the caller.
        if url == "ftp://github.com/owner/repo":
            # Edge-case: this DOES match because we don't anchor on the scheme.
            # That's fine — owner/repo is what matters; the API call uses HTTPS regardless.
            assert _parse_github_owner_repo(url) == ("owner", "repo")
        else:
            assert _parse_github_owner_repo(url) is None


class TestCheckoutStableComfyUI:
    """Verify checkout_stable_comfyui prefers local tag resolution over the
    GitHub API for `--version latest` (issue #440), and falls back when local
    resolution fails."""

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=("v0.20.1", True))
    def test_latest_uses_local_tag_no_api_call(self, mock_local, mock_api, mock_co):
        """When local tags resolve, the API is never consulted."""
        checkout_stable_comfyui("latest", "/repo")

        mock_local.assert_called_once_with("/repo")
        mock_api.assert_not_called()
        mock_co.assert_called_once_with("/repo", "v0.20.1")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=("v0.20.1", False))
    def test_latest_warns_on_stale_tag_when_fetch_failed(self, mock_local, mock_api, mock_co, capsys):
        """Fetch failed but a tag was found locally → warn the user it may be stale.

        Old behavior was to hard-fail via the API path; new behavior succeeds with
        whatever's on disk. Without this warning the user has no way to tell the
        clone is stale.
        """
        checkout_stable_comfyui("latest", "/repo")

        captured = capsys.readouterr()
        assert "could not refresh tags from remote" in captured.out
        assert "v0.20.1" in captured.out
        # Still uses the cached tag, no API call
        mock_api.assert_not_called()
        mock_co.assert_called_once_with("/repo", "v0.20.1")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=("v0.20.1", True))
    def test_latest_no_warning_when_fetch_succeeded(self, mock_local, mock_api, mock_co, capsys):
        """Happy path: fetch_ok=True → no stale-tag warning, quiet success."""
        checkout_stable_comfyui("latest", "/repo")

        captured = capsys.readouterr()
        assert "could not refresh tags" not in captured.out
        assert "querying GitHub API" not in captured.out

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_falls_back_to_api_when_local_empty(self, mock_local, mock_api, mock_co):
        """Fetch succeeded but the repo has no stable tags → API fallback runs."""
        mock_api.return_value = {"tag": "v0.20.1", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo")

        mock_local.assert_called_once_with("/repo")
        mock_api.assert_called_once_with("comfyanonymous", "ComfyUI")
        mock_co.assert_called_once_with("/repo", "v0.20.1")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_fallback_uses_fork_owner_repo_from_url(self, mock_local, mock_api, mock_co):
        """Fork case: API fallback must query the FORK's releases/latest, not upstream's.

        Otherwise we'd ask GitHub for `comfyanonymous/ComfyUI`'s latest tag and
        try to check it out in a fork that may not have it.
        """
        mock_api.return_value = {"tag": "v0.20.1-myfork", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo", url="https://github.com/myfork/ComfyUI")

        mock_api.assert_called_once_with("myfork", "ComfyUI")
        mock_co.assert_called_once_with("/repo", "v0.20.1-myfork")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_fallback_strips_branch_suffix_from_url(self, mock_local, mock_api, mock_co):
        """The setuptools-style `@branch` suffix in the install URL must not leak
        into the API call. `clone_comfyui` already strips it before cloning."""
        mock_api.return_value = {"tag": "v0.20.1", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo", url="https://github.com/myfork/ComfyUI.git@some-branch")

        mock_api.assert_called_once_with("myfork", "ComfyUI")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_fallback_defaults_to_upstream_for_non_github_url(self, mock_local, mock_api, mock_co):
        """Non-GitHub URLs (local paths, GitLab, etc.) fall back to upstream defaults
        — preserves prior behavior for users whose URL we can't parse."""
        mock_api.return_value = {"tag": "v0.20.1", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo", url="/local/path/to/comfyui")

        mock_api.assert_called_once_with("comfyanonymous", "ComfyUI")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_fallback_defaults_to_upstream_when_url_omitted(self, mock_local, mock_api, mock_co):
        """Backward compat: omitting the new `url` kwarg yields the prior behavior
        (querying upstream)."""
        mock_api.return_value = {"tag": "v0.20.1", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo")  # no url=

        mock_api.assert_called_once_with("comfyanonymous", "ComfyUI")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, False))
    def test_latest_warns_when_fetch_failed_before_api_fallback(self, mock_local, mock_api, mock_co, capsys):
        """When fetch failed AND local has no tags, surface the fetch failure
        so the user understands why we're falling back to the API."""
        mock_api.return_value = {"tag": "v0.20.1", "version": None, "download_url": "u"}

        checkout_stable_comfyui("latest", "/repo")

        captured = capsys.readouterr()
        assert "Could not refresh tags from the remote" in captured.out
        # Sanity: didn't print the wrong (success-fetch) branch
        assert "No stable release tags found locally" not in captured.out

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release", return_value=None)
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local", return_value=(None, True))
    def test_latest_exits_when_both_local_and_api_fail(self, mock_local, mock_api, mock_co):
        with pytest.raises(SystemExit):
            checkout_stable_comfyui("latest", "/repo")
        mock_co.assert_not_called()

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local")
    def test_specific_version_skips_both_local_and_api(self, mock_local, mock_api, mock_co):
        """`--version 0.20.1` must not consult the API or the local resolver."""
        checkout_stable_comfyui("0.20.1", "/repo")

        mock_local.assert_not_called()
        mock_api.assert_not_called()
        mock_co.assert_called_once_with("/repo", "v0.20.1")

    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    @patch("comfy_cli.command.install.get_latest_release")
    @patch("comfy_cli.command.install._resolve_latest_tag_from_local")
    def test_specific_version_with_v_prefix_passes_through(self, mock_local, mock_api, mock_co):
        checkout_stable_comfyui("v0.20.1", "/repo")

        mock_local.assert_not_called()
        mock_api.assert_not_called()
        mock_co.assert_called_once_with("/repo", "v0.20.1")

    @patch("comfy_cli.command.install.requests.get")
    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    def test_latest_with_rate_limited_api_when_no_local_tags(self, mock_co, mock_get, tmp_path):
        """End-to-end repro of issue #440: empty local clone + 60/hr exhausted IP.

        With no local tags, the resolver returns None and the API path runs;
        a 403 there must surface as GitHubRateLimitError exactly as before.
        """
        # Real but tag-less git repo
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

        rate_limited = Mock()
        rate_limited.status_code = 403
        rate_limited.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1777415867"}
        mock_get.return_value = rate_limited

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(GitHubRateLimitError, match="1777415867"):
                checkout_stable_comfyui("latest", str(tmp_path))

        mock_co.assert_not_called()

    @patch("comfy_cli.command.install.requests.get")
    @patch("comfy_cli.command.install.git_checkout_tag", return_value=True)
    def test_latest_with_local_tags_no_network_at_all(self, mock_co, mock_get, tmp_path):
        """The pre-fix repro of issue #440: with local tags present, no
        GitHub API call should be made even when the network is hostile."""
        TestResolveLatestTagFromLocal._make_repo(tmp_path, ["v0.19.5", "v0.20.0", "v0.20.1"])

        with patch.dict("os.environ", {}, clear=True):
            checkout_stable_comfyui("latest", str(tmp_path))

        # Resolved locally; never touched the API
        assert mock_get.call_count == 0
        mock_co.assert_called_once_with(str(tmp_path), "v0.20.1")


class TestInstallExecuteWithLatest:
    """Integration test for the FULL `install.execute()` flow with `--version latest`.

    Uses a real (synthetic) git repo on disk so `clone_comfyui`,
    `_resolve_latest_tag_from_local`, and `git_checkout_tag` actually run.
    The slow pip / venv steps are mocked. Most importantly, ``requests.get``
    inside ``install`` is wired to **raise** if invoked — so any future
    refactor that puts a GitHub API call back on the happy path of
    ``--version latest`` will fail this test loudly.

    This is the regression net the unit tests can't provide: it proves
    the clone-then-resolve-then-checkout ordering survives changes to
    ``execute()``.
    """

    @staticmethod
    def _make_comfy_repo(path):
        """Build a tag-bearing git repo at `path` that mimics ComfyUI's pattern.

        Each tag points at its own commit so ``git describe --exact-match HEAD``
        is unambiguous after checkout.
        """
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "x@x"], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "x"], check=True)
        for tag in ["v0.18.2", "v0.19.5", "v0.20.0", "v0.20.1", "v0.21.0-rc1"]:
            subprocess.run(
                ["git", "-C", str(path), "commit", "--allow-empty", "-m", f"release {tag}", "-q"],
                check=True,
            )
            subprocess.run(["git", "-C", str(path), "tag", tag], check=True)

    def test_full_execute_resolves_latest_locally_no_api_call(self, tmp_path, capsys):
        repo_dir = tmp_path / "ComfyUI"
        self._make_comfy_repo(repo_dir)

        api_calls = []

        def crash_on_api(*args, **kwargs):
            api_calls.append(("requests.get", args, kwargs))
            raise AssertionError(
                "Regression: install.execute('--version latest') made an unexpected "
                f"GitHub API call: args={args}, kwargs={kwargs}"
            )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("comfy_cli.command.install.requests.get", side_effect=crash_on_api),
            patch("comfy_cli.command.install.clone_comfyui") as mock_clone,
            patch("comfy_cli.command.install.ensure_workspace_python", return_value=sys.executable),
            patch("comfy_cli.command.install.pip_install_comfyui_dependencies"),
            patch("comfy_cli.command.install.update_node_id_cache"),
            patch.object(install_module.workspace_manager, "skip_prompting", True),
            patch.object(install_module.workspace_manager, "setup_workspace_manager"),
            patch("comfy_cli.command.install.WorkspaceManager") as mock_ws_class,
            patch("comfy_cli.config_manager.ConfigManager") as mock_cfg_class,
        ):
            mock_ws_class.return_value = Mock()
            mock_cfg_class.return_value = Mock()

            install_module.execute(
                url="https://github.com/comfyanonymous/ComfyUI",
                comfy_path=str(repo_dir),
                restore=False,
                skip_manager=True,
                version="latest",
            )

        # The core regression assertions:
        assert api_calls == [], "GitHub API was called on the --version latest happy path"
        mock_clone.assert_not_called()  # repo already exists at comfy_path

        # The right tag actually got checked out by the real git_checkout_tag call
        head = subprocess.run(
            ["git", "-C", str(repo_dir), "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert head.stdout.strip() == "v0.20.1", (
            f"Expected HEAD at v0.20.1 (highest stable tag), got: {head.stdout.strip()!r}"
        )

    def test_full_execute_with_specific_version_no_api_no_resolver(self, tmp_path):
        """`--version 0.20.0` must take the direct-tag path, not the resolver."""
        repo_dir = tmp_path / "ComfyUI"
        self._make_comfy_repo(repo_dir)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "comfy_cli.command.install.requests.get",
                side_effect=AssertionError("API must not be called for specific versions"),
            ),
            patch(
                "comfy_cli.command.install._resolve_latest_tag_from_local",
                side_effect=AssertionError("Local resolver must not be called for specific versions"),
            ),
            patch("comfy_cli.command.install.clone_comfyui"),
            patch("comfy_cli.command.install.ensure_workspace_python", return_value=sys.executable),
            patch("comfy_cli.command.install.pip_install_comfyui_dependencies"),
            patch("comfy_cli.command.install.update_node_id_cache"),
            patch.object(install_module.workspace_manager, "skip_prompting", True),
            patch.object(install_module.workspace_manager, "setup_workspace_manager"),
            patch("comfy_cli.command.install.WorkspaceManager") as mock_ws_class,
            patch("comfy_cli.config_manager.ConfigManager") as mock_cfg_class,
        ):
            mock_ws_class.return_value = Mock()
            mock_cfg_class.return_value = Mock()

            install_module.execute(
                url="https://github.com/comfyanonymous/ComfyUI",
                comfy_path=str(repo_dir),
                restore=False,
                skip_manager=True,
                version="0.20.0",
            )

        head = subprocess.run(
            ["git", "-C", str(repo_dir), "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert head.stdout.strip() == "v0.20.0"


if __name__ == "__main__":
    pytest.main([__file__])
