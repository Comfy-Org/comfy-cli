"""Tests for launch-time frontend PR functionality"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.install import PRInfo, handle_temporary_frontend_pr
from comfy_cli.pr_cache import PRCache


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_tracking_consent():
    with patch("comfy_cli.tracking.prompt_tracking_consent"):
        yield


@pytest.fixture
def sample_frontend_pr_info():
    return PRInfo(
        number=789,
        head_repo_url="https://github.com/testuser/ComfyUI_frontend.git",
        head_branch="test-feature",
        base_repo_url="https://github.com/Comfy-Org/ComfyUI_frontend.git",
        base_branch="main",
        title="Test feature for frontend",
        user="testuser",
        mergeable=True,
    )


@pytest.fixture
def mock_pr_cache():
    with patch("comfy_cli.pr_cache.PRCache") as mock_cache_cls:
        mock_cache = Mock()
        mock_cache_cls.return_value = mock_cache
        yield mock_cache


class TestLaunchWithFrontendPR:
    """Test launching with temporary frontend PR"""

    @patch("comfy_cli.command.install.verify_node_tools")
    def test_launch_frontend_pr_without_node(self, mock_verify):
        """Test launch with frontend PR when Node.js is missing"""
        mock_verify.return_value = False

        result = handle_temporary_frontend_pr("#123")
        assert result is None
        mock_verify.assert_called_once()

    @patch("comfy_cli.command.install.verify_node_tools")
    @patch("comfy_cli.command.install.parse_frontend_pr_reference")
    @patch("comfy_cli.command.install.fetch_pr_info")
    def test_launch_frontend_pr_with_cache_hit(
        self, mock_fetch, mock_parse, mock_verify, mock_pr_cache, sample_frontend_pr_info
    ):
        """Test launch with cached frontend PR"""
        mock_verify.return_value = True
        mock_parse.return_value = ("Comfy-Org", "ComfyUI_frontend", 789)
        mock_fetch.return_value = sample_frontend_pr_info

        # Mock cache hit
        cached_path = Path("/cache/frontend/pr-789/dist")
        mock_pr_cache.get_cached_frontend_path.return_value = cached_path

        result = handle_temporary_frontend_pr("#789")

        assert result == str(cached_path)
        mock_pr_cache.get_cached_frontend_path.assert_called_once()
        # Should not build if cache hit
        mock_pr_cache.save_cache_info.assert_not_called()

    @patch("pathlib.Path.mkdir")
    @patch("os.chdir")
    @patch("subprocess.run")
    @patch("comfy_cli.command.install.checkout_pr")
    @patch("comfy_cli.command.install.clone_comfyui")
    @patch("comfy_cli.command.install.verify_node_tools")
    @patch("comfy_cli.command.install.parse_frontend_pr_reference")
    @patch("comfy_cli.command.install.fetch_pr_info")
    def test_launch_frontend_pr_cache_miss_builds(
        self,
        mock_fetch,
        mock_parse,
        mock_verify,
        mock_clone,
        mock_checkout,
        mock_run,
        mock_chdir,
        mock_mkdir,
        mock_pr_cache,
        sample_frontend_pr_info,
    ):
        """Test launch builds frontend when not cached"""
        mock_verify.return_value = True
        mock_parse.return_value = ("Comfy-Org", "ComfyUI_frontend", 789)
        mock_fetch.return_value = sample_frontend_pr_info
        mock_checkout.return_value = True

        # Mock cache miss
        mock_pr_cache.get_cached_frontend_path.return_value = None
        cache_path = Path("/cache/frontend/pr-789")
        mock_pr_cache.get_frontend_cache_path.return_value = cache_path

        # Mock successful build
        mock_run.side_effect = [
            Mock(returncode=0),  # pnpm install
            Mock(returncode=0),  # vite build
        ]

        # Mock dist exists
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            result = handle_temporary_frontend_pr("#789")

        # Should return built path
        assert result == str(cache_path / "repo" / "dist")
        # Should save cache info
        mock_pr_cache.save_cache_info.assert_called_once_with(sample_frontend_pr_info, cache_path)


class TestPRCacheManagement:
    """Test PR cache functionality"""

    def test_pr_cache_get_frontend_path(self, sample_frontend_pr_info):
        """Test getting frontend cache path"""
        cache = PRCache()
        path = cache.get_frontend_cache_path(sample_frontend_pr_info)

        assert "frontend" in str(path)
        assert "testuser" in str(path)
        assert "789" in str(path)

    def test_pr_cache_list_empty(self):
        """Test listing empty cache"""
        cache = PRCache()
        with patch("pathlib.Path.exists", return_value=False):
            result = cache.list_cached_frontends()
        assert result == []

    def test_pr_cache_clean_specific(self, tmp_path):
        """Test cleaning specific PR cache"""
        cache = PRCache()
        cache.cache_dir = tmp_path / "test-cache"

        # Create mock cache structure
        frontend_cache = cache.cache_dir / "frontend" / "pr-123"
        frontend_cache.mkdir(parents=True)
        cache_info = frontend_cache / ".cache-info.json"
        cache_info.write_text('{"pr_number": 123}')

        # Clean specific PR
        cache.clean_frontend_cache(123)

        assert not frontend_cache.exists()

    def test_pr_cache_age_check(self, sample_frontend_pr_info, tmp_path):
        """Test cache age validation"""
        cache = PRCache()
        cache.cache_dir = tmp_path / "test-cache"
        cache_path = cache.get_frontend_cache_path(sample_frontend_pr_info)
        cache_path.mkdir(parents=True)

        # Create cache info with old timestamp
        old_time = datetime.now() - timedelta(days=10)
        cache_info = {
            "pr_number": sample_frontend_pr_info.number,
            "pr_title": sample_frontend_pr_info.title,
            "user": sample_frontend_pr_info.user,
            "head_branch": sample_frontend_pr_info.head_branch,
            "cached_at": old_time.isoformat(),
        }

        info_path = cache.get_cache_info_path(cache_path)
        with open(info_path, "w") as f:
            json.dump(cache_info, f)

        # Should be invalid due to age
        assert not cache.is_cache_valid(sample_frontend_pr_info, cache_path)

    def test_pr_cache_enforce_limits(self, tmp_path):
        """Test cache limit enforcement"""
        cache = PRCache()
        cache.cache_dir = tmp_path / "test-cache"
        cache.max_cache_items = 3  # Set low limit for testing

        # Create multiple cache entries
        for i in range(5):
            cache_dir = cache.cache_dir / "frontend" / f"pr-{i}"
            cache_dir.mkdir(parents=True)
            cache_info = {
                "pr_number": i,
                "pr_title": f"Test PR {i}",
                "cached_at": (datetime.now() - timedelta(hours=i)).isoformat(),
            }
            with open(cache_dir / ".cache-info.json", "w") as f:
                json.dump(cache_info, f)

        # Enforce limits
        cache.enforce_cache_limits()

        # Should only have 3 newest items
        remaining = list((cache.cache_dir / "frontend").iterdir())
        assert len(remaining) == 3
        # Check that newest items remain
        remaining_numbers = sorted([int(d.name.split("-")[1]) for d in remaining])
        assert remaining_numbers == [0, 1, 2]  # Newest 3

    def test_get_cache_age(self):
        """Test human-readable cache age"""
        cache = PRCache()

        # Test various ages
        now = datetime.now()
        assert cache.get_cache_age(now.isoformat()) == "just now"

        age_5_min = (now - timedelta(minutes=5)).isoformat()
        assert "5 minutes ago" in cache.get_cache_age(age_5_min)

        age_2_hours = (now - timedelta(hours=2)).isoformat()
        assert "2 hours ago" in cache.get_cache_age(age_2_hours)

        age_3_days = (now - timedelta(days=3)).isoformat()
        assert "3 days ago" in cache.get_cache_age(age_3_days)


class TestPRCacheCommands:
    """Test PR cache CLI commands"""

    def test_pr_cache_list_command(self, runner):
        """Test pr-cache list command"""
        with patch("comfy_cli.command.pr_command.PRCache") as mock_cache_cls:
            mock_cache = Mock()
            mock_cache.list_cached_frontends.return_value = []
            mock_cache_cls.return_value = mock_cache

            result = runner.invoke(app, ["pr-cache", "list"])
            assert result.exit_code == 0
            assert "No cached PR builds found" in result.output

    def test_pr_cache_clean_command_with_confirmation(self, runner):
        """Test pr-cache clean command with confirmation"""
        with patch("comfy_cli.command.pr_command.PRCache") as mock_cache_cls:
            mock_cache = Mock()
            mock_cache.list_cached_frontends.return_value = [
                {"pr_number": 123, "pr_title": "Test PR"}  # Mock some cached items
            ]
            mock_cache_cls.return_value = mock_cache

            # Simulate user saying "no"
            result = runner.invoke(app, ["pr-cache", "clean"], input="n\n")
            assert result.exit_code == 0
            assert "Cancelled" in result.output
            mock_cache.clean_frontend_cache.assert_not_called()

    def test_pr_cache_clean_command_with_yes_flag(self, runner):
        """Test pr-cache clean command with --yes flag"""
        with patch("comfy_cli.command.pr_command.PRCache") as mock_cache_cls:
            mock_cache = Mock()
            mock_cache_cls.return_value = mock_cache

            result = runner.invoke(app, ["pr-cache", "clean", "--yes"])
            assert result.exit_code == 0
            mock_cache.clean_frontend_cache.assert_called_once_with()
