import os
from unittest.mock import MagicMock, patch

from comfy_cli.workspace_manager import WorkspaceType, _paths_match


class TestPathsMatch:
    def test_identical_paths(self, tmp_path):
        d = tmp_path / "comfy"
        d.mkdir()
        assert _paths_match(str(d), str(d))

    def test_symlink_to_same_dir(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert _paths_match(str(real), str(link))

    def test_different_paths(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert not _paths_match(str(a), str(b))

    def test_nonexistent_paths_same(self):
        assert _paths_match("/nonexistent/same", "/nonexistent/same")

    def test_nonexistent_paths_different(self):
        assert not _paths_match("/nonexistent/a", "/nonexistent/b")

    def test_trailing_slash(self, tmp_path):
        d = tmp_path / "comfy"
        d.mkdir()
        assert _paths_match(str(d), str(d) + "/")

    def test_dot_components(self, tmp_path):
        d = tmp_path / "comfy"
        d.mkdir()
        assert _paths_match(str(d), str(d) + "/./")

    def test_parent_component(self, tmp_path):
        d = tmp_path / "comfy"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        assert _paths_match(str(d), str(sub) + "/..")

    def test_one_exists_one_not(self, tmp_path):
        d = tmp_path / "exists"
        d.mkdir()
        # samefile will raise because the second path doesn't exist;
        # fallback compares realpath strings, which will differ
        assert not _paths_match(str(d), "/nonexistent/path")

    def test_double_symlink(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link1 = tmp_path / "link1"
        link1.symlink_to(real)
        link2 = tmp_path / "link2"
        link2.symlink_to(link1)
        assert _paths_match(str(link1), str(link2))
        assert _paths_match(str(real), str(link2))


def _make_manager(*, use_here=None, specified_workspace=None, use_recent=None):
    """Create a fresh WorkspaceManager with reset singleton."""
    from comfy_cli.workspace_manager import WorkspaceManager

    WorkspaceManager._instances = {}
    mgr = WorkspaceManager()
    mgr.use_here = use_here
    mgr.use_recent = use_recent
    mgr.specified_workspace = specified_workspace
    return mgr


def _mock_config(mgr, default_workspace=None, recent_workspace=None):
    """Replace config_manager with a mock that returns the given values."""
    mock_cm = MagicMock()

    def _get(key):
        from comfy_cli import constants

        if key == constants.CONFIG_KEY_DEFAULT_WORKSPACE:
            return default_workspace
        if key == constants.CONFIG_KEY_RECENT_WORKSPACE:
            return recent_workspace
        return None

    mock_cm.get.side_effect = _get
    mgr.config_manager = mock_cm
    return mock_cm


class TestStep1Workspace:
    def test_workspace_flag_takes_priority(self):
        mgr = _make_manager(specified_workspace="/opt/comfy")
        _mock_config(mgr)
        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.SPECIFIED
        assert path == "/opt/comfy"

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_workspace_overrides_cwd_matching_default(self, mock_getcwd, mock_check):
        """--workspace wins even when cwd is the default workspace."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(specified_workspace="/other/ComfyUI")
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.SPECIFIED
        assert path == "/other/ComfyUI"


class TestStep3Here:
    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_here_flag_forces_current_dir_even_if_matches_default(self, mock_getcwd, mock_check):
        """--here always returns CURRENT_DIR, even when cwd IS the default."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(use_here=True)
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.CURRENT_DIR
        assert path == "/home/user/comfy/ComfyUI"

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_here_flag_non_comfy_dir_appends_comfyui(self, mock_getcwd, mock_check):
        """--here in a non-ComfyUI dir returns cwd/ComfyUI."""
        mock_getcwd.return_value = "/home/user/projects"
        mock_check.return_value = (False, None)

        mgr = _make_manager(use_here=True)
        _mock_config(mgr)

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.CURRENT_DIR
        assert path == os.path.join("/home/user/projects", "ComfyUI")


class TestStep4AutoDetect:
    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_cwd_matches_default_returns_default_type(self, mock_getcwd, mock_check):
        """Core fix: cwd is the configured default workspace -> DEFAULT."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        with patch("comfy_cli.workspace_manager._paths_match", return_value=True):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT
        assert path == "/home/user/comfy/ComfyUI"

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_cwd_different_repo_returns_current_dir(self, mock_getcwd, mock_check):
        """cwd is a ComfyUI repo but NOT the default -> CURRENT_DIR."""
        mock_getcwd.return_value = "/home/user/other/ComfyUI"
        mock_check.return_value = (True, "/home/user/other/ComfyUI")

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        with patch("comfy_cli.workspace_manager._paths_match", return_value=False):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.CURRENT_DIR
        assert path == "/home/user/other/ComfyUI"

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_cwd_repo_no_default_configured(self, mock_getcwd, mock_check):
        """cwd is a ComfyUI repo, no default configured -> CURRENT_DIR."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=None)

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.CURRENT_DIR
        assert path == "/home/user/comfy/ComfyUI"

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_cwd_repo_empty_default_returns_current_dir(self, mock_getcwd, mock_check):
        """default_workspace is empty string -> treated as not configured."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace="")

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.CURRENT_DIR

    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_paths_match_called_with_correct_args(self, mock_getcwd, mock_check):
        """Verify _paths_match receives resolved path and default_workspace."""
        mock_getcwd.return_value = "/cwd"
        mock_check.return_value = (True, "/resolved/ComfyUI")

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace="/configured/default")

        with patch("comfy_cli.workspace_manager._paths_match", return_value=False) as mock_pm:
            mgr.get_workspace_path()

        mock_pm.assert_called_once_with("/resolved/ComfyUI", "/configured/default")


class TestNoHereSkipsStep4:
    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_no_here_skips_cwd_detection(self, mock_getcwd, mock_check):
        """--no-here (use_here=False) skips step 4 entirely, falls to step 5."""
        mock_getcwd.return_value = "/home/user/comfy/ComfyUI"
        mock_check.return_value = (True, "/home/user/comfy/ComfyUI")

        mgr = _make_manager(use_here=False)
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT
        assert path == "/home/user/comfy/ComfyUI"
        # getcwd should never be called because step 4 is skipped
        mock_getcwd.assert_not_called()


class TestStep5ConfiguredDefault:
    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_not_comfy_repo_falls_through_to_default(self, mock_getcwd, mock_check):
        """cwd is NOT a ComfyUI repo -> falls through to configured default."""
        mock_getcwd.return_value = "/home/user/projects"
        mock_check.side_effect = lambda path: (
            (True, "/home/user/comfy/ComfyUI") if path == "/home/user/comfy/ComfyUI" else (False, None)
        )

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace="/home/user/comfy/ComfyUI")

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.DEFAULT
        assert path == "/home/user/comfy/ComfyUI"


class TestStep6RecentFallback:
    @patch("comfy_cli.workspace_manager.check_comfy_repo")
    @patch("comfy_cli.workspace_manager.os.getcwd")
    def test_no_default_falls_to_recent(self, mock_getcwd, mock_check):
        """No default configured, valid recent workspace -> RECENT."""
        mock_getcwd.return_value = "/home/user/projects"
        mock_check.side_effect = lambda path: (
            (True, "/home/user/recent/ComfyUI") if path == "/home/user/recent/ComfyUI" else (False, None)
        )

        mgr = _make_manager(use_here=None, use_recent=None)
        _mock_config(
            mgr,
            default_workspace=None,
            recent_workspace="/home/user/recent/ComfyUI",
        )

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.RECENT
        assert path == "/home/user/recent/ComfyUI"


class TestStep7FallbackDefault:
    @patch("comfy_cli.workspace_manager.utils.get_not_user_set_default_workspace")
    @patch("comfy_cli.workspace_manager.check_comfy_repo", return_value=(False, None))
    @patch("comfy_cli.workspace_manager.os.getcwd", return_value="/tmp/random")
    def test_all_fallbacks_exhausted(self, _cwd, _check, mock_fallback):
        """Nothing configured, cwd not a repo -> system fallback DEFAULT."""
        mock_fallback.return_value = "/home/user/comfy"
        mgr = _make_manager(use_here=None, use_recent=None)
        _mock_config(mgr, default_workspace=None, recent_workspace=None)

        path, ws_type = mgr.get_workspace_path()
        assert ws_type == WorkspaceType.DEFAULT
        assert path == "/home/user/comfy"


class TestFullIntegration:
    """Create a real git repo that looks like ComfyUI (with the right remote)
    and exercise the entire get_workspace_path flow with no mocked internals.
    Only os.getcwd and ConfigManager are faked."""

    @staticmethod
    def _create_comfy_repo(path):
        """Create a bare-minimum git repo with a ComfyUI remote."""
        import git as gitmodule

        repo = gitmodule.Repo.init(path)
        repo.create_remote("origin", "https://github.com/comfyanonymous/ComfyUI")
        # Need at least one commit for repo to be fully valid
        readme = os.path.join(path, "main.py")
        with open(readme, "w") as f:
            f.write("# ComfyUI\n")
        repo.index.add(["main.py"])
        repo.index.commit("init")
        return repo

    def test_cwd_is_default_workspace_real_repo(self, tmp_path):
        """Bug repro: cd into default workspace -> must return DEFAULT."""
        comfy_dir = str(tmp_path / "ComfyUI")
        self._create_comfy_repo(comfy_dir)

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=comfy_dir)

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=comfy_dir):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT
        assert path == comfy_dir

    def test_cwd_is_default_workspace_via_symlink(self, tmp_path):
        """Default stored as symlink, cwd is the real path -> DEFAULT."""
        real_dir = tmp_path / "real_comfy"
        self._create_comfy_repo(str(real_dir))
        link = tmp_path / "comfy_link"
        link.symlink_to(real_dir)

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=str(link))

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=str(real_dir)):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT

    def test_cwd_is_subdir_of_default_workspace(self, tmp_path):
        """cd into custom_nodes/ inside default workspace -> DEFAULT."""
        comfy_dir = tmp_path / "ComfyUI"
        self._create_comfy_repo(str(comfy_dir))
        subdir = comfy_dir / "custom_nodes"
        subdir.mkdir()

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=str(comfy_dir))

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=str(subdir)):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT
        assert path == str(comfy_dir)

    def test_two_repos_cwd_in_non_default(self, tmp_path):
        """Two ComfyUI repos exist; cwd is in the non-default one -> CURRENT_DIR."""
        default_dir = tmp_path / "default_comfy"
        other_dir = tmp_path / "other_comfy"
        self._create_comfy_repo(str(default_dir))
        self._create_comfy_repo(str(other_dir))

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=str(default_dir))

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=str(other_dir)):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.CURRENT_DIR
        assert path == str(other_dir)

    def test_default_workspace_trailing_slash(self, tmp_path):
        """Config has trailing slash, git working_dir doesn't -> DEFAULT."""
        comfy_dir = str(tmp_path / "ComfyUI")
        self._create_comfy_repo(comfy_dir)

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=comfy_dir + "/")

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=comfy_dir):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT

    def test_here_flag_overrides_even_with_real_repo(self, tmp_path):
        """--here forces CURRENT_DIR even in a real default workspace repo."""
        comfy_dir = str(tmp_path / "ComfyUI")
        self._create_comfy_repo(comfy_dir)

        mgr = _make_manager(use_here=True)
        _mock_config(mgr, default_workspace=comfy_dir)

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=comfy_dir):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.CURRENT_DIR

    def test_not_in_any_repo_falls_to_configured_default(self, tmp_path):
        """cwd is a plain dir, configured default exists -> DEFAULT via step 5."""
        comfy_dir = str(tmp_path / "ComfyUI")
        self._create_comfy_repo(comfy_dir)
        plain_dir = str(tmp_path / "plain")
        os.makedirs(plain_dir)

        mgr = _make_manager(use_here=None)
        _mock_config(mgr, default_workspace=comfy_dir)

        with patch("comfy_cli.workspace_manager.os.getcwd", return_value=plain_dir):
            path, ws_type = mgr.get_workspace_path()

        assert ws_type == WorkspaceType.DEFAULT
        assert path == comfy_dir
