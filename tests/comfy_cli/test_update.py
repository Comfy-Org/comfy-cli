import json
import time
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock, patch

import pytest
import requests

from comfy_cli import update
from comfy_cli.update import (
    UPDATE_CHECK_DISABLE_ENV,
    check_for_newer_pypi_version,
    latest_upgrade_version,
    upgrade_cli,
)


def _mock_pypi_response(latest_version):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"info": {"version": latest_version}}
    return mock_resp


class TestCheckForNewerPypiVersion:
    @patch("comfy_cli.update.requests.get")
    def test_newer_version_available(self, mock_get):
        mock_get.return_value = _mock_pypi_response("99.0.0")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is True
        assert ver == "99.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_no_update_when_current(self, mock_get):
        mock_get.return_value = _mock_pypi_response("1.0.0")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is False
        assert ver == "1.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_network_failure_returns_false(self, mock_get):
        mock_get.side_effect = requests.Timeout("connection timed out")
        has_newer, ver = check_for_newer_pypi_version("comfy-cli", "1.0.0")
        assert has_newer is False
        assert ver == "1.0.0"

    @patch("comfy_cli.update.requests.get")
    def test_timeout_value_is_passed(self, mock_get):
        mock_get.return_value = _mock_pypi_response("1.0.0")
        check_for_newer_pypi_version("comfy-cli", "1.0.0")
        mock_get.assert_called_once_with("https://pypi.org/pypi/comfy-cli/json", timeout=5)


# TestCheckForUpdates was removed alongside the bright-blue
# "🔔 Update Available!" panel (Task 5 of the CLI UX consistency pass).
# ``check_for_newer_pypi_version`` is still tested directly above; the
# welcome banner consumes ``latest_upgrade_version`` (below) inline.


class TestLatestUpgradeVersion:
    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_returns_newer_version_and_writes_cache(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        result = latest_upgrade_version("1.0.0", tmp_path)
        assert result == "99.0.0"
        cache = tmp_path / "update-check.json"
        assert json.loads(cache.read_text())["latest"] == "99.0.0"

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_returns_none_when_current(self, mock_check, tmp_path):
        mock_check.return_value = (False, "1.0.0")
        assert latest_upgrade_version("1.0.0", tmp_path) is None

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_fresh_cache_skips_network(self, mock_check, tmp_path):
        cache = tmp_path / "update-check.json"
        cache.write_text(json.dumps({"checked_at": time.time(), "latest": "99.0.0"}))
        assert latest_upgrade_version("1.0.0", tmp_path) == "99.0.0"
        mock_check.assert_not_called()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_stale_cache_refreshes(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        cache = tmp_path / "update-check.json"
        cache.write_text(json.dumps({"checked_at": 0, "latest": "2.0.0"}))
        assert latest_upgrade_version("1.0.0", tmp_path) == "99.0.0"
        mock_check.assert_called_once()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_disabled_via_env(self, mock_check, tmp_path, monkeypatch):
        monkeypatch.setenv(UPDATE_CHECK_DISABLE_ENV, "1")
        assert latest_upgrade_version("1.0.0", tmp_path) is None
        mock_check.assert_not_called()

    @patch("comfy_cli.update.check_for_newer_pypi_version")
    def test_corrupt_cache_does_not_raise(self, mock_check, tmp_path):
        mock_check.return_value = (True, "99.0.0")
        cache = tmp_path / "update-check.json"
        cache.write_text("not json")
        assert latest_upgrade_version("1.0.0", tmp_path) is None


class TestUpgradeCli:
    @patch("comfy_cli.update.subprocess.run")
    def test_runs_pip_install_upgrade(self, mock_run, tmp_path):
        with (
            patch.object(update.sys, "prefix", str(tmp_path)),
            patch("comfy_cli.update.importlib.util.find_spec", return_value=ModuleSpec("pip", loader=None)),
        ):
            upgrade_cli()
        args = mock_run.call_args[0][0]
        assert args[1:] == ["-m", "pip", "install", "-U", "comfy-cli"]
        assert mock_run.call_args[1]["check"] is True
        assert mock_run.call_args[1].get("timeout") == 600

    @patch("comfy_cli.update.subprocess.run")
    def test_uv_tool_install_upgrades_through_uv(self, mock_run, tmp_path):
        """A `uv tool install` env has no pip, so `python -m pip` always fails there."""
        (tmp_path / "uv-receipt.toml").write_text("[tool]\n")
        with (
            patch.object(update.sys, "prefix", str(tmp_path)),
            patch("comfy_cli.update.resolve_binary", return_value="/opt/bin/uv"),
        ):
            upgrade_cli()
        assert mock_run.call_args[0][0] == ["/opt/bin/uv", "tool", "upgrade", "comfy-cli"]
        assert mock_run.call_args[1]["check"] is True
        assert mock_run.call_args[1].get("timeout") == 600

    @pytest.mark.parametrize("ambient_tool_dir", [None, "other", "current"])
    @patch("comfy_cli.update.subprocess.run")
    def test_uv_upgrade_targets_running_install_not_ambient_tool_dir(
        self, mock_run, tmp_path, monkeypatch, ambient_tool_dir
    ):
        tool_root = tmp_path / "custom-tools"
        prefix = tool_root / "comfy-cli"
        prefix.mkdir(parents=True)
        (prefix / "uv-receipt.toml").write_text("[tool]\n")
        previous = (
            None
            if ambient_tool_dir is None
            else str(tool_root if ambient_tool_dir == "current" else tmp_path / "other-tools")
        )
        if previous is None:
            monkeypatch.delenv("UV_TOOL_DIR", raising=False)
        else:
            monkeypatch.setenv("UV_TOOL_DIR", previous)
        monkeypatch.setenv("UV_SELF_UPGRADE_TEST_SENTINEL", "preserved")
        with (
            patch.object(update.sys, "prefix", str(prefix)),
            patch("comfy_cli.update.resolve_binary", return_value="/opt/bin/uv"),
        ):
            upgrade_cli()
        installer_env = mock_run.call_args.kwargs.get("env", {})
        assert installer_env.get("UV_TOOL_DIR") == str(tool_root)
        assert installer_env.get("UV_SELF_UPGRADE_TEST_SENTINEL") == "preserved"
        assert update.os.environ.get("UV_TOOL_DIR") == previous
        assert mock_run.call_args.args[0] == ["/opt/bin/uv", "tool", "upgrade", "comfy-cli"]
        assert mock_run.call_args.kwargs["timeout"] == 600

    @patch("comfy_cli.update.subprocess.run")
    def test_uv_tool_install_without_uv_on_path_is_unavailable(self, mock_run, tmp_path):
        (tmp_path / "uv-receipt.toml").write_text("[tool]\n")
        with (
            patch.object(update.sys, "prefix", str(tmp_path)),
            patch("comfy_cli.update.resolve_binary", return_value=None),
            pytest.raises(update.UpgradeUnavailable) as excinfo,
        ):
            upgrade_cli()
        assert "uv tool upgrade comfy-cli" in excinfo.value.hint
        mock_run.assert_not_called()

    @patch("comfy_cli.update.subprocess.run")
    def test_env_without_pip_is_unavailable(self, mock_run, tmp_path):
        with (
            patch.object(update.sys, "prefix", str(tmp_path)),
            patch("comfy_cli.update.importlib.util.find_spec", return_value=None),
            pytest.raises(update.UpgradeUnavailable),
        ):
            upgrade_cli()
        mock_run.assert_not_called()


@pytest.mark.parametrize("payload", ["[]", '"x"', '{"checked_at": "abc"}'])
def test_latest_upgrade_version_survives_malformed_cache(tmp_path, payload):
    (tmp_path / "update-check.json").write_text(payload)
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None


@patch("comfy_cli.update.check_for_newer_pypi_version")
def test_latest_upgrade_version_survives_missing_latest_key(mock_check, tmp_path):
    # {"checked_at": 0} is stale — triggers refresh; the payload itself lacks "latest"
    # but refresh returns a valid current_version, so we just confirm no crash.
    mock_check.return_value = (False, "1.0.0")
    (tmp_path / "update-check.json").write_text('{"checked_at": 0}')
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None


def test_latest_upgrade_version_survives_unparseable_latest(tmp_path):
    (tmp_path / "update-check.json").write_text(json.dumps({"checked_at": time.time(), "latest": "unknown"}))
    assert update.latest_upgrade_version("1.0.0", str(tmp_path)) is None


@pytest.mark.parametrize(
    "receipt",
    [
        '[tool]\nrequirements = [{ name = "comfy-cli", specifier = "==1.2.3" }]\n',
        '[tool]\nrequirements = [{ name = "comfy-cli", specifier = "===1.2.3" }]\n',
        '[tool]\nrequirements = [{ name = "Comfy_Cli", specifier = ">=1,==1.2.3" }]\n',
        '[tool]\nrequirements = ["comfy-cli==1.2.3"]\n',
        '[tool]\nrequirements = [{ name = "comfy-cli" }]\nconstraints = [{ name = "comfy-cli", specifier = "==1.2.3" }]\n',
    ],
    ids=["exact", "arbitrary-exact", "mixed-normalized", "legacy", "constraint"],
)
def test_uv_exact_pin_blocks_installer_and_preserves_receipt(tmp_path, monkeypatch, receipt):
    prefix = tmp_path / "original tools" / "comfy-cli"
    prefix.mkdir(parents=True)
    path = prefix / "uv-receipt.toml"
    path.write_text(receipt)
    monkeypatch.setattr(update.sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
    with patch("comfy_cli.update.subprocess.run") as installer, pytest.raises(update.UpgradeUnavailable) as failure:
        upgrade_cli()
    assert "pin" in str(failure.value).lower()
    assert "UV_TOOL_DIR" in failure.value.hint
    assert str(prefix.parent) in failure.value.hint
    assert "reinstall" in failure.value.hint.lower()
    assert path.read_text() == receipt
    installer.assert_not_called()


@pytest.mark.parametrize(
    "receipt",
    [
        "[tool]\n",
        '[tool]\nrequirements = [{ name = "comfy-cli" }]\n',
        '[tool]\nrequirements = [{ name = "comfy-cli", specifier = ">=1,<2" }]\n',
        '[tool]\nrequirements = [{ name = "comfy-cli", specifier = "==1.2.*" }]\n',
        '[tool]\nrequirements = [{ name = "comfy-cli", specifier = ">=1" }, { name = "helper", specifier = "==2" }]\nconstraints = [{ name = "helper", specifier = "==2" }]\n',
        '[tool]\nrequirements = ["comfy-cli>=1", "helper==2"]\n',
    ],
    ids=["legacy-empty", "unbounded", "range", "wildcard", "other-dependency", "legacy-range"],
)
def test_uv_unpinned_or_ranged_receipt_keeps_existing_upgrade(tmp_path, monkeypatch, receipt):
    prefix = tmp_path / "tools" / "comfy-cli"
    prefix.mkdir(parents=True)
    path = prefix / "uv-receipt.toml"
    path.write_text(receipt)
    monkeypatch.setattr(update.sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
    with patch("comfy_cli.update.subprocess.run") as installer:
        upgrade_cli()
    assert installer.call_count == 1
    assert installer.call_args.args[0] == ["/opt/bin/uv", "tool", "upgrade", "comfy-cli"]
    assert installer.call_args.kwargs["env"]["UV_TOOL_DIR"] == str(prefix.parent)
    assert installer.call_args.kwargs["check"] is True
    assert installer.call_args.kwargs["timeout"] == 600
    assert path.read_text() == receipt


@pytest.mark.parametrize(
    "receipt",
    [
        "[tool",
        'other = "invalid"\n',
        '[tool]\nrequirements = "invalid"\n',
        "[tool]\nrequirements = [7]\n",
        '[tool]\nrequirements = [{name="comfy-cli", specifier="token=secret"}]\n',
    ],
)
def test_malformed_uv_receipt_is_structured_and_never_runs_installer(tmp_path, monkeypatch, receipt):
    prefix = tmp_path / "tools" / "comfy-cli"
    prefix.mkdir(parents=True)
    path = prefix / "uv-receipt.toml"
    path.write_text(receipt)
    monkeypatch.setattr(update.sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
    with patch("comfy_cli.update.subprocess.run") as installer, pytest.raises(update.UpgradeUnavailable) as failure:
        upgrade_cli()
    assert "receipt" in str(failure.value).lower()
    assert "token=secret" not in str(failure.value)
    assert "UV_TOOL_DIR" in failure.value.hint
    installer.assert_not_called()


def test_unreadable_uv_receipt_is_unavailable_without_exposing_details(tmp_path, monkeypatch):
    prefix = tmp_path / "tools" / "comfy-cli"
    prefix.mkdir(parents=True)
    receipt = prefix / "uv-receipt.toml"
    receipt.write_text("[tool]\n")
    monkeypatch.setattr(update.sys, "prefix", str(prefix))
    monkeypatch.setattr(update, "resolve_binary", lambda name: "/opt/bin/uv")
    original_read = update.Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == receipt:
            raise PermissionError("sensitive-parser-details")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(update.Path, "read_text", guarded_read)
    with patch("comfy_cli.update.subprocess.run") as installer, pytest.raises(update.UpgradeUnavailable) as failure:
        upgrade_cli()
    assert "receipt" in str(failure.value).lower()
    assert "sensitive-parser-details" not in str(failure.value)
    installer.assert_not_called()
