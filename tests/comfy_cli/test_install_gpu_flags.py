"""``comfy install`` must reject a GPU flag that its platform cannot honor.

``--m-series`` selects ``GPU_OPTION.MAC_M_SERIES``, which installs the nightly
CPU torch wheels. That is right on Apple Silicon and wrong everywhere else: on
Linux/Windows it silently hands the user a CPU-only nightly build and no GPU
support. The check used to print a warning and fall through, so the bad install
still happened — it now exits like the symmetric ``--nvidia``-on-macOS guard.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from comfy_cli import constants
from comfy_cli.cmdline import app, g_exclusivity, g_gpu_exclusivity

runner = CliRunner()


@pytest.fixture
def stub_install_env():
    """Neutralize everything ``install`` touches before the platform checks."""
    # The mutual-exclusivity validators are module-level and accumulate across
    # in-process CliRunner invocations; without this the second GPU flag in the
    # file fails as "mutually exclusive" with the first test's.
    g_exclusivity.reset_for_testing()
    g_gpu_exclusivity.reset_for_testing()
    with (
        patch("comfy_cli.cmdline.EnvChecker") as checker,
        patch("comfy_cli.cmdline.workspace_manager.get_workspace_path", return_value=("/tmp/ComfyUI", None)),
        patch("comfy_cli.cmdline.check_comfy_repo", return_value=(False, None)),
        patch("comfy_cli.cmdline.install_inner.execute") as execute,
    ):
        checker.return_value.python_version = MagicMock(major=3, minor=12)
        yield execute


@pytest.mark.parametrize("plat", [constants.OS.LINUX, constants.OS.WINDOWS])
def test_m_series_on_non_macos_exits_without_installing(stub_install_env, plat):
    with patch("comfy_cli.cmdline.utils.get_os", return_value=plat):
        result = runner.invoke(app, ["install", "--m-series", "--skip-manager"])

    assert result.exit_code == 1
    # The whole point: no install runs with the Apple-Silicon GPU option.
    stub_install_env.assert_not_called()
    # The message must name the paths that do work, not just deny the flag.
    assert "--cpu" in result.output


def test_m_series_on_macos_still_installs(stub_install_env):
    with patch("comfy_cli.cmdline.utils.get_os", return_value=constants.OS.MACOS):
        result = runner.invoke(app, ["install", "--m-series", "--skip-manager"])

    assert result.exit_code == 0
    stub_install_env.assert_called_once()
    assert stub_install_env.call_args.kwargs["gpu"] is constants.GPU_OPTION.MAC_M_SERIES


def test_nvidia_on_macos_exits_without_installing(stub_install_env):
    """The guard --m-series is now symmetric with."""
    with patch("comfy_cli.cmdline.utils.get_os", return_value=constants.OS.MACOS):
        result = runner.invoke(app, ["install", "--nvidia", "--skip-manager"])

    assert result.exit_code == 1
    stub_install_env.assert_not_called()
