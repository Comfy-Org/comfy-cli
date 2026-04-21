from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager

# Unwrap the singleton to get fresh ConfigManager instances per test.
_ConfigManagerCls = ConfigManager.__closure__[0].cell_contents


@pytest.fixture
def tracking_module(tmp_path):
    """Yield comfy_cli.tracking with a fresh tmp-path ConfigManager and a mocked Mixpanel client."""
    config_dir = tmp_path / "comfy-cli"
    config_dir.mkdir()
    with patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)):
        cfg = _ConfigManagerCls()

    import comfy_cli.tracking as tracking_mod

    with (
        patch.object(tracking_mod, "config_manager", cfg),
        patch.object(tracking_mod, "user_id", None),
        patch.object(tracking_mod, "cli_version", "test-cli-version"),
        patch.object(tracking_mod, "tracing_id", "test-tracing-id"),
        patch.object(tracking_mod, "mp", MagicMock()),
    ):
        yield tracking_mod


class TestTrackEvent:
    def test_short_circuits_when_disabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_not_called()

    def test_short_circuits_when_not_configured(self, tracking_module):
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_not_called()

    def test_fires_when_enabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.mp.track.assert_called_once()
        _, kwargs = tracking_module.mp.track.call_args
        assert kwargs["event_name"] == "some_event"
        assert kwargs["properties"]["k"] == "v"
        assert "cli_version" in kwargs["properties"]
        assert "tracing_id" in kwargs["properties"]

    def test_properties_default_to_empty_dict(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_called_once()
        _, kwargs = tracking_module.mp.track.call_args
        assert set(kwargs["properties"].keys()) == {"cli_version", "tracing_id"}

    def test_swallows_mixpanel_errors(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.mp.track.side_effect = RuntimeError("boom")
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_called_once()


class TestInitTrackingRoundTrip:
    """End-to-end: init_tracking() writes the string "False"/"True", and track_event honors it.

    Regression for a prior bug where track_event used config_manager.get(), which returned
    the raw string "False" (a truthy value), so disabling via this code path had no effect.
    """

    def test_disable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(False)
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_not_called()

    def test_enable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(True)
        tracking_module.mp.track.reset_mock()
        tracking_module.track_event("some_event")
        tracking_module.mp.track.assert_called_once()

    def test_disable_persists_as_parseable_bool(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False

    def test_enable_generates_user_id(self, tracking_module):
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None
        tracking_module.init_tracking(True)
        generated_user_id = tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID)
        assert generated_user_id is not None
        assert tracking_module.user_id == generated_user_id
        _, kwargs = tracking_module.mp.track.call_args
        assert kwargs["distinct_id"] == generated_user_id

    def test_disable_does_not_generate_user_id(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None

    def test_install_event_fires_once_across_calls(self, tracking_module):
        tracking_module.init_tracking(True)
        assert tracking_module.mp.track.call_count == 1
        tracking_module.init_tracking(True)
        assert tracking_module.mp.track.call_count == 1
