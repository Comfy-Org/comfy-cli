from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager

# Unwrap the singleton to get fresh ConfigManager instances per test.
_ConfigManagerCls = ConfigManager.__closure__[0].cell_contents


@pytest.fixture
def tracking_module(tmp_path):
    """Yield comfy_cli.tracking with a fresh tmp-path ConfigManager and a single
    mocked TelemetryProvider in PROVIDERS so tests can assert on the fan-out.

    Exposes the mock as ``tracking_mod.provider`` for assertions.
    """
    config_dir = tmp_path / "comfy-cli"
    config_dir.mkdir()
    with patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)):
        cfg = _ConfigManagerCls()

    import comfy_cli.tracking as tracking_mod

    fake_provider = MagicMock()
    fake_provider.enabled = True
    # Mirror MixpanelProvider's no-op-on-missing-distinct-id behavior so opt-out
    # paths look identical from the test's perspective.
    fake_provider.track.return_value = None

    with (
        patch.object(tracking_mod, "config_manager", cfg),
        patch.object(tracking_mod, "user_id", None),
        patch.object(tracking_mod, "cli_version", "test-cli-version"),
        patch.object(tracking_mod, "tracing_id", "test-tracing-id"),
        patch.object(tracking_mod, "PROVIDERS", [fake_provider]),
        patch.object(tracking_mod, "_session_only_tracking", False),
    ):
        # Stash the mock on the module for convenient access from tests
        # without changing the fixture return contract.
        tracking_mod.provider = fake_provider  # type: ignore[attr-defined]
        try:
            yield tracking_mod
        finally:
            del tracking_mod.provider


def _last_track_call(provider):
    args, kwargs = provider.track.call_args
    # Provider.track(event_name, distinct_id=..., properties=...)
    event_name = args[0] if args else kwargs.get("event_name")
    distinct_id = kwargs.get("distinct_id", args[1] if len(args) > 1 else None)
    properties = kwargs.get("properties", args[2] if len(args) > 2 else {})
    return event_name, distinct_id, properties


class TestTrackEvent:
    def test_short_circuits_when_disabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_short_circuits_when_not_configured(self, tracking_module):
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_fires_when_enabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.provider.track.assert_called_once()
        event_name, _, properties = _last_track_call(tracking_module.provider)
        assert event_name == "some_event"
        assert properties["k"] == "v"
        assert "cli_version" in properties
        assert "tracing_id" in properties

    def test_properties_default_to_empty_dict(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert set(properties.keys()) == {"cli_version", "tracing_id"}

    def test_swallows_provider_errors(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.provider.track.side_effect = RuntimeError("boom")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()


class TestTrackCommandRedaction:
    """track_command must redact secret-bearing kwargs before they reach the tracking system."""

    def test_api_key_value_is_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, api_key=None):
            return None

        some_cmd(workflow="wf.json", api_key="sk-supersecret")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["api_key"] == "<redacted>"
        assert properties["workflow"] == "wf.json"
        assert "sk-supersecret" not in str(properties)

    def test_api_key_none_stays_none(self, tracking_module):
        # When the user didn't pass --api-key (or set $COMFY_API_KEY), we still
        # want to be able to see in the analytics that it was absent — not a
        # "<redacted>" sentinel that would imply they did pass one.
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, api_key=None):
            return None

        some_cmd(workflow="wf.json", api_key=None)

        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["api_key"] is None


class TestInitTrackingRoundTrip:
    """End-to-end: init_tracking() writes the string "False"/"True", and track_event honors it.

    Regression for a prior bug where track_event used config_manager.get(), which returned
    the raw string "False" (a truthy value), so disabling via this code path had no effect.
    """

    def test_disable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(False)
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_enable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(True)
        tracking_module.provider.track.reset_mock()
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()

    def test_disable_persists_as_parseable_bool(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False

    def test_enable_generates_user_id(self, tracking_module):
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None
        tracking_module.init_tracking(True)
        generated_user_id = tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID)
        assert generated_user_id is not None
        assert tracking_module.user_id == generated_user_id
        _, distinct_id, _ = _last_track_call(tracking_module.provider)
        assert distinct_id == generated_user_id

    def test_disable_does_not_generate_user_id(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None

    def test_install_event_fires_once_across_calls(self, tracking_module):
        tracking_module.init_tracking(True)
        assert tracking_module.provider.track.call_count == 1
        tracking_module.init_tracking(True)
        assert tracking_module.provider.track.call_count == 1


class TestPromptTrackingConsent:
    def test_enables_session_only_when_stdin_not_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=True),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is None
        assert tracking_module._session_only_tracking is True
        assert tracking_module.user_id is not None

    def test_enables_session_only_when_stdout_not_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=True),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is None
        assert tracking_module._session_only_tracking is True

    def test_session_only_tracking_fires_track_event(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.provider.track.assert_called_once()
        event_name, distinct_id, _ = _last_track_call(tracking_module.provider)
        assert event_name == "some_event"
        assert distinct_id is not None

    def test_session_only_persists_user_id(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
        persisted = tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID)
        assert persisted is not None
        assert persisted == tracking_module.user_id

    def test_session_only_survives_unwritable_config(self, tracking_module):
        # Read-only / missing config dir (fresh CI, restricted sandbox) must
        # not crash the caller mid-typer-callback — otherwise an agent gets
        # a Python traceback instead of a structured `failed` event.
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.config_manager, "set", side_effect=PermissionError("read-only fs")),
        ):
            tracking_module.prompt_tracking_consent()
        # In-memory state is still correct so this process tracks normally.
        assert tracking_module._session_only_tracking is True
        assert tracking_module.user_id is not None

    def test_session_only_reuses_existing_user_id(self, tracking_module):
        existing_id = "existing-uuid-from-prior-run"
        tracking_module.config_manager.set(constants.CONFIG_KEY_USER_ID, existing_id)
        with (
            patch.object(tracking_module, "user_id", existing_id),
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
            assert tracking_module.user_id == existing_id
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) == existing_id

    def test_prompts_when_both_are_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=True),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=True),
            patch.object(tracking_module.ui, "prompt_confirm_action", return_value=False) as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_called_once()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False
        assert tracking_module._session_only_tracking is False

    def test_skip_prompt_bypasses_tty_check(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent(skip_prompt=True, default_value=False)
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False
        assert tracking_module._session_only_tracking is False

    def test_no_op_when_already_configured(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is True
        assert tracking_module._session_only_tracking is False

    def test_session_only_is_idempotent(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
            first_user_id = tracking_module.user_id
            tracking_module.prompt_tracking_consent()
            assert tracking_module.user_id == first_user_id
