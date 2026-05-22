"""Provider-level tests for the dual-send telemetry refactor (MAR-52).

These cover the contract each provider has to honor — Mixpanel keeps legacy
event names via the ``mixpanel_name`` alias kwarg, PostHog stamps every event
with the standard CLI properties and aliases ``tracing_id`` to
``workflow_run_id`` on the canonical execution lifecycle events.
"""

from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager
from comfy_cli.tracking import EXECUTION_EVENTS, MixpanelProvider, PostHogProvider

_ConfigManagerCls = ConfigManager.__closure__[0].cell_contents


@pytest.fixture
def tracking_with_two_providers(tmp_path):
    """Yield comfy_cli.tracking with a MixpanelProvider + PostHogProvider pair
    whose underlying clients are MagicMocks. Lets tests assert on the fan-out
    without hitting the network."""
    config_dir = tmp_path / "comfy-cli"
    config_dir.mkdir()
    with patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)):
        cfg = _ConfigManagerCls()

    import comfy_cli.tracking as tracking_mod

    mixpanel_provider = MixpanelProvider("token-mp")
    mixpanel_provider.client = MagicMock()
    mixpanel_provider.enabled = True

    posthog_provider = PostHogProvider.__new__(PostHogProvider)
    posthog_provider.client = MagicMock()
    posthog_provider.enabled = True

    with (
        patch.object(tracking_mod, "config_manager", cfg),
        patch.object(tracking_mod, "user_id", "test-distinct-id"),
        patch.object(tracking_mod, "cli_version", "test-cli-version"),
        patch.object(tracking_mod, "tracing_id", "test-tracing-id"),
        patch.object(tracking_mod, "PROVIDERS", [mixpanel_provider, posthog_provider]),
        patch.object(tracking_mod, "_session_only_tracking", False),
    ):
        tracking_mod.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        yield tracking_mod, mixpanel_provider, posthog_provider


def _posthog_capture_kwargs(client_mock):
    """Return the last ``capture(...)`` keyword arguments as a dict."""
    args, kwargs = client_mock.capture.call_args
    if "event" not in kwargs and args:
        kwargs = {"event": args[0], **kwargs}
    return kwargs


class TestDualFanOut:
    def test_track_event_fans_out_to_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("some_event", {"k": "v"})

        mp_provider.client.track.assert_called_once()
        ph_provider.client.capture.assert_called_once()

    def test_opt_out_short_circuits_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        tracking_mod.track_event("some_event")

        mp_provider.client.track.assert_not_called()
        ph_provider.client.capture.assert_not_called()

    def test_one_provider_raising_does_not_block_the_other(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        mp_provider.client.track.side_effect = RuntimeError("mixpanel down")

        tracking_mod.track_event("some_event")

        # Mixpanel raised but PostHog still got the call.
        ph_provider.client.capture.assert_called_once()

    def test_provider_order_does_not_matter_for_failure_isolation(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        ph_provider.client.capture.side_effect = RuntimeError("posthog down")

        tracking_mod.track_event("some_event")

        # PostHog raised but Mixpanel still got the call (it ran first).
        mp_provider.client.track.assert_called_once()


class TestPostHogStandardProperties:
    def test_environment_surface_source_are_stamped(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("any_event")

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        props = capture_kwargs["properties"]
        assert props["environment"] == "cli"
        assert props["surface"] == "cli"
        assert props["source"] == "cli"
        assert props["trigger_source"] == "cli"
        assert props["cli_version"] == "test-cli-version"
        assert props["tracing_id"] == "test-tracing-id"

    def test_caller_properties_win_over_defaults(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("any_event", {"surface": "custom"})

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert capture_kwargs["properties"]["surface"] == "custom"

    def test_distinct_id_is_user_id(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("any_event")

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert capture_kwargs["distinct_id"] == "test-distinct-id"

    def test_mixpanel_does_not_receive_posthog_standard_props(self, tracking_with_two_providers):
        # The Mixpanel pipe has 2 years of history without these CLI-canonical
        # props; injecting them would dirty the schema. PostHogProvider owns
        # the env/surface/source stamping, not the shared track_event flow.
        tracking_mod, mp_provider, _ = tracking_with_two_providers
        tracking_mod.track_event("any_event")

        _, kwargs = mp_provider.client.track.call_args
        props = kwargs["properties"]
        assert "environment" not in props
        assert "surface" not in props
        assert "source" not in props


class TestWorkflowRunIdAlias:
    @pytest.mark.parametrize("event_name", sorted(EXECUTION_EVENTS))
    def test_execution_events_get_workflow_run_id(self, tracking_with_two_providers, event_name):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event(event_name)

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        props = capture_kwargs["properties"]
        assert props["workflow_run_id"] == "test-tracing-id"
        assert props["tracing_id"] == "test-tracing-id"

    def test_non_execution_events_do_not_get_workflow_run_id(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("install")

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert "workflow_run_id" not in capture_kwargs["properties"]

    def test_caller_workflow_run_id_is_not_overwritten(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("execution_start", {"workflow_run_id": "caller-supplied"})

        capture_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert capture_kwargs["properties"]["workflow_run_id"] == "caller-supplied"


class TestMixpanelLegacyNameAlias:
    def test_mixpanel_name_kwarg_routes_to_mixpanel_only(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("execution_start", mixpanel_name="run")

        _, mp_kwargs = mp_provider.client.track.call_args
        assert mp_kwargs["event_name"] == "run"

        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert ph_kwargs["event"] == "execution_start"

    def test_without_alias_both_providers_get_same_name(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("execution_success")

        _, mp_kwargs = mp_provider.client.track.call_args
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert mp_kwargs["event_name"] == "execution_success"
        assert ph_kwargs["event"] == "execution_success"


class TestProviderConstruction:
    def test_posthog_with_empty_token_is_disabled_and_silent(self):
        provider = PostHogProvider("", "https://t.comfy.org")
        assert provider.enabled is False
        # Calling .track on a disabled provider must not raise.
        provider.track("any_event", "distinct_id", {})

    def test_posthog_with_valid_token_constructs_client(self):
        provider = PostHogProvider("phc_test", "https://t.comfy.org")
        assert provider.enabled is True
        assert provider.client is not None

    def test_mixpanel_with_empty_token_is_disabled(self):
        provider = MixpanelProvider("")
        assert provider.enabled is False

    def test_posthog_track_skips_when_distinct_id_is_none(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        with patch.object(tracking_mod, "user_id", None):
            tracking_mod.track_event("execution_start")

        ph_provider.client.capture.assert_not_called()


class TestRedactionThroughFanOut:
    def test_api_key_redaction_reaches_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers

        @tracking_mod.track_command()
        def fake_cmd(workflow, api_key=None):
            return None

        fake_cmd(workflow="wf.json", api_key="sk-supersecret")

        _, mp_kwargs = mp_provider.client.track.call_args
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert mp_kwargs["properties"]["api_key"] == "<redacted>"
        assert ph_kwargs["properties"]["api_key"] == "<redacted>"
        assert "sk-supersecret" not in str(mp_kwargs["properties"])
        assert "sk-supersecret" not in str(ph_kwargs["properties"])


class TestAtexitFlush:
    def test_flush_all_providers_calls_each_flush(self):
        """The module registers ``_flush_all_providers`` with ``atexit`` at import
        time. Verify that helper drains every enabled provider so short-lived
        CLI invocations don't silently drop in-flight PostHog events."""
        import comfy_cli.tracking as tracking_mod

        p1 = MagicMock()
        p2 = MagicMock()
        with patch.object(tracking_mod, "PROVIDERS", [p1, p2]):
            tracking_mod._flush_all_providers()

        p1.flush.assert_called_once()
        p2.flush.assert_called_once()

    def test_flush_swallows_provider_errors(self):
        import comfy_cli.tracking as tracking_mod

        p1 = MagicMock()
        p1.flush.side_effect = RuntimeError("flush failed")
        p2 = MagicMock()
        with patch.object(tracking_mod, "PROVIDERS", [p1, p2]):
            tracking_mod._flush_all_providers()

        p2.flush.assert_called_once()
