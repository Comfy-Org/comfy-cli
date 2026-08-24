"""Provider-level tests for the dual-send telemetry refactor.

These cover the contract each provider has to honor — Mixpanel keeps legacy
event names via the ``mixpanel_name`` alias kwarg, PostHog stamps every event
with the standard CLI properties and aliases ``tracing_id`` to
``workflow_run_id`` on the canonical execution lifecycle events.
"""

import importlib
import logging
import threading
import time
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


def _mixpanel_track_kwargs(mp_provider):
    """Drain the Mixpanel worker, then return the last ``track(...)`` kwargs.

    ``MixpanelProvider`` dispatches through a bounded queue drained by a daemon
    worker, so the send has not necessarily happened by the time
    ``track_event()`` returns. Every assertion on the mocked client goes through
    a ``flush()`` first.
    """
    mp_provider.flush()
    _, kwargs = mp_provider.client.track.call_args
    return kwargs


class TestDualFanOut:
    def test_track_event_fans_out_to_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("some_event", {"k": "v"})

        mp_provider.flush()
        mp_provider.client.track.assert_called_once()
        ph_provider.client.capture.assert_called_once()

    def test_opt_out_short_circuits_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        tracking_mod.track_event("some_event")

        mp_provider.flush()
        mp_provider.client.track.assert_not_called()
        ph_provider.client.capture.assert_not_called()

    def test_one_provider_raising_does_not_block_the_other(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        mp_provider.client.track.side_effect = RuntimeError("mixpanel down")

        tracking_mod.track_event("some_event")

        # The Mixpanel send now raises on its worker thread, not the caller's;
        # flush() still returns (the worker's finally: task_done()).
        mp_provider.flush()
        # Mixpanel raised but PostHog still got the call.
        ph_provider.client.capture.assert_called_once()

    def test_provider_order_does_not_matter_for_failure_isolation(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        ph_provider.client.capture.side_effect = RuntimeError("posthog down")

        tracking_mod.track_event("some_event")

        # PostHog raised but Mixpanel still got the call (it was enqueued first).
        mp_provider.flush()
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

        props = _mixpanel_track_kwargs(mp_provider)["properties"]
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

        mp_kwargs = _mixpanel_track_kwargs(mp_provider)
        assert mp_kwargs["event_name"] == "run"

        # PostHog receives the canonical name, prefixed; Mixpanel keeps "run".
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert ph_kwargs["event"] == "cli:execution_start"

    def test_posthog_prefixes_event_while_mixpanel_stays_bare(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("execution_success")

        mp_kwargs = _mixpanel_track_kwargs(mp_provider)
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        # Mixpanel keeps the bare name for stream continuity; PostHog is namespaced.
        assert mp_kwargs["event_name"] == "execution_success"
        assert ph_kwargs["event"] == "cli:execution_success"


class TestPostHogEventPrefix:
    def test_top_level_event_is_prefixed(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("install")

        # Mixpanel bare, PostHog namespaced.
        mp_kwargs = _mixpanel_track_kwargs(mp_provider)
        assert mp_kwargs["event_name"] == "install"
        assert _posthog_capture_kwargs(ph_provider.client)["event"] == "cli:install"

    def test_sub_namespaced_event_composes_with_prefix(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        tracking_mod.track_event("node:install")

        assert _posthog_capture_kwargs(ph_provider.client)["event"] == "cli:node:install"


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

    def test_mixpanel_client_has_bounded_request_timeout(self):
        """Regression guard: mixpanel-python's default
        Consumer uses request_timeout=None → an unbounded requests.post that
        hangs the CLI forever on a blackholed endpoint. The provider must build
        its Mixpanel client with an explicit 10s consumer timeout so this can't
        silently regress to None. retry_limit must also be 1: the Consumer
        default is 4 with backoff, which would let a blackholed send run ~40s+
        across retries even with a 10s per-attempt timeout — and this send is
        synchronous on the main thread, not covered by the atexit deadline."""
        provider = MixpanelProvider("token-mp")
        assert provider.client is not None
        consumer = provider.client._consumer
        assert consumer._request_timeout == 10
        # retry_limit is threaded into the session's urllib3 Retry.total.
        adapter = consumer._session.get_adapter("https://api.mixpanel.com")
        assert adapter.max_retries.total == 1

    def test_posthog_unregisters_its_own_atexit_join(self):
        """Regression guard: Posthog's constructor registers its own
        ``atexit.register(self.join)``, which flushes synchronously on the main
        thread at shutdown, unbounded by ``_flush_all_providers``' 5s daemon
        deadline. Against a blackholed endpoint that join can block ~21s after
        the terminal envelope. The provider must unregister it so the bounded
        flush is the only shutdown drain path."""
        import comfy_cli.tracking as tracking_mod

        fake_client = MagicMock()
        with (
            # PostHogProvider imports `Posthog` lazily inside __init__ (see
            # `_get_providers`), so there's no module-level attribute to patch —
            # patch the source it's imported from instead.
            patch("posthog.Posthog", return_value=fake_client),
            patch.object(tracking_mod, "atexit") as fake_atexit,
        ):
            provider = PostHogProvider("phc_test", "https://t.comfy.org")

        assert provider.enabled is True
        fake_atexit.unregister.assert_called_once_with(fake_client.join)

    def test_posthog_track_skips_when_distinct_id_is_none(self, tracking_with_two_providers):
        tracking_mod, _, ph_provider = tracking_with_two_providers
        with patch.object(tracking_mod, "user_id", None):
            tracking_mod.track_event("execution_start")

        ph_provider.client.capture.assert_not_called()


def _mixpanel_provider_with_mock_client():
    """A real ``MixpanelProvider`` — bounded queue and daemon worker included —
    whose SDK client is a MagicMock, so the worker's send never hits the network."""
    provider = MixpanelProvider("token-mp")
    provider.client = MagicMock()
    return provider


def _sent_event_names(provider):
    return [call.kwargs["event_name"] for call in provider.client.track.call_args_list]


class TestMixpanelNonBlockingDispatch:
    """``track()`` used to post inline on the calling thread, so every
    consented invocation paid a synchronous HTTP round-trip (worst case ~10s
    against a blackholed endpoint) *before* the wrapped command body ran.
    Dispatch is now a bounded queue drained by a daemon worker."""

    def test_track_returns_while_a_send_is_still_in_flight(self):
        provider = _mixpanel_provider_with_mock_client()
        started, release = threading.Event(), threading.Event()

        def _blocking_send(**_kwargs):
            started.set()
            release.wait(timeout=30)

        provider.client.track.side_effect = _blocking_send
        try:
            # Occupy the worker so the next track() provably overlaps a live send.
            provider.track("blocker", "test-distinct-id", {})
            assert started.wait(timeout=5), "worker never picked up the queued event"

            start = time.monotonic()
            provider.track("payload", "test-distinct-id", {"k": "v"})
            elapsed = time.monotonic() - start
            assert elapsed < 1.0, f"track() blocked on the in-flight send ({elapsed:.1f}s)"
        finally:
            release.set()

        provider.flush()
        assert _sent_event_names(provider) == ["blocker", "payload"]
        payload_kwargs = provider.client.track.call_args_list[1].kwargs
        assert payload_kwargs["distinct_id"] == "test-distinct-id"
        assert payload_kwargs["event_name"] == "payload"
        assert payload_kwargs["properties"] == {"k": "v"}

    def test_flush_drains_every_queued_event_in_submission_order(self):
        # A single FIFO drained by a single worker preserves per-process ordering.
        provider = _mixpanel_provider_with_mock_client()
        for i in range(20):
            provider.track(f"event-{i}", "test-distinct-id", {"i": i})

        provider.flush()
        assert _sent_event_names(provider) == [f"event-{i}" for i in range(20)]

    def test_overflow_drops_events_without_blocking_or_raising(self, monkeypatch, caplog):
        """A wedged worker must never turn into back-pressure on the CLI. The
        queue is bounded; overflow drops with a debug line (never a warning —
        telemetry failures must not surface on the user's stderr)."""
        monkeypatch.setattr(MixpanelProvider, "_QUEUE_MAX", 2)
        provider = _mixpanel_provider_with_mock_client()
        started, release = threading.Event(), threading.Event()

        def _blocking_send(**_kwargs):
            started.set()
            release.wait(timeout=30)

        provider.client.track.side_effect = _blocking_send
        try:
            provider.track("blocker", "test-distinct-id", {})
            assert started.wait(timeout=5), "worker never picked up the queued event"

            with caplog.at_level(logging.DEBUG):
                start = time.monotonic()
                for i in range(50):
                    provider.track(f"event-{i}", "test-distinct-id", {})
                elapsed = time.monotonic() - start
            assert elapsed < 1.0, f"track() blocked once the queue filled ({elapsed:.1f}s)"
        finally:
            release.set()

        provider.flush()
        # Two slots behind the blocked send; everything past that is dropped.
        assert _sent_event_names(provider) == ["blocker", "event-0", "event-1"]
        drops = [r for r in caplog.records if "queue full" in r.getMessage()]
        assert drops, "an overflow drop must leave a debug breadcrumb"
        assert all(r.levelno == logging.DEBUG for r in drops)

    def test_flush_returns_even_when_the_send_raises(self):
        """The worker marks each item done in a ``finally``, so a raising send
        can't leave ``flush()``' ``queue.join()`` waiting forever."""
        provider = _mixpanel_provider_with_mock_client()
        provider.client.track.side_effect = RuntimeError("mixpanel down")
        provider.track("boom", "test-distinct-id", {})

        finished = threading.Event()
        # Run flush() off-thread so a wedge fails the test instead of hanging it.
        threading.Thread(target=lambda: (provider.flush(), finished.set()), daemon=True).start()
        assert finished.wait(timeout=10), "flush() wedged after a raising send"
        provider.client.track.assert_called_once()

    def test_exit_stays_bounded_when_a_send_hangs(self):
        """End to end against a blackholed endpoint: ``_flush_all_providers`` runs
        each provider's ``flush()`` in a daemon thread joined against the shared
        5s deadline, so a hung send costs the exit path that much and no more —
        same treatment as PostHog's internally-unbounded ``client.flush()``.
        (Mixpanel's ``flush()`` also self-bounds; this covers the
        caller-side guarantee, which is what holds for every provider.)"""
        import comfy_cli.tracking as tracking_mod

        provider = _mixpanel_provider_with_mock_client()
        release = threading.Event()
        provider.client.track.side_effect = lambda **_kwargs: release.wait(timeout=60)
        try:
            provider.track("hangs", "test-distinct-id", {})

            with patch.object(tracking_mod, "PROVIDERS", [provider]):
                start = time.monotonic()
                tracking_mod._flush_all_providers()
                elapsed = time.monotonic() - start

            budget = tracking_mod._FLUSH_DEADLINE_SECONDS + 3.0
            assert elapsed < budget, f"exit was not bounded by the flush deadline (took {elapsed:.1f}s)"
        finally:
            release.set()

    def test_worker_is_a_daemon_and_the_provider_registers_no_atexit_hook(self):
        """Design constraint: ``_flush_all_providers`` is the ONLY
        shutdown drain path. The worker must die with the process rather than
        join it, and the provider must not add a second, unbounded atexit hook —
        the exact mistake ``PostHogProvider`` has to actively unregister."""
        import comfy_cli.tracking as tracking_mod

        with patch.object(tracking_mod, "atexit") as fake_atexit:
            provider = MixpanelProvider("token-mp")

        assert provider._worker.daemon is True
        assert provider._worker.is_alive()
        fake_atexit.register.assert_not_called()

    def test_disabled_provider_track_and_flush_are_inert(self):
        # No token → no client, no queue, no worker: neither call may raise.
        provider = MixpanelProvider("")
        assert provider.enabled is False
        assert not hasattr(provider, "_queue")
        provider.track("any_event", "test-distinct-id", {})
        provider.flush()


class TestMixpanelWorkerSurvivability:
    """A single worker drains the whole queue and nothing restarts it, so any
    escape from the loop wedges telemetry for the rest of the process: every
    later event is dropped and every ``flush()`` burns its full deadline."""

    def test_a_baseexception_from_the_sdk_does_not_kill_the_worker(self):
        class _Boom(BaseException):
            """Not an ``Exception`` — e.g. a ``MemoryError``/``SystemExit`` escape."""

        provider = _mixpanel_provider_with_mock_client()
        provider.client.track.side_effect = [_Boom("kaboom"), None]

        provider.track("first", "test-distinct-id", {})
        provider.track("second", "test-distinct-id", {})
        provider.flush()

        assert provider._worker.is_alive(), "worker died on a non-Exception failure"
        assert _sent_event_names(provider) == ["first", "second"]

    def test_flush_gives_up_promptly_when_the_worker_is_gone(self):
        """``queue.join()`` would block forever here, and ``flush()`` is public —
        a direct caller has no deadline of its own to fall back on."""
        provider = _mixpanel_provider_with_mock_client()
        provider.track("stranded", "test-distinct-id", {})
        # Impersonate a dead worker with the event still outstanding.
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        provider._worker = dead
        provider._pending = 1

        start = time.monotonic()
        provider.flush()
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"flush() waited on a dead worker ({elapsed:.1f}s)"

    def test_flush_is_bounded_when_the_worker_never_finishes(self, monkeypatch):
        import comfy_cli.tracking as tracking_mod

        monkeypatch.setattr(tracking_mod, "_FLUSH_DEADLINE_SECONDS", 0.5)
        provider = _mixpanel_provider_with_mock_client()
        release = threading.Event()
        provider.client.track.side_effect = lambda **_kwargs: release.wait(timeout=60)
        try:
            provider.track("hangs", "test-distinct-id", {})

            start = time.monotonic()
            provider.flush()
            elapsed = time.monotonic() - start
            assert elapsed < 5.0, f"flush() ignored its deadline (took {elapsed:.1f}s)"
        finally:
            release.set()

    def test_worker_start_failure_degrades_to_an_inert_provider(self):
        """``Thread.start`` raises ``RuntimeError: can't start new thread`` under
        thread/FD/memory pressure. ``_get_providers`` reports a construction
        failure with ``logging.warning`` — i.e. on the user's stderr — so a
        telemetry-side resource problem must not escape as one."""
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("can't start new thread")):
            provider = MixpanelProvider("token-mp")

        assert provider.enabled is False
        assert provider.client is None
        # Still inert rather than raising, even though the queue exists.
        provider.track("any_event", "test-distinct-id", {})
        provider.flush()


class TestMixpanelPropertySnapshot:
    """``@track_command`` fires its event *before* running the wrapped body, and
    serialization now happens on the worker instead of inline, so the payload has
    to be snapshotted at enqueue time or the body can mutate it out from under
    the send."""

    def test_nested_values_are_snapshotted_at_track_time(self):
        provider = _mixpanel_provider_with_mock_client()
        started, release = threading.Event(), threading.Event()

        def _blocking_send(**kwargs):
            if kwargs["event_name"] == "blocker":
                started.set()
                release.wait(timeout=30)

        provider.client.track.side_effect = _blocking_send
        try:
            # Hold the worker so the mutation below provably lands before the send.
            provider.track("blocker", "test-distinct-id", {})
            assert started.wait(timeout=5), "worker never picked up the queued event"

            nested = {"flags": ["--fast-deps"]}
            provider.track("payload", "test-distinct-id", {"nested": nested})
            nested["flags"].append("--no-deps")
        finally:
            release.set()

        provider.flush()
        sent = provider.client.track.call_args_list[1].kwargs["properties"]
        assert sent == {"nested": {"flags": ["--fast-deps"]}}

    def test_an_uncopyable_value_falls_back_to_a_shallow_copy(self):
        provider = _mixpanel_provider_with_mock_client()

        class _NoDeepCopy:
            def __deepcopy__(self, memo):
                raise TypeError("cannot deepcopy this")

        sentinel = _NoDeepCopy()
        provider.track("payload", "test-distinct-id", {"obj": sentinel})
        provider.flush()

        assert provider.client.track.call_args.kwargs["properties"]["obj"] is sentinel


class TestHardExitDrain:
    """``comfy launch`` leaves through ``os._exit``, which skips atexit handlers.
    That was free while Mixpanel sent inline from ``track()`` (the ``launch``
    event was delivered before the command body ran); with queue-and-drain
    dispatch those paths have to drain explicitly or drop the event every time."""

    def test_flush_for_hard_exit_drains_and_swallows(self):
        import comfy_cli.tracking as tracking_mod

        provider = MagicMock()
        with patch.object(tracking_mod, "PROVIDERS", [provider]):
            tracking_mod.flush_for_hard_exit()
        provider.flush.assert_called_once()

        with patch.object(tracking_mod, "_flush_all_providers", side_effect=RuntimeError("drain blew up")):
            tracking_mod.flush_for_hard_exit()  # must not propagate into the exit path

    def test_launch_hard_exit_drains_before_os_exit(self):
        import comfy_cli.tracking as tracking_mod
        from comfy_cli.command import launch as launch_mod

        calls = []
        with (
            patch.object(tracking_mod, "flush_for_hard_exit", side_effect=lambda: calls.append("drain")),
            patch.object(launch_mod.os, "_exit", side_effect=lambda code: calls.append(("exit", code))),
        ):
            launch_mod._hard_exit(3)

        assert calls == ["drain", ("exit", 3)], "telemetry must be drained before the process is torn down"

    def test_launch_exits_even_if_the_drain_raises(self):
        import comfy_cli.tracking as tracking_mod
        from comfy_cli.command import launch as launch_mod

        exits = []
        with (
            patch.object(tracking_mod, "flush_for_hard_exit", side_effect=RuntimeError("drain blew up")),
            patch.object(launch_mod.os, "_exit", side_effect=lambda code: exits.append(code)),
        ):
            launch_mod._hard_exit(1)

        assert exits == [1]


class TestRedactionThroughFanOut:
    def test_api_key_redaction_reaches_both_providers(self, tracking_with_two_providers):
        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers

        @tracking_mod.track_command()
        def fake_cmd(workflow, api_key=None):
            return None

        fake_cmd(workflow="wf.json", api_key="sk-supersecret")

        mp_kwargs = _mixpanel_track_kwargs(mp_provider)
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        assert mp_kwargs["properties"]["api_key"] == "<redacted>"
        assert ph_kwargs["properties"]["api_key"] == "<redacted>"
        assert "sk-supersecret" not in str(mp_kwargs["properties"])
        assert "sk-supersecret" not in str(ph_kwargs["properties"])

    def test_download_credentials_never_reach_either_provider(self, tracking_with_two_providers):
        # The original kwarg shape: before the suffix matcher and the underscore
        # filter, the un-redacted token still shipped to PostHog because its
        # client coerces the unserializable _ctx instead of raising the way
        # Mixpanel's does.
        import click

        tracking_mod, mp_provider, ph_provider = tracking_with_two_providers

        @tracking_mod.track_command("model")
        def download(_ctx=None, url=None, set_civitai_api_token=None, set_hf_api_token=None):
            return None

        download(
            _ctx=click.Context(click.Command("download")),
            url="https://example.com/model.safetensors",
            set_civitai_api_token="civ-secret",
            set_hf_api_token="hf-secret",
        )

        mp_kwargs = _mixpanel_track_kwargs(mp_provider)
        ph_kwargs = _posthog_capture_kwargs(ph_provider.client)
        for properties in (mp_kwargs["properties"], ph_kwargs["properties"]):
            assert "_ctx" not in properties
            assert properties["set_civitai_api_token"] == "<redacted>"
            assert properties["set_hf_api_token"] == "<redacted>"
            assert "civ-secret" not in str(properties)
            assert "hf-secret" not in str(properties)


class TestLazyProviderConstruction:
    """Providers must be built on first dispatch, never at module import.

    Eager construction started PostHog's consumer thread, whose atexit join
    stalls every CLI exit by the full flush_interval — even for invocations
    that never send a single event (e.g. ``comfy --version`` with
    ``DO_NOT_TRACK=1``)."""

    def test_first_track_event_constructs_providers(self, tracking_with_two_providers):
        tracking_mod, _, _ = tracking_with_two_providers
        built = [MagicMock(), MagicMock()]
        with (
            patch.object(tracking_mod, "PROVIDERS", None),
            patch.object(tracking_mod, "MixpanelProvider", return_value=built[0]),
            patch.object(tracking_mod, "PostHogProvider", return_value=built[1]),
        ):
            assert tracking_mod.PROVIDERS is None
            tracking_mod.track_event("some_event")
            assert tracking_mod.PROVIDERS == built
            built[0].track.assert_called_once()
            built[1].track.assert_called_once()

    def test_disabled_tracking_never_constructs_providers(self, tracking_with_two_providers):
        tracking_mod, _, _ = tracking_with_two_providers
        tracking_mod.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        with patch.object(tracking_mod, "PROVIDERS", None):
            tracking_mod.track_event("some_event")
            assert tracking_mod.PROVIDERS is None

    def test_env_opt_out_never_constructs_providers(self, tracking_with_two_providers):
        tracking_mod, _, _ = tracking_with_two_providers
        with (
            patch.object(tracking_mod, "PROVIDERS", None),
            patch.dict("os.environ", {"DO_NOT_TRACK": "1"}),
        ):
            tracking_mod.track_event("some_event")
            assert tracking_mod.PROVIDERS is None

    def test_get_providers_constructs_once_and_caches(self):
        import comfy_cli.tracking as tracking_mod

        built = MagicMock()
        with (
            patch.object(tracking_mod, "PROVIDERS", None),
            patch.object(tracking_mod, "MixpanelProvider", return_value=built) as mp_cls,
            patch.object(tracking_mod, "PostHogProvider", return_value=built) as ph_cls,
        ):
            first = tracking_mod._get_providers()
            second = tracking_mod._get_providers()
            assert first is second
            mp_cls.assert_called_once()
            ph_cls.assert_called_once()

    def test_posthog_flush_interval_is_bounded(self):
        """The Posthog client must be constructed with an explicit, small
        flush_interval: its atexit join waits out the full interval on an
        empty queue, and the library default varies by version (0.5s → 5.0s),
        which would add multi-second dead time to every CLI exit."""
        with patch("posthog.Posthog") as posthog_cls:
            provider = PostHogProvider("phc_test", "https://t.comfy.org")
        assert provider.enabled is True
        kwargs = posthog_cls.call_args.kwargs
        # main replaced the branch's `flush_interval=0.2` with a
        # tighter, more direct bound: cap the consumer's drain budget, and
        # unregister posthog's own atexit join so `_flush_all_providers` is the
        # only shutdown drain and its deadline actually governs.
        assert kwargs["max_retries"] == 1
        assert kwargs["timeout"] <= 10


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

    def test_flush_is_noop_when_providers_never_constructed(self):
        """The atexit flush must not itself trigger provider construction:
        if no provider was ever built, no event was ever dispatched, so there
        is nothing to flush and no reason to pay the construction cost."""
        import comfy_cli.tracking as tracking_mod

        with (
            patch.object(tracking_mod, "PROVIDERS", None),
            patch.object(tracking_mod, "MixpanelProvider") as mp_cls,
            patch.object(tracking_mod, "PostHogProvider") as ph_cls,
        ):
            tracking_mod._flush_all_providers()
            assert tracking_mod.PROVIDERS is None
            mp_cls.assert_not_called()
            ph_cls.assert_not_called()

    def test_flush_swallows_provider_errors(self):
        import comfy_cli.tracking as tracking_mod

        p1 = MagicMock()
        p1.flush.side_effect = RuntimeError("flush failed")
        p2 = MagicMock()
        with patch.object(tracking_mod, "PROVIDERS", [p1, p2]):
            tracking_mod._flush_all_providers()

        p2.flush.assert_called_once()

    def test_flush_returns_before_deadline_when_a_provider_hangs(self):
        """A provider whose flush() blocks (e.g. a blackholed telemetry endpoint)
        must not wedge the atexit hook past its per-provider deadline. The hook
        runs each flush in a daemon thread and joins with a ~5s timeout, so a
        60s-hanging provider is abandoned rather than allowed to hang the CLI.
        Bounds the total at well under the 60s hang."""
        import comfy_cli.tracking as tracking_mod

        release = threading.Event()

        class _HangingProvider:
            def flush(self):
                # Blocks until the test tears down; the deadline must fire first.
                release.wait(timeout=60)

        fast = MagicMock()
        try:
            with patch.object(tracking_mod, "PROVIDERS", [_HangingProvider(), fast]):
                start = time.monotonic()
                tracking_mod._flush_all_providers()
                elapsed = time.monotonic() - start

            # Deadline is 5s per provider; allow generous slack but stay far
            # under the 60s the hanging provider would otherwise cost.
            assert elapsed < 8.0, f"flush did not honor the deadline (took {elapsed:.1f}s)"
            # The healthy provider after the hanging one is still drained.
            fast.flush.assert_called_once()
        finally:
            release.set()


def _override_warnings(caplog):
    """The $POSTHOG_API_KEY-was-ignored warnings captured so far."""
    return [r for r in caplog.records if "POSTHOG_API_KEY" in r.getMessage()]


class TestPostHogTokenResolution:
    """$POSTHOG_API_KEY is also what posthog-cli calls a *personal*
    (``phx_``) API key, which the ingestion endpoint rejects with a 401. Only a
    ``phc_`` project write key may override the committed default."""

    def test_unset_env_uses_the_committed_default(self, monkeypatch):
        import comfy_cli.tracking as tracking_mod

        monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
        assert tracking_mod._resolve_posthog_token() == tracking_mod._POSTHOG_DEFAULT_TOKEN

    def test_phc_override_is_honored_without_warning(self, monkeypatch, caplog):
        import comfy_cli.tracking as tracking_mod

        monkeypatch.setenv("POSTHOG_API_KEY", "phc_custom")
        with caplog.at_level(logging.WARNING):
            assert tracking_mod._resolve_posthog_token() == "phc_custom"
        assert _override_warnings(caplog) == []

    def test_empty_override_still_disables_the_provider(self, monkeypatch, caplog):
        # The pre-existing escape hatch: an empty string opts the PostHog pipe
        # out entirely (PostHogProvider.__init__ returns early on a falsy token —
        # see test_posthog_with_empty_token_is_disabled_and_silent).
        import comfy_cli.tracking as tracking_mod

        monkeypatch.setenv("POSTHOG_API_KEY", "")
        with caplog.at_level(logging.WARNING):
            assert tracking_mod._resolve_posthog_token() == ""
        assert _override_warnings(caplog) == []

        provider = PostHogProvider(tracking_mod._resolve_posthog_token(), tracking_mod.POSTHOG_HOST)
        assert provider.enabled is False

    @pytest.mark.parametrize("bad_value", ["phx_test", "not-a-key", "phc-typo"])
    def test_non_phc_override_warns_and_falls_back(self, monkeypatch, caplog, bad_value):
        import comfy_cli.tracking as tracking_mod

        monkeypatch.setenv("POSTHOG_API_KEY", bad_value)
        with caplog.at_level(logging.WARNING):
            assert tracking_mod._resolve_posthog_token() == tracking_mod._POSTHOG_DEFAULT_TOKEN

        warnings = _override_warnings(caplog)
        assert len(warnings) == 1
        # ERROR, not WARNING: the default LOG_LEVEL is ERROR, so this must be
        # visible to the user by default (see comfy_cli/tracking.py).
        assert warnings[0].levelno == logging.ERROR


class _FakePosthogClient:
    """Stands in for ``posthog.Posthog`` so provider construction records the
    token it was handed without starting a real consumer thread."""

    def __init__(self, *, project_api_key, host, **_kwargs):
        self.api_key = project_api_key
        self.host = host
        self.join = MagicMock()
        self.capture = MagicMock()
        self.flush = MagicMock()


class TestPostHogTokenResolutionThroughProviderBuild:
    def test_phx_env_still_builds_a_provider_on_the_default_key(self, monkeypatch, caplog):
        """End to end: a developer with a personal key exported gets working
        telemetry and exactly one warning, no matter how many events fire —
        the resolve happens inside the once-only provider factory."""
        import comfy_cli.tracking as tracking_mod

        monkeypatch.setenv("POSTHOG_API_KEY", "phx_test")
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.delenv("COMFY_NO_TELEMETRY", raising=False)

        with (
            # Both SDKs are imported lazily inside the providers, so patch the
            # source modules rather than a tracking attribute.
            patch("posthog.Posthog", _FakePosthogClient),
            patch("mixpanel.Mixpanel", return_value=MagicMock()),
            patch.object(tracking_mod, "PROVIDERS", None),
            patch.object(tracking_mod, "_session_only_tracking", True),
            patch.object(tracking_mod, "user_id", "test-distinct-id"),
            caplog.at_level(logging.WARNING),
        ):
            tracking_mod.track_event("first_event")
            tracking_mod.track_event("second_event")
            providers = list(tracking_mod.PROVIDERS)

        posthog_providers = [p for p in providers if type(p).__name__ == "PostHogProvider"]
        assert len(posthog_providers) == 1
        posthog_provider = posthog_providers[0]
        assert posthog_provider.enabled is True
        assert posthog_provider.client.api_key == tracking_mod._POSTHOG_DEFAULT_TOKEN
        # Two events, one build, one warning.
        assert len(_override_warnings(caplog)) == 1
        assert posthog_provider.client.capture.call_count == 2


class TestPostHogLoggerIsQuiet:
    """posthog-python logs every failed upload — 401, offline, DNS, 5xx — at
    ERROR on the single ``posthog`` logger, which comfy-cli's root handler
    prints in red. Telemetry is best-effort, so that must not reach stderr."""

    @pytest.fixture(autouse=True)
    def _restore_module_state(self, monkeypatch):
        posthog_logger = logging.getLogger("posthog")
        original_level = posthog_logger.level
        yield
        # Undo the reloads: leave the module (and the logger) exactly as a
        # normal import would, so test ordering can't leak either one.
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        import comfy_cli.tracking as tracking_mod

        importlib.reload(tracking_mod)
        posthog_logger.setLevel(original_level)

    def test_import_silences_the_posthog_logger(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        import comfy_cli.tracking as tracking_mod

        logging.getLogger("posthog").setLevel(logging.NOTSET)
        importlib.reload(tracking_mod)

        assert logging.getLogger("posthog").level == logging.CRITICAL

    def test_log_level_debug_leaves_the_posthog_logger_alone(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")  # setup_logging upper()s it too
        import comfy_cli.tracking as tracking_mod

        logging.getLogger("posthog").setLevel(logging.NOTSET)
        importlib.reload(tracking_mod)

        assert logging.getLogger("posthog").level == logging.NOTSET
