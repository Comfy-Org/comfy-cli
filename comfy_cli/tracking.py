from __future__ import annotations

import atexit
import functools
import json
import logging as logginglib
import os
import sys
import uuid
from typing import Any, Protocol

import typer
from mixpanel import Mixpanel
from posthog import Posthog

from comfy_cli import constants, logging, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager

# Ignore logs from urllib3 that Mixpanel/PostHog use.
logginglib.getLogger("urllib3").setLevel(logginglib.ERROR)

MIXPANEL_TOKEN = "93aeab8962b622d431ac19800ccc9f67"

# phc_* are public client-side write keys designed for embedding — safe to commit, same as MIXPANEL_TOKEN above.
# Override with $POSTHOG_API_KEY.
POSTHOG_TOKEN = os.environ.get(
    "POSTHOG_API_KEY",
    "phc_iKfK86id4xVYws9LybMje0h44eGtfwFgRPIBehmy8rO",
)
POSTHOG_HOST = "https://t.comfy.org"

# Only these events get the tracing_id --> workflow_run_id alias on PostHog.
EXECUTION_EVENTS = frozenset({"execution_start", "execution_success", "execution_error"})

# Namespace applied to event names on PostHog only, matching the
# app:/hub:/registry: surface-prefix convention in the shared project. Mixpanel
# keeps the bare legacy names (see ``mixpanel_name`` in track_event) so its
# historical streams stay continuous.
POSTHOG_EVENT_PREFIX = "cli:"

# Sanitize command kwargs before sending them as telemetry: _is_sensitive()
# masks credential-bearing names, _is_trackable() drops ctx/private/unserializable
# values, and _scrub_value() strips query strings off URL values.

_SENSITIVE_SUFFIXES = ("_token", "_api_key", "_secret", "_password")
# `token` is the publish PAT; `changelog` is bulky free text with no analytics
# value beyond its presence. Sensitive values become "<redacted>" (the key is
# kept so we can still tell the option was supplied).
_SENSITIVE_EXACT = frozenset({"api_key", "token", "password", "secret", "changelog"})


def _is_sensitive(name: str) -> bool:
    """True if *name* looks like a credential. Case-insensitive; matches the
    snake_case suffixes only (Typer kwargs are always snake_case)."""
    lower = name.lower()
    return lower in _SENSITIVE_EXACT or lower.endswith(_SENSITIVE_SUFFIXES)


def _is_trackable(name: str, value: object) -> bool:
    """True if the (name, value) kwarg is safe to send. Drops ctx/context,
    underscore-prefixed names, and values json can't serialize -- posthog-python
    coerces unserializable values and ships them (e.g. a Click Context) rather
    than raising the way Mixpanel does, so we must reject them ourselves."""
    if name in ("ctx", "context"):
        return False
    if name.startswith("_"):
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return False
    return True


def _scrub_value(value: object) -> object:
    """Strip the query string and fragment from URL values; CivitAI download
    links carry the token as ?token=. Only top-level http(s) strings are touched."""
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return value.partition("?")[0].partition("#")[0]
    return value


# Generate a unique tracing ID per command.
config_manager = ConfigManager()
cli_version = config_manager.get_cli_version()

# tracking all events for a single user
user_id = config_manager.get(constants.CONFIG_KEY_USER_ID)
# tracking all events for a single command
tracing_id = str(uuid.uuid4())
workspace_manager = WorkspaceManager()

# Process-scoped opt-in used when running non-interactively before the
# user has ever recorded a consent choice. Captures agentic usage without
# persisting the consent flag, so a later interactive run can still
# prompt the human. The anonymous user_id is persisted separately for
# stable agent identity in analytics.
_session_only_tracking = False


def _telemetry_disabled_by_env() -> bool:
    """Return True if telemetry is suppressed via environment variable.

    Honors the cross-tool ``DO_NOT_TRACK`` convention
    (https://consoledonottrack.com/) and the project-specific
    ``COMFY_NO_TELEMETRY``. Per the spec, any value other than empty or
    ``"0"`` opts out.
    """
    for name in ("DO_NOT_TRACK", "COMFY_NO_TELEMETRY"):
        val = os.environ.get(name, "")
        if val and val != "0":
            return True
    return False


class TelemetryProvider(Protocol):
    enabled: bool

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


class MixpanelProvider:
    def __init__(self, token: str):
        self.client = Mixpanel(token) if token else None
        self.enabled = self.client is not None

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None:
        if not self.enabled or distinct_id is None:
            return
        self.client.track(distinct_id=distinct_id, event_name=event_name, properties=properties)

    def flush(self) -> None:
        # mixpanel-python ships per-call over sync HTTP; nothing to drain.
        return


class PostHogProvider:
    _STANDARD_PROPERTIES = {
        "environment": "cli",
        "surface": "cli",
        "source": "cli",
        "trigger_source": "cli",
    }

    def __init__(self, token: str, host: str):
        self.client: Posthog | None = None
        self.enabled = False
        if not token:
            return
        # disable_geoip=False lets PostHog enrich events with IP-derived location.
        self.client = Posthog(project_api_key=token, host=host, disable_geoip=False)
        self.enabled = True

    def track(self, event_name: str, distinct_id: str | None, properties: dict[str, Any]) -> None:
        if not self.enabled or self.client is None or distinct_id is None:
            return
        merged = {**self._STANDARD_PROPERTIES, **properties}
        # Membership check uses the canonical (unprefixed) name; the prefix is
        # cosmetic to the PostHog taxonomy and applied only at capture time.
        if event_name in EXECUTION_EVENTS and "tracing_id" in merged:
            merged.setdefault("workflow_run_id", merged["tracing_id"])
        self.client.capture(event=f"{POSTHOG_EVENT_PREFIX}{event_name}", distinct_id=distinct_id, properties=merged)

    def flush(self) -> None:
        if self.client is None:
            return
        # posthog-python ships asynchronously; without flush, short-lived CLI invocations silently drop in-flight events
        self.client.flush()


PROVIDERS: list[TelemetryProvider] = [
    MixpanelProvider(MIXPANEL_TOKEN),
    PostHogProvider(POSTHOG_TOKEN, POSTHOG_HOST),
]

app = typer.Typer()


@app.command()
def enable():
    init_tracking(True)
    typer.echo(f"Tracking is now {'enabled'}.")
    init_tracking(True)


@app.command()
def disable():
    init_tracking(False)
    typer.echo(f"Tracking is now {'disabled'}.")


def track_event(event_name: str, properties: Any = None, *, mixpanel_name: str | None = None):
    """Fire ``event_name`` to every enabled telemetry provider.

    ``mixpanel_name``, if supplied, overrides the event name on the Mixpanel pipe only — used to keep
    legacy Mixpanel event names while PostHog receives the canonical name.
    """
    if _telemetry_disabled_by_env():
        return
    if properties is None:
        properties = {}
    logging.debug(f"tracking event called with event_name: {event_name} and properties: {properties}")
    enable_tracking = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if not enable_tracking and not _session_only_tracking:
        return

    properties = {**properties, "cli_version": cli_version, "tracing_id": tracing_id}

    for provider in PROVIDERS:
        provider_event_name = (
            mixpanel_name if (mixpanel_name is not None and isinstance(provider, MixpanelProvider)) else event_name
        )
        try:
            provider.track(provider_event_name, distinct_id=user_id, properties=dict(properties))
        except Exception as e:
            logging.warning(f"Failed to track event via {type(provider).__name__}: {e}")


def filter_command_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop untrackable kwargs (see ``_is_trackable``), redact sensitive values
    (see ``_is_sensitive``), and strip credentials embedded in URL values
    (see ``_scrub_value``)."""
    return {
        k: ("<redacted>" if v is not None else None) if _is_sensitive(k) else _scrub_value(v)
        for k, v in kwargs.items()
        if _is_trackable(k, v)
    }


def track_command(sub_command: str = None):
    """
    A decorator factory that logs the command function name and selected arguments when it's called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            command_name = f"{sub_command}:{func.__name__}" if sub_command is not None else func.__name__
            filtered_kwargs = filter_command_kwargs(kwargs)
            logging.debug(f"Tracking command: {command_name} with arguments: {filtered_kwargs}")
            track_event(command_name, properties=filtered_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def prompt_tracking_consent(skip_prompt: bool = False, default_value: bool = False):
    global _session_only_tracking, user_id

    # Env-var opt-out short-circuits everything below: no prompt, no
    # auto-enable in non-TTY, no user_id persistence. Per-process only —
    # the on-disk consent flag is left untouched so a later run without
    # the env var still gets the normal prompt path.
    if _telemetry_disabled_by_env():
        return

    if _session_only_tracking:
        return

    tracking_enabled = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if tracking_enabled is not None:
        return

    if skip_prompt:
        init_tracking(default_value)
        return

    # When stdin or stdout is not a TTY (subprocess pipe, redirect, CI),
    # blocking on the consent prompt would either hang the caller forever
    # or corrupt their output stream. Enable tracking for this process and
    # persist a stable anonymous user_id so repeat agentic usage from the
    # same machine attributes to one identity. The consent flag itself
    # stays unset so a later interactive run can still ask the human; if
    # they consent, init_tracking will reuse this user_id.
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _session_only_tracking = True
        if user_id is None:
            user_id = str(uuid.uuid4())
            # Best-effort persistence — a read-only config dir (fresh CI,
            # restricted sandbox) must not crash the caller. If the write
            # fails we keep the in-memory user_id so this process still
            # tracks normally; the next run on a writable host will retry.
            try:
                config_manager.set(constants.CONFIG_KEY_USER_ID, user_id)
            except OSError:
                pass
        return

    enable_tracking = ui.prompt_confirm_action("Do you agree to enable tracking to improve the application?", False)
    init_tracking(enable_tracking)


def init_tracking(enable_tracking: bool):
    """
    Initialize the tracking system by setting the user identifier and tracking enabled status.
    """
    global user_id
    logging.debug(f"Initializing tracking with enable_tracking: {enable_tracking}")
    config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, str(enable_tracking))
    if not enable_tracking:
        return

    curr_user_id = config_manager.get(constants.CONFIG_KEY_USER_ID)
    logging.debug(f'User identifier for tracking user_id found: {curr_user_id}."')
    if curr_user_id is None:
        curr_user_id = str(uuid.uuid4())
        config_manager.set(constants.CONFIG_KEY_USER_ID, curr_user_id)
        logging.debug(f'Setting user identifier for tracking user_id: {curr_user_id}."')
    user_id = curr_user_id

    # Note: only called once when the user interacts with the CLI for the
    #  first time iff the permission is granted.
    install_event_triggered = config_manager.get_bool(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED)
    if not install_event_triggered:
        logging.debug("Tracking install event.")
        config_manager.set(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED, "True")
        track_event("install")


def _flush_all_providers() -> None:
    for provider in PROVIDERS:
        try:
            provider.flush()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Failed to flush telemetry provider {type(provider).__name__}: {e}")


atexit.register(_flush_all_providers)
