import functools
import logging as logginglib
import sys
import uuid

import typer
from mixpanel import Mixpanel

from comfy_cli import constants, logging, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager

# Ignore logs from urllib3 that Mixpanel uses.
logginglib.getLogger("urllib3").setLevel(logginglib.ERROR)

MIXPANEL_TOKEN = "93aeab8962b622d431ac19800ccc9f67"
mp = Mixpanel(MIXPANEL_TOKEN) if MIXPANEL_TOKEN else None

# Kwargs whose values must never reach tracking system.
# The key is kept (with a redacted marker) so we can still see whether the option was supplied.
SENSITIVE_TRACKING_KEYS = frozenset({"api_key"})

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


def track_event(event_name: str, properties: any = None):
    if properties is None:
        properties = {}
    logging.debug(f"tracking event called with event_name: {event_name} and properties: {properties}")
    enable_tracking = config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING)
    if not enable_tracking and not _session_only_tracking:
        return

    try:
        properties["cli_version"] = cli_version
        properties["tracing_id"] = tracing_id
        mp.track(distinct_id=user_id, event_name=event_name, properties=properties)
    except Exception as e:
        logging.warning(f"Failed to track event: {e}")  # Log the error but do not raise


def track_command(sub_command: str = None):
    """
    A decorator factory that logs the command function name and selected arguments when it's called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            command_name = f"{sub_command}:{func.__name__}" if sub_command is not None else func.__name__

            # Copy kwargs to avoid mutating original dictionary
            # Remove context and ctx from the dictionary as they are not needed for tracking and not serializable.
            filtered_kwargs = {
                k: ("<redacted>" if v is not None else None) if k in SENSITIVE_TRACKING_KEYS else v
                for k, v in kwargs.items()
                if k != "ctx" and k != "context"
            }

            logging.debug(f"Tracking command: {command_name} with arguments: {filtered_kwargs}")
            track_event(command_name, properties=filtered_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def prompt_tracking_consent(skip_prompt: bool = False, default_value: bool = False):
    global _session_only_tracking, user_id

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
