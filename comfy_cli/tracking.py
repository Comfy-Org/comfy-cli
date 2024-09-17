import functools
import logging as logginglib
import uuid
import threading

import typer
from mixpanel import Mixpanel

from comfy_cli import constants, logging, ui
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager

# Ignore logs from urllib3 that Mixpanel uses.
logginglib.getLogger("urllib3").setLevel(logginglib.ERROR)

MIXPANEL_TOKEN = "93aeab8962b622d431ac19800ccc9f67"
mp = Mixpanel(MIXPANEL_TOKEN) if MIXPANEL_TOKEN else None

# Generate a unique tracing ID per command.
config_manager = ConfigManager()
cli_version = config_manager.get_cli_version()

# tracking all events for a single user
user_id = config_manager.get(constants.CONFIG_KEY_USER_ID)
# tracking all events for a single command
tracing_id = str(uuid.uuid4())
workspace_manager = WorkspaceManager()

app = typer.Typer()


@app.command()
def enable():
    set_tracking_enabled(True)
    typer.echo(f"Tracking is now {'enabled'}.")
    init_tracking(True)


@app.command()
def disable():
    set_tracking_enabled(False)
    typer.echo(f"Tracking is now {'disabled'}.")


def track_event(event_name: str, properties: any = None, timeout: float = 10):
    if properties is None:
        properties = {}
    logging.debug(f"tracking event called with event_name: {event_name} and properties: {properties}")
    enable_tracking = config_manager.get(constants.CONFIG_KEY_ENABLE_TRACKING)
    if not enable_tracking:
        return

    try:
        properties["cli_version"] = cli_version
        properties["tracing_id"] = tracing_id

        # Define a function to wrap the mp.track call to enable timeout
        def track_call():
            mp.track(distinct_id=user_id, event_name=event_name, properties=properties)

        # Create a thread to run the track_call function
        track_thread = threading.Thread(target=track_call)

        # Start the thread
        track_thread.start()

        # Wait for the thread to finish within the timeout period
        track_thread.join(timeout)

        # Check if the thread is still active after the timeout
        if track_thread.is_alive():
            logging.warning(f"Tracking event '{event_name}' timed out after {timeout} seconds")
        else:
            logging.debug(f"Successfully tracked event '{event_name}'")

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
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "ctx" and k != "context"}

            logging.debug(f"Tracking command: {command_name} with arguments: {filtered_kwargs}")
            track_event(command_name, properties=filtered_kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def prompt_tracking_consent(skip_prompt: bool = False, default_value: bool = False):
    tracking_enabled = config_manager.get(constants.CONFIG_KEY_ENABLE_TRACKING)
    if tracking_enabled is not None:
        return

    if skip_prompt:
        init_tracking(default_value)
    else:
        enable_tracking = ui.prompt_confirm_action("Do you agree to enable tracking to improve the application?", False)
        init_tracking(enable_tracking)


def init_tracking(enable_tracking: bool):
    """
    Initialize the tracking system by setting the user identifier and tracking enabled status.
    """
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

    # Note: only called once when the user interacts with the CLI for the
    #  first time iff the permission is granted.
    install_event_triggered = config_manager.get(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED)
    if not install_event_triggered:
        logging.debug("Tracking install event.")
        config_manager.set(constants.CONFIG_KEY_INSTALL_EVENT_TRIGGERED, "True")
        track_event("install")


def set_tracking_enabled(enabled: bool):
    config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, str(enabled))
    return enabled
