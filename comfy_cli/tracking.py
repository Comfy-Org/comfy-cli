import functools
import os
import uuid

from mixpanel import Mixpanel

from comfy_cli import logging

MIXPANEL_TOKEN = "93aeab8962b622d431ac19800ccc9f67"
DISABLE_TELEMETRY = os.getenv('DISABLE_TELEMETRY', False)
mp = Mixpanel(MIXPANEL_TOKEN) if MIXPANEL_TOKEN else None

# Generate a unique tracing ID per command.
tracing_id = str(uuid.uuid4())


def track_event(event_name: str, properties: any):
  # TODO: Implement a way to prompt the user to opt-in to telemetry.
  if not DISABLE_TELEMETRY:
    mp.track(distinct_id=tracing_id, event_name=event_name, properties=properties)


def track_command(sub_command: str = None):
  """
  A decorator factory that logs the command function name and selected arguments when it's called.
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      command_name = f"{sub_command}:{func.__name__}" if sub_command is not None else func.__name__
      input_arguments = kwargs  # Example to pass all args and kwargs

      logging.debug(f"Tracking command: {command_name} with arguments: {input_arguments}")
      track_event(command_name, properties=input_arguments)
      return func(*args, **kwargs)

    return wrapper

  return decorator
