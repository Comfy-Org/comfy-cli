import os
from typing import Optional

import typer

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager
from comfy_cli.utils import singleton


@singleton
class WorkspaceManager:
  def __init__(self):
    self.config_manager = ConfigManager()

  @staticmethod
  def update_context(context: typer.Context, workspace: Optional[str], recent: Optional[bool]):
    """
    Updates the context object with the workspace and recent flags.
    """
    if workspace:
      context.obj[constants.CONTEXT_KEY_WORKSPACE] = workspace
    if recent:
      context.obj[constants.CONTEXT_KEY_RECENT] = recent

  def set_recent_workspace(self, path: str):
    """
    Sets the most recent workspace path in the configuration.
    """
    self.config_manager.set(constants.CONFIG_KEY_RECENT_WORKSPACE, path)

  def set_default_workspace(self, path: str):
    """
    Sets the default workspace path in the configuration.
    """
    self.config_manager.set(constants.CONTEXT_KEY_WORKSPACE, path)

  def get_workspace_comfy_path(self, context: typer.Context) -> str:
    """
    Retrieves the workspace path and appends '/ComfyUI' to it.
    """

    return self.get_workspace_path(context) + '/ComfyUI'

  def get_workspace_path(self, context: typer.Context) -> str:
    """
    Retrieves the workspace path based on the following precedence:
    1. Specified Workspace
    2. Most Recent (if use_recent is True)
    3. User Set Default Workspace
    4. Current Directory (if it contains a ComfyUI setup)
    5. Most Recent Workspace
    6. Fallback Default Workspace ('~/comfy')

    Raises:
        FileNotFoundError: If no valid workspace is found.
    """
    specified_workspace = context.obj.get(constants.CONTEXT_KEY_WORKSPACE)
    use_recent = context.obj.get(constants.CONTEXT_KEY_RECENT)

    # Check for explicitly specified workspace first
    if specified_workspace:
      specified_path = os.path.expanduser(specified_workspace)
      if os.path.exists(specified_path):
        return specified_path

    # Check for recent workspace if requested
    if use_recent:
      recent_workspace = self.config_manager.get(constants.CONFIG_KEY_RECENT_WORKSPACE)
      if recent_workspace and os.path.exists(recent_workspace):
        return recent_workspace

    # Check for user-set default workspace
    default_workspace = self.config_manager.get(constants.CONFIG_KEY_DEFAULT_WORKSPACE)
    if default_workspace and os.path.exists(default_workspace):
      return default_workspace

    # Check the current directory for a ComfyUI setup
    current_directory = os.getcwd()
    if os.path.exists(os.path.join(current_directory, 'ComfyUI')):
      return current_directory

    # Fallback to the most recent workspace if it exists
    if not use_recent:
      recent_workspace = self.config_manager.get(constants.CONFIG_KEY_RECENT_WORKSPACE)
      if recent_workspace and os.path.exists(recent_workspace):
        return recent_workspace

    # Final fallback to a hardcoded default workspace
    fallback_default = os.path.expanduser('~/comfy')
    if not os.path.exists(fallback_default):
      os.makedirs(fallback_default)  # Ensure the directory exists if not found
    return fallback_default
