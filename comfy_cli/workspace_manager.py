import os
import sys
from typing import Optional

import typer

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import EnvChecker
from comfy_cli.utils import singleton
from rich import print


@singleton
class WorkspaceManager:
    def __init__(self):
        self.config_manager = ConfigManager()

    @staticmethod
    def update_context(
        context: typer.Context,
        workspace: Optional[str],
        recent: Optional[bool],
        here: Optional[bool],
    ):
        """
        Updates the context object with the workspace and recent flags.
        """

        if [workspace is not None, recent, here].count(True) > 1:
            print(
                f"--workspace, --recent, and --here options cannot be used together.",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        if workspace:
            context.obj[constants.CONTEXT_KEY_WORKSPACE] = workspace
        if recent:
            context.obj[constants.CONTEXT_KEY_RECENT] = recent
        if here:
            context.obj[constants.CONTEXT_KEY_HERE] = here

    def set_recent_workspace(self, path: str):
        """
        Sets the most recent workspace path in the configuration.
        """
        self.config_manager.set(
            constants.CONFIG_KEY_RECENT_WORKSPACE, os.path.abspath(path)
        )

    def set_default_workspace(self, path: str):
        """
        Sets the default workspace path in the configuration.
        """
        self.config_manager.set(
            constants.CONFIG_KEY_DEFAULT_WORKSPACE, os.path.abspath(path)
        )

    def get_workspace_comfy_path(self, context: typer.Context) -> str:
        """
        Retrieves the workspace path and appends '/ComfyUI' to it.
        """

        return self.get_workspace_path(context) + "/ComfyUI"

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
        use_here = context.obj.get(constants.CONTEXT_KEY_HERE)

        # Check for explicitly specified workspace first
        if specified_workspace:
            specified_path = os.path.expanduser(specified_workspace)
            if os.path.exists(specified_path):
                if os.path.exists(os.path.join(specified_path, "ComfyUI")):
                    return specified_path

            print(
                f"[bold red]warn: The specified workspace does not contain ComfyUI directory.[/bold red]"
            )  # If a path has been explicitly specified, cancel the command for safety.
            raise typer.Exit(code=1)

        # Check for recent workspace if requested
        if use_recent:
            recent_workspace = self.config_manager.get(
                constants.CONFIG_KEY_RECENT_WORKSPACE
            )
            if recent_workspace and os.path.exists(recent_workspace):
                if os.path.exists(os.path.join(recent_workspace, "ComfyUI")):
                    return recent_workspace

            print(
                f"[bold red]warn: The specified workspace does not contain ComfyUI directory.[/bold red]"
            )  # If a path has been explicitly specified, cancel the command for safety.
            raise typer.Exit(code=1)

        # Check for current workspace if requested
        if use_here:
            current_directory = os.getcwd()
            if os.path.exists(os.path.join(current_directory, "ComfyUI")):
                return current_directory

            comfy_repo = EnvChecker().comfy_repo
            if comfy_repo is not None:
                if os.path.basename(comfy_repo.working_dir) == "ComfyUI":
                    return os.path.abspath(os.path.join(comfy_repo.working_dir, ".."))
                else:
                    print(
                        f"[bold red]warn: The path name of the ComfyUI executed through 'comfy-cli' must be 'ComfyUI'. The current ComfyUI is being ignored.[/bold red]"
                    )
                    raise typer.Exit(code=1)

            print(
                f"[bold red]warn: The specified workspace does not contain ComfyUI directory.[/bold red]"
            )  # If a path has been explicitly specified, cancel the command for safety.
            raise typer.Exit(code=1)

        # Check for user-set default workspace
        default_workspace = self.config_manager.get(
            constants.CONFIG_KEY_DEFAULT_WORKSPACE
        )
        if default_workspace and os.path.exists(
            os.path.join(default_workspace, "ComfyUI")
        ):
            return default_workspace

        # Check the current directory for a ComfyUI setup
        current_directory = os.getcwd()
        if os.path.exists(os.path.join(current_directory, "ComfyUI")):
            return current_directory

        # Check the current directory for a ComfyUI repo
        comfy_repo = EnvChecker().comfy_repo
        if (
            comfy_repo is not None
            and os.path.basename(comfy_repo.working_dir) == "ComfyUI"
        ):
            return os.path.abspath(os.path.join(comfy_repo.working_dir, ".."))

        # Fallback to the most recent workspace if it exists
        if not use_recent:
            recent_workspace = self.config_manager.get(
                constants.CONFIG_KEY_RECENT_WORKSPACE
            )
            if recent_workspace and os.path.exists(
                os.path.join(recent_workspace, "ComfyUI")
            ):
                return recent_workspace

        # Final fallback to a hardcoded default workspace
        fallback_default = os.path.expanduser("~/comfy")
        if not os.path.exists(fallback_default):
            os.makedirs(fallback_default)  # Ensure the directory exists if not found
        return fallback_default
