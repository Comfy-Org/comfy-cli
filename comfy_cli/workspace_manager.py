import concurrent.futures
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import git
import typer
import yaml
from rich import print

from comfy_cli import constants, logging, utils
from comfy_cli.config_manager import ConfigManager
from comfy_cli.utils import singleton


@dataclass
class ModelPath:
    path: str


@dataclass
class Model:
    name: Optional[str] = None
    url: Optional[str] = None
    paths: List[ModelPath] = field(default_factory=list)
    hash: Optional[str] = None
    type: Optional[str] = None


@dataclass
class Basics:
    name: Optional[str] = None
    updated_at: datetime = None


@dataclass
class CustomNode:
    # Todo: Add custom node fields for comfy-lock.yaml
    pass


@dataclass
class ComfyLockYAMLStruct:
    basics: Basics
    models: List[Model] = field(default_factory=list)
    custom_nodes: List[CustomNode] = field(default_factory=list)


def check_comfy_repo(path) -> Tuple[bool, Optional[git.Repo]]:
    if not os.path.exists(path):
        return False, None
    try:
        repo = git.Repo(path, search_parent_directories=True)
        path_is_comfy_repo = any(remote.url in constants.COMFY_ORIGIN_URL_CHOICES for remote in repo.remotes)

        # If it's within the custom node repo, lookup from the parent directory.
        if not path_is_comfy_repo and "custom_nodes" in path:
            parts = path.split(os.sep)
            try:
                index = parts.index("custom_nodes")
                path = os.sep.join(parts[:index])

                repo = git.Repo(path, search_parent_directories=True)
                path_is_comfy_repo = any(remote.url in constants.COMFY_ORIGIN_URL_CHOICES for remote in repo.remotes)
            except ValueError:
                pass

        if path_is_comfy_repo:
            return path_is_comfy_repo, repo
        else:
            return False, None
    # Not in a git repo at all
    # pylint: disable=E1101  # no-member
    except git.exc.InvalidGitRepositoryError:
        return False, None


# Generate and update this following method using chatGPT
# def load_yaml(file_path: str) -> ComfyLockYAMLStruct:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = yaml.safe_load(file)
#         basics = Basics(
#             name=data.get("basics", {}).get("name"),
#             updated_at=(
#                 datetime.fromisoformat(data.get("basics", {}).get("updated_at"))
#                 if data.get("basics", {}).get("updated_at")
#                 else None
#             ),
#         )
#         models = [
#             Model(
#                 name=m.get("model"),
#                 url=m.get("url"),
#                 paths=[ModelPath(path=p.get("path")) for p in m.get("paths", [])],
#                 hash=m.get("hash"),
#                 type=m.get("type"),
#             )
#             for m in data.get("models", [])
#         ]
#         custom_nodes = []


# Generate and update this following method using chatGPT
def save_yaml(file_path: str, metadata: ComfyLockYAMLStruct):
    data = {
        "basics": {
            "name": metadata.basics.name,
            "updated_at": metadata.basics.updated_at.isoformat(),
        },
        "models": [
            {
                "model": m.name,
                "url": m.url,
                "paths": [{"path": p.path} for p in m.paths],
                "hash": m.hash,
                "type": m.type,
            }
            for m in metadata.models
        ],
        "custom_nodes": [],
    }
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, default_flow_style=False, allow_unicode=True)


# Function to check if the file is config.json
def check_file_is_model(path):
    if path.name.endswith(constants.SUPPORTED_PT_EXTENSIONS):
        return str(path)


class WorkspaceType(Enum):
    CURRENT_DIR = "current_dir"
    DEFAULT = "default"
    SPECIFIED = "specified"
    RECENT = "recent"


@singleton
class WorkspaceManager:
    def __init__(
        self,
    ):
        self.config_manager = ConfigManager()
        self.metadata = ComfyLockYAMLStruct(basics=Basics(), models=[])
        self.specified_workspace = None
        self.use_here = None
        self.use_recent = None
        self.workspace_path = None
        self.workspace_type = None
        self.skip_prompting = None

    def setup_workspace_manager(
        self,
        specified_workspace: Optional[str] = None,
        use_here: Optional[bool] = None,
        use_recent: Optional[bool] = None,
        skip_prompting: Optional[bool] = None,
    ):
        self.specified_workspace = specified_workspace
        self.use_here = use_here
        self.use_recent = use_recent
        self.workspace_path, self.workspace_type = self.get_workspace_path()
        self.skip_prompting = skip_prompting

    def set_recent_workspace(self, path: str):
        """
        Sets the most recent workspace path in the configuration.
        """
        self.config_manager.set(constants.CONFIG_KEY_RECENT_WORKSPACE, os.path.abspath(path))

    def set_default_workspace(self, path: str):
        """
        Sets the default workspace path in the configuration.
        """
        self.config_manager.set(constants.CONFIG_KEY_DEFAULT_WORKSPACE, os.path.abspath(path))

    def set_default_launch_extras(self, extras: str):
        """
        Sets the default workspace path in the configuration.
        """
        self.config_manager.set(constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, extras.strip())

    def __get_specified_workspace(self) -> Optional[str]:
        if self.specified_workspace is None:
            return None

        return os.path.abspath(os.path.expanduser(self.specified_workspace))

    def get_workspace_path(self) -> Tuple[str, WorkspaceType]:
        """
        Retrieves a workspace path based on user input and defaults. This function does not validate the existence of a validate ComfyUI workspace.
        1. Specified Workspace (--workspace)
        2. Most Recent (if --recent is True)
        3. Current Directory (if --here is True)
        4. Current Directory (if current dir is ComfyUI repo and --no-here is not True)
        5. Default Workspace (if a default workspace has been set using `comfy set-default`)
        6. Most Recent Workspace (if --no-recent is not True)
        7. Fallback Default Workspace ('~/comfy' for linux or ~/Documents/comfy for windows/macos)

        """
        # Check for explicitly specified workspace first
        specified_workspace = self.__get_specified_workspace()
        if specified_workspace:
            return specified_workspace, WorkspaceType.SPECIFIED

        # Check for recent workspace if requested
        if self.use_recent:
            recent_workspace = self.config_manager.get(constants.CONFIG_KEY_RECENT_WORKSPACE)
            if recent_workspace:
                return recent_workspace, WorkspaceType.RECENT
            else:
                print(
                    "[bold red]warn: No recent workspace has been set.[/bold red]"
                )  # If a path has been explicitly specified, cancel the command for safety.
                raise typer.Exit(code=1)

        # Check for current workspace if requested
        if self.use_here is True:
            current_directory = os.getcwd()
            found_comfy_repo, comfy_repo = check_comfy_repo(current_directory)
            if found_comfy_repo:
                return comfy_repo.working_dir, WorkspaceType.CURRENT_DIR
            else:
                return (
                    os.path.join(current_directory, "ComfyUI"),
                    WorkspaceType.CURRENT_DIR,
                )

        # Check the current directory for a ComfyUI
        if self.use_here is None:
            current_directory = os.getcwd()
            found_comfy_repo, comfy_repo = check_comfy_repo(os.path.join(current_directory))
            # If it's in a sub dir of the ComfyUI repo, get the repo working dir
            if found_comfy_repo:
                return comfy_repo.working_dir, WorkspaceType.CURRENT_DIR

        # Check for user-set default workspace
        default_workspace = self.config_manager.get(constants.CONFIG_KEY_DEFAULT_WORKSPACE)

        if default_workspace and check_comfy_repo(default_workspace)[0]:
            return default_workspace, WorkspaceType.DEFAULT

        # Fallback to the most recent workspace if it exists
        if self.use_recent is None:
            recent_workspace = self.config_manager.get(constants.CONFIG_KEY_RECENT_WORKSPACE)
            if recent_workspace and check_comfy_repo(recent_workspace)[0]:
                return recent_workspace, WorkspaceType.RECENT
            else:
                print(
                    f"[bold red]warn: The recent workspace {recent_workspace} is not a valid ComfyUI path.[/bold red]"
                )

        # Check for comfy-cli default workspace
        default_workspace = utils.get_not_user_set_default_workspace()
        return default_workspace, WorkspaceType.DEFAULT

    def get_comfyui_manager_path(self):
        if self.workspace_path is None:
            return None

        # To check more robustly, verify up to the `.git` path.
        manager_path = os.path.join(self.workspace_path, "custom_nodes", "ComfyUI-Manager")
        return manager_path

    def is_comfyui_manager_installed(self):
        if self.workspace_path is None:
            return False

        # To check more robustly, verify up to the `.git` path.
        manager_git_path = os.path.join(self.workspace_path, "custom_nodes", "ComfyUI-Manager", ".git")
        return os.path.exists(manager_git_path)

    def scan_dir(self):
        if not self.workspace_path:
            return []

        logging.info(f"Scanning directory: {self.workspace_path}")
        model_files = []
        for root, _dirs, files in os.walk(self.workspace_path):
            for file in files:
                if file.endswith(constants.SUPPORTED_PT_EXTENSIONS):
                    model_files.append(os.path.join(root, file))
        return model_files

    def scan_dir_concur(self):
        base_path = Path(".")
        model_files = []

        # Use ThreadPoolExecutor to manage concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_file_is_model, p) for p in base_path.rglob("*")]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    model_files.append(future.result())

        return model_files

    def load_metadata(self):
        file_path = os.path.join(self.workspace_path, constants.COMFY_LOCK_YAML_FILE)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        else:
            return {}

    def save_metadata(self):
        file_path = os.path.join(self.workspace_path, constants.COMFY_LOCK_YAML_FILE)
        save_yaml(file_path, self.metadata)

    def fill_print_table(self, table):
        table.add_row(
            "Current selected workspace",
            f"[bold green]→ {self.workspace_path}[/bold green]",
        )
