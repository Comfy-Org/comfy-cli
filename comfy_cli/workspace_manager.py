import concurrent.futures
import os
import pdb
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
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
from comfy_cli.env_checker import EnvChecker
from comfy_cli.utils import get_os, singleton


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
    remote: str
    name: Optional[str] = None
    updated_at: datetime = None


@dataclass
class CustomNode:
    # Todo: Add custom node fields for comfy-lock.yaml
    path: str
    enabled: bool
    is_git: Optional[bool] = False
    remote: Optional[str] = None


@dataclass
class ComfyLockYAMLStruct:
    basics: Basics
    models: List[Model] = field(default_factory=list)
    custom_nodes: List[CustomNode] = field(default_factory=list)


def check_comfy_repo(path):
    if not os.path.exists(path):
        return False, None
    try:
        repo = git.Repo(path, search_parent_directories=True)
        path_is_comfy_repo = any(
            remote.url in constants.COMFY_ORIGIN_URL_CHOICES for remote in repo.remotes
        )
        if path_is_comfy_repo:
            return path_is_comfy_repo, repo
        else:
            return False, None
    # Not in a git repo at all
    except git.exc.InvalidGitRepositoryError:
        return False, None


def _serialize_dataclass(data):
    """Serialize dataclasses, lists, and datetimes, filtering out None values."""
    if is_dataclass(data):
        # Convert dataclass to dict, omitting keys with None values
        return {
            k: _serialize_dataclass(v) for k, v in asdict(data).items() if v is not None
        }
    elif isinstance(data, list):
        # Serialize each item in the list
        return [_serialize_dataclass(item) for item in data]
    # TODO: keep in mind for any other data type to create condition to serialize it properly
    elif isinstance(data, datetime):
        return data.isoformat()
    return data


def load_yaml(file_path: str) -> ComfyLockYAMLStruct:
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

        basics = Basics(
            remote=data.get("basics", {}).get("remote", ""),
            name=data.get("basics", {}).get("name"),
            updated_at=(
                datetime.fromisoformat(data.get("basics", {}).get("updated_at"))
                if data.get("basics", {}).get("updated_at")
                else None
            ),
        )

        models = [
            Model(
                name=m.get("name"),
                url=m.get("url"),
                paths=[ModelPath(path=p["path"]) for p in m.get("paths", [])],
                hash=m.get("hash"),
                type=m.get("type"),
            )
            for m in data.get("models", [])
        ]

        custom_nodes = [
            CustomNode(
                path=cn.get("path"),
                enabled=cn.get("enabled", False),
                is_git=cn.get("is_git"),
                remote=cn.get("remote"),
            )
            for cn in data.get("custom_nodes", [])
        ]

        return ComfyLockYAMLStruct(
            basics=basics, models=models, custom_nodes=custom_nodes
        )


def save_yaml(file_path: str, metadata: ComfyLockYAMLStruct):
    # Serialize the dataclass to a dictionary, handling nested dataclasses and lists
    data = _serialize_dataclass(metadata)

    # Write the dictionary to a YAML file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(
            "# Beta Feature: This file is generated with comfy-cli to track ComfyUI state\n"
        )
        yaml.safe_dump(
            data, file, default_flow_style=False, allow_unicode=True, sort_keys=True
        )


# Function to check if the file is config.json
def check_file_is_model(path):
    if path.name.endswith(constants.SUPPORTED_PT_EXTENSIONS):
        return str(path)


def check_folder_is_git(path) -> Tuple[bool, Optional[git.Repo]]:
    try:
        repo = git.Repo(path)
        return True, repo
    except git.exc.InvalidGitRepositoryError:
        return False, None


class WorkspaceType(Enum):
    CURRENT_DIR = "current_dir"
    DEFAULT = "default"
    SPECIFIED = "specified"
    RECENT = "recent"
    NOT_FOUND = "not_found"


@singleton
class WorkspaceManager:
    def __init__(
        self,
    ):
        self.config_manager = ConfigManager()
        self.metadata = ComfyLockYAMLStruct(basics=Basics(remote=""), models=[])
        self.specified_workspace = None
        self.use_here = None
        self.use_recent = None
        self.workspace_path = None
        self.workspace_type = None
        self.workspace_repo = None
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
        (
            self.workspace_path,
            self.workspace_type,
            self.workspace_repo,
        ) = self.get_workspace_path()
        if self.workspace_type != WorkspaceType.NOT_FOUND:
            self.metadata.basics.remote = self.workspace_repo.remotes[0].url
        self.skip_prompting = skip_prompting

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

    def set_default_launch_extras(self, extras: str):
        """
        Sets the default workspace path in the configuration.
        """
        self.config_manager.set(
            constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, extras.strip()
        )

    def get_specified_workspace(self):
        if self.specified_workspace is None:
            return None

        return os.path.abspath(os.path.expanduser(self.specified_workspace))

    def get_workspace_path(self) -> Tuple[str, WorkspaceType, Optional[git.Repo]]:
        """
        Retrieves the workspace path and type based on the following precedence:
        1. Specified Workspace (--workspace)
        2. Most Recent (if --recent is True)
        3. Current Directory (if --here is True)
        4. Current Directory (if current dir is ComfyUI repo and --no-here is not True)
        5. Default Workspace (if a default workspace has been set using `comfy set-default`)
        6. Most Recent Workspace (if --no-recent is not True)
        7. Fallback Default Workspace ('~/comfy' for linux or ~/Documents/comfy for windows/macos)

        Raises:
            FileNotFoundError: If no valid workspace is found.
        """
        # Check for explicitly specified workspace first
        specified_workspace = self.get_specified_workspace()
        if specified_workspace:
            found_comfy_repo, repo = check_comfy_repo(specified_workspace)
            if found_comfy_repo:
                return specified_workspace, WorkspaceType.SPECIFIED, repo

            print(
                "[bold red]warn: The specified workspace is not ComfyUI directory.[/bold red]"
            )  # If a path has been explicitly specified, cancel the command for safety.
            raise typer.Exit(code=1)

        # Check for recent workspace if requested
        if self.use_recent:
            recent_workspace = self.config_manager.get(
                constants.CONFIG_KEY_RECENT_WORKSPACE
            )
            if recent_workspace:
                found_comfy_repo, repo = check_comfy_repo(recent_workspace)
                if found_comfy_repo:
                    return recent_workspace, WorkspaceType.RECENT, repo
            else:
                print(
                    "[bold red]warn: No recent workspace has been set.[/bold red]"
                )  # If a path has been explicitly specified, cancel the command for safety.
                raise typer.Exit(code=1)

            print(
                "[bold red]warn: The recent workspace is not ComfyUI.[/bold red]"
            )  # If a path has been explicitly specified, cancel the command for safety.
            raise typer.Exit(code=1)

        # Check for current workspace if requested
        if self.use_here:
            current_directory = os.getcwd()
            found_comfy_repo, comfy_repo = check_comfy_repo(current_directory)
            if found_comfy_repo:
                return comfy_repo.working_dir, WorkspaceType.CURRENT_DIR, comfy_repo

            print(
                "[bold red]warn: you are not current in a ComfyUI directory.[/bold red]"
            )
            raise typer.Exit(code=1)

        # Check the current directory for a ComfyUI
        if self.use_here is None:
            current_directory = os.getcwd()
            found_comfy_repo, comfy_repo = check_comfy_repo(
                os.path.join(current_directory)
            )
            # If it's in a sub dir of the ComfyUI repo, get the repo working dir
            if found_comfy_repo:
                return comfy_repo.working_dir, WorkspaceType.CURRENT_DIR, comfy_repo

        # Check for user-set default workspace
        default_workspace = self.config_manager.get(
            constants.CONFIG_KEY_DEFAULT_WORKSPACE
        )

        if default_workspace:
            found_comfy_repo, repo = check_comfy_repo(default_workspace)
            return default_workspace, WorkspaceType.DEFAULT, repo

        # Fallback to the most recent workspace if it exists
        if self.use_recent is None:
            recent_workspace = self.config_manager.get(
                constants.CONFIG_KEY_RECENT_WORKSPACE
            )
            if recent_workspace:
                found_comfy_repo, repo = check_comfy_repo(recent_workspace)
                if found_comfy_repo:
                    return recent_workspace, WorkspaceType.RECENT, repo

        # Check for comfy-cli default workspace
        default_workspace = utils.get_not_user_set_default_workspace()
        found_comfy_repo, repo = check_comfy_repo(default_workspace)
        if found_comfy_repo:
            return default_workspace, WorkspaceType.DEFAULT, repo

        return None, WorkspaceType.NOT_FOUND, None

    def get_comfyui_manager_path(self):
        if self.workspace_path is None:
            return None

        # To check more robustly, verify up to the `.git` path.
        manager_path = os.path.join(
            self.workspace_path, "custom_nodes", "ComfyUI-Manager"
        )
        return manager_path

    def is_comfyui_manager_installed(self):
        if self.workspace_path is None:
            return False

        # To check more robustly, verify up to the `.git` path.
        manager_git_path = os.path.join(
            self.workspace_path, "custom_nodes", "ComfyUI-Manager", ".git"
        )
        return os.path.exists(manager_git_path)

    def scan_dir(self):
        logging.info(f"Scanning directory: {self.workspace_path}")
        model_files = []
        custom_node_folders = []
        counter = 0
        for root, _dirs, files in os.walk(self.workspace_path):
            for file in files:
                counter += 1
                if file.endswith(constants.SUPPORTED_PT_EXTENSIONS):
                    model_files.append(
                        Model(
                            name=os.path.basename(file),
                            paths=[ModelPath(path=file)],
                        )
                    )
        for custom_node_path in os.listdir(
            os.path.join(self.workspace_path, "custom_nodes")
        ):
            if not os.path.isdir(custom_node_path):
                continue
            if custom_node_path in constants.IGNORE_CUSTOM_NODE_FOLDERS:
                continue
            is_git_custom_node, repo = check_folder_is_git(custom_node_path)
            if is_git_custom_node:
                custom_node_folders.append(
                    CustomNode(
                        path=custom_node_path,
                        enabled=not custom_node_path.endswith(".disabled"),
                        is_git=True,
                        remote=repo.remotes[0].url if repo.remotes else None,
                    )
                )
            else:
                custom_node_folders.append(
                    CustomNode(
                        path=custom_node_path,
                        enabled=not custom_node_path.endswith(".disabled"),
                        is_git=False,
                        remote=None,
                    )
                )

        self.metadata.custom_nodes = custom_node_folders
        self.metadata.models = model_files
        self.metadata.basics.updated_at = datetime.now()
        save_yaml(
            os.path.join(self.workspace_path, constants.COMFY_LOCK_YAML_FILE),
            self.metadata,
        )

    def scan_dir_concur(self):
        base_path = Path(".")
        model_files = []

        # Use ThreadPoolExecutor to manage concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(check_file_is_model, p) for p in base_path.rglob("*")
            ]
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
