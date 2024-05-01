import concurrent.futures
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from comfy_cli import constants
from comfy_cli.env_checker import EnvChecker
from comfy_cli.utils import singleton
from comfy_cli.workspace_manager import WorkspaceManager


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


# Generate and update this following method using chatGPT
def load_yaml(file_path: str) -> ComfyLockYAMLStruct:
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        basics = Basics(
            name=data.get("basics", {}).get("name"),
            updated_at=(
                datetime.fromisoformat(data.get("basics", {}).get("updated_at"))
                if data.get("basics", {}).get("updated_at")
                else None
            ),
        )
        models = [
            Model(
                name=m.get("model"),
                url=m.get("url"),
                paths=[ModelPath(path=p.get("path")) for p in m.get("paths", [])],
                hash=m.get("hash"),
                type=m.get("type"),
            )
            for m in data.get("models", [])
        ]
        custom_nodes = []


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
def check_file(path):
    if path.name.endswith(constants.SUPPORTED_PT_EXTENSIONS):
        return str(path)


@singleton
class MetadataManager:
    """
    Manages the metadata (comfy.yaml) for ComfyUI when running comfy cli, including loading,
    validating, and saving metadata to a file.
    """

    def __init__(self):
        self.env_checker = EnvChecker()
        self.workspace_manager = WorkspaceManager()
        self.metadata = ComfyLockYAMLStruct(basics=Basics(), models=[])

    def scan_dir(self):
        model_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(constants.SUPPORTED_PT_EXTENSIONS):
                    model_files.append(os.path.join(root, file))
        return model_files

    def scan_dir_concur(self):
        base_path = Path(".")
        model_files = []

        # Use ThreadPoolExecutor to manage concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_file, p) for p in base_path.rglob("*")]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    model_files.append(future.result())

        return model_files

    def load_metadata(self):
        if os.path.exists(self.com):
            with open(self.metadata_file, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        else:
            return {}

    def save_metadata(self):
        save_yaml(self.metadata_file, self.metadata)


if __name__ == "__main__":
    manager = MetadataManager()

    # model_name = "example_model"
    # model_path = "/path/to/example_model"
    # manager.update_model_metadata(model_name, model_path)

    # retrieved_path = manager.get_model_path(model_name)
    # print(f"Retrieved path for {model_name}: {retrieved_path}")
