from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NodeVersion:
    changelog: str
    dependencies: List[str]
    deprecated: bool
    id: str
    version: str
    download_url: str


@dataclass
class Node:
    id: str
    name: str
    description: str
    author: Optional[str] = None
    license: Optional[str] = None
    icon: Optional[str] = None
    repository: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    latest_version: Optional[NodeVersion] = None


@dataclass
class PublishNodeVersionResponse:
    node_version: NodeVersion
    signedUrl: str


@dataclass
class URLs:
    homepage: str = ""
    documentation: str = ""
    repository: str = ""
    issues: str = ""


@dataclass
class Model:
    location: str
    model_url: str


@dataclass
class ComfyConfig:
    publisher_id: str = ""
    display_name: str = ""
    icon: str = ""
    models: List[Model] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    banner_url: str = ""
    web: Optional[str] = None


@dataclass
class License:
    file: str = ""
    text: str = ""


@dataclass
class ProjectConfig:
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    requires_python: str = ">= 3.9"
    dependencies: List[str] = field(default_factory=list)
    license: License = field(default_factory=License)
    urls: URLs = field(default_factory=URLs)
    supported_os: List[str] = field(default_factory=list)
    supported_accelerators: List[str] = field(default_factory=list)
    supported_comfyui_version: str = ""
    supported_comfyui_frontend_version: str = ""


@dataclass
class PyProjectConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    tool_comfy: ComfyConfig = field(default_factory=ComfyConfig)
