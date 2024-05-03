from dataclasses import dataclass, field
from typing import List


@dataclass
class NodeVersion:
    changelog: str
    dependencies: List[str]
    deprecated: bool
    id: str
    version: str


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


@dataclass
class ProjectConfig:
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    requires_python: str = ">= 3.9"
    dependencies: List[str] = field(default_factory=list)
    license: str = ""
    urls: URLs = field(default_factory=URLs)


@dataclass
class PyProjectConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    tool_comfy: ComfyConfig = field(default_factory=ComfyConfig)
