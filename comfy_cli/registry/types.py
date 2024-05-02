from dataclasses import dataclass
from typing import List


@dataclass
class NodeConfiguration:
    publisher_id: str
    node_id: str
    display_name: str
    description: str
    version: str
    license: str
    dependencies: List[str]
    tags: List[str]
    repository: str
    documentation: str
    author: str
    issues: str
    icon: str
