from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import yaml

@dataclass
class ModelPath:
    path: str

@dataclass
class Model:
    model: str
    url: str
    paths: List[ModelPath]
    hash: str
    type: str

@dataclass
class Basics:
    name: str
    updated_at: datetime

@dataclass
class YAMLStructure:
    basics: Basics
    models: List[Model]

    def to_yaml(self, file_path: str):
        data = {
            "basics": [{"name": b.name, "updated_at": b.updated_at.isoformat()} for b in self.basics],
            "models": [
                {
                    "model": m.model,
                    "url": m.url,
                    "paths": [{"path": p.path} for p in m.paths],
                    "hash": m.hash,
                    "type": m.type
                } for m in self.models
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            basics = [Basics(name=b["name"], updated_at=datetime.fromisoformat(b["updated_at"])) for b in data["basics"]]
            models = [
                Model(
                    model=m["model"],
                    url=m["url"],
                    paths=[ModelPath(path=p["path"]) for p in m["paths"]],
                    hash=m["hash"],
                    type=m["type"]
                ) for m in data["models"]
            ]
            return cls(basics=basics, models=models)

# Example usage:
data = YAMLStructure(
    basics=[Basics(name="Example Name", updated_at=datetime.now())],
    models=[
        Model(
            model="Model Name",
            url="https://huggingface.co/example",
            paths=[ModelPath(path="/path/to/model1"), ModelPath(path="/path/to/model2")],
            hash="abc123def456",
        )
    ]
)

# Serialize to YAML
file_path = 'data.yaml'
data.to_yaml(file_path)

# Deserialize from YAML
loaded_data = YAMLStructure.from_yaml(file_path)