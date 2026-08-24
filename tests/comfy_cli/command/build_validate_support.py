from __future__ import annotations

from pathlib import Path

from comfy_cli.command.build_spec import JsonObject, write_build_spec


def write_spec(root: Path, *, models: list[JsonObject], nodes: list[JsonObject]) -> Path:
    path = root / "comfy-build.yaml"
    spec: JsonObject = {
        "schema": "comfy-build/1",
        "id": None,
        "name": "Fixture",
        "description": "",
        "syncedRevision": None,
        "definition": {
            "schema": "distribution-definition/0",
            "models": models,
            "customNodes": nodes,
        },
    }
    write_build_spec(path, spec)
    return path


def local_model(**extra) -> JsonObject:
    return {
        "type": "checkpoints",
        "filename": "base.safetensors",
        "localPath": "checkpoints/base.safetensors",
        "source": "local",
        **extra,
    }


def local_node(**extra) -> JsonObject:
    return {"name": "local-node", "localPath": "local-node", "source": "local", **extra}


def remote_model(filename: str | None, index: int = 0) -> JsonObject:
    entry: JsonObject = {"type": f"group-{index:03}", "blobId": f"blob-{index}"}
    if filename is not None:
        entry["filename"] = filename
    return entry


class ResolveRecorder:
    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def __call__(self, url, target, *, method="GET", body: JsonObject | None = None, timeout=30.0, max_bytes):
        assert url == "https://builder.test/v1/models/resolve"
        assert body is not None
        self.calls.append({"method": method, "body": body})
        results = []
        filenames = body["filenames"]
        assert isinstance(filenames, list)
        for filename in filenames:
            assert isinstance(filename, str)
            if filename == "public.safetensors":
                results.append({"filename": filename, "candidates": [{"sourceUri": "https://models.example/public"}]})
            elif filename == "outage.safetensors":
                results.append({"filename": filename, "candidates": [], "error": "providers unavailable"})
            else:
                results.append({"filename": filename, "candidates": []})
        return 200, {"results": results}
