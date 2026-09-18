from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.build_spec import JsonObject, JsonValue, read_build_spec, write_build_spec


class RecordingBuilder:
    def __init__(self) -> None:
        self.calls: list[JsonObject] = []
        self.blobs: list[JsonObject] = []
        self.uploaded: list[bytes] = []
        self.snapshots: list[JsonObject] = []
        self.model_candidates: dict[str, list[JsonObject]] = {}
        self.checked_nodes: list[JsonObject] | None = None
        self.remote_revisions: dict[str, str] = {}
        self.created_id = "build-created"
        self.revision_number = 0
        self.stale_updates = 0
        self.always_stale = False
        self.build_targets: list[JsonObject] = [
            {"target": {"os": "linux", "gpu": "nvidia"}, "label": "Linux NVIDIA", "artifactKind": "image"},
            {"target": {"os": "linux", "gpu": "cpu"}, "label": "Linux CPU", "artifactKind": "image"},
        ]

    def _revision(self) -> str:
        self.revision_number += 1
        return f"revision-{self.revision_number}"

    def resolve_snapshot(self, snapshot: JsonObject) -> JsonObject:
        self.calls.append({"method": "resolve_snapshot", "snapshot": snapshot})
        self.snapshots.append(snapshot)
        if self.checked_nodes is not None:
            checked = self.checked_nodes
        else:
            snapshots = snapshot["snapshots"]
            assert isinstance(snapshots, list)
            payload = snapshots[0]
            assert isinstance(payload, dict)
            raw_nodes = payload["customNodes"]
            assert isinstance(raw_nodes, list)
            checked = []
            for raw in raw_nodes:
                assert isinstance(raw, dict)
                if raw["type"] == "cnr":
                    checked.append(
                        {
                            "name": raw["dirName"],
                            "id": raw["id"],
                            "registryVersion": raw["version"],
                        }
                    )
                else:
                    checked.append({"name": raw["dirName"], "repository": raw["url"]})
        return {"definition": {"customNodes": checked}, "report": {}}

    def resolve_models(self, filenames: list[str]) -> list[JsonObject]:
        self.calls.append({"method": "resolve_models", "filenames": filenames})
        return [{"filename": filename, "candidates": self.model_candidates.get(filename, [])} for filename in filenames]

    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str]:
        blob_id = f"blob-{len(self.blobs) + 1}"
        self.blobs.append(
            {"kind": kind, "filename": filename, "sha256": sha256, "sizeBytes": size_bytes, "blobId": blob_id}
        )
        self.calls.append({"method": "create_blob", "blobId": blob_id})
        return blob_id, f"https://uploads.example/{blob_id}"

    def upload_blob(self, upload_url: str, path: Path) -> None:
        self.calls.append({"method": "upload_blob", "url": upload_url})
        self.uploaded.append(path.read_bytes())

    def create_build(self, name: str, definition: JsonObject, description: str | None = None) -> str:
        revision = self._revision()
        self.remote_revisions[self.created_id] = revision
        self.calls.append(
            {
                "method": "create_build",
                "id": self.created_id,
                "name": name,
                "description": description,
                "definition": definition,
            }
        )
        return self.created_id

    def get_build(self, build_id: str) -> JsonObject:
        revision = self.remote_revisions.setdefault(build_id, self._revision())
        self.calls.append({"method": "get_build", "id": build_id, "updatedAt": revision})
        return {"id": build_id, "updatedAt": revision}

    def update_build(
        self,
        build_id: str,
        definition: JsonObject,
        expected_updated_at: str | None,
        *,
        name: str,
        description: str,
    ) -> JsonObject:
        self.calls.append(
            {
                "method": "update_build",
                "id": build_id,
                "definition": definition,
                "expectedUpdatedAt": expected_updated_at,
                "name": name,
                "description": description,
            }
        )
        if self.always_stale or self.stale_updates > 0:
            self.stale_updates = max(0, self.stale_updates - 1)
            raise stale_error()
        if expected_updated_at != self.remote_revisions.get(build_id):
            raise stale_error()
        revision = self._revision()
        self.remote_revisions[build_id] = revision
        return {"id": build_id, "updatedAt": revision, "name": name, "description": description}

    def list_build_targets(self) -> list[JsonObject]:
        self.calls.append({"method": "list_build_targets"})
        return self.build_targets

    def create_release(self, build_id: str, targets: list[JsonObject] | None = None) -> tuple[str, str]:
        # The real client refuses an implicit target before issuing a request; a
        # fake that accepted one would hide exactly the bug this guards.
        if not isinstance(targets, list) or not targets:
            raise ValueError("create_release requires a non-empty list of targets")
        self.calls.append({"method": "create_release", "id": build_id, "targets": targets})
        release_id = f"release-{sum(1 for call in self.calls if call['method'] == 'create_release')}"
        return release_id, f"https://builder.test/v1/releases/{release_id}"


def stale_error() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://builder.test/v1/builds/build-1",
        409,
        "Conflict",
        {},
        io.BytesIO(b'{"error":"STALE"}'),
    )


def make_workspace(root: Path) -> Path:
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "custom_nodes" / "local-node").mkdir(parents=True)
    (root / "models" / "checkpoints" / "base.safetensors").write_bytes(b"MODEL")
    (root / "custom_nodes" / "local-node" / "nodes.py").write_bytes(b"NODE")
    return root


def local_model(**extra: JsonValue) -> JsonObject:
    return {
        "type": "checkpoints",
        "filename": "base.safetensors",
        "localPath": "checkpoints/base.safetensors",
        "source": "local",
        **extra,
    }


def local_node(**extra: JsonValue) -> JsonObject:
    return {"name": "local-node", "localPath": "local-node", "source": "local", **extra}


def write_spec(
    root: Path,
    *,
    build_id: str | None = None,
    revision: str | None = None,
    name: str = "Fixture",
    description: str = "Fixture description",
    models: list[JsonObject] | None = None,
    nodes: list[JsonObject] | None = None,
    definition_extra: JsonObject | None = None,
) -> Path:
    definition: JsonObject = {
        "schema": "distribution-definition/0",
        "models": [*models] if models is not None else [local_model()],
        "customNodes": [*nodes] if nodes is not None else [local_node()],
        **(definition_extra or {}),
    }
    spec: JsonObject = {
        "schema": "comfy-build/1",
        "id": build_id,
        "name": name,
        "description": description,
        "syncedRevision": revision,
        "definition": definition,
    }
    path = root / "comfy-build.yaml"
    write_build_spec(path, spec)
    return path


def invoke_push(root: Path, *args: str, agentic: bool = True):
    return CliRunner().invoke(
        cli_app,
        ["build", "push", *args, str(root)],
        env={
            "AI_AGENT": "1" if agentic else None,
            "COMFY_OUTPUT": "json" if agentic else "pretty",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": None,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def envelope(result) -> JsonObject:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.output
    parsed = json.loads(lines[-1])
    assert isinstance(parsed, dict)
    return parsed


def reloaded(root: Path) -> JsonObject:
    return read_build_spec(root / "comfy-build.yaml")
