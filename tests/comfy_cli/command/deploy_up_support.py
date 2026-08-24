from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

from comfy_cli.command.build_spec import JsonObject


def deployment(
    deployment_id: str,
    *,
    release_id: str = "release-5",
    status: str = "ready",
    gpu: str = "l4",
    region: str = "US-MO-2",
    minimum: int = 0,
    maximum: int = 1,
    deleted_at: str | None = None,
) -> JsonObject:
    return {
        "id": deployment_id,
        "distributionVersionId": release_id,
        "status": status,
        "computeConfig": {"gpuClass": gpu, "region": region, "min": minimum, "max": maximum},
        "createdAt": f"2026-08-23T12:00:{deployment_id[-1].zfill(2) if deployment_id[-1].isdigit() else '00'}Z",
        "deletedAt": deleted_at,
    }


class FakeBuilder:
    def __init__(self, releases: list[JsonObject] | None = None) -> None:
        self.releases = releases or [{"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True}]
        self.calls: list[tuple[str, str]] = []

    def get_release(self, release_id: str) -> JsonObject:
        self.calls.append(("get_release", release_id))
        return next(release for release in self.releases if release["id"] == release_id)

    def list_releases(self, build_id: str) -> list[JsonObject]:
        self.calls.append(("list_releases", build_id))
        return copy.deepcopy(self.releases)


class FakeDeploy:
    """Thread-safe in-memory control plane; mutation is its documented purpose."""

    def __init__(
        self,
        rows: list[JsonObject] | None = None,
        *,
        generation_barrier: threading.Barrier | None = None,
        tombstone_first_create: bool = False,
        tombstone_all_creates: bool = False,
        get_statuses: list[str] | None = None,
    ) -> None:
        self.rows = {str(row["id"]): copy.deepcopy(row) for row in rows or []}
        self.generation_barrier = generation_barrier
        self.tombstone_first_create = tombstone_first_create
        self.tombstone_all_creates = tombstone_all_creates
        self.get_statuses = list(get_statuses or [])
        self.create_keys: list[str] = []
        self.generation_deleted_counts: list[int] = []
        self.update_calls: list[str] = []
        self.start_calls: list[str] = []
        self.catalog_calls = 0
        self._keys: dict[str, str] = {}
        self._tombstoned_once = False
        self._local = threading.local()
        self._lock = threading.Lock()

    def list_all_deployments(self) -> list[JsonObject]:
        call_count = getattr(self._local, "list_count", 0) + 1
        self._local.list_count = call_count
        if call_count == 2 and self.generation_barrier is not None:
            self.generation_barrier.wait(timeout=2)
        with self._lock:
            snapshot = copy.deepcopy(list(self.rows.values()))
            if call_count >= 2:
                self.generation_deleted_counts.append(
                    sum(
                        row.get("distributionVersionId") == "release-5" and row.get("deletedAt") is not None
                        for row in snapshot
                    )
                )
            return snapshot

    def create_deployment(
        self,
        build_version_id: str,
        compute_config: JsonObject,
        *,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        assert idempotency_key is not None
        with self._lock:
            self.create_keys.append(idempotency_key)
            existing_id = self._keys.get(idempotency_key)
            if existing_id is not None:
                existing = self.rows[existing_id]
                return {"id": existing_id, "status": existing["status"]}
            deployment_id = f"dep-{len(self._keys) + 1}"
            row = deployment(deployment_id, release_id=build_version_id)
            row["computeConfig"] = copy.deepcopy(compute_config)
            self.rows[deployment_id] = row
            self._keys[idempotency_key] = deployment_id
            if self.tombstone_all_creates or (self.tombstone_first_create and not self._tombstoned_once):
                row["deletedAt"] = "2026-08-23T12:30:00Z"
                self._tombstoned_once = True
            return {"id": deployment_id, "status": row["status"]}

    def get_deployment(self, deployment_id: str) -> JsonObject:
        with self._lock:
            row = self.rows[deployment_id]
            if self.get_statuses:
                row["status"] = self.get_statuses.pop(0)
            return copy.deepcopy(row)

    def update_deployment(self, deployment_id: str, compute_config: JsonObject) -> JsonObject:
        with self._lock:
            self.update_calls.append(deployment_id)
            self.rows[deployment_id]["computeConfig"] = copy.deepcopy(compute_config)
            return copy.deepcopy(self.rows[deployment_id])

    def start_deployment(self, deployment_id: str) -> JsonObject:
        with self._lock:
            self.start_calls.append(deployment_id)
            self.rows[deployment_id]["status"] = "queued"
            return copy.deepcopy(self.rows[deployment_id])

    def get_compute_catalog(self) -> JsonObject:
        self.catalog_calls += 1
        return {
            "regions": [
                {
                    "region": "US-MO-2",
                    "label": "Missouri",
                    "gpus": [
                        {"gpuClass": "l4", "label": "NVIDIA L4", "vramGb": 24},
                        {"gpuClass": "a100", "label": "NVIDIA A100", "vramGb": 80},
                    ],
                },
                {
                    "region": "EU-RO-1",
                    "label": "Romania",
                    "gpus": [{"gpuClass": "l4", "label": "NVIDIA L4", "vramGb": 24}],
                },
            ]
        }

    def soft_delete(self, deployment_id: str) -> None:
        with self._lock:
            self.rows[deployment_id]["deletedAt"] = "2026-08-23T12:30:00Z"


def write_spec(root: Path) -> Path:
    path = root / "comfy-build.json"
    path.write_text(
        json.dumps(
            {
                "schema": "comfy-build/1",
                "id": "build-1",
                "name": "example",
                "description": "",
                "syncedRevision": None,
                "definition": {},
            }
        ),
        encoding="utf-8",
    )
    return path
