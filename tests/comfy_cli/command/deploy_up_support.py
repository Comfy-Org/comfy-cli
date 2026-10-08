from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import typer

from comfy_cli.command import deploy
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError


def option_names(*path: str) -> set[str]:
    """Every option flag on a `comfy deploy` leaf, read from Typer rather than from
    rendered help. Paths are relative to `deploy` — `("refs", "compute")`.

    `--help` output cannot be substring-matched: rich colorizes flag names whenever it
    detects a CI terminal, which splits `--gpu` across ANSI escapes, so the match fails
    on GitHub Actions while passing locally. `build_tree_support` reads the build tree
    from Typer for the same reason.
    """
    command = typer.main.get_command(deploy.app)
    for name in path:
        command = command.commands[name]
    return {flag for param in command.params for flag in (*param.opts, *param.secondary_opts)}


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
    **served: str | None,
) -> JsonObject:
    """A deployment as comfy-deploy lists it. ``name=`` adds the name a server
    serving names answers with, null included; left out, the row is one from a
    server that predates names."""
    return {
        **served,
        "id": deployment_id,
        "releaseId": release_id,
        "status": status,
        "computeConfig": {"gpuClass": gpu, "region": region, "min": minimum, "max": maximum},
        "createdAt": f"2026-08-23T12:00:{deployment_id[-1].zfill(2) if deployment_id[-1].isdigit() else '00'}Z",
        "deletedAt": deleted_at,
    }


class FakeBuilder:
    def __init__(self, releases: list[JsonObject] | None = None, builds: list[JsonObject] | None = None) -> None:
        self.releases = releases or [{"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True}]
        self.builds = builds or [{"id": "build-1", "name": "example"}]
        self.calls: list[tuple[str, str]] = []

    def list_builds(self) -> list[JsonObject]:
        self.calls.append(("list_builds", ""))
        return copy.deepcopy(self.builds)

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
        estimate: JsonObject | Exception | None = None,
        move: str = "landed",
        get_patches: list[JsonObject] | None = None,
        strip_move_reply: bool = False,
        serves_names: bool = False,
        name_races: int = 0,
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
        # A service too old to estimate answers 404, which is what an unset
        # estimate plays, so a case not about the estimate never meets one.
        self.estimate = estimate
        self.estimate_calls: list[tuple[str, str, str]] = []
        # How a move answers: "landed" moves at once (200), "pending" waits on
        # the new release's copy (202), and "unchanged" answers 200 at the
        # revision it was asked against.
        self.move = move
        # A reply the service's rollout check failed to answer: no revision
        # and no pendingUpdate, though the move went through.
        self.strip_move_reply = strip_move_reply
        # A comfy-deploy that names deployments, as comfy-deploy's own names
        # rule does; otherwise one that predates names and answers none.
        self.serves_names = serves_names
        self.name_races = name_races
        self.create_names: list[str | None] = []
        self.rename_calls: list[tuple[str, str]] = []
        self.move_calls: list[tuple[str, int, str]] = []
        self.promote_calls: list[tuple[str, int, str]] = []
        self.rollback_calls: list[tuple[str, int, int | None]] = []
        # Each deployment's revisions, oldest first, as the service lists them.
        self.revisions: dict[str, list[JsonObject]] = {}
        # Raised by the next worker-bounds edit, when set.
        self.update_error: DeployAPIError | None = None
        # Merged into the row on each read after a move, one per read.
        self.get_patches = list(get_patches or [])
        self.get_ids: list[str] = []
        self._keys: dict[str, str] = {}
        self._tombstoned_once = False
        self._local = threading.local()
        self._lock = threading.Lock()

    def list_all_deployments(self) -> list[JsonObject]:
        call_count = getattr(self._local, "list_count", 0) + 1
        self._local.list_count = call_count
        with self._lock:
            snapshot = copy.deepcopy(list(self.rows.values()))
            self.generation_deleted_counts.append(
                sum(row.get("releaseId") == "release-5" and row.get("deletedAt") is not None for row in snapshot)
            )
        # The first list is the one a create's first key is read from, so both
        # callers read it before either creates.
        if call_count == 1 and self.generation_barrier is not None:
            self.generation_barrier.wait(timeout=2)
        return snapshot

    def create_deployment(
        self,
        release_id: str,
        compute_config: JsonObject,
        *,
        idempotency_key: str | None = None,
        name: str | None = None,
    ) -> JsonObject:
        assert idempotency_key is not None
        with self._lock:
            self.create_keys.append(idempotency_key)
            self.create_names.append(name)
            existing_id = self._keys.get(idempotency_key)
            if existing_id is not None:
                existing = self.rows[existing_id]
                return {"id": existing_id, "status": existing["status"], **self._served_name(existing)}
            if self.serves_names and name is not None:
                self._refuse_taken(name)
            if name is None and self.name_races > 0:
                self.name_races -= 1
                raise DeployAPIError(
                    "deploy_conflict",
                    "concurrent creates in this build took each default name tried; send the create again",
                    status=409,
                    details={"server_code": "NAME_RACE"},
                )
            deployment_id = f"dep-{len(self._keys) + 1}"
            row = deployment(deployment_id, release_id=release_id)
            if self.serves_names:
                row["name"] = name or self._default_name()
            row["computeConfig"] = copy.deepcopy(compute_config)
            self.rows[deployment_id] = row
            self._keys[idempotency_key] = deployment_id
            if self.tombstone_all_creates or (self.tombstone_first_create and not self._tombstoned_once):
                row["deletedAt"] = "2026-08-23T12:30:00Z"
                self._tombstoned_once = True
            return {"id": deployment_id, "status": row["status"], **self._served_name(row)}

    def rename_deployment(self, deployment_id: str, name: str) -> JsonObject:
        with self._lock:
            self.rename_calls.append((deployment_id, name))
            if not self.serves_names:
                # A body with neither computeConfig nor baseRevision.
                raise DeployAPIError(
                    "deploy_bad_request", "nothing to update", status=400, details={"server_code": "INVALID_REQUEST"}
                )
            self._refuse_taken(name, keep=deployment_id)
            self.rows[deployment_id]["name"] = name
            return copy.deepcopy(self.rows[deployment_id])

    def _served_name(self, row: JsonObject) -> JsonObject:
        return {"name": row.get("name")} if self.serves_names else {}

    def _live_names(self, keep: str | None = None) -> set[str]:
        return {
            row["name"]
            for row in self.rows.values()
            if row.get("deletedAt") is None and row["id"] != keep and isinstance(row.get("name"), str)
        }

    def _default_name(self) -> str:
        taken = self._live_names()
        if not taken:
            return "production"
        return next(f"deployment-{n}" for n in range(1, len(taken) + 2) if f"deployment-{n}" not in taken)

    def _refuse_taken(self, name: str, keep: str | None = None) -> None:
        if name in self._live_names(keep):
            raise DeployAPIError(
                "deploy_conflict",
                f"this build already has a live deployment named {name!r}",
                status=409,
                details={"server_code": "NAME_TAKEN"},
            )

    def get_deployment(self, deployment_id: str) -> JsonObject:
        with self._lock:
            self.get_ids.append(deployment_id)
            row = self.rows[deployment_id]
            if (self.move_calls or self.promote_calls or self.rollback_calls) and self.get_patches:
                row.update(self.get_patches.pop(0))
            if self.get_statuses:
                row["status"] = self.get_statuses.pop(0)
            return copy.deepcopy(row)

    def update_deployment(self, deployment_id: str, compute_config: JsonObject) -> JsonObject:
        with self._lock:
            self.update_calls.append(deployment_id)
            if self.update_error is not None:
                raise self.update_error
            self.rows[deployment_id]["computeConfig"] = copy.deepcopy(compute_config)
            return copy.deepcopy(self.rows[deployment_id])

    def move_deployment(self, deployment_id: str, base_revision: int, release_id: str) -> JsonObject:
        with self._lock:
            self.move_calls.append((deployment_id, base_revision, release_id))
            return self._apply_move(deployment_id, base_revision, release_id)

    def promote_deployment(self, deployment_id: str, base_revision: int, from_deployment_id: str) -> JsonObject:
        with self._lock:
            self.promote_calls.append((deployment_id, base_revision, from_deployment_id))
            release_id = self.rows[from_deployment_id]["releaseId"]
            reply = self._apply_move(deployment_id, base_revision, release_id)
            if self.strip_move_reply:
                reply.pop("revision", None)
                reply.pop("pendingUpdate", None)
            return reply

    def rollback_deployment(self, deployment_id: str, base_revision: int, to_revision: int | None = None) -> JsonObject:
        with self._lock:
            self.rollback_calls.append((deployment_id, base_revision, to_revision))
            row = self.rows[deployment_id]
            current = row.get("revision")
            if current == 1:
                raise DeployAPIError(
                    "deploy_conflict", "no earlier revision", status=409, details={"server_code": "NO_EARLIER_REVISION"}
                )
            goal = (current - 1) if to_revision is None else to_revision
            release_id = next(item["releaseId"] for item in self.revisions[deployment_id] if item["revision"] == goal)
            moved = self._apply_move(deployment_id, base_revision, release_id)
            if moved.get("revision") != base_revision:
                self.revisions[deployment_id].append(
                    {
                        "revision": moved["revision"],
                        "releaseId": release_id,
                        "kind": "rollback",
                        "createdBy": "user-1",
                        "createdAt": "2026-10-07T12:00:00Z",
                        "fromRevision": goal,
                    }
                )
            reply = {key: moved[key] for key in ("id", "revision", "releaseId") if key in moved}
            reply["kind"] = "rollback"
            if isinstance(moved.get("pendingUpdate"), dict):
                reply["pendingUpdate"] = {**moved["pendingUpdate"], "kind": "rollback"}
            return reply

    def get_deployment_events(self, deployment_id: str) -> JsonObject:
        return {"deploymentId": deployment_id, "events": []}

    def get_deployment_logs(self, deployment_id: str) -> JsonObject:
        return {"deploymentId": deployment_id, "capturedAt": None, "comfyuiLog": ""}

    def get_deployment_revisions(self, deployment_id: str) -> JsonObject:
        with self._lock:
            return {"items": copy.deepcopy(self.revisions.get(deployment_id, []))}

    def _apply_move(self, deployment_id: str, base_revision: int, release_id: str) -> JsonObject:
        """Call with the lock held."""
        row = self.rows[deployment_id]
        # The service answers at the same revision, before it checks the base,
        # when the deployment already serves the release.
        if row.get("releaseId") == release_id and not isinstance(row.get("pendingUpdate"), dict):
            return copy.deepcopy(row)
        if row.get("revision") != base_revision:
            raise DeployAPIError("deploy_conflict", "stale revision", status=409)
        if self.move == "landed":
            row["releaseId"] = release_id
            row["revision"] = base_revision + 1
        elif self.move == "pending":
            row["pendingUpdate"] = {
                "releaseId": release_id,
                "baseRevision": base_revision,
                "status": "provisioning",
                "since": "2026-10-07T12:00:00Z",
                "kind": "update",
            }
        else:
            # Another change landed the release first, so the service
            # answers at the revision it was asked against.
            row["releaseId"] = release_id
        return copy.deepcopy(row)

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

    def get_deploy_estimate(self, release_id: str, gpu_class: str, region: str) -> JsonObject:
        self.estimate_calls.append((release_id, gpu_class, region))
        if self.estimate is None:
            raise DeployAPIError("deploy_not_found", "the deploy service has no estimate route", status=404)
        if isinstance(self.estimate, Exception):
            raise self.estimate
        return copy.deepcopy(self.estimate)

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
