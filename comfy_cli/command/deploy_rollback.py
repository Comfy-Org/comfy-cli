"""Roll a deployment back to an earlier release, and read the releases it ran."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from comfy_cli.command.build_paths import resolve_build_paths
from comfy_cli.command.build_spec import JsonObject, read_build_spec
from comfy_cli.command.deploy_resolve import (
    AmbiguousDeploymentError,
    BuilderReleaseClient,
    BuildNotPushedError,
    DeployResolveError,
    find_build_release,
)
from comfy_cli.command.deploy_types import (
    DeployUpClient,
    move_changed,
    optional_revision,
    release_summary,
    required_int,
    required_string,
    server_shape_error,
)
from comfy_cli.command.deploy_up import build_deployments, raise_unless_landed, refuse_unmovable, running_first
from comfy_cli.deploy_api_errors import DeployAPIError


class DeployRollbackClient(DeployUpClient, Protocol):
    def rollback_deployment(
        self, deployment_id: str, base_revision: int, to_revision: int | None = None
    ) -> JsonObject: ...

    def get_deployment_revisions(self, deployment_id: str) -> JsonObject: ...


class RevisionsUnavailableError(DeployResolveError):
    code = "deploy_updates_unavailable"

    def __init__(self, deployment_id: str, command: str) -> None:
        self.hint = (
            "run `comfy deploy up --create --release <id>` to start a separate deployment on an earlier release"
            if command == "rollback"
            else "run `comfy deploy events` to see what the deployment did"
        )
        self.details = {"deployment_id": deployment_id}
        super().__init__(
            f"deployment {deployment_id} carries no revision, so `{command}` has no history to work from; "
            "deployment updates are most likely not on for this workspace yet"
        )


class NoDeploymentToRollBackError(DeployResolveError):
    code = "deploy_not_found"
    hint = "run `comfy deploy ls` to find one, then pass `--deployment <id>`"

    def __init__(self, build_id: str) -> None:
        self.details = {"buildId": build_id}
        super().__init__(f"Build {build_id} has no deployment to roll back")


class RollbackAmbiguousError(AmbiguousDeploymentError):
    hint = "pass `--deployment <id>` to roll back one of them"

    def __init__(self, build_id: str, candidate_ids: list[str]) -> None:
        super().__init__(build_id, candidate_ids)
        self.args = (f"Build {build_id} has {len(candidate_ids)} deployments and `rollback` moves only one",)


class ReleaseNeverRanError(DeployResolveError):
    code = "deploy_bad_request"

    def __init__(self, deployment_id: str, selector: str) -> None:
        self.hint = f"run `comfy deploy history --deployment {deployment_id}` to see the releases it ran"
        self.details = {"deployment_id": deployment_id, "to": selector}
        super().__init__(f"deployment {deployment_id} never ran release {selector} before its current revision")


# The service's reasons for refusing a rollback, said plainly.
_REFUSALS = {
    "NO_EARLIER_REVISION": ("has no earlier release to roll back to", None),
    "STALE_REVISION": (
        "changed after it was read, so the rollback was refused",
        "run the rollback again to roll back from where it is now",
    ),
    "PENDING_UPDATE_EXISTS": (
        "is waiting on an update, so the rollback was refused",
        "wait until `comfy deploy status --deployment {id}` shows no update, then roll back",
    ),
    "RELEASE_DELETED": (
        "cannot go back to that release, because it was deleted",
        "run `comfy deploy history --deployment {id}` and pick another with `--to`",
    ),
}


@dataclass(frozen=True, slots=True)
class RollbackRequest:
    path: str | None
    deployment_id: str | None
    to: str | None


@dataclass(frozen=True, slots=True)
class RollbackResult:
    deployment: JsonObject
    release: JsonObject
    previous_release: JsonObject
    changed: bool

    @property
    def waiting(self) -> bool:
        return isinstance(self.deployment.get("pendingUpdate"), dict)

    def payload(self) -> JsonObject:
        deployment: JsonObject = {
            "id": required_string(self.deployment, "id"),
            "status": required_string(self.deployment, "status"),
        }
        revision = optional_revision(self.deployment)
        if revision is not None:
            deployment["revision"] = revision
        return {"deployment": deployment, "release": self.release, "previousRelease": self.previous_release}


@dataclass(frozen=True, slots=True)
class History:
    deployment_id: str
    revisions: list[JsonObject]

    def payload(self) -> JsonObject:
        return {"deploymentId": self.deployment_id, "revisions": self.revisions}


def _picked_deployment(builder: BuilderReleaseClient, client: DeployUpClient, request: RollbackRequest) -> str:
    """The deployment named, else the Build's only one up, as `up` picks it."""
    if request.deployment_id is not None:
        return request.deployment_id
    spec = read_build_spec(resolve_build_paths(request.path).spec_file)
    build_id = spec.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise BuildNotPushedError
    candidates = build_deployments(client.list_all_deployments(), builder.list_releases(build_id))
    pool = running_first(candidates)
    if not pool:
        raise NoDeploymentToRollBackError(build_id)
    if len(pool) > 1:
        raise RollbackAmbiguousError(build_id, [required_string(row, "id") for row in pool])
    return required_string(pool[0], "id")


def _revisions(client: DeployRollbackClient, deployment_id: str) -> list[JsonObject]:
    items = client.get_deployment_revisions(deployment_id).get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise server_shape_error("the deployment revisions response has no items array")
    return sorted(items, key=lambda item: required_int(item, "revision"))


def _versions(builder: BuilderReleaseClient, deployment: JsonObject) -> tuple[list[JsonObject], dict[str, JsonObject]]:
    """The Build's releases, and each one's summary by id, read from the release the deployment runs."""
    current = builder.get_release(required_string(deployment, "releaseId"))
    releases = builder.list_releases(required_string(current, "buildId"))
    known = {}
    for release in releases:
        version = release.get("version")
        known[required_string(release, "id")] = (
            release_summary(release) if isinstance(version, int) else {"id": release["id"]}
        )
    return releases, known


def _target_revision(
    deployment_id: str, revisions: Sequence[JsonObject], current: int, releases: list[JsonObject], selector: str | None
) -> JsonObject:
    """The earlier revision a rollback returns to: the one before, or the latest that ran ``selector``."""
    earlier = [item for item in revisions if required_int(item, "revision") < current]
    if selector is None:
        before = [item for item in earlier if required_int(item, "revision") == current - 1]
        if not before:
            raise DeployAPIError(
                "deploy_conflict",
                f"deployment {deployment_id} has no earlier release to roll back to",
                details={"deploymentId": deployment_id, "server_code": "NO_EARLIER_REVISION"},
            )
        return before[0]
    release = find_build_release(releases, selector.strip())
    ran = [item for item in earlier if release is not None and item.get("releaseId") == release.get("id")]
    if not ran:
        raise ReleaseNeverRanError(deployment_id, selector)
    return ran[-1]


def _plain(error: DeployAPIError, deployment_id: str) -> DeployAPIError:
    server_code = (error.details or {}).get("server_code")
    refusal = _REFUSALS.get(server_code) if isinstance(server_code, str) else None
    if refusal is None:
        return error
    reason, hint = refusal
    return DeployAPIError(
        error.code,
        f"deployment {deployment_id} {reason}",
        status=error.status,
        details=error.details,
        hint=hint.format(id=deployment_id) if hint else error.hint,
    )


def rollback(builder: BuilderReleaseClient, client: DeployRollbackClient, request: RollbackRequest) -> RollbackResult:
    deployment_id = _picked_deployment(builder, client, request)
    target = client.get_deployment(deployment_id)
    base_revision = optional_revision(target)
    if base_revision is None:
        raise RevisionsUnavailableError(deployment_id, "rollback")
    refuse_unmovable(target, command="rollback")
    releases, known = _versions(builder, target)
    previous_id = required_string(target, "releaseId")
    goal = _target_revision(deployment_id, _revisions(client, deployment_id), base_revision, releases, request.to)
    release_id = required_string(goal, "releaseId")
    try:
        moved = client.rollback_deployment(
            deployment_id, base_revision, None if request.to is None else required_int(goal, "revision")
        )
    except DeployAPIError as error:
        raise _plain(error, deployment_id) from error
    changed = move_changed(moved, base_revision, release_id, previous_id)
    # The reply carries no status, so the deployment is read for it.
    deployment = client.get_deployment(deployment_id)
    return RollbackResult(
        deployment,
        known.get(release_id) or {"id": release_id},
        known.get(previous_id) or {"id": previous_id},
        changed,
    )


def finish_rollback(result: RollbackResult, watched: JsonObject) -> RollbackResult:
    """The watched result, once the rollback it followed landed."""
    raise_unless_landed(watched, result.release, result.previous_release)
    return replace(result, deployment=watched)


def history(builder: BuilderReleaseClient, client: DeployRollbackClient, deployment_id: str) -> History:
    deployment = client.get_deployment(deployment_id)
    if optional_revision(deployment) is None:
        raise RevisionsUnavailableError(deployment_id, "history")
    _, known = _versions(builder, deployment)
    revisions = _revisions(client, deployment_id)
    current = revisions[-1]["revision"] if revisions else None
    rows: list[JsonObject] = []
    for item in reversed(revisions):
        row: JsonObject = {
            "revision": required_int(item, "revision"),
            "releaseId": required_string(item, "releaseId"),
            "kind": required_string(item, "kind"),
            "createdBy": required_string(item, "createdBy"),
            "createdAt": required_string(item, "createdAt"),
            "current": item["revision"] == current,
        }
        version = known.get(row["releaseId"], {}).get("version")
        if isinstance(version, int):
            row["releaseVersion"] = version
        from_revision = item.get("fromRevision")
        if isinstance(from_revision, int) and not isinstance(from_revision, bool):
            row["fromRevision"] = from_revision
        rows.append(row)
    return History(deployment_id, rows)
