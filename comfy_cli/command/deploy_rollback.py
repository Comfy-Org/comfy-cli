"""Roll a deployment back to an earlier release, and read the releases it ran."""

from __future__ import annotations

import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from comfy_cli.command.build_paths import resolve_build_paths
from comfy_cli.command.build_spec import JsonObject, read_build_spec
from comfy_cli.command.deploy_resolve import (
    AmbiguousDeploymentError,
    BuilderReleaseClient,
    BuildNotPushedError,
    DeployResolveError,
    deployment_id_for,
    find_build_release,
    release_version_selector,
)
from comfy_cli.command.deploy_types import (
    DeployUpClient,
    MoveResult,
    move_changed,
    optional_revision,
    release_summary,
    required_int,
    required_string,
    server_shape_error,
)
from comfy_cli.command.deploy_up import build_deployments, refuse_unmovable, release_or_id, running_first
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge


class DeployRollbackClient(DeployUpClient, Protocol):
    def rollback_deployment(
        self, deployment_id: str, base_revision: int, to_revision: int | None = None
    ) -> JsonObject: ...

    def get_deployment_revisions(self, deployment_id: str) -> JsonObject: ...


class RevisionsUnavailableError(DeployResolveError):
    code = "deploy_updates_unavailable"

    def __init__(self, deployment_id: str, command: str) -> None:
        self.hint = (
            "run `comfy build release ls --id <build>` to find the earlier release, then "
            "`comfy deploy up --create --release <id> --gpu <class> --region <region>` from the Build folder "
            "to start a separate deployment on it"
            if command == "rollback"
            else "run `comfy deploy events` to see what the deployment did"
        )
        self.details = {"deployment_id": deployment_id}
        super().__init__(
            f"deployment {deployment_id} carries no revision, so `{command}` has no revisions to work from; "
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


class ReleaseNotListedError(DeployResolveError):
    code = "deploy_bad_request"

    def __init__(self, build_id: str, deployment_id: str, selector: str) -> None:
        self.hint = f"run `comfy deploy history --deployment {deployment_id}` and pass the release id with `--to`"
        self.details = {"deployment_id": deployment_id, "buildId": build_id, "to": selector}
        super().__init__(f"Build {build_id} lists no release {selector}; it may have been deleted")


class ReleaseNeverRanError(DeployResolveError):
    code = "deploy_bad_request"

    def __init__(self, deployment_id: str, selector: str) -> None:
        self.hint = f"run `comfy deploy history --deployment {deployment_id}` to see the releases it ran"
        self.details = {"deployment_id": deployment_id, "to": selector}
        super().__init__(f"deployment {deployment_id} never ran release {selector} before its current revision")


# The service's reasons for refusing a rollback, said plainly.
_REFUSALS = {
    "NO_EARLIER_REVISION": (
        "has no earlier release to roll back to",
        "it still runs the release it was created on; `comfy deploy up --deployment {id}` moves it to a newer one",
    ),
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
class History:
    deployment_id: str
    revisions: list[JsonObject]

    def payload(self) -> JsonObject:
        return {"deploymentId": self.deployment_id, "revisions": self.revisions}


# A Build's id and the releases it lists.
BuildReleases = tuple[str, list[JsonObject]]


def _picked_deployment(
    builder: BuilderReleaseClient, client: DeployUpClient, request: RollbackRequest
) -> tuple[str, BuildReleases | None]:
    """The deployment named, else the Build's only one up, as `up` picks it,
    with the Build's releases where picking read them."""
    if request.deployment_id is not None:
        return deployment_id_for(builder, client, request.deployment_id, path=request.path), None
    spec = read_build_spec(resolve_build_paths(request.path).spec_file)
    build_id = spec.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise BuildNotPushedError
    releases = builder.list_releases(build_id)
    pool = running_first(build_deployments(client.list_all_deployments(), releases))
    if not pool:
        raise NoDeploymentToRollBackError(build_id)
    if len(pool) > 1:
        raise RollbackAmbiguousError(build_id, [required_string(row, "id") for row in pool])
    return required_string(pool[0], "id"), (build_id, releases)


def _revisions(client: DeployRollbackClient, deployment_id: str) -> list[JsonObject]:
    items = client.get_deployment_revisions(deployment_id).get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise server_shape_error("the deployment revisions response has no items array")
    return sorted(items, key=lambda item: required_int(item, "revision"))


def _build_of(builder: BuilderReleaseClient, release_id: str) -> BuildReleases:
    """The Build the release belongs to, and its releases."""
    build_id = required_string(builder.get_release(release_id), "buildId")
    return build_id, builder.list_releases(build_id)


def _summaries(releases: list[JsonObject]) -> dict[str, JsonObject]:
    known = {}
    for release in releases:
        version = release.get("version")
        known[required_string(release, "id")] = (
            release_summary(release) if isinstance(version, int) else {"id": release["id"]}
        )
    return known


def _labels(builder: BuilderReleaseClient, release_id: str) -> dict[str, JsonObject]:
    """Each of the Build's releases by id, for their versions only.

    A failed lookup costs the versions in the output, not the command.
    """
    try:
        return _summaries(_build_of(builder, release_id)[1])
    except (DeployAPIError, ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError):
        return {}


def _listed_release_id(build: BuildReleases, deployment_id: str, selector: str) -> str:
    build_id, releases = build
    listed = find_build_release(releases, selector)
    if listed is None:
        raise ReleaseNotListedError(build_id, deployment_id, selector)
    return required_string(listed, "id")


def _latest_run(
    deployment_id: str, revisions: Sequence[JsonObject], current: int, release_id: str, selector: str
) -> int:
    """The latest revision before ``current`` that ran ``release_id``."""
    ran = [
        required_int(item, "revision")
        for item in revisions
        if required_int(item, "revision") < current and item.get("releaseId") == release_id
    ]
    if not ran:
        raise ReleaseNeverRanError(deployment_id, selector)
    return max(ran)


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
        hint=hint.format(id=deployment_id),
    )


def rollback(builder: BuilderReleaseClient, client: DeployRollbackClient, request: RollbackRequest) -> MoveResult:
    deployment_id, build = _picked_deployment(builder, client, request)
    target = client.get_deployment(deployment_id)
    base_revision = optional_revision(target)
    if base_revision is None:
        raise RevisionsUnavailableError(deployment_id, "rollback")
    refuse_unmovable(target, command="rollback")
    previous_id = required_string(target, "releaseId")
    # Without `--to` the service picks the revision before the current one.
    to_revision = None
    if request.to is not None:
        selector = request.to.strip()
        # A version is looked up in the Build's releases; a release id is
        # matched against the revisions, so a release the Build no longer
        # lists can still be returned to.
        release_id = selector
        if release_version_selector(selector) is not None:
            build = build or _build_of(builder, previous_id)
            release_id = _listed_release_id(build, deployment_id, selector)
        if release_id == previous_id:
            # Already there, so nothing is sent, as `up` treats a release the deployment serves.
            known = _summaries(build[1]) if build else _labels(builder, previous_id)
            current = known.get(previous_id) or {"id": previous_id}
            return MoveResult(target, current, current, False)
        to_revision = _latest_run(deployment_id, _revisions(client, deployment_id), base_revision, release_id, selector)
    try:
        moved = client.rollback_deployment(deployment_id, base_revision, to_revision)
    except DeployAPIError as error:
        raise _plain(error, deployment_id) from error
    # The reply names the release it moved to, or, while that waits, the release it moves to on the pending update.
    pending = moved.get("pendingUpdate")
    release_id = required_string(pending if isinstance(pending, dict) else moved, "releaseId")
    changed = move_changed(moved, base_revision, release_id, previous_id)
    # The reply carries no status, so the deployment is read for it.
    deployment = client.get_deployment(deployment_id)
    # Read after the move, so a failed lookup costs only the versions.
    known = _summaries(build[1]) if build else _labels(builder, previous_id)
    release = known.get(release_id) or release_or_id(builder, release_id)
    return MoveResult(deployment, release, known.get(previous_id) or {"id": previous_id}, changed)


def history(builder: BuilderReleaseClient, client: DeployRollbackClient, deployment_id: str) -> History:
    deployment = client.get_deployment(deployment_id)
    # The deployment's own revision is the one it serves.
    current = optional_revision(deployment)
    if current is None:
        raise RevisionsUnavailableError(deployment_id, "history")
    known = _labels(builder, required_string(deployment, "releaseId"))
    revisions = _revisions(client, deployment_id)
    rows: list[JsonObject] = []
    for item in reversed(revisions):
        row: JsonObject = {
            "revision": required_int(item, "revision"),
            "releaseId": required_string(item, "releaseId"),
            "kind": required_string(item, "kind"),
            "createdBy": required_string(item, "createdBy"),
            "createdAt": required_string(item, "createdAt"),
        }
        row["current"] = row["revision"] == current
        version = known.get(row["releaseId"], {}).get("version")
        if isinstance(version, int):
            row["releaseVersion"] = version
        from_revision = item.get("fromRevision")
        if isinstance(from_revision, int) and not isinstance(from_revision, bool):
            row["fromRevision"] = from_revision
        rows.append(row)
    return History(deployment_id, rows)
