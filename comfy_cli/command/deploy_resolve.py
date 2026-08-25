"""Resolve deploy command options to a deployment id or Builder release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol

from comfy_cli.command.build_spec import BuildSpec, JsonObject, JsonValue
from comfy_cli.command.deploy_types import required_string, server_shape_error

# Wave 8 status formatting must import this table rather than redefine the order.
_STATUS_RANK: Final[dict[str, int]] = {
    "ready": 8,
    "unhealthy": 7,
    "starting": 6,
    "provisioning": 5,
    "queued": 4,
    "stopping": 3,
    "stop_failed": 2,
    "stopped": 1,
    "failed": 0,
}


class BuilderReleaseClient(Protocol):
    def get_release(self, release_id: str, /) -> JsonObject: ...

    def list_releases(self, build_id: str, /) -> list[JsonObject]: ...


class DeploymentListClient(Protocol):
    def list_all_deployments(self) -> list[JsonObject]: ...


@dataclass(frozen=True, slots=True)
class ReleaseResolveRequest:
    deployment_id: str | None = None
    release_id: str | None = None
    spec: BuildSpec | None = None


@dataclass(frozen=True, slots=True)
class DeploymentReference:
    deployment_id: str


@dataclass(frozen=True, slots=True)
class ReleaseReference:
    release: JsonObject
    build_id: str | None


class DeployResolveError(Exception):
    code: str
    hint: str
    details: JsonObject


class BuildNotPushedError(DeployResolveError):
    code = "deploy_build_not_pushed"
    hint = "run `comfy build push`"

    def __init__(self) -> None:
        self.details = {"buildId": None}
        super().__init__("the local build spec has no id")


class NoDeployableReleaseError(DeployResolveError):
    code = "deploy_no_deployable_release"

    def __init__(self, build_id: str, release_count: int) -> None:
        self.details = {"buildId": build_id, "releaseCount": release_count}
        if release_count == 0:
            self.hint = "run `comfy build release create --target linux/nvidia`"
            message = f"Build {build_id} has no releases"
        else:
            self.hint = (
                "no `linux/nvidia` artifact exists in this Build's releases; "
                "run `comfy build release create --target linux/nvidia`"
            )
            message = f"Build {build_id} has releases, but none has a deployable linux/nvidia artifact"
        super().__init__(message)


class AmbiguousDeploymentError(DeployResolveError):
    code = "deploy_ambiguous_deployment"
    hint = "pass `--deployment <id>` to select one deployment explicitly"

    def __init__(self, build_id: str, candidate_ids: list[str]) -> None:
        ordered_ids = sorted(candidate_ids)
        candidate_values: list[JsonValue] = [*ordered_ids]
        self.details = {"buildId": build_id, "candidateIds": candidate_values}
        super().__init__(f"Build {build_id} has indistinguishable deployments: {', '.join(ordered_ids)}")


def _release_version(release: JsonObject) -> int:
    version = release.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise KeyError("version")
    return version


def _required_string(value: JsonObject, key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str):
        raise KeyError(key)
    return field


def deployment_selection_key(deployment: JsonObject) -> tuple[int, datetime]:
    """Rank one deployment for "which deployment did the user mean".

    Every failure is a server-shape failure, never a traceback: an unknown
    status or an unparsable ``createdAt`` surfaces as ``deploy_server_error``
    like any other bad field. A naive timestamp is read as UTC because the wire
    format is RFC 3339 — leaving it naive would make the ``max()`` over these
    keys raise on a batch that mixes aware and naive values.
    """
    status = required_string(deployment, "status")
    created_at = required_string(deployment, "createdAt")
    try:
        rank = _STATUS_RANK[status]
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise server_shape_error("the deployment has an invalid status or createdAt", status=status) from error
    return rank, created if created.tzinfo is not None else created.replace(tzinfo=timezone.utc)


def resolve_release(
    builder: BuilderReleaseClient,
    request: ReleaseResolveRequest,
) -> DeploymentReference | ReleaseReference:
    """Apply deployment, release, then local-spec precedence without hidden I/O."""
    if request.deployment_id is not None:
        return DeploymentReference(deployment_id=request.deployment_id)

    if request.release_id is not None:
        release = builder.get_release(request.release_id)
        build_id_value = release.get("buildId")
        build_id = build_id_value if isinstance(build_id_value, str) else None
        return ReleaseReference(release=release, build_id=build_id)

    build_id_value = request.spec.get("id") if request.spec is not None else None
    if not isinstance(build_id_value, str) or not build_id_value:
        raise BuildNotPushedError

    releases = builder.list_releases(build_id_value)
    deployable = [release for release in releases if release.get("deployable") is True]
    if not deployable:
        raise NoDeployableReleaseError(build_id_value, len(releases))

    release = max(deployable, key=_release_version)
    return ReleaseReference(release=release, build_id=build_id_value)


def resolve_deployment(
    builder: BuilderReleaseClient,
    deploy: DeploymentListClient,
    build_id: str,
    *,
    include_deleted: bool = False,
) -> JsonObject | None:
    """Return the preferred deployment joined to every release of the Build."""
    release_ids = {_required_string(release, "id") for release in builder.list_releases(build_id)}
    deployments = deploy.list_all_deployments()
    candidates = [
        deployment
        for deployment in deployments
        if _required_string(deployment, "distributionVersionId") in release_ids
        and (include_deleted or deployment.get("deletedAt") is None)
    ]
    if not candidates:
        return None

    ranked = [(deployment, deployment_selection_key(deployment)) for deployment in candidates]
    winning_key = max(key for _, key in ranked)
    tied = [deployment for deployment, key in ranked if key == winning_key]
    if len(tied) > 1:
        raise AmbiguousDeploymentError(build_id, [_required_string(deployment, "id") for deployment in tied])
    return tied[0]
