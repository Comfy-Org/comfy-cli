"""Resolve deploy command options to a deployment id or Builder release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from comfy_cli.command.build_spec import BuildSpec, JsonObject, JsonValue
from comfy_cli.command.deploy_types import required_string, server_shape_error
from comfy_cli.utils import parse_rfc3339

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


class UnrelatedDeploymentError(DeployResolveError):
    code = "deploy_unrelated_deployment"
    hint = "pick one of `details.candidateIds`, which lists every deployment this command can act on"

    def __init__(self, build_id: str, deployment_id: str, candidate_ids: list[str], scope: str) -> None:
        ordered_ids = sorted(candidate_ids)
        candidate_values: list[JsonValue] = [*ordered_ids]
        self.details = {
            "buildId": build_id,
            "deploymentId": deployment_id,
            "candidateIds": candidate_values,
            "scope": scope,
        }
        # Naming the valid set is a dead end when the valid set is empty, and
        # empty is an ordinary first-use state here: a freshly cut release, or a
        # Build nothing has deployed yet.
        if not ordered_ids:
            self.hint = f"{scope} holds no deployment yet; drop `--deployment` to let the command pick or create one"
        super().__init__(f"Deployment {deployment_id} is not among {scope}")


def select_deployment(
    candidates: list[JsonObject], build_id: str, deployment_id: str | None, *, scope: str
) -> JsonObject:
    """The one deployment the user meant, out of the candidates in *scope*.

    Named explicitly, it is looked up rather than ranked — that selection is the
    whole point of ``--deployment``, and silently ranking past an id the user
    typed would act on a different deployment than the one they asked for. An id
    that matches nothing refuses here rather than returning "no deployment",
    which on the ``up`` path would fall through and *create* a second, billable
    deployment on a typo. Otherwise the highest status rank and newest creation
    time win, and a tie is reported rather than broken arbitrarily.

    ``scope`` names the set actually searched, because the callers search
    different ones — every deployment of the Build, or only those on the release
    being reconciled — and a refusal that named the wrong one sent the user to a
    ``comfy deploy ls`` that lists the very id it just called unrelated.
    """
    if deployment_id is not None:
        for deployment in candidates:
            if required_string(deployment, "id") == deployment_id:
                return deployment
        raise UnrelatedDeploymentError(
            build_id, deployment_id, [required_string(deployment, "id") for deployment in candidates], scope
        )
    ranked = [(deployment, deployment_selection_key(deployment)) for deployment in candidates]
    winning_key = max(key for _, key in ranked)
    tied = [deployment for deployment, key in ranked if key == winning_key]
    if len(tied) > 1:
        raise AmbiguousDeploymentError(build_id, [required_string(deployment, "id") for deployment in tied])
    return tied[0]


def _release_version(release: JsonObject) -> int:
    version = release.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise KeyError("version")
    return version


def deployment_selection_key(deployment: JsonObject) -> tuple[int, datetime]:
    """Rank one deployment for "which deployment did the user mean".

    Every failure is a server-shape failure, never a traceback: an unknown
    status or an unparsable ``createdAt`` surfaces as ``deploy_server_error``
    like any other bad field. The two are reported apart, and each names the
    value it rejected, because a message hedging across both fields sent
    everyone after the wrong one — a valid ``provisioning`` was printed as the
    evidence while the timestamp that actually failed went unmentioned.
    """
    status = required_string(deployment, "status")
    created_at = required_string(deployment, "createdAt")
    rank = _STATUS_RANK.get(status)
    if rank is None:
        raise server_shape_error("the deployment has an unknown status", status=status)
    try:
        created = parse_rfc3339(created_at)
    except ValueError as error:
        raise server_shape_error("the deployment has an unparsable createdAt", createdAt=created_at) from error
    return rank, created


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
    deployment_id: str | None = None,
) -> JsonObject | None:
    """Return the preferred deployment joined to every release of the Build.

    Identity fields go through the shared strict ``required_string``, which
    refuses an empty string. A lax local copy used to accept ``""`` here, so a
    release with a blank id built a ``release_ids`` set containing ``""`` and
    then adopted every deployment whose ``releaseId`` was also blank as
    belonging to this Build — feeding the wrong deployment to a lifecycle
    mutation instead of reporting the bad server shape.
    """
    release_ids = {required_string(release, "id") for release in builder.list_releases(build_id)}
    deployments = deploy.list_all_deployments()
    candidates = [
        deployment
        for deployment in deployments
        if required_string(deployment, "releaseId") in release_ids
        and (include_deleted or deployment.get("deletedAt") is None)
    ]
    if deployment_id is None and not candidates:
        return None
    scope = "the deployments" if include_deleted else "the live deployments"
    return select_deployment(candidates, build_id, deployment_id, scope=f"{scope} of Build {build_id}")
