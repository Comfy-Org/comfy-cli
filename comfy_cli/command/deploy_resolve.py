"""Resolve deploy command options to a deployment id or Builder release."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from comfy_cli.command.build_paths import BuildSpecNotFoundError, resolve_build_paths
from comfy_cli.command.build_spec import BuildSpec, JsonObject, JsonValue, read_build_spec
from comfy_cli.command.deploy_types import deployment_name, required_string, server_shape_error
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
# A deployment id, today's or an older row's; comfy-deploy refuses a name
# starting with dep-, so a value with either prefix is never a name.
_ID_PREFIXES: Final = ("dep-", "dep_")


class BuilderReleaseClient(Protocol):
    def get_release(self, release_id: str, /) -> JsonObject: ...

    def list_releases(self, build_id: str, /) -> list[JsonObject]: ...

    def list_builds(self) -> list[JsonObject]: ...


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


class ReleaseNotInBuildError(DeployResolveError):
    code = "deploy_bad_request"

    def __init__(self, build_id: str, release: str) -> None:
        self.hint = f"run `comfy build release ls --id {build_id}` to see the Build's releases"
        self.details = {"buildId": build_id, "release": release}
        super().__init__(f"Build {build_id} has no release {release}")


def release_version_selector(value: str) -> int | None:
    """The version `v5` or `5` names, or None where the value is a release id."""
    digits = value[1:] if value[:1] in {"v", "V"} else value
    if not (digits.isascii() and digits.isdigit()):
        return None
    version = int(digits)
    return version if version > 0 else None


def find_build_release(releases: list[JsonObject], selector: str) -> JsonObject | None:
    """The release among a Build's that `selector` names: a version such as v5
    or 5, or a release id."""
    version = release_version_selector(selector)
    for release in releases:
        if version is None and release.get("id") == selector:
            return release
        if version is not None and release.get("version") == version:
            return release
    return None


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
    candidates = _build_rows(builder, deploy, build_id, include_deleted=include_deleted)
    if deployment_id is None and not candidates:
        return None
    scope = "the deployments" if include_deleted else "the live deployments"
    return select_deployment(candidates, build_id, deployment_id, scope=f"{scope} of Build {build_id}")


def _build_rows(
    builder: BuilderReleaseClient, deploy: DeploymentListClient, build_id: str, *, include_deleted: bool = False
) -> list[JsonObject]:
    """The Build's deployments, which comfy-deploy lists only workspace-wide, live ones alone unless asked."""
    release_ids = {required_string(release, "id") for release in builder.list_releases(build_id)}
    return [
        deployment
        for deployment in deploy.list_all_deployments()
        if required_string(deployment, "releaseId") in release_ids
        and (include_deleted or deployment.get("deletedAt") is None)
    ]


class DeploymentNameNotFoundError(DeployResolveError):
    code = "deploy_name_not_found"

    def __init__(self, build_id: str, name: str, names: list[str], *, live: bool) -> None:
        held: list[JsonValue] = [*names]
        self.details = {"buildId": build_id, "name": name, "names": held}
        if names:
            self.hint = f"pick one of {', '.join(names)}, the names Build {build_id}'s live deployments hold"
        elif live:
            # comfy-deploy omits an unset name, so an older server reads the same.
            self.hint = (
                f"none of Build {build_id}'s live deployments has a name: they predate names, or comfy-deploy "
                "does not serve names yet; pass the id, which `comfy deploy ls` lists"
            )
        else:
            self.hint = "drop `--deployment` to let the command pick or create one"
        message = f"no live deployment of Build {build_id} is named {name}"
        super().__init__(message if live else f"Build {build_id} has no live deployment, so none is named {name}")


class BuildNotFoundError(DeployResolveError):
    code = "deploy_build_not_found"
    hint = "run `comfy build ls` to see each Build's name and id; a teammate's Build is named by its id"

    def __init__(self, build: str) -> None:
        self.details = {"build": build}
        super().__init__(f"no Build is named {build} or has that id")


class NameOutsideBuildError(DeployResolveError):
    """A bare deployment name where no Build's folder says whose deployment it is."""

    code = "deploy_build_not_found"

    def __init__(self, name: str) -> None:
        self.details = {"build": None, "name": name}
        self.hint = f"run it from the Build's folder, or name the Build as `<build>/{name}`"
        super().__init__(f"no Build's folder is here to say whose deployment {name} is")


class AmbiguousBuildError(DeployResolveError):
    code = "deploy_ambiguous_build"
    hint = "name the Build by its id, as `<build id>/<deployment name>`"

    def __init__(self, build: str, build_ids: list[str]) -> None:
        ids: list[JsonValue] = [*build_ids]
        self.details = {"build": build, "buildIds": ids}
        super().__init__(f"{len(build_ids)} Builds are named {build}: {', '.join(build_ids)}")


def deployment_id_for(
    builder: BuilderReleaseClient,
    deploy: DeploymentListClient,
    value: str,
    *,
    path: str | None = None,
    build_id: str | None = None,
) -> str:
    """The id of the deployment `value` names, for every command taking one.

    ``value`` is an id, a name among the live deployments of the Build in the
    folder at ``path`` (or of ``build_id`` where the caller already knows it),
    or ``<build>/<name>`` with the Build given by its name or id.

    The Build's live deployments come from the whole list, as
    ``resolve_deployment`` reads them, rather than from comfy-deploy's name
    filter: a refusal lists every name they hold, and a comfy-deploy that
    predates names ignores the filter and answers every deployment. A row
    with no name, from either kind of server, is unnamed.
    """
    build, slash, name = value.rpartition("/")
    is_id = name.startswith(_ID_PREFIXES)
    if is_id and not slash:
        return name
    if slash:
        build_id = _build_id_named(builder, build)
        live = _named_build_rows(builder, deploy, build, build_id)
    else:
        if build_id is None:
            build_id = _folder_build_id(path, name)
        live = _build_rows(builder, deploy, build_id)
    if is_id:
        # The Build named is held to, as it is for a name.
        ids = [required_string(deployment, "id") for deployment in live]
        if name not in ids:
            raise UnrelatedDeploymentError(build_id, name, ids, f"Build {build_id}'s live deployments")
        return name
    for deployment in live:
        if deployment_name(deployment) == name:
            return required_string(deployment, "id")
    names = sorted(held for deployment in live if (held := deployment_name(deployment)) is not None)
    raise DeploymentNameNotFoundError(build_id, name, names, live=bool(live))


def _build_id_named(builder: BuilderReleaseClient, build: str) -> str:
    """The id of the Build ``build`` names by id or by name; names are not unique in a workspace.

    Outside enterprise the Build list holds only the caller's own Builds, so a
    value it does not name is taken as a teammate's Build id.
    """
    builds = builder.list_builds()
    if any(listed.get("id") == build for listed in builds):
        return build
    named = sorted(required_string(listed, "id") for listed in builds if listed.get("name") == build)
    if len(named) > 1:
        raise AmbiguousBuildError(build, named)
    if named:
        return named[0]
    # Empty or dot-only, it would name another path rather than a Build.
    if not build.strip("."):
        raise BuildNotFoundError(build)
    return build


def _named_build_rows(
    builder: BuilderReleaseClient, deploy: DeploymentListClient, build: str, build_id: str
) -> list[JsonObject]:
    try:
        return _build_rows(builder, deploy, build_id)
    except urllib.error.HTTPError as error:
        # comfy-builder answers 404 for a Build the workspace does not hold.
        if error.code == 404:
            raise BuildNotFoundError(build) from error
        raise


def _folder_build_id(path: str | None, name: str) -> str:
    try:
        spec = read_build_spec(resolve_build_paths(path).spec_file)
    except BuildSpecNotFoundError as error:
        # A PATH given and mistyped is the mistake to name, as it is without a name.
        if path is not None:
            raise
        raise NameOutsideBuildError(name) from error
    build_id = spec.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise BuildNotPushedError
    return build_id
