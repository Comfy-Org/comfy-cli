"""Typed values and wire-shape parsing for deploy commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.command.deploy_progress import progress_of
from comfy_cli.deploy_api_errors import DeployAPIError


class DeployUpClient(Protocol):
    def list_all_deployments(self) -> list[JsonObject]: ...

    def create_deployment(
        self,
        release_id: str,
        compute_config: JsonObject,
        *,
        idempotency_key: str | None = None,
        name: str | None = None,
    ) -> JsonObject: ...

    def get_deployment(self, deployment_id: str) -> JsonObject: ...

    def update_deployment(self, deployment_id: str, compute_config: JsonObject) -> JsonObject: ...

    def start_deployment(self, deployment_id: str) -> JsonObject: ...

    def move_deployment(self, deployment_id: str, base_revision: int, release_id: str) -> JsonObject: ...

    def promote_deployment(self, deployment_id: str, base_revision: int, from_deployment_id: str) -> JsonObject: ...

    def get_compute_catalog(self) -> JsonObject: ...

    def get_deploy_estimate(self, release_id: str, gpu_class: str, region: str) -> JsonObject: ...

    def get_deployment_revisions(self, deployment_id: str) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class UpRequest:
    release: JsonObject
    build_id: str
    gpu: str | None
    region: str | None
    # ``None`` is "flag omitted", never `--min 0 --max 1`: only omission keeps
    # the scale a previous `comfy deploy scale` set on a live deployment.
    minimum: int | None
    maximum: int | None
    # The deployment `--deployment` named, when the Build has more than one the
    # ranking cannot separate.
    deployment_id: str | None = None
    # `--create`: add a deployment on this release beside any the Build has,
    # instead of moving the one it has onto the release.
    create: bool = False
    # `--name`: the name a created deployment gets, where comfy-deploy would
    # otherwise pick one.
    name: str | None = None
    # Whether the caller follows the deployment once the request is accepted.
    # A move cannot carry new worker bounds, so they are applied after it lands,
    # and only a watch is there to see it land.
    watch: bool = True


@dataclass(frozen=True, slots=True)
class UpResult:
    deployment: JsonObject
    release: JsonObject
    compute_config: JsonObject
    supersedes: list[JsonObject]
    created: bool
    changed: bool
    # Flags the caller supplied that this reconcile could not apply. Restarting
    # a stopped deployment is a start, not an edit, so bounds passed alongside
    # it are dropped — silently discarding explicit input is the same defect as
    # silently resetting it, so the renderer says so.
    dropped_bounds: tuple[str, ...] = ()
    # The service's estimate of how long a new deployment takes to come up,
    # asked just before the create. ``None`` on every other branch, and on a
    # create the service could not estimate: the estimate is advice and never
    # stops a deploy.
    estimate: JsonObject | None = None
    # Set when this run moved an existing deployment onto the release: the
    # release it served before.
    previous_release: JsonObject | None = None
    # Worker bounds to apply once the move lands, since the service refuses
    # them in the same request.
    pending_bounds: JsonObject | None = None
    # The name `--name` asked a create for, to say so where comfy-deploy took none.
    requested_name: str | None = None

    def payload(self) -> JsonObject:
        supersedes: list[JsonValue] = [*self.supersedes]
        deployment = {
            "id": required_string(self.deployment, "id"),
            "name": deployment_name(self.deployment),
            "status": required_string(self.deployment, "status"),
            "created": self.created,
        }
        revision = self.deployment.get("revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            deployment["revision"] = revision
        payload: JsonObject = {
            "deployment": deployment,
            "release": self.release,
            "computeConfig": self.compute_config,
            "supersedes": supersedes,
        }
        # Present only while the deployment is coming up, and absent rather than
        # null otherwise: an older service never sends it, and a settled
        # deployment has nothing left to narrate.
        progress = progress_of(self.deployment)
        if progress is not None:
            payload["progress"] = progress
        if self.estimate is not None:
            payload["estimate"] = self.estimate
        if self.previous_release is not None:
            payload["previousRelease"] = self.previous_release
        return payload


def optional_revision(deployment: JsonObject) -> int | None:
    """The deployment's revision, or ``None`` when deployment updates are off.

    The service sends ``revision`` only to a workspace inside the rollout of
    deployment updates; outside it the move, rollback and history routes all
    answer 404, so its absence is how a command tells the two apart.
    """
    revision = deployment.get("revision")
    if revision is None:
        return None
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise server_shape_error("the deploy service returned an invalid revision", field="revision")
    return revision


# A deployment `up` will not move: one already shutting down, or one whose
# stop did not take and may still be billing. A stopped or failed deployment
# moves, since the service starts the new release's copy for it.
NOT_MOVABLE: Final = frozenset({"stopping", "stop_failed"})


@dataclass(frozen=True, slots=True)
class WatchedMove:
    """The move a watch follows: the release it moves to, the one the
    deployment served before, and the revision it was asked from, where the
    move's reply names one."""

    release_id: str
    previous_id: str | None
    base: int | None


def watched_move(release: JsonObject, previous: JsonObject | None, reply: JsonObject) -> WatchedMove:
    """The move to ``release`` from ``previous``, as the service's ``reply`` to it says it waits."""
    pending = reply.get("pendingUpdate")
    base = pending.get("baseRevision") if isinstance(pending, dict) else None
    return WatchedMove(
        required_string(release, "id"),
        (previous or {}).get("id"),
        base if isinstance(base, int) and not isinstance(base, bool) else None,
    )


def move_settled(outcome: Callable[[JsonObject], str | None]) -> Callable[[JsonObject], bool]:
    """When a watch on a move can stop, given each read's ``outcome``: the move
    landed or its copy failed, or two reads that say anything about it show it
    replaced or dropped, with no read between them showing it waiting. Each
    watch takes its own, since it counts.

    The service assembles a read from several queries, so one that straddles
    a change can show the old release with nothing waiting, which is how a
    dropped move reads too, or still show an update a newer move has just
    replaced; the next read shows where the deployment went.
    """
    ended_reads = 0

    def settled(snapshot: JsonObject) -> bool:
        nonlocal ended_reads
        found = outcome(snapshot)
        if found in ("dropped", "replaced"):
            ended_reads += 1
            return ended_reads >= 2
        if found in ("unknown", "past"):
            return False
        ended_reads = 0
        return found is not None

    return settled


def move_outcome(snapshot: JsonObject, move: WatchedMove, after_base: str | None = None) -> str | None:
    """``landed``, ``failed``, ``replaced``, ``dropped``, ``past``, ``unknown``, or ``None`` while ``move`` still waits.

    It lands when the deployment serves the release and nothing waits. It
    failed when the copy it waits on failed. A newer update replaced it when
    the update waiting is to another release from the same base, or when
    nothing waits and the deployment serves, at the revision after the base,
    a release that is neither this move's nor the one it served before: a
    newer move onto a ready release it kept lands at once. It dropped when
    nothing waits and the deployment still serves that earlier release: the
    service dropped the move, a cancel ended it, or a move back to it did.

    It went ``past`` when the deployment has gone beyond the revision after
    the base, a later update waiting or a later revision serving another
    release. That alone does not say this move made that revision: a cancel
    leaves the revision where it was, so a later move takes it. ``after_base``,
    the release the deployment's history shows at that revision, settles it:
    landed where it is this move's release, replaced otherwise.

    A read with no revision is ``unknown`` unless it already shows the
    release, since the service leaves revision and pendingUpdate out whenever
    its rollout check fails.
    """
    pending = snapshot.get("pendingUpdate")
    if isinstance(pending, dict):
        waiting = pending.get("releaseId")
        if isinstance(waiting, str) and waiting != move.release_id:
            asked = pending.get("baseRevision")
            if move.base is not None and isinstance(asked, int) and asked > move.base:
                return _went_past(move, after_base)
            return "replaced"
        return "failed" if pending.get("status") == "failed" else None
    serving = snapshot.get("releaseId")
    revision = optional_revision(snapshot)
    if revision is None:
        return "landed" if serving == move.release_id else "unknown"
    if serving == move.release_id:
        return "landed"
    if move.base is not None and revision > move.base + 1:
        return _went_past(move, after_base)
    if move.previous_id is not None and serving != move.previous_id:
        return "replaced"
    return "dropped"


def _went_past(move: WatchedMove, after_base: str | None) -> str:
    if after_base is None:
        return "past"
    return "landed" if after_base == move.release_id else "replaced"


def revision_release(revisions: object, revision: int) -> str | None:
    """The release the deployment's history shows at ``revision``, if it lists one."""
    if not isinstance(revisions, list):
        return None
    for item in revisions:
        if isinstance(item, dict) and item.get("revision") == revision and isinstance(item.get("releaseId"), str):
            return item["releaseId"]
    return None


def move_changed(moved: JsonObject, base_revision: int, release_id: str, previous_id: str) -> bool:
    """Whether the reply to a move onto ``release_id`` says anything changed.

    The service answers at the same revision when the deployment already
    serves the release. A reply with no revision is one its rollout check
    failed to answer, so only the releases say whether a move was asked.
    """
    if isinstance(moved.get("pendingUpdate"), dict):
        return True
    revision = optional_revision(moved)
    return release_id != previous_id if revision is None else revision > base_revision


def moved_release(moved: JsonObject, fallback: str) -> str:
    """The release the reply to a move says the deployment moves to.

    A waiting move names it on its pending update. A reply with no revision is
    one the service's rollout check failed to answer, so it carries neither,
    and ``fallback``, the release the move asked for, stands in.
    """
    pending = moved.get("pendingUpdate")
    if isinstance(pending, dict):
        return required_string(pending, "releaseId")
    if optional_revision(moved) is None:
        return fallback
    return required_string(moved, "releaseId")


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Where a promote or rollback left the deployment it moved."""

    deployment: JsonObject
    release: JsonObject
    previous_release: JsonObject
    changed: bool
    source_id: str | None = None

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
        payload: JsonObject = {"deployment": deployment}
        if self.source_id is not None:
            payload["source"] = {"id": self.source_id}
        payload.update(release=self.release, previousRelease=self.previous_release, waiting=self.waiting)
        return payload


class ComputeRequiredError(Exception):
    pass


def server_shape_error(message: str, **details: JsonValue) -> DeployAPIError:
    return DeployAPIError("deploy_server_error", message, details=details)


def required_string(value: JsonObject, key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise server_shape_error(f"the deploy service returned no {key}", field=key)
    return field


def deployment_name(deployment: JsonObject) -> str | None:
    """The deployment's name, or None for an unnamed one.

    comfy-deploy omits an unset name, and one that predates names sends none.
    A name is only ever shown, so a malformed one reads as none rather than
    failing the command that shows it.
    """
    name = deployment.get("name")
    return name if isinstance(name, str) and name else None


def deployment_label(deployment: JsonObject) -> str:
    """``name (id)`` as a sentence names a deployment, or the id alone for an unnamed one."""
    deployment_id = required_string(deployment, "id")
    name = deployment_name(deployment)
    return deployment_id if name is None else f"{name} ({deployment_id})"


def required_int(value: JsonObject, key: str) -> int:
    field = value.get(key)
    if not isinstance(field, int) or isinstance(field, bool):
        raise server_shape_error(f"the deploy service returned an invalid {key}", field=key)
    return field


def compute_config(deployment: JsonObject) -> JsonObject:
    """The deployment's compute configuration, as the service models it.

    ``min``/``max`` are optional server-side and are carried only when stored:
    the create handler writes them solely when the caller sent them, so a
    deployment made without bounds — by the web UI, or by a direct API call —
    legitimately has neither. Requiring them refused that row outright.
    """
    raw = deployment.get("computeConfig")
    if not isinstance(raw, dict):
        raise server_shape_error("the deployment has no computeConfig")
    config: JsonObject = {
        "gpuClass": required_string(raw, "gpuClass"),
        "region": required_string(raw, "region"),
    }
    for bound in ("min", "max"):
        if bound in raw:
            config[bound] = required_int(raw, bound)
    return config


def release_summary(release: JsonObject) -> JsonObject:
    return {"id": required_string(release, "id"), "version": required_int(release, "version")}


def release_label(release: JsonObject) -> str:
    """`release v5`, or `release <id>` for a release whose version is unknown."""
    version = release.get("version")
    return f"release v{version}" if isinstance(version, int) else f"release {release.get('id')}"
