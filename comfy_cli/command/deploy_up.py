"""Reconcile and render deploy-up operations."""

import http.client
import urllib.error
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import Final

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    ChoiceRefusedError,
    NameTakenError,
    UnrelatedDeploymentError,
    build_deployments,
    deployment_id_for,
    name_refusal,
    select_deployment,
    valid_name,
)
from comfy_cli.command.deploy_runtime import terminal_status_error
from comfy_cli.command.deploy_types import NOT_MOVABLE as _NOT_MOVABLE
from comfy_cli.command.deploy_types import (
    ComputeRequiredError,
    DeployUpClient,
    MoveResult,
    UpRequest,
    UpResult,
    WatchedMove,
    deployment_label,
    deployment_name,
    release_label,
    watched_move,
)
from comfy_cli.command.deploy_types import compute_config as _compute_config
from comfy_cli.command.deploy_types import move_changed as _move_changed
from comfy_cli.command.deploy_types import move_outcome as _move_outcome
from comfy_cli.command.deploy_types import optional_revision as _optional_revision
from comfy_cli.command.deploy_types import release_summary as _release_summary
from comfy_cli.command.deploy_types import required_int as _required_int
from comfy_cli.command.deploy_types import required_string as _required_string
from comfy_cli.command.deploy_types import revision_release as _revision_release
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge

# This literal is a permanent protocol constant. Regenerating it would silently
# change every idempotency key and allow duplicate deployments.
_IDEMPOTENCY_NAMESPACE: Final = uuid.UUID("86e81377-21c8-5a10-9db8-33797ad495f1")
_CREATE_ATTEMPTS: Final = 3
_HOLDS_COMPUTE: Final = frozenset({"queued", "provisioning", "starting", "ready", "unhealthy"})
_DEFAULT_MINIMUM: Final = 0
_DEFAULT_MAXIMUM: Final = 1
# The service's refusals of a bounds edit: it was answered, and nothing changed.
_REFUSED: Final = frozenset({400, 402, 409, 422})


def _idempotency_key(
    build_id: str, release_id: str, generation: int, live: int = 0, *, name: str | None = None, made: int = 0
) -> str:
    # `live` counts the deployments `--create` adds beside the ones already on
    # the release, and is 0 for every other create, so their keys never change.
    # A name joins the key, since comfy-deploy refuses a retry of a key under
    # another name; `=` keeps it apart from `live`, as no name holds one. A
    # named key also counts the deployments the Build has ever had, which only
    # grows, so one renamed or moved off the release never hands its key back.
    seed = f"{build_id}:{release_id}:{generation}" + (f":{live}" if live else "")
    seed += f":name={name}:{made}" if name else ""
    return str(uuid.uuid5(_IDEMPOTENCY_NAMESPACE, seed))


def _live_on_release(deployments: Sequence[JsonObject], release_id: str) -> int:
    return sum(
        deployment.get("releaseId") == release_id and deployment.get("deletedAt") is None for deployment in deployments
    )


class UpAmbiguousDeploymentError(ChoiceRefusedError):
    """`up` found more than one deployment it could move and was not told which."""

    hint = "pass `--deployment <name|id>` to update one of them, or `--create` to add another deployment"

    def __init__(self, build_id: str, rows: Sequence[JsonObject], releases: Sequence[JsonObject]) -> None:
        super().__init__(build_id, "up", rows, releases)


def _soft_deleted_generation(deployments: Sequence[JsonObject], release_id: str) -> int:
    return sum(
        deployment.get("releaseId") == release_id and deployment.get("deletedAt") is not None
        for deployment in deployments
    )


def _existing_deployment(
    deployments: Sequence[JsonObject], release_id: str, build_id: str, deployment_id: str | None = None
) -> JsonObject | None:
    candidates = [
        deployment
        for deployment in deployments
        if deployment.get("releaseId") == release_id and deployment.get("deletedAt") is None
    ]
    # A named deployment is resolved even when the release has none, so an id
    # that matches nothing refuses instead of reporting "no deployment" and
    # falling through to the create branch below with a second billable
    # deployment as the result.
    if deployment_id is None and not candidates:
        return None
    return select_deployment(
        list(candidates),
        build_id,
        deployment_id,
        scope=f"the live deployments of release {release_id} of Build {build_id}",
    )


def _supersedes(
    deployments: Sequence[JsonObject], releases: Sequence[JsonObject], current_release_id: str
) -> list[JsonObject]:
    versions = {_required_string(release, "id"): _required_int(release, "version") for release in releases}
    rows = []
    for deployment in deployments:
        release_id = deployment.get("releaseId")
        status = deployment.get("status")
        if (
            isinstance(release_id, str)
            and release_id != current_release_id
            and release_id in versions
            and status in _HOLDS_COMPUTE
            and deployment.get("deletedAt") is None
        ):
            rows.append(
                {
                    "id": _required_string(deployment, "id"),
                    "status": status,
                    "release": {"version": versions[release_id]},
                }
            )
    return sorted(rows, key=lambda row: str(row["id"]))


_ESTIMATE_FIELDS: Final = ("etaSecondsLow", "etaSecondsHigh", "bytesToFetch")
# Optional in the answer, but `--json` passes the whole answer on, so each one
# present must hold what deploy_up.json promises for it.
_OPTIONAL_COUNTS: Final = ("bytesTotal", "bytesHeld")
_OPTIONAL_FLAGS: Final = ("atLeast", "measured")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _deploy_estimate(client: DeployUpClient, release_id: str, compute: JsonObject) -> JsonObject | None:
    """The service's estimate for this create, or ``None`` when it has none.

    Advice only: a service too old to serve it, any refusal, a connection that
    drops or a response cut short, and any answer missing the numbers it is
    read for all leave the create exactly as it was. So does any ``enabled`` but
    ``true``: ``false`` is the service saying the estimate is switched off, and
    nothing is printed or added to the output, whatever else the answer carries.
    """
    try:
        estimate = client.get_deploy_estimate(release_id, str(compute["gpuClass"]), str(compute["region"]))
    except (DeployAPIError, ResponseTooLarge, OSError, http.client.HTTPException):
        return None
    if estimate.get("enabled") is not True:
        return None
    if not all(_is_count(estimate.get(field)) for field in _ESTIMATE_FIELDS):
        return None
    if any(field in estimate and not _is_count(estimate[field]) for field in _OPTIONAL_COUNTS):
        return None
    if any(field in estimate and not isinstance(estimate[field], bool) for field in _OPTIONAL_FLAGS):
        return None
    # The service quotes the short end from its fast rates and the long end from
    # its slow ones; an answer that runs backwards is not one to repeat.
    if estimate["etaSecondsLow"] > estimate["etaSecondsHigh"]:
        return None
    return estimate


def _minutes_or_hours(low_seconds: int, high_seconds: int) -> str:
    # The short end rounds down and the long end up, so rounding never narrows
    # what the service quoted: a short end under a minute is 0. A short end under
    # half an hour stays in minutes, since the smallest half hour would raise it.
    low = low_seconds // 60
    high = max(low, -(-high_seconds // 60))
    if high < 120 or low < 30:
        return f"{low} min" if low == high else f"{low}-{high} min"
    low_hours = (low // 30) / 2
    high_hours = -(-high // 30) / 2
    return f"{low_hours:g}-{high_hours:g} h"


def estimate_line(estimate: JsonObject) -> str:
    """One line for a person: the time until ready and what has to download."""
    time = _minutes_or_hours(int(estimate["etaSecondsLow"]), int(estimate["etaSecondsHigh"]))
    fetch = int(estimate["bytesToFetch"])
    at_least = estimate.get("atLeast") is True
    if estimate.get("measured") is False:
        what = "the release's models were never measured, so this counts starting the endpoint alone"
    elif fetch == 0:
        what = "some models have no recorded size" if at_least else "nothing to download"
    else:
        size = f"{fetch / 1024**3:.1f} GB of models to download"
        what = f"at least {size}" if at_least else size
    return f"Expected ready in {time} ({what})."


def _deployments_made(deployments: Sequence[JsonObject], releases: Sequence[JsonObject]) -> int:
    """Every deployment the Build has had, deleted ones included."""
    build_release_ids = {_required_string(release, "id") for release in releases}
    return sum(row.get("releaseId") in build_release_ids for row in deployments)


def _create_live_deployment(
    client: DeployUpClient, request: UpRequest, compute: JsonObject, deployments: Sequence[JsonObject], made: int
) -> JsonObject:
    """Create on the release, the first key read from the list the name check read, so a rerun's matches the first run's."""
    release_id = _required_string(request.release, "id")
    for attempt in range(_CREATE_ATTEMPTS):
        exhaustive = deployments if attempt == 0 else client.list_all_deployments()
        generation = _soft_deleted_generation(exhaustive, release_id)
        live = _live_on_release(exhaustive, release_id) if request.create else 0
        key = _idempotency_key(request.build_id, release_id, generation, live, name=request.name, made=made)
        try:
            created = client.create_deployment(
                release_id,
                compute,
                idempotency_key=key,
                name=request.name,
            )
        except DeployAPIError as error:
            # Concurrent creates in the Build took the default name, and comfy-deploy
            # stored nothing under the key, so the create is sent again, its key read afresh.
            if (error.details or {}).get("server_code") != "NAME_RACE":
                refusal = error if request.name is None else name_refusal(error, request.name)
                if refusal is error:
                    raise
                raise refusal from error
            if attempt + 1 < _CREATE_ATTEMPTS:
                continue
            raise DeployAPIError(
                "deploy_conflict",
                "concurrent creates in this Build kept taking the default name",
                details={"attempts": _CREATE_ATTEMPTS, "releaseId": release_id},
                hint="run it again, or pass `--name` to pick one",
            ) from error
        deployment_id = _required_string(created, "id")
        snapshot = client.get_deployment(deployment_id)
        if snapshot.get("deletedAt") is not None:
            if attempt + 1 < _CREATE_ATTEMPTS:
                continue
            raise DeployAPIError(
                "deploy_conflict",
                "a concurrent delete kept invalidating the idempotency key",
                details={"attempts": _CREATE_ATTEMPTS, "releaseId": release_id},
            )
        status = _required_string(snapshot, "status")
        if status in {"stopping", "stop_failed"}:
            raise DeployAPIError(
                "deploy_conflict",
                f"deployment {deployment_id} entered {status} before creation could be confirmed",
                details={"deploymentId": deployment_id, "status": status},
            )
        # Correct as of this authoritative GET only; a later DELETE needs a
        # server-side CAS or delete-intent change and is deliberately out of scope.
        return snapshot
    raise AssertionError("bounded create loop exhausted without returning or raising")


def _deployment_holding_name(request: UpRequest, candidates: list[JsonObject]) -> UpRequest:
    """`up --name` again: the deployment already holding the name is the one to reconcile."""
    holder = next((row for row in candidates if deployment_name(row) == request.name), None)
    if holder is None:
        return request
    return replace(request, deployment_id=_required_string(holder, "id"))


def _name_on_update(request: UpRequest, deployment_id: str | None = None) -> DeployAPIError:
    """`--name` where `up` would update a deployment, the one named or one of several, rather than create one."""
    which = f"deployment {deployment_id}" if deployment_id else f"one of Build {request.build_id}'s"
    rename = (
        f"--deployment {deployment_id} {request.name}" if deployment_id else f"--deployment <name|id> {request.name}"
    )
    return DeployAPIError(
        "deploy_bad_request",
        f"--name names a new deployment, and `up` would update {which} instead",
        details={"buildId": request.build_id, "deploymentId": deployment_id, "name": request.name},
        hint=f"pass `--create` to add a deployment named {request.name}, or run `comfy deploy rename {rename}`",
    )


def _refuse_name_on_update(request: UpRequest, deployment: JsonObject) -> None:
    if request.name is not None and deployment_name(deployment) != request.name:
        raise _name_on_update(request, _required_string(deployment, "id"))


def _dropped_bounds(request: UpRequest, compute: JsonObject) -> tuple[str, ...]:
    """Name the bound flags that were supplied but would not change the live value.

    Only the restart branches consult this: they hand back ``compute`` untouched,
    so a bound the caller actually typed is discarded. A bound equal to what is
    already live is not reported — nothing was lost.
    """
    supplied = (("--min", request.minimum, compute.get("min")), ("--max", request.maximum, compute.get("max")))
    return tuple(flag for flag, value, live in supplied if value is not None and value != live)


class MoveFailedError(Exception):
    """A move `up` followed did not land; the deployment serves ``serving_release_id``."""

    outcome = "failed"

    def __init__(self, deployment_id: str, release: JsonObject, serving: JsonObject, status: object) -> None:
        super().__init__(
            f"the update of deployment {deployment_id} to {release_label(release)} {self.outcome}; "
            f"it still serves {release_label(serving)}"
        )
        self.deployment_id = deployment_id
        self.release = release
        self.serving_release_id = serving.get("id")
        self.status = status


class MoveEndedError(MoveFailedError):
    """A move `up` followed ended with nothing failed: cancelled, dropped, or a move back ended it."""

    outcome = "ended before it landed"


class MoveReplacedError(Exception):
    """A newer update replaced the move `up` followed; ``replacing_release_id`` is the release it moves to."""

    def __init__(self, deployment_id: str, release: JsonObject, replacing_release_id: object) -> None:
        super().__init__(replaced_message(deployment_id, release, {"id": replacing_release_id}))
        self.deployment_id = deployment_id
        self.release = release
        self.replacing_release_id = replacing_release_id


def replaced_message(deployment_id: str, release: JsonObject, replacing: JsonObject) -> str:
    """What a replaced move says: the release it moved to and the one replacing it."""
    return (
        f"the update of deployment {deployment_id} to {release_label(release)} "
        f"was replaced by an update to {release_label(replacing)}"
    )


class MoveOutcomes:
    """Each read's outcome for one watched move.

    Where a read shows the deployment gone past the revision the move would
    make, it reads the deployment's history for the release at that
    revision, which stays put once there, and keeps it.
    """

    def __init__(self, client: DeployUpClient, deployment_id: str, move: WatchedMove) -> None:
        self.client = client
        self.deployment_id = deployment_id
        self.move = move
        self.after_base: str | None = None

    def __call__(self, snapshot: JsonObject) -> str | None:
        outcome = _move_outcome(snapshot, self.move, self.after_base)
        if outcome != "past" or self.move.base is None:
            return outcome
        try:
            revisions = self.client.get_deployment_revisions(self.deployment_id).get("items")
        except (DeployAPIError, ResponseTooLarge, OSError, http.client.HTTPException):
            # A dropped connection included, as the estimate's read takes it.
            # The next read asks again; a watch that never gets an answer
            # ends as a lost watch, never as a landed move.
            return outcome
        self.after_base = _revision_release(revisions, self.move.base + 1)
        return _move_outcome(snapshot, self.move, self.after_base)


def raise_unless_landed(watched: JsonObject, release: JsonObject, previous: JsonObject, outcomes: MoveOutcomes) -> None:
    """Raise MoveFailedError, or MoveReplacedError, unless the watched move onto ``release`` landed."""
    deployment_id = _required_string(watched, "id")
    outcome = outcomes(watched)
    if outcome == "landed":
        return
    pending = watched.get("pendingUpdate")
    serving_id = watched.get("releaseId")
    if outcome == "replaced":
        if outcomes.after_base is not None:
            replacing = outcomes.after_base
        else:
            replacing = pending.get("releaseId") if isinstance(pending, dict) else serving_id
        raise MoveReplacedError(deployment_id, release, replacing)
    serving = previous if serving_id == previous.get("id") else {"id": serving_id}
    failed = MoveEndedError if outcome == "dropped" else MoveFailedError
    raise failed(deployment_id, release, serving, watched.get("status"))


def landed_result(result: MoveResult, watched: JsonObject, outcomes: MoveOutcomes) -> MoveResult:
    """The watched promote or rollback, once the move it followed landed."""
    raise_unless_landed(watched, result.release, result.previous_release, outcomes)
    return replace(result, deployment=watched)


def release_or_id(builder: BuilderReleaseClient, release_id: str) -> JsonObject:
    """The release's summary, read after the move was accepted.

    A failed lookup then costs the version in the output, not the watch.
    """
    try:
        return _release_summary(builder.get_release(release_id))
    except (BuilderAuthError, DeployAPIError, ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError):
        return {"id": release_id}


def refuse_unmovable(target: JsonObject, command: str = "up") -> None:
    deployment_id = _required_string(target, "id")
    status = _required_string(target, "status")
    if status in _NOT_MOVABLE:
        raise DeployAPIError(
            "deploy_conflict",
            f"deployment {deployment_id} is {status}, so `{command}` will not move it to another release",
            details={"deploymentId": deployment_id, "status": status},
            hint=f"wait until `comfy deploy status --deployment {deployment_id}` shows it stopped, "
            "running `comfy deploy stop` again if the stop failed",
        )


def _refuse_compute_change(request: UpRequest, deployment: JsonObject, compute: JsonObject) -> None:
    if (request.gpu is not None and request.gpu != compute["gpuClass"]) or (
        request.region is not None and request.region != compute["region"]
    ):
        raise DeployAPIError(
            "deploy_immutable_compute",
            "an existing deployment cannot change gpuClass or region in place",
            details={"deploymentId": _required_string(deployment, "id"), "computeConfig": compute},
        )


def _merged_bounds(request: UpRequest, compute: JsonObject) -> JsonObject:
    # An omitted bound keeps the live value, exactly as `comfy deploy scale`
    # merges: re-running `up` after a release must not silently unscale.
    desired = {**compute}
    for bound, requested in (("min", request.minimum), ("max", request.maximum)):
        if requested is not None:
            desired[bound] = requested
    return desired


# A deployment down in one of these is a leftover the Build stopped using,
# which a choice between deployments leaves out while any other is running.
_DOWN: Final = frozenset({"stopped", "failed", "stop_failed"})
# A settled status a command that brings a deployment up reports as not ok.
_TERMINAL: Final = frozenset({"failed", "stopped", "stop_failed", "unhealthy"})


def running_first(candidates: list[JsonObject]) -> list[JsonObject]:
    """The deployments a pick chooses among: those up, else every one."""
    return [row for row in candidates if row.get("status") not in _DOWN] or candidates


def _move_target(
    client: DeployUpClient,
    candidates: list[JsonObject],
    releases: Sequence[JsonObject],
    request: UpRequest,
    release_id: str,
) -> tuple[JsonObject, int] | None:
    """The Build's deployment `up` moves onto the release, read fresh, with its revision.

    ``None`` keeps today's behaviour: the workspace is outside the rollout of
    deployment updates, so a new release gets a deployment of its own. The read
    is made only when the answer could differ from that behaviour, which keeps
    a plain re-run of `up` on the release it already serves to its one list.

    Unnamed, the choice is among the deployments that are up, so a stopped one
    an older `up` left behind never makes it ambiguous; a Build whose only
    deployment is down moves that one, which keeps its URL for the fix.
    """
    if not candidates:
        return None
    named = request.deployment_id
    if named is not None:
        pick = next((row for row in candidates if row.get("id") == named), None)
        if pick is None or pick.get("releaseId") == release_id:
            return None
        pool = [pick]
    else:
        pool = running_first(candidates)
        if len(pool) == 1 and pool[0].get("releaseId") == release_id:
            return None
    snapshot = client.get_deployment(_required_string(pool[0], "id"))
    revision = _optional_revision(snapshot)
    if revision is None:
        return None
    if len(pool) > 1:
        raise UpAmbiguousDeploymentError(request.build_id, pool, releases)
    return snapshot, revision


def _release_of(builder: BuilderReleaseClient, releases: Sequence[JsonObject], release_id: str) -> JsonObject:
    # A release cut after the list was read is asked for by id.
    listed = next((release for release in releases if release.get("id") == release_id), None)
    return _release_summary(listed if listed is not None else builder.get_release(release_id))


def _move(
    builder: BuilderReleaseClient,
    client: DeployUpClient,
    request: UpRequest,
    target: JsonObject,
    base_revision: int,
    releases: Sequence[JsonObject],
    supersedes: list[JsonObject],
) -> UpResult:
    release_id = _required_string(request.release, "id")
    deployment_id = _required_string(target, "id")
    compute = _compute_config(target)
    refuse_unmovable(target)
    _refuse_compute_change(request, target, compute)
    desired = _merged_bounds(request, compute)
    pending_bounds = desired if desired != compute else None
    if pending_bounds is not None and not request.watch:
        raise DeployAPIError(
            "deploy_bad_request",
            "--min and --max are applied once the update lands, which only a watch sees",
            details={"deploymentId": deployment_id},
            hint="drop --no-watch, or run `comfy deploy scale` once the update lands",
        )
    previous = _release_of(builder, releases, _required_string(target, "releaseId"))
    moved = client.move_deployment(deployment_id, base_revision, release_id)
    changed = _move_changed(moved, base_revision, release_id, previous["id"])
    result = UpResult(
        moved,
        _release_summary(request.release),
        compute,
        [row for row in supersedes if row["id"] != deployment_id],
        False,
        changed,
        previous_release=previous,
        pending_bounds=pending_bounds,
    )
    # The service answers at the same revision when the deployment already
    # serves the release, so there is nothing to wait for before the bounds.
    if changed:
        return result
    move = watched_move(result.release, result.previous_release, moved)
    return finish_move(client, result, moved, MoveOutcomes(client, deployment_id, move))


def finish_move(client: DeployUpClient, result: UpResult, watched: JsonObject, outcomes: MoveOutcomes) -> UpResult:
    """Confirm the move landed, then apply the bounds it could not carry.

    The move has landed by the time the bounds go, so a refused bounds edit is
    reported as bounds that had no effect rather than as a failed `up`.
    """
    deployment_id = _required_string(watched, "id")
    raise_unless_landed(watched, result.release, result.previous_release or {}, outcomes)
    bounds = result.pending_bounds
    result = replace(result, deployment=watched, pending_bounds=None)
    if bounds is None:
        return result
    try:
        updated = client.update_deployment(deployment_id, bounds)
    except DeployAPIError as error:
        if error.status not in _REFUSED:
            # Unanswered, or not ours to judge: the bounds may have applied, so
            # say what is known, that the move landed, and pass the error on.
            raise DeployAPIError(
                error.code,
                f"the update to {release_label(result.release)} landed, but the --min/--max edit failed: {error}",
                status=error.status,
                details=error.details,
                hint=error.hint,
            ) from error
        live = result.compute_config
        dropped = tuple(flag for flag, key in (("--min", "min"), ("--max", "max")) if bounds.get(key) != live.get(key))
        return replace(result, dropped_bounds=dropped)
    return replace(result, deployment=updated, compute_config=bounds)


def reconcile_up(builder: BuilderReleaseClient, client: DeployUpClient, request: UpRequest) -> UpResult:
    if request.name is not None:
        valid_name(request.name)
    release_id = _required_string(request.release, "id")
    releases = builder.list_releases(request.build_id)
    deployments = client.list_all_deployments()
    supersedes = _supersedes(deployments, releases, release_id)
    existing: JsonObject | None = None
    if request.create:
        if request.deployment_id is not None:
            raise DeployAPIError("deploy_bad_request", "--create makes a new deployment, so it takes no --deployment")
    else:
        if request.name is not None and request.deployment_id is None:
            request = _deployment_holding_name(request, build_deployments(deployments, releases))
        if request.deployment_id is not None:
            named = deployment_id_for(builder, client, request.deployment_id, build_id=request.build_id)
            request = replace(request, deployment_id=named)
        candidates = build_deployments(deployments, releases)
        try:
            target = _move_target(client, candidates, releases, request, release_id)
        except UpAmbiguousDeploymentError as error:
            if request.name is None:
                raise
            raise _name_on_update(request) from error
        if target is not None:
            _refuse_name_on_update(request, target[0])
        if target is not None and target[0].get("releaseId") != release_id:
            return _move(builder, client, request, *target, releases, supersedes)
        if target is not None:
            # Another change moved it onto the release after the list was read.
            existing = target[0]
        else:
            try:
                existing = _existing_deployment(deployments, release_id, request.build_id, request.deployment_id)
            except UnrelatedDeploymentError as error:
                error.hint = f"{error.hint}, or pass `--create` to add a deployment on this release"
                raise
            if existing is not None:
                _refuse_name_on_update(request, existing)
    if existing is None:
        # A create would reuse the key of the one that made this deployment on
        # the release it has since left, and comfy-deploy would answer with it.
        if request.name is not None and any(
            deployment_name(row) == request.name for row in build_deployments(deployments, releases)
        ):
            raise NameTakenError(request.name)
        if request.gpu is None or request.region is None:
            raise ComputeRequiredError
        minimum = _DEFAULT_MINIMUM if request.minimum is None else request.minimum
        # A library-level contract, not a CLI one: `_require_paired_bounds`
        # refuses a lone `--min` before this runs, so the command line cannot
        # reach a floor without a ceiling. A direct `reconcile_up` caller can,
        # and an omitted ceiling has to clear the requested floor or the create
        # is refused against the placeholder maximum of 1.
        maximum = max(_DEFAULT_MAXIMUM, minimum) if request.maximum is None else request.maximum
        compute = {
            "gpuClass": request.gpu,
            "region": request.region,
            "min": minimum,
            "max": maximum,
        }
        estimate = _deploy_estimate(client, release_id, compute)
        made = _deployments_made(deployments, releases)
        snapshot = _create_live_deployment(client, request, compute, deployments, made)
        return UpResult(
            snapshot,
            _release_summary(request.release),
            compute,
            supersedes,
            True,
            True,
            estimate=estimate,
            requested_name=request.name,
        )

    compute = _compute_config(existing)
    _refuse_compute_change(request, existing, compute)
    deployment_id = _required_string(existing, "id")
    status = _required_string(existing, "status")
    dropped = _dropped_bounds(request, compute)
    if status in {"stopped", "failed"}:
        started = client.start_deployment(deployment_id)
        return UpResult(started, _release_summary(request.release), compute, supersedes, False, True, dropped)
    if status == "stop_failed":
        return UpResult(existing, _release_summary(request.release), compute, supersedes, False, False, dropped)
    desired = _merged_bounds(request, compute)
    if desired != compute:
        updated = client.update_deployment(deployment_id, desired)
        return UpResult(updated, _release_summary(request.release), desired, supersedes, False, True)
    return UpResult(existing, _release_summary(request.release), compute, supersedes, False, False)


def move_line(result: UpResult, deployment_id: str) -> str | None:
    if result.previous_release is None:
        return None
    return move_text(deployment_id, result.deployment, result.release, result.previous_release, result.changed)


def move_text(
    deployment_id: str, deployment: JsonObject, release: JsonObject, previous: JsonObject, changed: bool
) -> str:
    """Where a move onto ``release`` stands, for `up` and `promote` alike."""
    label, was = release_label(release), release_label(previous)
    if not changed:
        return f"Deployment {deployment_id} already serves {label}."
    # A read with no revision says nothing of the move, so the move waits
    # until the deployment reads the new release.
    unconfirmed = _optional_revision(deployment) is None and deployment.get("releaseId") != release.get("id")
    if isinstance(deployment.get("pendingUpdate"), dict) or unconfirmed:
        if deployment.get("status") in _DOWN:
            return f"Deployment {deployment_id} starts on {label}."
        return f"Deployment {deployment_id} moves to {label} once it is ready; {was} serves until then."
    serving_id = deployment.get("releaseId")
    if serving_id != release.get("id"):
        # A read caught as the move lands, or a watch interrupted on a dropped read.
        serving = was if serving_id == previous.get("id") else release_label({"id": serving_id})
        return (
            f"Deployment {deployment_id} read as serving {serving}, with no update to {label} waiting; "
            f"run `comfy deploy show --deployment {deployment_id}` to see where it settled."
        )
    return f"Deployment {deployment_id} now serves {label} (was {was})."


def warn_status(renderer, deployment_id: str, status: str, *, watch: bool) -> None:
    """Warn about a deployment left billing without serving, or settled down."""
    if status == "stop_failed":
        renderer.warn(
            f"Deployment {deployment_id} could not stop and may still be billing.",
            hint=f"run `comfy deploy stop --deployment {deployment_id}` again",
        )
    elif status == "unhealthy":
        # `up` leaves an unhealthy deployment as it is, so saying nothing would
        # read as success for one that is billing and not serving.
        renderer.warn(
            f"Deployment {deployment_id} is unhealthy: it came up, then its endpoint degraded. "
            "It is still billing, and the service moves it back to ready if it recovers.",
            hint=f"run `comfy deploy logs --deployment {deployment_id}` to see why, "
            f"or `comfy deploy stop --deployment {deployment_id}` to stop billing",
        )
    elif watch and status in {"failed", "stopped"}:
        renderer.warn(f"Deployment {deployment_id} reached terminal status {status}.")


def ends_terminal(deployment: JsonObject, status: str, *, moving: bool) -> bool:
    # A move still waiting is judged by the move, whatever the old copy's
    # status: one that was down stays down until the new release lands.
    waiting = moving and isinstance(deployment.get("pendingUpdate"), dict)
    return not waiting and status in _TERMINAL


def _render_result(renderer, result: UpResult, *, watch: bool) -> None:
    status = _required_string(result.deployment, "status")
    deployment_id = _required_string(result.deployment, "id")
    name = deployment_name(result.deployment)
    if renderer.is_pretty():
        label = deployment_label(result.deployment)
        renderer.success(move_line(result, deployment_id) or f"Deployment {label}: {status}")
    if result.created and result.requested_name is not None and name != result.requested_name:
        # It bills either way, so it is reported rather than refused.
        took = "without a name" if name is None else f"named {name}"
        renderer.warn(
            f"Deployment {deployment_id} was created {took}, not {result.requested_name}: "
            "comfy-deploy may not serve deployment names yet.",
            hint=f"rename it with `comfy deploy rename --deployment {deployment_id} {result.requested_name}`, "
            "or use its id wherever a command takes a deployment",
        )
    warn_status(renderer, deployment_id, status, watch=watch)
    if result.dropped_bounds:
        joined = " and ".join(result.dropped_bounds)
        renderer.warn(
            f"{joined} had no effect; deployment {deployment_id} kept its existing worker bounds.",
            # `scale` is only actionable once the deployment settles: the API
            # rejects an edit unless it is ready or stopped (`run_scale` re-wraps
            # that as `deploy_conflict`), so a `stop_failed` deployment is sent
            # to the stop remedy warned about just above instead.
            hint=None
            if status == "stop_failed"
            else f"run `comfy deploy scale --deployment {deployment_id} --min <n> --max <n>` to change them",
        )
    # A new release gets a new deployment, so the old one keeps billing. The
    # JSON envelope carries `supersedes`; a person reading the terminal needs it
    # said, on stderr under --json as with every other warning here.
    for row in result.supersedes:
        old_id = row["id"]
        version = row["release"]["version"]
        renderer.warn(
            f"Deployment {old_id} (release v{version}, {row['status']}) is still running and billing.",
            hint=f"run `comfy deploy stop --deployment {old_id}` if you no longer need it",
        )
    terminal = ends_terminal(result.deployment, status, moving=result.previous_release is not None)
    renderer.emit(
        result.payload(),
        command="deploy up",
        changed=result.changed,
        ok=not terminal,
        error=terminal_status_error(deployment_id, status) if terminal else None,
    )
    if terminal:
        raise typer.Exit(code=1)
