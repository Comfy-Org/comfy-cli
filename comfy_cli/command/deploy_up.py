"""Reconcile and render deploy-up operations."""

import http.client
import uuid
from collections.abc import Sequence
from typing import Final

import typer

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    select_deployment,
)
from comfy_cli.command.deploy_runtime import terminal_status_error
from comfy_cli.command.deploy_types import ComputeRequiredError, DeployUpClient, UpRequest, UpResult
from comfy_cli.command.deploy_types import compute_config as _compute_config
from comfy_cli.command.deploy_types import release_summary as _release_summary
from comfy_cli.command.deploy_types import required_int as _required_int
from comfy_cli.command.deploy_types import required_string as _required_string
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge

# This literal is a permanent protocol constant. Regenerating it would silently
# change every idempotency key and allow duplicate deployments.
_IDEMPOTENCY_NAMESPACE: Final = uuid.UUID("86e81377-21c8-5a10-9db8-33797ad495f1")
_CREATE_ATTEMPTS: Final = 3
_HOLDS_COMPUTE: Final = frozenset({"queued", "provisioning", "starting", "ready", "unhealthy"})
_DEFAULT_MINIMUM: Final = 0
_DEFAULT_MAXIMUM: Final = 1


def _idempotency_key(build_id: str, release_id: str, generation: int) -> str:
    return str(uuid.uuid5(_IDEMPOTENCY_NAMESPACE, f"{build_id}:{release_id}:{generation}"))


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


def _create_live_deployment(client: DeployUpClient, request: UpRequest, compute: JsonObject) -> JsonObject:
    release_id = _required_string(request.release, "id")
    for attempt in range(_CREATE_ATTEMPTS):
        exhaustive = client.list_all_deployments()
        generation = _soft_deleted_generation(exhaustive, release_id)
        created = client.create_deployment(
            release_id,
            compute,
            idempotency_key=_idempotency_key(request.build_id, release_id, generation),
        )
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


def _dropped_bounds(request: UpRequest, compute: JsonObject) -> tuple[str, ...]:
    """Name the bound flags that were supplied but would not change the live value.

    Only the restart branches consult this: they hand back ``compute`` untouched,
    so a bound the caller actually typed is discarded. A bound equal to what is
    already live is not reported — nothing was lost.
    """
    supplied = (("--min", request.minimum, compute.get("min")), ("--max", request.maximum, compute.get("max")))
    return tuple(flag for flag, value, live in supplied if value is not None and value != live)


def reconcile_up(builder: BuilderReleaseClient, client: DeployUpClient, request: UpRequest) -> UpResult:
    release_id = _required_string(request.release, "id")
    releases = builder.list_releases(request.build_id)
    deployments = client.list_all_deployments()
    supersedes = _supersedes(deployments, releases, release_id)
    existing = _existing_deployment(deployments, release_id, request.build_id, request.deployment_id)
    if existing is None:
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
        snapshot = _create_live_deployment(client, request, compute)
        return UpResult(snapshot, _release_summary(request.release), compute, supersedes, True, True, estimate=estimate)

    compute = _compute_config(existing)
    if (request.gpu is not None and request.gpu != compute["gpuClass"]) or (
        request.region is not None and request.region != compute["region"]
    ):
        raise DeployAPIError(
            "deploy_immutable_compute",
            "an existing deployment cannot change gpuClass or region in place",
            details={"deploymentId": _required_string(existing, "id"), "computeConfig": compute},
        )
    deployment_id = _required_string(existing, "id")
    status = _required_string(existing, "status")
    dropped = _dropped_bounds(request, compute)
    if status in {"stopped", "failed"}:
        started = client.start_deployment(deployment_id)
        return UpResult(started, _release_summary(request.release), compute, supersedes, False, True, dropped)
    if status == "stop_failed":
        return UpResult(existing, _release_summary(request.release), compute, supersedes, False, False, dropped)
    # An omitted bound keeps the live value, exactly as `comfy deploy scale`
    # merges: re-running `up` after a release must not silently unscale.
    desired = {**compute}
    for bound, requested in (("min", request.minimum), ("max", request.maximum)):
        if requested is not None:
            desired[bound] = requested
    if desired != compute:
        updated = client.update_deployment(deployment_id, desired)
        return UpResult(updated, _release_summary(request.release), desired, supersedes, False, True)
    return UpResult(existing, _release_summary(request.release), compute, supersedes, False, False)


def _render_result(renderer, result: UpResult, *, watch: bool) -> None:
    status = _required_string(result.deployment, "status")
    deployment_id = _required_string(result.deployment, "id")
    if renderer.is_pretty():
        renderer.success(f"Deployment {deployment_id}: {status}")
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
    terminal = status in {"failed", "stopped", "stop_failed", "unhealthy"}
    renderer.emit(
        result.payload(),
        command="deploy up",
        changed=result.changed,
        ok=not terminal,
        error=terminal_status_error(deployment_id, status) if terminal else None,
    )
    if terminal:
        raise typer.Exit(code=1)
