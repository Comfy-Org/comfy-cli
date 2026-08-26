"""Reconcile and render deploy-up operations."""

import uuid
from collections.abc import Sequence
from typing import Final

import typer

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_resolve import (
    AmbiguousDeploymentError,
    BuilderReleaseClient,
    deployment_selection_key,
)
from comfy_cli.command.deploy_types import ComputeRequiredError, DeployUpClient, UpRequest, UpResult
from comfy_cli.command.deploy_types import compute_config as _compute_config
from comfy_cli.command.deploy_types import release_summary as _release_summary
from comfy_cli.command.deploy_types import required_int as _required_int
from comfy_cli.command.deploy_types import required_string as _required_string
from comfy_cli.deploy_api_errors import DeployAPIError

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


def _existing_deployment(deployments: Sequence[JsonObject], release_id: str, build_id: str) -> JsonObject | None:
    candidates = [
        deployment
        for deployment in deployments
        if deployment.get("releaseId") == release_id and deployment.get("deletedAt") is None
    ]
    if not candidates:
        return None
    ranked = [(deployment, deployment_selection_key(deployment)) for deployment in candidates]
    best = max(key for _, key in ranked)
    tied = [deployment for deployment, key in ranked if key == best]
    if len(tied) > 1:
        raise AmbiguousDeploymentError(build_id, [_required_string(item, "id") for item in tied])
    return tied[0]


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
    existing = _existing_deployment(deployments, release_id, request.build_id)
    if existing is None:
        if request.gpu is None or request.region is None:
            raise ComputeRequiredError
        minimum = _DEFAULT_MINIMUM if request.minimum is None else request.minimum
        # A create sends both bounds, so the API's min <= max check has neither
        # one to skip: an omitted `--max` has to clear the requested floor, or
        # `--min 3` would be refused against the placeholder ceiling of 1.
        maximum = max(_DEFAULT_MAXIMUM, minimum) if request.maximum is None else request.maximum
        compute = {
            "gpuClass": request.gpu,
            "region": request.region,
            "min": minimum,
            "max": maximum,
        }
        snapshot = _create_live_deployment(client, request, compute)
        return UpResult(snapshot, _release_summary(request.release), compute, supersedes, True, True)

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
            hint=f"run `comfy deploy stop {deployment_id}` again",
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
            hint=None if status == "stop_failed" else f"run `comfy deploy scale {deployment_id}` to change them",
        )
    renderer.emit(
        result.payload(),
        command="deploy up",
        changed=result.changed,
        ok=status not in {"failed", "stopped", "stop_failed"},
    )
    if status in {"failed", "stopped", "stop_failed"}:
        raise typer.Exit(code=1)
