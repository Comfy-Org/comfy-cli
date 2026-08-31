"""Typed values and wire-shape parsing for deploy commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.deploy_api_errors import DeployAPIError


class DeployUpClient(Protocol):
    def list_all_deployments(self) -> list[JsonObject]: ...

    def create_deployment(
        self, release_id: str, compute_config: JsonObject, *, idempotency_key: str | None = None
    ) -> JsonObject: ...

    def get_deployment(self, deployment_id: str) -> JsonObject: ...

    def update_deployment(self, deployment_id: str, compute_config: JsonObject) -> JsonObject: ...

    def start_deployment(self, deployment_id: str) -> JsonObject: ...

    def get_compute_catalog(self) -> JsonObject: ...


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

    def payload(self) -> JsonObject:
        supersedes: list[JsonValue] = [*self.supersedes]
        deployment = {
            "id": required_string(self.deployment, "id"),
            "status": required_string(self.deployment, "status"),
            "created": self.created,
        }
        return {
            "deployment": deployment,
            "release": self.release,
            "computeConfig": self.compute_config,
            "supersedes": supersedes,
        }


class ComputeRequiredError(Exception):
    pass


def server_shape_error(message: str, **details: JsonValue) -> DeployAPIError:
    return DeployAPIError("deploy_server_error", message, details=details)


def required_string(value: JsonObject, key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise server_shape_error(f"the deploy service returned no {key}", field=key)
    return field


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
