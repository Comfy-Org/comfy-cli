from __future__ import annotations

from dataclasses import dataclass, field

BASE_URL = "https://deploy.test/"
BASE = "https://deploy.test"
MAX_JSON = 5 * 1024 * 1024
MAX_LOG_JSON = 32 * 1024 * 1024
COMPUTE = {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 1}


@dataclass(frozen=True)
class Wire:
    operation: str
    method_name: str
    http_method: str
    url: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    body: dict | None = None
    headers: dict[str, str] | None = None
    max_bytes: int = MAX_JSON


WIRES = [
    Wire(
        "create",
        "create_deployment",
        "POST",
        f"{BASE}/v1/deployments",
        args=("v1", COMPUTE),
        kwargs={"idempotency_key": "idem-1"},
        body={"buildVersionId": "v1", "computeConfig": COMPUTE},
        headers={"Idempotency-Key": "idem-1"},
    ),
    Wire(
        "list",
        "list_deployments",
        "GET",
        f"{BASE}/v1/deployments?status=ready&limit=20",
        kwargs={"status": "ready", "limit": 20},
    ),
    Wire("get", "get_deployment", "GET", f"{BASE}/v1/deployments/dep-1", args=("dep-1",)),
    Wire(
        "scale",
        "update_deployment",
        "PATCH",
        f"{BASE}/v1/deployments/dep-1",
        args=("dep-1", COMPUTE),
        body={"computeConfig": COMPUTE},
    ),
    Wire("delete", "delete_deployment", "DELETE", f"{BASE}/v1/deployments/dep-1", args=("dep-1",)),
    Wire("start", "start_deployment", "POST", f"{BASE}/v1/deployments/dep-1/start", args=("dep-1",)),
    Wire("stop", "stop_deployment", "POST", f"{BASE}/v1/deployments/dep-1/stop", args=("dep-1",)),
    Wire("events", "get_deployment_events", "GET", f"{BASE}/v1/deployments/dep-1/events", args=("dep-1",)),
    Wire(
        "logs",
        "get_deployment_logs",
        "GET",
        f"{BASE}/v1/deployments/dep-1/logs",
        args=("dep-1",),
        max_bytes=MAX_LOG_JSON,
    ),
    Wire("compute", "get_compute_catalog", "GET", f"{BASE}/v1/compute-catalog"),
]


@dataclass(frozen=True)
class StatusCase:
    operation: str
    status: int
    code: str
    message: str = "server rejected the operation"


OPERATIONS = tuple(wire.operation for wire in WIRES)
STATUS_CASES = [
    *[StatusCase(op, 400, "deploy_bad_request", "invalid request parameter") for op in OPERATIONS],
    *[StatusCase(op, 401, "deploy_not_signed_in") for op in OPERATIONS],
    *[StatusCase(op, 402, "deploy_payment_required") for op in OPERATIONS],
    *[StatusCase(op, 403, "deploy_forbidden") for op in OPERATIONS],
    *[StatusCase(op, 404, "deploy_not_found") for op in OPERATIONS],
    *[StatusCase(op, 409, "deploy_conflict", "current status is provisioning") for op in OPERATIONS],
    *[StatusCase(op, 429, "deploy_quota_exceeded") for op in OPERATIONS],
    *[StatusCase(op, 503, "deploy_server_error") for op in OPERATIONS],
]
for index, case in enumerate(STATUS_CASES):
    if case.operation in {"create", "scale", "start"} and case.status == 400:
        STATUS_CASES[index] = StatusCase(
            case.operation,
            400,
            "deploy_compute_unavailable",
            'computeConfig.gpuClass "l4" is not available in region "US-MO-2"',
        )
    if case.operation == "scale" and case.status == 409:
        STATUS_CASES[index] = StatusCase(
            "scale",
            409,
            "deploy_immutable_compute",
            "stop before changing gpuClass or region",
        )
    if case.operation == "start" and case.status == 409:
        STATUS_CASES[index] = StatusCase("start", 409, "deploy_deleted", "deployment was deleted")
