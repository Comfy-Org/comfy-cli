"""HTTP fakes and fixtures for the deploy half of the shared auth matrix."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never
from urllib.parse import urlsplit

import click
import pytest
import typer
from build_auth_support import BUILD_ID, RELEASE_ID
from build_push_support import write_spec
from typer.testing import CliRunner

from comfy_cli.auth.store import CloudSession
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import deploy
from comfy_cli.command.build_spec import JsonObject

DEPLOYMENT_ID = "dep-matrix"
ENDPOINT_ORIGIN = f"https://{DEPLOYMENT_ID}.run.comfy.app"


class DeployFixtureKind(Enum):
    UP = "up"
    STATUS = "status"
    LS = "ls"
    SHOW = "show"
    LOGS = "logs"
    EVENTS = "events"
    SCALE = "scale"
    STOP = "stop"
    START = "start"
    DELETE = "delete"
    RUN = "run"
    REFS_COMPUTE = "refs compute"


@dataclass(frozen=True, slots=True)
class DeployAuthCase:
    fixture: DeployFixtureKind
    command: str


DEPLOY_AUTH_CASES = tuple(DeployAuthCase(kind, kind.value) for kind in DeployFixtureKind)


def _deployment() -> JsonObject:
    return {
        "id": DEPLOYMENT_ID,
        "status": "ready",
        "distributionVersionId": RELEASE_ID,
        "buildVersionId": RELEASE_ID,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
        "deletedAt": None,
        "computeConfig": {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 1},
        "endpointUrl": ENDPOINT_ORIGIN,
        "serving": None,
    }


class DeployRecordingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        url: str,
        _target,
        *,
        method: str = "GET",
        body: JsonObject | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_bytes: int,
    ) -> tuple[int, JsonObject]:
        self.calls.append(url)
        path = urlsplit(url).path
        row = _deployment()
        routes: dict[tuple[str, str], tuple[int, JsonObject]] = {
            ("GET", "/v1/deployments"): (200, {"deployments": [row]}),
            ("GET", f"/v1/deployments/{DEPLOYMENT_ID}"): (200, row),
            ("POST", f"/v1/deployments/{DEPLOYMENT_ID}/stop"): (202, {**row, "status": "stopping"}),
            ("POST", f"/v1/deployments/{DEPLOYMENT_ID}/start"): (202, {**row, "status": "queued"}),
            ("DELETE", f"/v1/deployments/{DEPLOYMENT_ID}"): (204, {}),
            ("GET", f"/v1/deployments/{DEPLOYMENT_ID}/logs"): (
                200,
                {"capturedAt": None, "comfyuiLog": "", "deploymentId": DEPLOYMENT_ID},
            ),
            ("GET", f"/v1/deployments/{DEPLOYMENT_ID}/events"): (
                200,
                {"deploymentId": DEPLOYMENT_ID, "events": []},
            ),
            ("GET", "/v1/compute-catalog"): (200, {"regions": []}),
        }
        if method == "PATCH" and path == f"/v1/deployments/{DEPLOYMENT_ID}":
            return 200, copy.deepcopy(row)
        route = routes.get((method, path))
        if route is None:
            pytest.fail(f"unexpected deploy control-plane call: {method} {url}")
        status, payload = route
        return status, copy.deepcopy(payload)


class JobRecordingTransport:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        url: str,
        _target,
        *,
        method: str = "GET",
        body: JsonObject | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_bytes: int,
    ) -> tuple[int, JsonObject]:
        self.calls.append(url)
        return 201, {
            "id": "job-matrix",
            "status": "queued",
            "outputs": [],
            "metrics": {},
            "urls": {
                "self": "/api/v2/jobs/job-matrix",
                "events": "/api/v2/jobs/job-matrix/events",
                "cancel": "/api/v2/jobs/job-matrix/cancel",
            },
        }


def prepare_deploy(kind: DeployFixtureKind, root: Path) -> list[str]:
    match kind:
        case DeployFixtureKind.UP:
            return ["deploy", "up", "--release", RELEASE_ID, "--gpu", "l4", "--region", "US-MO-2"]
        case DeployFixtureKind.STATUS:
            write_spec(root, build_id=BUILD_ID, models=[], nodes=[])
            return ["deploy", "status", str(root)]
        case DeployFixtureKind.LS:
            write_spec(root, build_id=BUILD_ID, models=[], nodes=[])
            return ["deploy", "ls", str(root)]
        case DeployFixtureKind.SHOW | DeployFixtureKind.LOGS | DeployFixtureKind.EVENTS:
            return ["deploy", kind.value, "--deployment", DEPLOYMENT_ID]
        case DeployFixtureKind.SCALE:
            return ["deploy", "scale", "--deployment", DEPLOYMENT_ID, "--max", "1"]
        case DeployFixtureKind.STOP | DeployFixtureKind.START:
            return ["deploy", kind.value, "--deployment", DEPLOYMENT_ID]
        case DeployFixtureKind.DELETE:
            return ["deploy", "delete", "--deployment", DEPLOYMENT_ID, "--yes"]
        case DeployFixtureKind.RUN:
            workflow = root / "workflow.json"
            root.mkdir(parents=True, exist_ok=True)
            workflow.write_text(json.dumps({"1": {"class_type": "Test", "inputs": {}}}), encoding="utf-8")
            return [
                "deploy",
                "run",
                "--workflow",
                str(workflow),
                "--deployment",
                DEPLOYMENT_ID,
                "--no-wait",
            ]
        case DeployFixtureKind.REFS_COMPUTE:
            return ["deploy", "refs", "compute"]
        case unreachable:
            assert_never(unreachable)


def deploy_leaf_commands() -> set[str]:
    leaves: set[str] = set()

    def walk(commands: Mapping[str, object], prefix: tuple[str, ...]) -> None:
        for name, child in commands.items():
            subcommands = getattr(child, "commands", None)
            if subcommands:
                walk(subcommands, (*prefix, name))
            else:
                leaves.add(" ".join((*prefix, name)))

    command = typer.main.get_command(deploy.app)
    assert isinstance(command, click.Group)
    walk(command.commands, ())
    return leaves


def invoke_deploy(args: list[str], token: str | None):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        args,
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": "json",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_DEPLOY_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
            "COMFY_DEPLOY_URL": "https://control.test",
        },
    )


def session(token: str | None) -> CloudSession | None:
    if token is None:
        return None
    return CloudSession(
        base_url="https://api.comfy.org",
        resource="comfy-cli",
        client_id="matrix",
        scope="openid",
        access_token=token,
        refresh_token=None,
        token_type="Bearer",
        expires_at=None,
        saved_at="2026-01-01T00:00:00Z",
    )
