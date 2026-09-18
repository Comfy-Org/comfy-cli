"""Orchestrate one deployment workflow job across the control and data planes."""

from __future__ import annotations

import time
import urllib.error
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, JsonValue
from comfy_cli.command.deploy_read import ReadRequest, resolve_deployment_id
from comfy_cli.command.deploy_resolve import DeploymentListClient, DeployResolveError
from comfy_cli.command.deploy_run_assets import AssetResolveContext, ResolvedAssets, resolve_assets
from comfy_cli.command.deploy_run_watch import DeployRunTimeoutError
from comfy_cli.command.deploy_run_watch import watch_with_timeout as _watch_with_timeout
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import required_string, server_shape_error
from comfy_cli.command.deploy_workflow import (
    DeployWorkflowAssetError,
    DeployWorkflowEmptyError,
    DeployWorkflowFormatUIError,
    DeployWorkflowNotApiFormatError,
    load_deploy_workflow,
    resolve_asset_roots,
)
from comfy_cli.command.run.loader import WorkflowLoadError
from comfy_cli.credentials import resolve_partner_credential
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.deploy_assets import DeployAssetClient
from comfy_cli.deploy_download import (
    OutputDownloadRequest,
    download_job_outputs,
    resolve_endpoint_link,
    validate_endpoint_origin,
)
from comfy_cli.deploy_events import JobEventCallbacks, JobWatchRequest, watch_job
from comfy_cli.deploy_jobs import DeployJobClient, JobSubmitRequest
from comfy_cli.http import ResponseTooLarge, request_json
from comfy_cli.interaction import require_option
from comfy_cli.output import get_renderer
from comfy_cli.output.renderer import Renderer
from comfy_cli.target import Target

_MAX_CANCEL_JSON = 64 * 1024


@runtime_checkable
class RunControlClient(DeploymentListClient, Protocol):
    target: Target

    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class DeployRunRequest:
    path: str | None
    workflow: str | None
    deployment_id: str | None
    wait: bool
    output_dir: Path
    timeout: float | None
    no_upload: bool
    asset_roots: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class _RunResult:
    deployment_id: str
    endpoint_origin: str
    job: JsonObject
    assets: ResolvedAssets
    outputs: list[JsonObject]

    def payload(self) -> JsonObject:
        metrics = self.job.get("metrics", {})
        if not isinstance(metrics, dict):
            raise server_shape_error("the data-plane job returned invalid metrics")
        output_values: list[JsonValue] = [*self.outputs]
        deployment: JsonObject = {"id": self.deployment_id, "endpointUrl": self.endpoint_origin}
        job: JsonObject = {"id": required_string(self.job, "id"), "status": required_string(self.job, "status")}
        assets: JsonValue = self.assets.payload()
        metric_values: JsonValue = metrics
        return {
            "deployment": deployment,
            "job": job,
            "assets": assets,
            "outputs": output_values,
            "metrics": metric_values,
        }


@dataclass(slots=True)
class _RunState:
    """Mutable state exists only to make submission-aware SIGINT cancellation one-shot."""

    job: JsonObject | None = None
    endpoint_origin: str | None = None
    target: Target | None = None
    cancel_attempted: bool = False


class DeployJobCanceledError(DeployAPIError):
    code = "deploy_job_canceled"

    def __init__(self, job: JsonObject) -> None:
        super().__init__(self.code, f"deploy job {job.get('id')} was canceled", details={"job": job})


def _job_link(job: JsonObject, field: str, endpoint_origin: str) -> str:
    urls = job.get("urls")
    if not isinstance(urls, dict):
        raise server_shape_error("the submitted job returned no urls object")
    return resolve_endpoint_link(endpoint_origin, required_string(urls, field))


def _cancel_once(state: _RunState, renderer: Renderer) -> None:
    if state.cancel_attempted or state.job is None or state.endpoint_origin is None or state.target is None:
        return
    state.cancel_attempted = True
    try:
        cancel_url = _job_link(state.job, "cancel", state.endpoint_origin)
        request_json(cancel_url, state.target, method="POST", max_bytes=_MAX_CANCEL_JSON)
    except KeyboardInterrupt:
        raise
    except (DeployAPIError, ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.warn(f"Failed to cancel deploy job: {error}")


def _emit_cancelled(renderer: Renderer, state: _RunState) -> None:
    """Report the interrupt, carrying the job id there is still work behind.

    `typer.Exit(130)` alone bypasses cmdline's global `cancelled` envelope, so a
    `--json` caller got exit 130 with empty stdout *and* empty stderr — no
    outcome, and no id to reconcile against a job that was already submitted and
    may still be settling.
    """
    job_id = state.job.get("id") if state.job is not None else None
    renderer.error(
        code="cancelled",
        message="Interrupted by user",
        details={"job_id": job_id, "cancel_requested": state.cancel_attempted},
        exit_code=130,
    )


def _emit_result(renderer: Renderer, result: _RunResult) -> None:
    payload = result.payload()
    if renderer.is_pretty():
        renderer.success(
            f"Deployment job {required_string(result.job, 'id')} is {required_string(result.job, 'status')}"
        )
    renderer.emit(payload, command="deploy run", changed=True)


def _terminal_result(job: JsonObject) -> None:
    status = required_string(job, "status")
    if status == "canceled":
        raise DeployJobCanceledError(job)
    if status == "expired":
        raise DeployAPIError("deploy_job_failed", f"deploy job {job.get('id')} expired", details={"job": job})


def run_deploy(ctx: typer.Context, request: DeployRunRequest) -> None:
    renderer = get_renderer()
    state = _RunState()
    try:
        workflow_file = require_option(
            "--workflow",
            request.workflow,
            prompt_fn=lambda: typer.prompt("Workflow file"),
            error_code="deploy_missing_input",
            ctx=ctx,
        )
        plan = load_deploy_workflow(
            Path(workflow_file),
            asset_roots=resolve_asset_roots(request.path, extra_roots=request.asset_roots),
        )
        partner_credential = resolve_partner_credential()
        builder, candidate = _command_clients()
        if not isinstance(candidate, RunControlClient):
            raise server_shape_error("the deploy client cannot resolve deployment jobs")
        deployment_id = resolve_deployment_id(builder, candidate, ReadRequest(request.path, request.deployment_id))
        deployment = candidate.get_deployment(deployment_id)
        status = required_string(deployment, "status")
        if status != "ready":
            raise DeployAPIError(
                "deploy_not_ready",
                f"deployment {deployment_id} is {status}, not ready",
                details={"deployment_id": deployment_id, "status": status},
            )
        endpoint_value = deployment.get("endpointUrl")
        endpoint_url = endpoint_value if isinstance(endpoint_value, str) else None
        endpoint_origin = validate_endpoint_origin(deployment_id, endpoint_url)
        cloud_token = candidate.target.auth_token
        if not isinstance(cloud_token, str) or not cloud_token:
            raise DeployAPIError("deploy_not_signed_in", "the control-plane session has no Cloud JWT")
        data_target = Target(kind="cloud", base_url=endpoint_origin, path_prefix="/api/v2", auth_token=cloud_token)
        assets = resolve_assets(
            plan,
            AssetResolveContext(DeployAssetClient(endpoint_origin, cloud_token), renderer, not request.no_upload),
        )
        submitted = DeployJobClient(endpoint_origin, cloud_token).submit_job(
            JobSubmitRequest(assets.workflow, str(uuid.uuid4()), deployment_id, partner_credential),
            candidate,
        )
        state.job = submitted
        state.endpoint_origin = endpoint_origin
        state.target = data_target
        if not request.wait:
            _emit_result(renderer, _RunResult(deployment_id, endpoint_origin, submitted, assets, []))
            return
        watch_request = JobWatchRequest(
            data_target,
            _job_link(submitted, "self", endpoint_origin),
            _job_link(submitted, "events", endpoint_origin),
        )
        watched = (
            watch_job(watch_request, JobEventCallbacks(), time.sleep)
            if request.timeout is None
            else _watch_with_timeout(watch_request, request.timeout, time.sleep)
        )
        _terminal_result(watched.job)
        outputs = download_job_outputs(
            OutputDownloadRequest(tuple(watched.outputs), endpoint_origin, cloud_token, request.output_dir),
            renderer,
        )
        _emit_result(renderer, _RunResult(deployment_id, endpoint_origin, watched.job, assets, outputs))
    except KeyboardInterrupt as error:
        try:
            _cancel_once(state, renderer)
        except KeyboardInterrupt:
            _emit_cancelled(renderer, state)
            raise typer.Exit(code=130) from None
        _emit_cancelled(renderer, state)
        raise typer.Exit(code=130) from error
    except DeployRunTimeoutError as error:
        _cancel_once(state, renderer)
        renderer.error(
            code="deploy_server_error",
            message=str(error),
            details={"timeout": error.seconds, "job_id": state.job.get("id") if state.job is not None else None},
        )
        raise typer.Exit(code=1) from error
    except DeployWorkflowFormatUIError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint)
        raise typer.Exit(code=1) from error
    except (DeployWorkflowEmptyError, DeployWorkflowNotApiFormatError) as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint)
        raise typer.Exit(code=1) from error
    except DeployWorkflowAssetError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except WorkflowLoadError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint)
        raise typer.Exit(code=1) from error
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except DeployResolveError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except DeployAPIError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError, OSError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
