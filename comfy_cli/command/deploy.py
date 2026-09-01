"""Deployment lifecycle commands."""

import urllib.error
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command import deploy_lifecycle as _deploy_lifecycle
from comfy_cli.command import deploy_ls as _deploy_ls
from comfy_cli.command import deploy_read as _deploy_read
from comfy_cli.command import deploy_refs as _deploy_refs
from comfy_cli.command import deploy_run as _deploy_run
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError
from comfy_cli.command.deploy_compute import prompt_gpu as _prompt_gpu
from comfy_cli.command.deploy_compute import prompt_region as _prompt_region
from comfy_cli.command.deploy_resolve import DeployResolveError
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import poll_deployment as _poll_deployment
from comfy_cli.command.deploy_runtime import render_spec_error as _render_spec_error
from comfy_cli.command.deploy_runtime import resolved_up_request as _resolved_up_request
from comfy_cli.command.deploy_runtime import sleep as _sleep
from comfy_cli.command.deploy_status import run_status as _run_status
from comfy_cli.command.deploy_types import ComputeRequiredError
from comfy_cli.command.deploy_types import UpRequest as UpRequest
from comfy_cli.command.deploy_types import (
    required_string as _required_string,
)
from comfy_cli.command.deploy_up import (
    _IDEMPOTENCY_NAMESPACE as _IDEMPOTENCY_NAMESPACE,
)
from comfy_cli.command.deploy_up import (
    _idempotency_key as _idempotency_key,
)
from comfy_cli.command.deploy_up import (
    _render_result,
    reconcile_up,
)
from comfy_cli.command.deploy_up import (
    _soft_deleted_generation as _soft_deleted_generation,
)
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.interaction import require_option
from comfy_cli.output import get_renderer

app = typer.Typer(no_args_is_help=True, help="Create and manage serverless deployments.")
refs_app = typer.Typer(no_args_is_help=True, help="Inspect deployment reference catalogs.")
app.add_typer(refs_app, name="refs")
DeployPath = Annotated[
    str | None,
    typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
]
DeploymentOption = Annotated[str | None, typer.Option("--deployment", help="Select this deployment id.")]


def _require_paired_bounds(renderer, minimum: int | None, maximum: int | None) -> None:
    """Refuse one worker bound without the other.

    The service measures an omitted bound against the effective default the
    provider applies (``min``->0, ``max``->1), not against the stored value, so
    ``--min 3`` alone is compared to a ceiling of 1 and refused — against a
    limit nobody set, on a deployment that legitimately stores no bounds because
    the web UI created it. Naming both is the only shape that means what it
    says, and refusing here says so in the user's own vocabulary rather than
    relaying a 400 about a `max` they never typed.
    """
    supplied = [flag for flag, value in (("--min", minimum), ("--max", maximum)) if value is not None]
    if len(supplied) != 1:
        return
    missing = "--max" if supplied[0] == "--min" else "--min"
    renderer.error(
        code="deploy_missing_input",
        message=f"{supplied[0]} requires {missing}: worker bounds are set as a pair.",
        details={"missing": [missing]},
    )
    raise typer.Exit(code=1)


@app.command("run", help="Submit an API-format workflow to a ready deployment.")
@tracking.track_command("deploy")
def run_cmd(
    ctx: typer.Context,
    path: DeployPath = None,
    workflow: Annotated[str | None, typer.Option("--workflow", help="API-format workflow JSON file.")] = None,
    deployment_id: DeploymentOption = None,
    wait: Annotated[bool, typer.Option("--wait/--no-wait", help="Wait for the job and download outputs.")] = True,
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Directory for completed outputs.")] = Path(
        "outputs"
    ),
    timeout: Annotated[float | None, typer.Option("--timeout", min=0.001, help="Maximum wait in seconds.")] = None,
    no_upload: Annotated[bool, typer.Option("--no-upload", help="Fail when a local asset is not deduped.")] = False,
    asset_root: Annotated[
        list[Path] | None,
        typer.Option("--asset-root", help="Extra directory workflow inputs may name files inside. Repeatable."),
    ] = None,
) -> None:
    _deploy_run.run_deploy(
        ctx,
        _deploy_run.DeployRunRequest(
            path,
            workflow,
            deployment_id,
            wait,
            output_dir,
            timeout,
            no_upload,
            tuple(asset_root or ()),
        ),
    )


@app.command("scale", help="Edit a deployment's worker bounds or stopped compute configuration.")
@tracking.track_command("deploy")
def scale_cmd(
    path: DeployPath = None,
    deployment_id: DeploymentOption = None,
    minimum: Annotated[
        int | None, typer.Option("--min", min=0, max=20, help="Minimum warm workers. Requires --max.")
    ] = None,
    maximum: Annotated[
        int | None, typer.Option("--max", min=1, max=20, help="Maximum workers. Requires --min.")
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="GPU class; deployment must be stopped.")] = None,
    region: Annotated[str | None, typer.Option("--region", help="Region; deployment must be stopped.")] = None,
) -> None:
    _require_paired_bounds(get_renderer(), minimum, maximum)
    target = _deploy_read.ReadRequest(path, deployment_id)
    _deploy_lifecycle.run_scale(_deploy_lifecycle.ScaleRequest(target, minimum, maximum, gpu, region))


@app.command("stop", help="Pause a deployment while retaining its endpoint and staged models.")
@tracking.track_command("deploy")
def stop_cmd(path: DeployPath = None, deployment_id: DeploymentOption = None) -> None:
    _deploy_lifecycle.run_stop(_deploy_read.ReadRequest(path, deployment_id))


@app.command("start", help="Resume a stopped or failed deployment.")
@tracking.track_command("deploy")
def start_cmd(path: DeployPath = None, deployment_id: DeploymentOption = None) -> None:
    _deploy_lifecycle.run_start(_deploy_read.ReadRequest(path, deployment_id))


@app.command("delete", help="Enqueue teardown and soft-delete a deployment record.")
@tracking.track_command("deploy")
def delete_cmd(
    ctx: typer.Context,
    path: DeployPath = None,
    deployment_id: DeploymentOption = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt.")] = False,
) -> None:
    _deploy_lifecycle.run_delete(_deploy_read.ReadRequest(path, deployment_id), yes=yes, ctx=ctx)


@refs_app.command("compute", help="List deployable regions and GPU classes with availability.")
@tracking.track_command("deploy")
def refs_compute_cmd(
    region: Annotated[str | None, typer.Option("--region", help="Filter to one region.")] = None,
) -> None:
    _deploy_refs.run_compute(region)


@app.command("ls", help="List deployments for this Build, or the whole workspace.")
@tracking.track_command("deploy")
def ls_cmd(
    path: DeployPath = None,
    include_deleted: Annotated[bool, typer.Option("--all", "-a", help="Include deleted deployments.")] = False,
    workspace: Annotated[bool, typer.Option("--workspace", help="List every Build's deployments.")] = False,
    status: Annotated[str | None, typer.Option("--status", help="Filter by deployment status server-side.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100, help="Maximum visible deployments.")] = 20,
) -> None:
    _deploy_ls.run_ls(_deploy_ls.ListRequest(path, include_deleted, workspace, status, limit))


@app.command("show", help="Show one deployment's raw control-plane record.")
@tracking.track_command("deploy")
def show_cmd(path: DeployPath = None, deployment_id: DeploymentOption = None) -> None:
    _deploy_read.run_show(_deploy_read.ReadRequest(path, deployment_id))


@app.command("logs", help="Show one deployment's captured ComfyUI log snapshot.")
@tracking.track_command("deploy")
def logs_cmd(path: DeployPath = None, deployment_id: DeploymentOption = None) -> None:
    _deploy_read.run_logs(_deploy_read.ReadRequest(path, deployment_id))


@app.command("events", help="Show one deployment's status events in server order.")
@tracking.track_command("deploy")
def events_cmd(path: DeployPath = None, deployment_id: DeploymentOption = None) -> None:
    _deploy_read.run_events(_deploy_read.ReadRequest(path, deployment_id))


@app.command("status", help="Report deployment health, release freshness, and serving activity for this Build.")
@tracking.track_command("deploy")
def status_cmd(
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    deployment_id: DeploymentOption = None,
    watch: Annotated[bool, typer.Option("--watch", help="Poll until the deployment reaches a terminal state.")] = False,
) -> None:
    _run_status(path, deployment_id=deployment_id, watch=watch)


@app.command("up", help="Create or reconcile a deployment for the selected Build release.")
@tracking.track_command("deploy")
def up_cmd(
    ctx: typer.Context,
    path: Annotated[
        str | None,
        typer.Argument(help="ComfyUI install directory or build spec path. Default: the current directory."),
    ] = None,
    gpu: Annotated[str | None, typer.Option("--gpu", help="GPU class for a new deployment.")] = None,
    region: Annotated[str | None, typer.Option("--region", help="Region for a new deployment.")] = None,
    minimum: Annotated[
        int | None,
        typer.Option(
            "--min", min=0, max=20, help="Minimum warm workers. Requires --max. Default: keep the current value, or 0."
        ),
    ] = None,
    maximum: Annotated[
        int | None,
        typer.Option(
            "--max",
            min=1,
            max=20,
            help="Maximum workers. Requires --min. Default: keep the current value, else 1.",
        ),
    ] = None,
    release: Annotated[str | None, typer.Option("--release", help="Deploy this release id.")] = None,
    deployment_id: DeploymentOption = None,
    watch: Annotated[bool, typer.Option("--watch", help="Poll until the deployment reaches a terminal state.")] = False,
) -> None:
    renderer = get_renderer()
    _require_paired_bounds(renderer, minimum, maximum)
    try:
        builder, client = _command_clients()
        request = replace(
            _resolved_up_request(builder, path, release),
            gpu=gpu,
            region=region,
            minimum=minimum,
            maximum=maximum,
            deployment_id=deployment_id,
        )
        try:
            result = reconcile_up(builder, client, request)
        except ComputeRequiredError:
            missing = [name for name, value in (("--gpu", gpu), ("--region", region)) if value is None]
            selected_gpu = require_option(
                "--gpu",
                gpu,
                prompt_fn=lambda: _prompt_gpu(client),
                error_code="deploy_missing_input",
                missing=missing,
                ctx=ctx,
            )
            selected_region = require_option(
                "--region",
                region,
                prompt_fn=lambda: _prompt_region(client, selected_gpu),
                error_code="deploy_missing_input",
                missing=missing,
                ctx=ctx,
            )
            result = reconcile_up(builder, client, replace(request, gpu=selected_gpu, region=selected_region))
        if watch:
            watched = _poll_deployment(client, _required_string(result.deployment, "id"), _sleep)
            result = replace(result, deployment=watched)
        _render_result(renderer, result, watch=watch)
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        _render_spec_error(renderer, error)
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
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
