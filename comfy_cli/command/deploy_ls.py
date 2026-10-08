"""List deployments with Build and deletion filtering."""

from __future__ import annotations

import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError, resolve_build_paths
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, JsonValue, read_build_spec
from comfy_cli.command.deploy_resolve import BuildNotPushedError, DeployResolveError
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import required_string, server_shape_error
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer


@dataclass(frozen=True, slots=True)
class ListRequest:
    path: str | None
    include_deleted: bool
    workspace: bool
    status: str | None
    limit: int


@runtime_checkable
class DeploymentPageClient(Protocol):
    def iter_deployments(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> Iterator[JsonObject]: ...


def _version_of(release: JsonObject) -> int | None:
    version = release.get("version")
    return version if isinstance(version, int) and not isinstance(version, bool) else None


def _release_column(deployment: JsonObject) -> str:
    version = deployment.get("releaseVersion")
    if isinstance(version, int):
        return f"  v{version}"
    release_id = deployment.get("releaseId")
    return f"  {release_id}" if isinstance(release_id, str) and release_id else ""


def run_ls(request: ListRequest) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentPageClient):
            raise server_shape_error("the deploy client cannot list deployment pages")

        # The Build's own releases, each with its version where it has one;
        # under `--workspace` rows span Builds, so each keeps only its release id.
        versions: dict[str, int | None] | None = None
        if not request.workspace:
            paths = resolve_build_paths(request.path)
            spec = read_build_spec(paths.spec_file)
            build_id = spec.get("id")
            if not isinstance(build_id, str) or not build_id:
                raise BuildNotPushedError
            versions = {
                required_string(release, "id"): _version_of(release) for release in builder.list_releases(build_id)
            }

        deployments: list[JsonObject] = []
        for page in candidate.iter_deployments(status=request.status, limit=request.limit):
            page_rows = page.get("deployments")
            if not isinstance(page_rows, list):
                raise server_shape_error("the deployment list page has no deployments array")
            for row in page_rows:
                if not isinstance(row, dict):
                    raise server_shape_error("the deployment list contains a non-object row")
                version: int | None = None
                if versions is not None:
                    release_id = required_string(row, "releaseId")
                    if release_id not in versions:
                        continue
                    version = versions[release_id]
                if not request.include_deleted and row.get("deletedAt") is not None:
                    continue
                deployments.append(row if version is None else {**row, "releaseVersion": version})
                if len(deployments) == request.limit:
                    break
            if len(deployments) == request.limit:
                break

        if renderer.is_pretty():
            if not deployments:
                renderer.info("No deployments found.")
            for deployment in deployments:
                marker = " (deleted)" if deployment.get("deletedAt") is not None else ""
                renderer.print(
                    f"  {required_string(deployment, 'id')}  {required_string(deployment, 'status')}"
                    f"{_release_column(deployment)}{marker}"
                )
        payload_rows: list[JsonValue] = [*deployments]
        renderer.emit({"deployments": payload_rows}, command="deploy ls", changed=False)
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
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
