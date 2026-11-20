"""Model files added straight to a deployment, with no new release: `comfy deploy models add`, `ls` and `rm`."""

from __future__ import annotations

import http.client
import os
import shlex
import urllib.error
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Protocol, runtime_checkable

import requests
import typer

from comfy_cli import tracking
from comfy_cli.builder_api import BlobFileError, BuilderAuthError, BuilderClient, PartsUpload
from comfy_cli.command.build import DEFAULT_BUILDER_URL
from comfy_cli.command.build import _without_signed_query as without_signed_query
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, JsonValue
from comfy_cli.command.build_upload_progress import UploadProgressReporter
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    DeploymentListClient,
    DeployResolveError,
    deployment_id_for,
)
from comfy_cli.command.deploy_runtime import DEPLOY_POLL_SECONDS, render_spec_error
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import sleep as _sleep
from comfy_cli.command.deploy_types import required_string, server_shape_error
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.hashing import file_digests
from comfy_cli.http import ResponseTooLarge
from comfy_cli.interaction import confirm
from comfy_cli.output import get_renderer
from comfy_cli.output.progress import human_bytes
from comfy_cli.output.renderer import Renderer

app = typer.Typer(no_args_is_help=True, help="Add model files straight to a deployment, list them, remove them.")
DeploymentArgument = Annotated[
    str, typer.Argument(help="Deployment name or id, or <build>/<name>.", show_default=False)
]

_NOT_ON: Final = "Adding models to a deployment is not on for this workspace yet."
_REMOVE_QUESTION: Final = (
    "Remove {file}? It is deleted from this deployment's disk, and jobs that name it will fail. A job already "
    "running may still finish, and a worker that already loaded it may keep using it for a few minutes."
)
# comfy-deploy's own 404 codes on the models route; a 404 with neither is the route not being on.
_KNOWN_NOT_FOUND: Final = frozenset({"NOT_FOUND", "UPLOAD_NOT_FOUND"})


@runtime_checkable
class DeploymentModelsClient(DeploymentListClient, Protocol):
    def add_deployment_model(self, deployment_id: str, blob_id: str, folder: str, /) -> JsonObject: ...

    def list_deployment_models(self, deployment_id: str, /) -> JsonObject: ...

    def remove_deployment_model(self, deployment_id: str, model_id: str, /) -> JsonObject: ...


class ModelsUnavailableError(DeployResolveError):
    code = "deploy_models_unavailable"

    def __init__(self, deployment_id: str, file: Path, folder: str) -> None:
        self.hint = (
            "To upload it to the workspace library instead: "
            f"comfy assets library upload {shlex.quote(str(file))} --type {shlex.quote(folder)}"
        )
        self.details = {"deploymentId": deployment_id}
        super().__init__(_NOT_ON)


class ModelFileError(DeployResolveError):
    """The model file was removed or became unreadable after it was checked, while hashing or sending it."""

    code = "deploy_asset_upload_failed"

    def __init__(self, file: Path, error: OSError) -> None:
        if isinstance(error, FileNotFoundError):
            reason, message = "file_not_found", f"{file} does not exist or is not a file"
        else:
            reason, message = "file_unreadable", f"{file} cannot be read ({error.strerror or error})"
        self.hint = "check the file is still there and readable, then add it again"
        self.details = {"local_path": str(file), "reason": reason}
        super().__init__(message)


class ModelNotAddedError(DeployResolveError):
    code = "deploy_model_failed"

    def __init__(self, deployment_id: str, entry: JsonObject) -> None:
        file = required_string(entry, "filename")
        self.hint = f"run `comfy deploy models rm {deployment_id} {file}`, then add it again"
        self.details = {"deploymentId": deployment_id, "model": _shown(entry)}
        super().__init__(_failed_line(file))


class ModelNotFoundError(DeployResolveError):
    code = "deploy_model_not_found"

    def __init__(self, deployment_id: str, name: str, matches: list[JsonObject]) -> None:
        shown: list[JsonValue] = [_shown(entry) for entry in matches]
        self.details = {"deploymentId": deployment_id, "name": name, "matches": shown}
        if matches:
            self.hint = (
                f"name it as <folder>/<file>, or by the id `comfy deploy models ls {deployment_id} --json` gives"
            )
            message = f"{name} names {len(matches)} model files of deployment {deployment_id}"
        else:
            self.hint = f"run `comfy deploy models ls {deployment_id}` to see its model files"
            message = f"deployment {deployment_id} has no added model file named {name}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _Upload:
    """The file as ``UploadProgressReporter`` reads it."""

    kind: str
    filename: str
    size_bytes: int


def _blob_client() -> BuilderClient:
    return BuilderClient.from_credentials(os.environ.get("COMFY_BUILDER_URL") or DEFAULT_BUILDER_URL)


def _clients() -> tuple[BuilderReleaseClient, DeploymentModelsClient]:
    builder, candidate = _command_clients()
    if not isinstance(candidate, DeploymentModelsClient):
        raise server_shape_error("the deploy client cannot manage a deployment's model files")
    return builder, candidate


def _failed_line(file: str) -> str:
    return f"{file} was not added to this deployment. Remove it and add it again."


def _shown(entry: JsonObject) -> JsonObject:
    """The entry as the CLI shows it: never its failure text, which is for comfy-deploy's log."""
    return {key: value for key, value in entry.items() if key != "failure"}


def _entries(answer: JsonObject) -> list[JsonObject]:
    models = answer.get("models")
    if not isinstance(models, list) or not all(isinstance(entry, dict) for entry in models):
        raise server_shape_error("the deployment's model list has no models array", field="models")
    return [entry for entry in models if isinstance(entry, dict)]


def _added(client: DeploymentModelsClient, deployment_id: str, blob_id: str, folder: str, file: Path) -> JsonObject:
    try:
        return client.add_deployment_model(deployment_id, blob_id, folder)
    except DeployAPIError as error:
        if error.status == 404 and error.details.get("server_code") not in _KNOWN_NOT_FOUND:
            raise ModelsUnavailableError(deployment_id, file, folder) from error
        raise


def _send(blobs: BuilderClient, upload: PartsUpload, file: Path, item: _Upload, renderer: Renderer) -> None:
    reporter = UploadProgressReporter(renderer)
    # As `comfy build push` reports a file the builder turns out to hold: planned, then complete as held.
    reporter.plan([item], already_held=0)
    if upload.deduplicated:
        reporter.deduplicated(item, 1, 1)
        return
    with reporter.uploading(item, 1, 1) as progress:
        try:
            if upload.upload_url is not None:
                blobs.upload_blob(upload.upload_url, file, progress=progress)
            else:
                blobs.upload_blob_parts(upload.blob_id, upload.parts, file, progress)
        except BlobFileError as error:
            raise ModelFileError(file, error.error) from error
        except (FileNotFoundError, PermissionError, IsADirectoryError) as error:
            # The one PUT opening a file removed or locked since it was hashed; a dropped connection is another OSError.
            raise ModelFileError(file, error) from error


def _wait_ready(
    client: DeploymentModelsClient, deployment_id: str, entry: JsonObject, renderer: Renderer
) -> JsonObject:
    """Read the entry until it is ready, saying each status it passes; failed, or removed, ends the wait."""
    entry_id = required_string(entry, "id")
    said: str | None = None
    while True:
        status = required_string(entry, "status")
        if status != said and renderer.is_pretty():
            renderer.info(f"{required_string(entry, 'filename')}: {status}")
        said = status
        if status == "ready":
            return entry
        if status != "adding":
            raise ModelNotAddedError(deployment_id, entry)
        _sleep(DEPLOY_POLL_SECONDS)
        listed = [row for row in _entries(client.list_deployment_models(deployment_id)) if row.get("id") == entry_id]
        entry = listed[0] if listed else {**entry, "status": "removing"}


def add(file: Path, deployment: str, folder: str, *, wait: bool) -> JsonObject:
    renderer = get_renderer()
    builder, client = _clients()
    deployment_id = deployment_id_for(builder, client, deployment)
    blobs = _blob_client()
    if renderer.is_pretty():
        renderer.info(f"Hashing {file.name}...")
    try:
        digests = file_digests(file)
    except OSError as error:  # removed or locked since typer checked it; nothing is sent yet
        raise ModelFileError(file, error) from error
    blake3 = digests.blake3.removeprefix("blake3:")
    upload = blobs.create_blob_parts("model", file.name, digests.sha256, digests.size, blake3=blake3)
    item = _Upload("model", file.name, digests.size)
    if upload.upload_url is None:
        # Held, or sent in pieces: added before any byte moves, so a refusal sends none.
        entry = _added(client, deployment_id, upload.blob_id, folder, file)
        _send(blobs, upload, file, item, renderer)
    else:
        # A builder without the pieces routes: comfy-deploy takes a one-PUT upload once it is whole,
        # so here the whole file is sent before a refusal, `deploy_models_unavailable` included, is known.
        _send(blobs, upload, file, item, renderer)
        entry = _added(client, deployment_id, upload.blob_id, folder, file)
    if wait:
        entry = _wait_ready(client, deployment_id, entry, renderer)
    return {"deploymentId": deployment_id, "model": _shown(entry), "uploaded": not upload.deduplicated}


@contextmanager
def _reported(renderer: Renderer) -> Iterator[None]:
    try:
        yield
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except (DeployResolveError, DeployAPIError) as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except (
        ResponseTooLarge,
        TimeoutError,
        urllib.error.URLError,
        requests.RequestException,
        http.client.HTTPException,
        OSError,
        KeyError,
    ) as error:
        # A connection dropped or an answer cut short mid-read comes as http.client's or a bare OSError, which
        # neither client maps. A failed piece PUT quotes its presigned URL, whose query string is the credential.
        renderer.error(code="deploy_server_error", message=without_signed_query(error) or type(error).__name__)
        raise typer.Exit(code=1) from error


@app.command("add", help="Upload a model file and add it to a deployment, with no new release.")
@tracking.track_command("deploy")
def add_cmd(
    deployment: DeploymentArgument,
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="The model file.", show_default=False)
    ],
    folder: Annotated[str, typer.Option("--type", help="Model folder it goes in, such as loras or checkpoints.")],
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait until the deployment can use it; --no-wait returns once sent."),
    ] = True,
) -> None:
    renderer = get_renderer()
    with _reported(renderer):
        result = add(file, deployment, folder, wait=wait)
        if renderer.is_pretty():
            model, deployment_id = result["model"], result["deploymentId"]
            assert isinstance(model, dict)
            where = f"{model.get('type')}/{model.get('filename')}"
            if model.get("status") == "ready":
                renderer.success(f"Added {where} to deployment {deployment_id}; it is ready.")
            else:
                renderer.success(
                    f"Sent {where} to deployment {deployment_id}; it reads {model.get('status')}. "
                    f"`comfy deploy models ls {deployment_id}` shows when it is ready."
                )
        renderer.emit(result, command="deploy models add", changed=True)


@app.command("ls", help="List the model files added to a deployment: folder, name, size and status.")
@tracking.track_command("deploy")
def ls_cmd(deployment: DeploymentArgument) -> None:
    renderer = get_renderer()
    with _reported(renderer):
        builder, client = _clients()
        deployment_id = deployment_id_for(builder, client, deployment)
        models = [_shown(entry) for entry in _entries(client.list_deployment_models(deployment_id))]
        if renderer.is_pretty():
            if not models:
                renderer.info(f"No model files added to deployment {deployment_id}.")
            for entry in models:
                size = entry.get("sizeBytes")
                shown = human_bytes(size) if isinstance(size, int) else "-"
                filename = required_string(entry, "filename")
                status = required_string(entry, "status")
                renderer.info(f"  {entry.get('type')}  {filename}  {shown}  {status}")
                if status == "failed":
                    renderer.info(f"    {_failed_line(filename)}")
        rows: list[JsonValue] = [*models]
        renderer.emit({"deploymentId": deployment_id, "models": rows}, command="deploy models ls", changed=False)


def _named(models: list[JsonObject], name: str) -> list[JsonObject]:
    return [
        entry
        for entry in models
        if name in (entry.get("id"), entry.get("filename"), f"{entry.get('type')}/{entry.get('filename')}")
    ]


@app.command("rm", help="Remove a model file added to a deployment, deleting it from the deployment's disk.")
@tracking.track_command("deploy")
def rm_cmd(
    ctx: typer.Context,
    deployment: DeploymentArgument,
    # `file`, not `name`: the tracked kwargs redact `file`, and a model's name can be private. NAME on screen.
    file: Annotated[
        str, typer.Argument(metavar="NAME", help="The file's name, <folder>/<name>, or its id.", show_default=False)
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Remove without asking.")] = False,
) -> None:
    renderer = get_renderer()
    with _reported(renderer):
        builder, client = _clients()
        deployment_id = deployment_id_for(builder, client, deployment)
        matches = _named(_entries(client.list_deployment_models(deployment_id)), file)
        if len(matches) != 1:
            raise ModelNotFoundError(deployment_id, file, matches)
        entry = matches[0]
        question = _REMOVE_QUESTION.format(file=required_string(entry, "filename"))
        details = {"deploymentId": deployment_id, "modelId": required_string(entry, "id")}
        if not confirm(question, yes=yes, error_code="deploy_model_remove_needs_confirm", details=details, ctx=ctx):
            renderer.info("Aborted.")
            return
        if yes:
            renderer.info(question)
        removed = client.remove_deployment_model(deployment_id, required_string(entry, "id"))
        if renderer.is_pretty():
            renderer.success(f"Removing {entry.get('type')}/{entry.get('filename')} from deployment {deployment_id}.")
        renderer.emit(
            {"deploymentId": deployment_id, "model": _shown(removed or entry)}, command="deploy models rm", changed=True
        )
