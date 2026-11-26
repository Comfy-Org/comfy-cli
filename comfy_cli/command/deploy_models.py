from __future__ import annotations

import http.client
import os
import shlex
import sys
import time
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Protocol, runtime_checkable

import requests
import typer

from comfy_cli import tracking
from comfy_cli.builder_api import (
    BlobFileError,
    BlobPlanError,
    BuilderAuthError,
    BuilderClient,
    BuilderCredentialRefused,
    PartsUpload,
)
from comfy_cli.command.build import DEFAULT_BUILDER_URL
from comfy_cli.command.build import _report_builder_error as report_builder_error
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
from comfy_cli.command.deploy_runtime import (
    _UNANSWERED_DELAYS,
    _UNANSWERED_WINDOW,
    DEPLOY_POLL_SECONDS,
    EXIT_WATCH_LOST,
    _unanswered,
    render_spec_error,
)
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
_BUILDER_REFUSED: Final = "the builder answered and refused it; change what its message names, then run it again"
_KNOWN_NOT_FOUND: Final = frozenset({"NOT_FOUND", "UPLOAD_NOT_FOUND"})
MODEL_WAIT_SECONDS: Final = 1800.0
_clock = time.monotonic


@dataclass(frozen=True, slots=True)
class _Given:
    value: str
    placeholder: str


def _operand(file: Path) -> str:
    value = str(file)
    return os.path.join(os.curdir, value) if value.startswith("-") else value


def _command(*args: str | _Given) -> str:
    if sys.platform == "win32":
        return " ".join(["comfy", *(arg.placeholder if isinstance(arg, _Given) else arg for arg in args)])
    return shlex.join(["comfy", *(arg.value if isinstance(arg, _Given) else arg for arg in args)])


def _where(*given: _Given) -> str:
    if sys.platform != "win32":
        return ""
    return ", where " + " and ".join(f"{arg.placeholder} is {arg.value}" for arg in given)


@runtime_checkable
class DeploymentModelsClient(DeploymentListClient, Protocol):
    def add_deployment_model(self, deployment_id: str, blob_id: str, folder: str, /) -> JsonObject: ...

    def list_deployment_models(self, deployment_id: str, /) -> JsonObject: ...

    def remove_deployment_model(self, deployment_id: str, model_id: str, /) -> JsonObject: ...


class ModelsUnavailableError(DeployResolveError):
    code = "deploy_models_unavailable"

    def __init__(self, deployment_id: str, file: Path, folder: str) -> None:
        given = (_Given(_operand(file), "<file>"), _Given(folder, "<folder>"))
        upload = _command("assets", "library", "upload", given[0], "--type", given[1])
        self.hint = (
            f"To upload it to the workspace library instead: {upload}{_where(_Given(str(file), '<file>'), given[1])}"
        )
        self.details = {"deploymentId": deployment_id}
        super().__init__(_NOT_ON)


class ModelFileError(DeployResolveError):
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
        deployment = _Given(deployment_id, "<deployment>")
        rm = _command("deploy", "models", "rm", deployment, _Given(required_string(entry, "id"), "<id>"))
        listing = _command("deploy", "models", "ls", deployment)
        self.hint = f"run `{rm}`, wait until `{listing}` no longer lists it, then add it again"
        self.details = {"deploymentId": deployment_id, "model": _shown(entry)}
        super().__init__(_failed_line(file))


class ModelRemovedError(DeployResolveError):
    code = "deploy_model_removed"

    def __init__(self, deployment_id: str, entry: JsonObject, file: Path, folder: str) -> None:
        name = required_string(entry, "filename")
        add = _command(
            "deploy",
            "models",
            "add",
            _Given(deployment_id, "<deployment>"),
            _Given(_operand(file), "<file>"),
            "--type",
            _Given(folder, "<folder>"),
        )
        self.details = {"deploymentId": deployment_id, "model": _shown(entry)}
        if required_string(entry, "status") != "removing":
            self.hint = f"add it again: `{add}`"
            super().__init__(f"{name} is no longer on deployment {deployment_id}; it was removed while being added")
            return
        listing = _command("deploy", "models", "ls", _Given(deployment_id, "<deployment>"))
        self.hint = f"wait until `{listing}` no longer lists it, then add it again: `{add}`"
        super().__init__(
            f"{name} is being removed from deployment {deployment_id}; its removal was started while it was being added"
        )


class ModelWaitTimeoutError(DeployResolveError):
    code = "deploy_model_wait_timeout"
    exit_code = EXIT_WATCH_LOST

    def __init__(
        self, deployment_id: str, entry: JsonObject, seconds: float, unanswered: Exception | None = None
    ) -> None:
        name = required_string(entry, "filename")
        listing = _command("deploy", "models", "ls", _Given(deployment_id, "<deployment>"))
        self.hint = f"run `{listing}` to see when it is ready"
        self.details = {"deploymentId": deployment_id, "model": _shown(entry), "waitedSeconds": seconds}
        if unanswered is not None:
            super().__init__(
                f"the deploy API stopped answering while waiting on {name} ({unanswered}); it may still be adding"
            )
            return
        super().__init__(
            f"{name} was still adding after {int(seconds // 60)} minutes of waiting; it may still be adding"
        )


class ModelNotFoundError(DeployResolveError):
    code = "deploy_model_not_found"

    def __init__(self, deployment_id: str, name: str, matches: list[JsonObject]) -> None:
        shown: list[JsonValue] = [_shown(entry) for entry in matches]
        self.details = {"deploymentId": deployment_id, "name": name, "matches": shown}
        if matches:
            listing = _command("--json", "deploy", "models", "ls", _Given(deployment_id, "<deployment>"))
            self.hint = f"name it as <folder>/<file>, or by the id `{listing}` gives"
            message = f"{name} names {len(matches)} model files of deployment {deployment_id}"
        else:
            listing = _command("deploy", "models", "ls", _Given(deployment_id, "<deployment>"))
            self.hint = f"run `{listing}` to see its model files"
            message = f"deployment {deployment_id} has no added model file named {name}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _Upload:
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
    return {key: value for key, value in entry.items() if key != "failure"}


def _entries(answer: JsonObject) -> list[JsonObject]:
    models = answer.get("models")
    if not isinstance(models, list) or not all(isinstance(entry, dict) for entry in models):
        raise server_shape_error("the deployment's model list has no models array", field="models")
    return [entry for entry in models if isinstance(entry, dict)]


def _added(client: DeploymentModelsClient, deployment_id: str, blob_id: str, folder: str, file: Path) -> JsonObject:
    try:
        entry = client.add_deployment_model(deployment_id, blob_id, folder)
    except DeployAPIError as error:
        if error.status == 404 and error.details.get("server_code") not in _KNOWN_NOT_FOUND:
            raise ModelsUnavailableError(deployment_id, file, folder) from error
        raise
    required_string(entry, "id")
    required_string(entry, "status")
    return entry


def _send(
    blobs: BuilderClient,
    upload: PartsUpload,
    file: Path,
    item: _Upload,
    renderer: Renderer,
    join: Callable[[], None] | None = None,
) -> None:
    reporter = UploadProgressReporter(renderer)
    reporter.plan([item], already_held=0)
    if upload.deduplicated:
        reporter.deduplicated(item, 1, 1)
        return
    with reporter.uploading(item, 1, 1) as progress:
        try:
            if upload.upload_url is not None:
                blobs.upload_blob(upload.upload_url, file, progress=progress)
            else:
                blobs.send_blob_parts(upload.blob_id, upload.parts, file, progress)
        except BlobFileError as error:
            raise ModelFileError(file, error.error) from error
        except (FileNotFoundError, PermissionError, IsADirectoryError) as error:
            raise ModelFileError(file, error) from error
        if join is not None:
            join()


def _listing(client: DeploymentModelsClient, deployment_id: str, entry: JsonObject, started: float) -> list[JsonObject]:
    failures, since = 0, 0.0
    while True:
        try:
            answer = client.list_deployment_models(deployment_id)
        except (DeployAPIError, ConnectionError, http.client.HTTPException) as error:
            if not _unanswered(error):
                raise
            if failures == 0:
                since = _clock()
            if failures == len(_UNANSWERED_DELAYS) or _clock() - since >= _UNANSWERED_WINDOW:
                raise ModelWaitTimeoutError(deployment_id, entry, _clock() - started, error) from error
            _sleep(_UNANSWERED_DELAYS[failures])
            failures += 1
            continue
        return _entries(answer)


def _wait_ready(
    client: DeploymentModelsClient, deployment_id: str, entry: JsonObject, renderer: Renderer, file: Path, folder: str
) -> JsonObject:
    entry_id = required_string(entry, "id")
    said: str | None = None
    started = _clock()
    while True:
        status = required_string(entry, "status")
        if status != said and renderer.is_pretty():
            renderer.info(f"{required_string(entry, 'filename')}: {status}")
        said = status
        if status == "ready":
            return entry
        if status == "removing":
            raise ModelRemovedError(deployment_id, entry, file, folder)
        if status != "adding":
            raise ModelNotAddedError(deployment_id, entry)
        if _clock() - started >= MODEL_WAIT_SECONDS:
            raise ModelWaitTimeoutError(deployment_id, entry, MODEL_WAIT_SECONDS)
        _sleep(DEPLOY_POLL_SECONDS)
        listed = [row for row in _listing(client, deployment_id, entry, started) if row.get("id") == entry_id]
        if not listed:
            raise ModelRemovedError(deployment_id, entry, file, folder)
        entry = listed[0]


def _refused(error: BaseException) -> bool:
    return isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500


def _withdraw(
    client: DeploymentModelsClient,
    deployment_id: str,
    entry: JsonObject,
    error: BaseException,
    renderer: Renderer,
    *,
    maybe_stored: bool = False,
) -> None:
    entry_id, name = entry.get("id"), entry.get("filename")
    if not isinstance(entry_id, str) or not isinstance(name, str):
        return
    deployment, entry_given = _Given(deployment_id, "<deployment>"), _Given(entry_id, "<id>")
    listing = _command("deploy", "models", "ls", deployment)
    rm = _command("deploy", "models", "rm", deployment, entry_given)
    stays = (
        f"{name} stays on the deployment reading adding: remove it with `{rm}`, wait until `{listing}` no longer "
        f"lists it, then add it again{_where(deployment, entry_given)}"
    )
    if maybe_stored:
        hint = (
            f"{name} stays on the deployment, as the builder may have stored it, and it may still be adding: run "
            f"`{listing}` to see when it is ready. If it shows it failed, remove it with `{rm}`, wait until "
            f"`{listing}` no longer lists it, then add it again; the same `{rm}` cancels it while it is adding"
            f"{_where(deployment, entry_given)}"
        )
    elif not isinstance(error, Exception):
        hint = stays
    else:
        try:
            client.remove_deployment_model(deployment_id, entry_id)
        except Exception:
            hint = stays
        else:
            hint = (
                f"{name} was taken off the deployment again: wait until `{listing}` no longer lists it, then add it "
                f"again{_where(deployment)}"
            )
    if not isinstance(error, Exception):
        renderer.warn(hint)
    error.after_send = hint  # type: ignore[attr-defined]


def _after_send(error: BaseException, hint: str | None) -> str | None:
    after = getattr(error, "after_send", None)
    if after is None:
        return hint
    return after if hint is None else f"{hint}; {after}"


def add(file: Path, deployment: str, folder: str, *, wait: bool) -> JsonObject:
    renderer = get_renderer()
    builder, client = _clients()
    deployment_id = deployment_id_for(builder, client, deployment)
    blobs = _blob_client()
    if renderer.is_pretty():
        renderer.info(f"Hashing {file.name}...")
    try:
        digests = file_digests(file)
    except OSError as error:
        raise ModelFileError(file, error) from error
    blake3 = digests.blake3.removeprefix("blake3:")
    upload = blobs.create_blob_parts("model", file.name, digests.sha256, digests.size, blake3=blake3)
    item = _Upload("model", file.name, digests.size)
    if upload.upload_url is None:
        entry = _added(client, deployment_id, upload.blob_id, folder, file)
        joining = False

        def join() -> None:
            nonlocal joining
            joining = True
            blobs.complete_blob(upload.blob_id)

        try:
            _send(blobs, upload, file, item, renderer, join)
        except BaseException as error:
            _withdraw(client, deployment_id, entry, error, renderer, maybe_stored=joining and not _refused(error))
            raise
    else:
        _send(blobs, upload, file, item, renderer)
        entry = _added(client, deployment_id, upload.blob_id, folder, file)
    if wait:
        entry = _wait_ready(client, deployment_id, entry, renderer, file, folder)
    return {"deploymentId": deployment_id, "model": _shown(entry), "uploaded": not upload.deduplicated}


@contextmanager
def _reported(renderer: Renderer) -> Iterator[None]:
    try:
        yield
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except (DeployResolveError, DeployAPIError) as error:
        exit_code = getattr(error, "exit_code", 1)
        hint = getattr(error, "after_send", None) or error.hint
        renderer.error(code=error.code, message=str(error), hint=hint, details=error.details, exit_code=exit_code)
        raise typer.Exit(code=exit_code) from error
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except BuilderCredentialRefused as error:
        hint = _after_send(error, error.hint)
        renderer.error(code="deploy_not_signed_in", message="the builder refused the credential (401)", hint=hint)
        raise typer.Exit(code=1) from error
    except (
        BlobPlanError,
        ResponseTooLarge,
        TimeoutError,
        urllib.error.URLError,
        requests.RequestException,
        http.client.HTTPException,
        OSError,
        KeyError,
    ) as error:
        if isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500:
            report_builder_error(renderer, error, hint=_after_send(error, _BUILDER_REFUSED))
        else:
            message = without_signed_query(error) or type(error).__name__
            renderer.error(code="deploy_server_error", message=message, hint=_after_send(error, None))
        raise typer.Exit(code=1) from error


@app.command("add", help="Upload a model file and add it to a deployment, with no new release.")
@tracking.track_command("deploy")
def models_add_cmd(
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
                listing = _command("deploy", "models", "ls", _Given(str(deployment_id), "<deployment>"))
                renderer.success(
                    f"Sent {where} to deployment {deployment_id}; it reads {model.get('status')}. "
                    f"`{listing}` shows when it is ready."
                )
        renderer.emit(result, command="deploy models add", changed=True)


@app.command("ls", help="List the model files added to a deployment: folder, name, size and status.")
@tracking.track_command("deploy")
def models_ls_cmd(deployment: DeploymentArgument) -> None:
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
def models_rm_cmd(
    ctx: typer.Context,
    deployment: DeploymentArgument,
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
