from __future__ import annotations

import copy
import http.client
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import typer
from click.testing import Result
from deploy_up_support import option_names
from typer.testing import CliRunner
from typing_extensions import NotRequired, TypedDict

from comfy_cli import deploy_jobs
from comfy_cli.cmdline import app
from comfy_cli.command import deploy, deploy_run, deploy_workflow
from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.deploy_assets import AssetResolveRequest, AssetResolveResult
from comfy_cli.deploy_events import JobWatchResult
from comfy_cli.deploy_jobs import JobSubmitRequest
from comfy_cli.discovery import COMMAND_SCHEMAS
from comfy_cli.hashing import blake3_file
from comfy_cli.target import Target

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"


def published_validator() -> jsonschema.Draft202012Validator:
    """The very schema ``comfy discover`` publishes for this command.

    Resolved through ``COMMAND_SCHEMAS`` rather than by filename so a payload
    that drifts from its *advertised* contract fails here — the schema is only
    a contract for consumers who can find it.
    """
    name = COMMAND_SCHEMAS["comfy deploy run"]
    return jsonschema.Draft202012Validator(json.loads((SCHEMAS_DIR / f"{name}.json").read_text()))


class Envelope(TypedDict):
    data: NotRequired[JsonObject]
    error: NotRequired[JsonObject]


class FakeBuilder:
    def list_releases(self, _build_id: str) -> list[JsonObject]:
        return [{"id": "release-1", "deployable": True, "version": 1}]


class FakeControl:
    def __init__(self, endpoint_url: str | None, *, status: str = "ready") -> None:
        self.target = Target(
            kind="cloud", base_url="https://control.test", path_prefix="/v1", auth_token="cloud-secret"
        )
        self.row: JsonObject = {"id": "dep-id", "status": status, "endpointUrl": endpoint_url}
        self.calls: list[str] = []

    def list_all_deployments(self) -> list[JsonObject]:
        self.calls.append("list")
        return [copy.deepcopy(self.row)]

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(f"get:{deployment_id}")
        return copy.deepcopy({**self.row, "id": deployment_id})


class FakeAssetClient:
    def __init__(self, result: AssetResolveResult | None = None) -> None:
        self.result = result
        self.requests: list[AssetResolveRequest] = []

    def resolve_asset(self, request: AssetResolveRequest) -> AssetResolveResult:
        self.requests.append(request)
        assert self.result is not None
        return self.result


class FakeJobClient:
    def __init__(self, response: JsonObject) -> None:
        self.response = response
        self.requests: list[JobSubmitRequest] = []

    def submit_job(self, request: JobSubmitRequest, control_plane: FakeControl) -> JsonObject:
        self.requests.append(request)
        assert control_plane is not None
        return copy.deepcopy(self.response)


def job(status: str = "queued", *, outputs: list[JsonObject] | None = None) -> JsonObject:
    output_values: list[JsonValue] = [*(outputs or [])]
    return {
        "id": "job-1",
        "status": status,
        "outputs": output_values,
        "metrics": {"queue_ms": 9, "execution_ms": 42} if status == "succeeded" else {},
        "urls": {
            "self": "/api/v2/jobs/job-1",
            "events": "/api/v2/jobs/job-1/events",
            "cancel": "/api/v2/jobs/job-1/cancel",
        },
    }


def write_workflow(path: Path, *, local_asset: bool = False) -> Path:
    inputs: JsonObject = {"text": "hello"}
    if local_asset:
        # `input/` is one of the three ComfyUI asset directories the deploy
        # scanner may read; a file merely sitting beside the workflow is not.
        asset_dir = path.parent / "input"
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "input.png").write_bytes(b"input")
        inputs["image"] = "input.png"
    path.write_text(json.dumps({"1": {"class_type": "Test", "inputs": inputs}}), encoding="utf-8")
    return path


def envelope(result: Result) -> Envelope:
    parsed = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert isinstance(parsed, dict)
    resolved: Envelope = {}
    data = parsed.get("data")
    if data is not None:
        assert isinstance(data, dict)
        resolved["data"] = data
    error = parsed.get("error")
    if error is not None:
        assert isinstance(error, dict)
        resolved["error"] = error
    return resolved


def envelope_data(result: Result) -> JsonObject:
    data = envelope(result).get("data")
    assert isinstance(data, dict)
    return data


def envelope_error(result: Result) -> JsonObject:
    error = envelope(result).get("error")
    assert isinstance(error, dict)
    return error


def install_run(
    monkeypatch: pytest.MonkeyPatch,
    control: FakeControl,
    job_client: FakeJobClient,
    *,
    asset_client: FakeAssetClient | None = None,
    watched: JobWatchResult | None = None,
) -> None:
    selected_asset_client = asset_client or FakeAssetClient()
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: (FakeBuilder(), control))
    monkeypatch.setattr(deploy_run, "DeployAssetClient", lambda *_args, **_kwargs: selected_asset_client)
    monkeypatch.setattr(deploy_run, "DeployJobClient", lambda *_args, **_kwargs: job_client)
    if watched is not None:
        monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: watched)


def invoke(
    workflow: Path | None,
    *args: str,
    input_text: str | None = None,
    agentic: bool = True,
) -> Result:
    command = ["--json" if agentic else "--no-json", "deploy", "run", *args]
    if workflow is not None:
        command.extend(["--workflow", str(workflow)])
    return CliRunner().invoke(
        app,
        command,
        input=input_text,
        env={"AI_AGENT": "1" if agentic else None, "NO_COLOR": "1", "COLUMNS": "400"},
    )


def test_deploy_run_is_registered_with_the_complete_option_surface() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "run", "--help"])

    # Then
    assert result.exit_code == 0
    assert {
        "--workflow",
        "--deployment",
        "--wait",
        "--no-wait",
        "--output-dir",
        "--timeout",
        "--no-upload",
        "--asset-root",
        "--allow-upload",
    } <= option_names("run")


def test_deploy_tree_is_exactly_the_seventeen_designed_commands() -> None:
    # Given / When
    command = typer.main.get_command(deploy.app)
    assert hasattr(command, "commands"), command
    command_names = set(command.commands)

    # Then
    assert command_names == {
        "up",
        "promote",
        "rollback",
        "rename",
        "history",
        "cancel",
        "status",
        "ls",
        "show",
        "logs",
        "events",
        "scale",
        "stop",
        "start",
        "delete",
        "run",
        "refs",
    }


def test_happy_wait_resolves_assets_downloads_outputs_and_emits_the_full_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json", local_asset=True)
    output: JsonObject = {
        "node_id": "9",
        "name": "image.png",
        "type": "image",
        "id": "output-1",
        "url": "https://storage.googleapis.com/signed",
    }
    asset_client = FakeAssetClient(AssetResolveResult({"id": "asset-1"}, uploaded=0, deduped=1, bytes=0))
    job_client = FakeJobClient(job())
    watched = JobWatchResult(job("succeeded", outputs=[output]), [output])
    install_run(
        monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client, asset_client=asset_client, watched=watched
    )
    monkeypatch.setattr(
        deploy_run,
        "download_job_outputs",
        lambda *_args, **_kwargs: [
            {"node_id": "9", "name": "image.png", "type": "image", "id": "output-1", "path": "outputs/image.png"}
        ],
    )

    # When
    result = invoke(workflow, str(tmp_path), "--deployment", "dep-id")

    # Then
    data = envelope_data(result)
    assert result.exit_code == 0, result.stderr
    assert data["assets"] == {
        "uploaded": 0,
        "deduped": 1,
        "bytes": 0,
        "files": [
            {
                "local_path": str((tmp_path / "input" / "input.png").resolve()),
                "file_path": "input.png",
                "bytes": 5,
                "uploaded": False,
            }
        ],
    }
    assert data["job"] == {"id": "job-1", "status": "succeeded"}
    assert data["metrics"] == {"queue_ms": 9, "execution_ms": 42}
    output_values = data["outputs"]
    assert isinstance(output_values, list)
    first_output = output_values[0]
    assert isinstance(first_output, dict)
    assert first_output["path"] == "outputs/image.png"
    submitted_node = job_client.requests[0].workflow["1"]
    assert isinstance(submitted_node, dict)
    submitted_inputs = submitted_node["inputs"]
    assert isinstance(submitted_inputs, dict)
    submitted_image = submitted_inputs["image"]
    assert isinstance(submitted_image, dict)
    submitted_info = submitted_image["info"]
    assert isinstance(submitted_info, dict)
    assert submitted_info["id"] == "asset-1"


def test_no_wait_returns_after_submission_without_watch_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)
    monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: pytest.fail("watch called"))
    monkeypatch.setattr(deploy_run, "download_job_outputs", lambda *_args, **_kwargs: pytest.fail("download called"))

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    data = envelope_data(result)
    assert result.exit_code == 0
    assert data["job"] == {"id": "job-1", "status": "queued"}
    assert data["outputs"] == []


def test_the_full_wait_payload_validates_against_the_published_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserting emitted keys equal a literal (as the tests above do) pins the
    producer against itself, and so cannot notice a schema that forbids a key the
    producer always sends — which is how `assets.files` shipped."""
    # Given
    workflow = write_workflow(tmp_path / "workflow.json", local_asset=True)
    output: JsonObject = {
        "node_id": "9",
        "name": "image.png",
        "type": "image",
        "id": "output-1",
        "url": "https://storage.googleapis.com/signed",
    }
    asset_client = FakeAssetClient(AssetResolveResult({"id": "asset-1"}, uploaded=1, deduped=0, bytes=5))
    watched = JobWatchResult(job("succeeded", outputs=[output]), [output])
    install_run(
        monkeypatch,
        FakeControl("https://dep-id.run.comfy.app"),
        FakeJobClient(job()),
        asset_client=asset_client,
        watched=watched,
    )
    monkeypatch.setattr(
        deploy_run,
        "download_job_outputs",
        lambda *_args, **_kwargs: [
            {"node_id": "9", "name": "image.png", "type": "image", "id": "output-1", "path": "outputs/image.png"}
        ],
    )

    # When
    result = invoke(workflow, str(tmp_path), "--deployment", "dep-id")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope_data(result)
    assets = data["assets"]
    assert isinstance(assets, dict)
    assert assets["files"], "the disclosure this schema must cover was absent from the payload"
    published_validator().validate(data)


def test_the_asset_free_no_wait_payload_validates_against_the_published_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`assets.files` is emitted as `[]` rather than omitted when no local file is
    referenced, so the leanest run is every bit as much a schema client as the
    richest one."""
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope_data(result)
    assert data["assets"] == {"uploaded": 0, "deduped": 0, "bytes": 0, "files": []}
    published_validator().validate(data)


def test_each_invocation_uses_a_fresh_job_idempotency_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)

    # When
    first = invoke(workflow, "--deployment", "dep-id", "--no-wait")
    second = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert first.exit_code == second.exit_code == 0
    assert len({request.idempotency_key for request in job_client.requests}) == 2


# --- the partner credential rides with the submission -------------------------


def write_partner_workflow(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "1": {"class_type": "GeminiImage2Node", "inputs": {"prompt": "a fox"}},
                "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param(("auth_token_comfy_org", "session-token"), id="oauth-session"),
        pytest.param(("api_key_comfy_org", "comfyui-key"), id="api-key"),
        pytest.param(None, id="none-configured"),
    ],
)
def test_the_resolved_credential_is_carried_into_the_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    credential: tuple[str, str] | None,
) -> None:
    """Given a resolved credential, When a workflow is submitted, Then it rides along.

    Whichever field the resolver returns is what travels: an interactively
    signed-in caller holds a session token and no API key.
    """
    # Given
    workflow = write_partner_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)
    monkeypatch.setattr(deploy_run, "resolve_partner_credential", lambda: credential)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert result.exit_code == 0, result.stdout
    assert job_client.requests[0].partner_credential == credential


@pytest.mark.parametrize(
    ("partner_env", "partner_key"),
    [(None, "comfyui-team"), ("comfyui-partner-billing", "comfyui-partner-billing")],
    ids=["workspace-key", "explicit-partner-key-kept"],
)
def test_a_key_is_sent_to_the_endpoint_and_to_partner_nodes_in_place_of_a_sign_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partner_env: str | None, partner_key: str
) -> None:
    """Given a control-plane client holding a workspace API key, When a workflow runs,
    Then the deployment's endpoint and its partner nodes both receive that key.

    platform-gateway reads a key only as a Bearer value, and the job runs in the
    key's workspace, so a sign-in's token must not reach partner nodes beside it.
    """
    # Given
    workflow = write_partner_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    control = FakeControl("https://dep-id.run.comfy.app")
    control.target = Target(kind="cloud", base_url="https://control.test", path_prefix="/v1", api_key="comfyui-team")
    install_run(monkeypatch, control, job_client)
    endpoint_credentials: list[str] = []
    monkeypatch.setattr(
        deploy_run, "DeployJobClient", lambda _origin, credential: endpoint_credentials.append(credential) or job_client
    )
    # The sign-in is never consulted, so its refresh never runs.
    monkeypatch.setattr(deploy_run, "resolve_partner_credential", lambda: pytest.fail("sign-in consulted"))
    if partner_env is None:
        monkeypatch.delenv("COMFY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("COMFY_API_KEY", partner_env)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert result.exit_code == 0, result.stdout
    assert (endpoint_credentials, job_client.requests[0].partner_credential) == (
        ["comfyui-team"],
        ("api_key_comfy_org", partner_key),
    )


# --- a workflow over the size limit ---------------------------------------------


def test_an_oversize_workflow_is_refused_after_its_assets_are_uploaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size is measured on the request that carries asset ids, so the files go first.

    The refusal must not claim nothing was sent: the local file is already an
    asset. Only the job request is withheld.
    """
    # Given
    workflow = write_workflow(tmp_path / "workflow.json", local_asset=True)
    nodes = json.loads(workflow.read_text(encoding="utf-8"))
    nodes["2"] = {"class_type": "PreviewAny", "inputs": {"source": "x" * 10_100_000}}
    workflow.write_text(json.dumps(nodes), encoding="utf-8")
    asset_client = FakeAssetClient(AssetResolveResult({"id": "asset-1"}, uploaded=1, deduped=0, bytes=5))
    job_requests: list[object] = []
    monkeypatch.setattr(
        deploy_run, "_command_clients", lambda: (FakeBuilder(), FakeControl("https://dep-id.run.comfy.app"))
    )
    monkeypatch.setattr(deploy_run, "DeployAssetClient", lambda *_args, **_kwargs: asset_client)
    monkeypatch.setattr(deploy_run, "resolve_partner_credential", lambda: None)
    monkeypatch.setattr(deploy_jobs, "request_json", lambda *args, **_kwargs: job_requests.append(args))

    # When
    result = invoke(workflow, str(tmp_path), "--deployment", "dep-id", "--no-wait")

    # Then
    error = envelope_error(result)
    assert error["code"] == "deploy_workflow_too_large"
    assert len(asset_client.requests) == 1
    assert job_requests == []
    message = str(error["message"])
    assert "nothing was sent" not in message
    assert "the job was not submitted, so no job was created" in message


def test_the_release_version_the_gateway_names_is_printed_and_emitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(
        monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient({**job(), "release_version": 7})
    )

    # When
    agentic = invoke(workflow, "--deployment", "dep-id", "--no-wait")
    pretty = invoke(workflow, "--deployment", "dep-id", "--no-wait", agentic=False)

    # Then
    assert agentic.exit_code == 0, agentic.stderr
    data = envelope_data(agentic)
    assert data["job"] == {"id": "job-1", "status": "queued", "releaseVersion": 7}
    published_validator().validate(data)
    assert pretty.exit_code == 0, pretty.stderr
    assert "Deployment job job-1 (release v7) is queued" in pretty.output


def test_a_waited_run_names_the_release_version_of_the_finished_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    watched = JobWatchResult({**job("succeeded"), "release_version": 7}, [])
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()), watched=watched)

    # When
    agentic = invoke(workflow, "--deployment", "dep-id")
    pretty = invoke(workflow, "--deployment", "dep-id", agentic=False)

    # Then
    assert agentic.exit_code == 0, agentic.stderr
    data = envelope_data(agentic)
    assert data["job"] == {"id": "job-1", "status": "succeeded", "releaseVersion": 7}
    published_validator().validate(data)
    assert pretty.exit_code == 0, pretty.stderr
    assert "Deployment job job-1 (release v7) is succeeded" in pretty.output


def test_a_waited_run_keeps_the_submit_answers_version_when_the_finished_job_names_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    watched = JobWatchResult(job("succeeded"), [])
    install_run(
        monkeypatch,
        FakeControl("https://dep-id.run.comfy.app"),
        FakeJobClient({**job(), "release_version": 7}),
        watched=watched,
    )

    # When
    agentic = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert agentic.exit_code == 0, agentic.stderr
    assert envelope_data(agentic)["job"] == {"id": "job-1", "status": "succeeded", "releaseVersion": 7}


@pytest.mark.parametrize("version", [None, "7", 7.5, True, 0])
def test_a_missing_or_unusable_release_version_leaves_the_result_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: JsonValue,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    submitted = job() if version is None else {**job(), "release_version": version}
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(submitted))

    # When
    agentic = invoke(workflow, "--deployment", "dep-id", "--no-wait")
    pretty = invoke(workflow, "--deployment", "dep-id", "--no-wait", agentic=False)

    # Then
    assert agentic.exit_code == 0, agentic.stderr
    assert envelope_data(agentic)["job"] == {"id": "job-1", "status": "queued"}
    assert pretty.exit_code == 0, pretty.stderr
    assert "Deployment job job-1 is queued" in pretty.output


# --- the sign-in token expiring during a run ------------------------------------


class _RotatingSignIn:
    """The stored sign-in, whose token the test swaps as if it had been refreshed."""

    def __init__(self) -> None:
        self.token = "token-at-start"

    def __call__(self, *, refresh=True, force=False, allow_clear=True, rejected_access_token=None):
        return SimpleNamespace(access_token=self.token, base_url="https://cloud.comfy.org")


@pytest.mark.parametrize("refreshes", [True, False], ids=["sign-in", "token-handed-directly"])
def test_a_waited_run_reads_the_sign_in_afresh_after_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refreshes: bool
) -> None:
    """Given a sign-in that rotates while the job runs, When the run waits and downloads,
    Then the watch and the download carry the rotated token, not the one the run started with.

    The token lasts fifteen minutes; a run that sent the starting token for the
    whole job was refused (401) after the job succeeded, and lost its outputs.
    A token the client was handed directly is never swapped.
    """
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    output: JsonObject = {"node_id": "9", "name": "a.png", "type": "image", "id": "o-1", "url": "https://x"}
    control = FakeControl("https://dep-id.run.comfy.app")
    control.sign_in_base_url = "https://cloud.comfy.org" if refreshes else None  # type: ignore[attr-defined]
    install_run(monkeypatch, control, FakeJobClient(job()))
    monkeypatch.setattr(deploy_run, "resolve_partner_credential", lambda: None)
    sign_in = _RotatingSignIn()
    monkeypatch.setattr("comfy_cli.credentials.get_session", sign_in)
    seen: dict[str, str] = {}

    def watch(request, *_args, **_kwargs):
        sign_in.token = "token-after-expiry"
        seen["watch"] = request.current_target().auth_token
        return JobWatchResult(job("succeeded", outputs=[output]), [output])

    def download(request, _renderer):
        seen["download"] = request.token.current()
        return []

    monkeypatch.setattr(deploy_run, "watch_job", watch)
    monkeypatch.setattr(deploy_run, "download_job_outputs", download)

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 0, result.stdout
    expected = "token-after-expiry" if refreshes else "cloud-secret"
    assert seen == {"watch": expected, "download": expected}


def test_an_unreadable_sign_in_at_cancel_still_cancels_with_the_token_the_run_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a waited run interrupted while the sign-in store cannot be read,
    When it cancels the job, Then it still sends the cancel and reports the interrupt."""
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    control = FakeControl("https://dep-id.run.comfy.app")
    control.sign_in_base_url = "https://cloud.comfy.org"  # type: ignore[attr-defined]
    install_run(monkeypatch, control, FakeJobClient(job()))
    monkeypatch.setattr(deploy_run, "resolve_partner_credential", lambda: None)
    store = {"readable": True}

    def get_session(*, refresh=True, force=False, allow_clear=True, rejected_access_token=None):
        if not store["readable"]:
            raise OSError("auth store lock unavailable")
        return SimpleNamespace(access_token="cloud-secret", base_url="https://cloud.comfy.org")

    monkeypatch.setattr("comfy_cli.credentials.get_session", get_session)

    def interrupted_watch(*_args, **_kwargs):
        store["readable"] = False
        raise KeyboardInterrupt

    cancels: list[str | None] = []
    monkeypatch.setattr(deploy_run, "watch_job", interrupted_watch)
    monkeypatch.setattr(
        deploy_run, "request_json", lambda _url, target, **_kwargs: cancels.append(target.auth_token) or (202, {})
    )

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 130, result.stdout
    assert cancels == ["cloud-secret"]
    assert envelope_error(result)["details"] == {"job_id": "job-1", "cancel_requested": True}


# --- the upload plan a person approves ---------------------------------------


def two_portraits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A Build folder (the cwd) and the tracked workspace each hold a different
    `portrait.png`. The Build folder also holds `customer.csv`, which a custom
    node names as `input/customer.csv`. Returns the Build folder, the workspace
    and the API workflow."""
    build = tmp_path / "build"
    (build / "input").mkdir(parents=True)
    (build / "comfy-build.yaml").write_text("schema: build-spec/1\n", encoding="utf-8")
    (build / "input" / "portrait.png").write_bytes(b"build portrait")
    (build / "input" / "customer.csv").write_bytes(b"name,email\n")
    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    (workspace / "input" / "portrait.png").write_bytes(b"workspace portrait")

    class TrackedWorkspace:
        workspace_path = str(workspace)

    monkeypatch.setattr(deploy_workflow, "WorkspaceManager", TrackedWorkspace)
    monkeypatch.chdir(build)
    workflow = tmp_path / "api.json"
    workflow.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "portrait.png"}},
                "2": {"class_type": "CustomerLookup", "inputs": {"table": "input/customer.csv"}},
            }
        ),
        encoding="utf-8",
    )
    return build, workspace, workflow


def deployment_holds_nothing(monkeypatch: pytest.MonkeyPatch, job_client: FakeJobClient) -> list[str]:
    """Run against the real asset client, whose every hash probe misses.

    Returns the names of the files it uploads; each upload answers with a new asset.
    """
    control = FakeControl("https://dep-id.run.comfy.app")
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: (FakeBuilder(), control))
    monkeypatch.setattr(deploy_run, "DeployJobClient", lambda *_args, **_kwargs: job_client)

    def probe(url: str, *_args, **_kwargs):
        raise urllib.error.HTTPError(url, 404, "not found", http.client.HTTPMessage(), io.BytesIO(b""))

    uploaded: list[str] = []

    class Created:
        status = 201

        def __init__(self, request: urllib.request.Request) -> None:
            body = request.data
            assert body is not None and not isinstance(body, bytes)
            chunks = b"".join(body)  # type: ignore[arg-type]
            uploaded.append(chunks.split(b'filename="', 1)[1].split(b'"', 1)[0].decode())
            self._body = io.BytesIO(json.dumps({"id": f"asset-{len(uploaded)}"}).encode())

        def __enter__(self) -> Created:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, amount: int = -1) -> bytes:
            return self._body.read(amount)

    monkeypatch.setattr("comfy_cli.deploy_assets.request_json", probe)
    monkeypatch.setattr("comfy_cli.deploy_assets.no_redirect_urlopen", lambda request, *, timeout: Created(request))
    return uploaded


def test_a_no_upload_refusal_lists_the_file_the_scanner_picked_by_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given two different `portrait.png` files, When a `--no-upload` run refuses,
    Then it lists every file it would send, with the full path it picked: the
    Build folder's `portrait.png`, not the workspace's, and the custom node's
    `customer.csv`. This list is what a person approves."""
    # Given
    build, _workspace, workflow = two_portraits(tmp_path, monkeypatch)
    job_client = FakeJobClient(job())
    uploaded = deployment_holds_nothing(monkeypatch, job_client)
    portrait = (build / "input" / "portrait.png").resolve()
    customers = (build / "input" / "customer.csv").resolve()

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-upload")

    # Then
    assert result.exit_code == 1, result.stdout
    error = envelope_error(result)
    assert error["code"] == "deploy_asset_missing"
    details = error["details"]
    assert isinstance(details, dict)
    assert details["missing"] == [
        {"file_path": "portrait.png", "hash": blake3_file(portrait), "local_path": str(portrait), "bytes": 14},
        {"file_path": "customer.csv", "hash": blake3_file(customers), "local_path": str(customers), "bytes": 11},
    ]
    assert uploaded == []
    assert job_client.requests == []


def test_allow_upload_sends_only_the_approved_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a person approved the workspace's `portrait.png`, When the run picks
    the Build folder's one instead, Then nothing is uploaded and the refusal
    names the file it picked. Approving the listed hashes uploads exactly those."""
    # Given
    build, workspace, workflow = two_portraits(tmp_path, monkeypatch)
    job_client = FakeJobClient(job())
    uploaded = deployment_holds_nothing(monkeypatch, job_client)
    portrait = (build / "input" / "portrait.png").resolve()
    customers = (build / "input" / "customer.csv").resolve()
    wrong = blake3_file(workspace / "input" / "portrait.png")

    # When the approval names the other portrait
    refused = invoke(workflow, "--deployment", "dep-id", "--no-wait", "--allow-upload", wrong)

    # Then
    assert refused.exit_code == 1, refused.stdout
    details = envelope_error(refused)["details"]
    assert isinstance(details, dict)
    missing = details["missing"]
    assert isinstance(missing, list)
    assert [entry["local_path"] for entry in missing if isinstance(entry, dict)] == [str(portrait), str(customers)]
    assert uploaded == []
    assert job_client.requests == []

    # When the approval names the plan the refusal listed
    allowed = invoke(
        workflow,
        "--deployment",
        "dep-id",
        "--no-wait",
        "--allow-upload",
        blake3_file(portrait),
        "--allow-upload",
        blake3_file(customers),
    )

    # Then
    assert allowed.exit_code == 0, allowed.stdout
    assert uploaded == ["portrait.png", "customer.csv"]
    assets = envelope_data(allowed)["assets"]
    assert isinstance(assets, dict)
    files = assets["files"]
    assert isinstance(files, list)
    assert [entry["local_path"] for entry in files if isinstance(entry, dict)] == [str(portrait), str(customers)]
    assert len(job_client.requests) == 1


def test_a_refused_file_fails_the_run_before_an_approved_one_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a person approved only the second of two missing files, When the
    first is refused, Then the run fails before anything is uploaded, so the
    deployment is not left holding an approved file no job uses."""
    # Given
    build, _workspace, workflow = two_portraits(tmp_path, monkeypatch)
    job_client = FakeJobClient(job())
    uploaded = deployment_holds_nothing(monkeypatch, job_client)
    portrait = (build / "input" / "portrait.png").resolve()
    customers = (build / "input" / "customer.csv").resolve()

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait", "--allow-upload", blake3_file(customers))

    # Then
    assert result.exit_code == 1, result.stdout
    error = envelope_error(result)
    assert error["code"] == "deploy_asset_missing"
    details = error["details"]
    assert isinstance(details, dict)
    missing = details["missing"]
    assert isinstance(missing, list)
    assert [entry["local_path"] for entry in missing if isinstance(entry, dict)] == [str(portrait)]
    assert uploaded == []
    assert job_client.requests == []


def test_an_approved_upload_reports_the_bytes_it_sent(tmp_path: Path) -> None:
    """Given an approved file that is replaced by other approved bytes after the
    probe, When it uploads, Then `assets.files[].bytes` is what was sent, not
    the size read before the probe."""
    from comfy_cli.command import deploy_run_assets
    from comfy_cli.deploy_assets import DeployAssetMissingError

    # Given
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(b"abc")
    plan = deploy_workflow.WorkflowAssetPlan(
        workflow={"1": {"inputs": {"image": "marker-1"}}},
        assets=(deploy_workflow.WorkflowAsset("marker-1", portrait, "portrait.png"),),
    )

    class ProbeThenUpload:
        def resolve_asset(self, request: AssetResolveRequest) -> AssetResolveResult:
            if not request.upload_enabled:
                raise DeployAssetMissingError(request, "hash-a", 3)
            return AssetResolveResult(asset={"id": "asset-1"}, uploaded=1, deduped=0, bytes=7)

    context = deploy_run_assets.AssetResolveContext(
        client=ProbeThenUpload(),  # type: ignore[arg-type]
        renderer=SimpleNamespace(is_pretty=lambda: False),  # type: ignore[arg-type]
        upload_enabled=True,
        allowed_hashes=frozenset({"hash-a", "hash-b"}),
    )

    # When
    resolved = deploy_run_assets.resolve_assets(plan, context)

    # Then
    assert [file.bytes for file in resolved.files] == [7]
    assert resolved.bytes == 7
