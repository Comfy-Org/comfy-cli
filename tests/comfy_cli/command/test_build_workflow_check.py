"""`comfy build validate --workflow` and `comfy build push --workflow`.

The builder's POST /v1/workflows/validate compares a
workflow with a definition the caller sends. These pin that the CLI sends the
LOCAL spec's definition, stops a push before anything moves when a class is
missing or cannot be verified, and treats an old builder or a strange answer as
nothing verified. The answers are the shapes that PR's own tests pin.
"""

from __future__ import annotations

import copy
import io
import json
import urllib.error
from pathlib import Path

import jsonschema
import pytest
from build_push_support import RecordingBuilder, envelope, local_model, local_node, make_workspace, write_spec
from typer.testing import CliRunner

from comfy_cli.builder_api import BuilderClient
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject

VALIDATE_URL = "https://builder.test/v1/workflows/validate"

#: An API export using one custom class twice, beside a built-in one.
API_WORKFLOW: JsonObject = {
    "1": {"class_type": "KSampler", "inputs": {}},
    "2": {"class_type": "UltimateSDUpscale", "inputs": {}},
    "3": {"class_type": "UltimateSDUpscale", "inputs": {}},
}
#: The same graph as the editor saves it.
UI_WORKFLOW: JsonObject = {
    "nodes": [
        {"id": 1, "type": "KSampler", "widgets_values": []},
        {"id": 2, "type": "UltimateSDUpscale", "widgets_values": []},
        {"id": 3, "type": "UltimateSDUpscale", "widgets_values": []},
    ],
    "links": [],
}

MISSING: JsonObject = {
    "format": "api",
    "complete": False,
    "covered": 0,
    "missing": [
        {
            "packId": "ultimate-sd-upscale",
            "name": "Ultimate SD Upscale",
            "latestVersion": "1.2.0",
            "classes": ["UltimateSDUpscale", "UltimateSDUpscale"],
        }
    ],
    "unverified": [],
}
COVERED: JsonObject = {"format": "api", "complete": True, "covered": 1, "missing": [], "unverified": []}
UNVERIFIED: JsonObject = {
    "format": "api",
    "complete": False,
    "covered": 0,
    "missing": [],
    "unverified": [{"classType": "SilentNode", "reason": "lookupFailed"}],
}

#: (status, body, error code): every answer that must not read as verified.
NOT_VERIFIED = [
    pytest.param(200, UNVERIFIED, "build_workflow_unverified", id="unverified"),
    pytest.param(503, {"error": "UNAVAILABLE", "message": "registry down"}, "build_builder_error", id="5xx"),
    pytest.param(404, {"error": "NOT_FOUND"}, "build_workflow_check_unavailable", id="old-server-404"),
    pytest.param(405, {}, "build_workflow_check_unavailable", id="old-server-405"),
    pytest.param(501, {}, "build_workflow_check_unavailable", id="old-server-501"),
    pytest.param(200, {"complete": True}, "build_workflow_check_unavailable", id="no-lists"),
    pytest.param(200, {**COVERED, "complete": "yes"}, "build_workflow_check_unavailable", id="complete-not-bool"),
    pytest.param(
        200, {**COVERED, "unverified": [{"classType": "X"}]}, "build_workflow_check_unavailable", id="bad-entry"
    ),
    pytest.param(200, None, "build_workflow_check_unavailable", id="not-json"),
]


class CoverageTransport:
    """The builder at the ``request_json`` seam: the workflow check answers
    ``answer``; every other route fails the test, so a saved-Build read shows."""

    def __init__(self, status: int = 200, answer: JsonObject | None = COVERED) -> None:
        self.status = status
        self.answer = answer
        self.calls: list[JsonObject] = []

    def __call__(self, url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        self.calls.append({"url": url, "method": method, "body": body})
        if (method, url) == ("GET", "https://builder.test/v1/model-directories"):
            return 200, {"directories": ["checkpoints", "loras"]}
        if (method, url) != ("POST", VALIDATE_URL):
            pytest.fail(f"unexpected Builder call: {method} {url}")
        if self.status >= 400:
            raise urllib.error.HTTPError(url, self.status, "refused", {}, io.BytesIO(json.dumps(self.answer).encode()))
        return self.status, copy.deepcopy(self.answer)

    def checks(self) -> list[JsonObject]:
        return [call for call in self.calls if call["url"] == VALIDATE_URL]


class CoverageBuilder(RecordingBuilder):
    """The push fake, whose workflow check is the real client over the transport."""

    def validate_workflow(self, workflow: JsonObject, definition: JsonObject) -> JsonObject:
        self.calls.append({"method": "validate_workflow", "workflow": workflow, "definition": definition})
        return BuilderClient("https://builder.test", "tok_test").validate_workflow(workflow, definition)


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def _workflow_file(root: Path, workflow: JsonObject = API_WORKFLOW, name: str = "wf.json") -> Path:
    path = root / name
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return path


def _invoke(command: str, root: Path, *args: str, token: str | None = "tok_test", output: str = "json"):
    return CliRunner().invoke(
        cli_app,
        ["build", command, *args, str(root)],
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": output,
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def _said(result) -> str:
    return " ".join(result.output.split())


def _schema(name: str) -> JsonObject:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_node() -> JsonObject:
    return {"name": "upscale", "id": "ultimate-sd-upscale", "registryVersion": "1.2.0"}


# --- validate -----------------------------------------------------------------


def test_validate_names_each_missing_class_once_in_json(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    transport = CoverageTransport(answer=MISSING)
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    write_spec(workspace, models=[], nodes=[])

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)))

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == "build_workflow_nodes_missing"
    assert error["details"]["missing"] == [
        {
            "packId": "ultimate-sd-upscale",
            "classes": ["UltimateSDUpscale"],
            "name": "Ultimate SD Upscale",
            "latestVersion": "1.2.0",
        }
    ]
    assert error["details"]["unverified"] == []
    assert "ultimate-sd-upscale: UltimateSDUpscale" in error["message"]
    assert error["message"].count("UltimateSDUpscale") == 1
    assert "workflow_coverage" not in json.dumps(envelope(result))


def test_validate_names_each_missing_class_once_to_a_person(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport(answer=MISSING))
    write_spec(workspace, models=[], nodes=[])

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)), output="pretty")

    # Then
    assert result.exit_code == 1, result.output
    said = _said(result)
    assert "ultimate-sd-upscale: UltimateSDUpscale" in said
    assert said.count("UltimateSDUpscale") == 1
    assert "Build spec is valid" not in said


def test_validate_tells_each_missing_pack_the_source_it_can_be_added_from(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pack the registry does not publish comes back with its clone URL as
    ``packId``, its ``repository``, and no ``latestVersion``: it can only be added
    from that repository, so its line must not send the user to the registry."""
    # Given
    clone = "https://github.com/example/comfyui-unpublished"
    answer = copy.deepcopy(MISSING)
    answer["missing"].insert(0, {"packId": clone, "repository": clone, "classes": ["UnpublishedNode"]})
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport(answer=answer))
    write_spec(workspace, models=[], nodes=[])

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)))

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == "build_workflow_nodes_missing"
    assert error["details"]["missing"][0] == {"packId": clone, "classes": ["UnpublishedNode"], "repository": clone}
    [unpublished] = [line for line in error["message"].splitlines() if line.startswith(f"  {clone}:")]
    [published] = [line for line in error["message"].splitlines() if line.startswith("  ultimate-sd-upscale:")]
    assert f"from its repository: repository {clone}, gitRef" in unpublished
    assert "registry" not in unpublished
    assert "from the registry: id ultimate-sd-upscale, registryVersion 1.2.0" in published
    assert "registry id at the latest version" not in error["hint"]
    assert "`repository` + `gitRef`" in error["hint"]


@pytest.mark.parametrize("workflow", [API_WORKFLOW, UI_WORKFLOW], ids=["api-export", "ui-export"])
def test_validate_passes_a_covered_workflow_in_either_export(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, workflow: JsonObject
) -> None:
    # Given
    transport = CoverageTransport(answer={**COVERED, "format": "ui" if "nodes" in workflow else "api"})
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    write_spec(workspace, models=[], nodes=[_registry_node()])
    path = _workflow_file(workspace, workflow)

    # When
    result = _invoke("validate", workspace, "--workflow", str(path))

    # Then
    assert result.exit_code == 0, result.output
    data = envelope(result)["data"]
    assert data["workflow_coverage"] == {
        "path": str(path),
        "format": "ui" if "nodes" in workflow else "api",
        "complete": True,
        "covered": 1,
    }
    jsonschema.Draft202012Validator(_schema("build_validate.json")).validate(data)
    # The file goes as parsed; the builder tells the two formats apart.
    assert [call["body"]["workflow"] for call in transport.checks()] == [workflow]


def test_validate_tells_a_person_the_workflow_is_covered(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport())
    write_spec(workspace, models=[], nodes=[_registry_node()])

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)), output="pretty")

    # Then
    assert result.exit_code == 0, result.output
    assert "installs all 1 custom node class(es)" in _said(result)


def test_validate_checks_the_local_edit_not_the_saved_build(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The spec names a saved Build, and the local definition has gained a pack
    since it synced. The check reads the local copy and never the remote one (the
    transport fails any Build read)."""
    # Given
    transport = CoverageTransport()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    spec_path = write_spec(workspace, build_id="build-1", revision="revision-1", models=[], nodes=[_registry_node()])
    before = spec_path.read_bytes()

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)))

    # Then
    assert result.exit_code == 0, result.output
    assert [call["url"] for call in transport.calls] == [VALIDATE_URL]
    assert transport.calls[0]["body"]["definition"]["customNodes"] == [_registry_node()]
    assert spec_path.read_bytes() == before


def test_validate_sends_a_local_node_as_the_upload_a_push_makes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local node has no source on the wire until it uploads; sent bare, its
    pack would read as missing. It goes as an upload, which the builder can only
    leave unverified, and the authoring fields stay home."""
    # Given
    transport = CoverageTransport()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    write_spec(workspace, models=[], nodes=[local_node(repository="https://github.com/example/node")])

    # When
    result = _invoke("validate", workspace, "--workflow", str(_workflow_file(workspace)))

    # Then
    assert result.exit_code == 0, result.output
    assert transport.calls[0]["body"]["definition"]["customNodes"] == [
        {"name": "local-node", "blobId": "pending-upload"}
    ]


@pytest.mark.parametrize(("status", "answer", "code"), NOT_VERIFIED)
def test_validate_never_reads_an_unchecked_workflow_as_verified(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, status: int, answer: JsonObject | None, code: str
) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport(status, answer))
    write_spec(workspace, models=[], nodes=[])
    path = _workflow_file(workspace)

    # When
    result = _invoke("validate", workspace, "--workflow", str(path))
    pretty = _invoke("validate", workspace, "--workflow", str(path), output="pretty")

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == code
    assert str(path) in error["message"] + (error.get("hint") or "")
    assert envelope(result)["data"] is None
    assert pretty.exit_code == 1
    assert "Build spec is valid" not in _said(pretty)
    assert "installs all" not in _said(pretty)


@pytest.mark.parametrize(
    "contents",
    [pytest.param(None, id="no-file"), pytest.param("{not json", id="invalid-json"), pytest.param("[]", id="a-list")],
)
def test_validate_refuses_an_unreadable_workflow_before_any_http(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None
) -> None:
    # Given
    transport = CoverageTransport()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    write_spec(workspace, models=[], nodes=[])
    path = workspace / "wf.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    # When: signed out too, so a sign-in refusal would show first if it came first.
    result = _invoke("validate", workspace, "--workflow", str(path), token=None)

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == "build_workflow_invalid"
    assert error["details"] == {"path": str(path)}
    assert transport.calls == []


def test_validate_without_a_workflow_keeps_its_shape_and_signs_in_for_nothing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))
    write_spec(workspace, models=[], nodes=[])

    # When
    result = _invoke("validate", workspace, token=None)

    # Then
    assert result.exit_code == 0, result.output
    assert set(envelope(result)["data"]) == {"spec_file", "remote", "wire_definition", "model_lookups"}


# --- push ---------------------------------------------------------------------


def _install(
    monkeypatch: pytest.MonkeyPatch, status: int = 200, answer: JsonObject | None = COVERED
) -> CoverageBuilder:
    client = CoverageBuilder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport(status, answer))
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)
    return client


_MOVES = {"create_blob", "upload_blob", "create_build", "update_build", "create_release", "resolve_snapshot"}


def _moves(client: RecordingBuilder) -> list[str]:
    return [call["method"] for call in client.calls if call["method"] in _MOVES]


@pytest.mark.parametrize(("build_id", "revision"), [(None, None), ("build-1", "revision-1")], ids=["create", "update"])
def test_push_with_a_missing_pack_moves_nothing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, build_id: str | None, revision: str | None
) -> None:
    # Given: a local model and node waiting to upload, and a release asked for.
    client = _install(monkeypatch, answer=MISSING)
    client.remote_revisions = {"build-1": "revision-1"}
    spec_path = write_spec(
        workspace, build_id=build_id, revision=revision, models=[local_model()], nodes=[local_node()]
    )
    before = spec_path.read_bytes()
    args = ("--workflow", str(_workflow_file(workspace)), "--release", "--target", "linux/nvidia")

    # When
    result = _invoke("push", workspace, *args)
    pretty = _invoke("push", workspace, *args, output="pretty")

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == "build_workflow_nodes_missing"
    assert error["details"]["missing"][0]["classes"] == ["UltimateSDUpscale"]
    assert pretty.exit_code == 1
    assert "ultimate-sd-upscale: UltimateSDUpscale" in _said(pretty)
    assert _moves(client) == []
    assert client.uploaded == []
    assert spec_path.read_bytes() == before


@pytest.mark.parametrize(("status", "answer", "code"), NOT_VERIFIED)
def test_push_moves_nothing_for_a_workflow_it_could_not_verify(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, status: int, answer: JsonObject | None, code: str
) -> None:
    # Given
    client = _install(monkeypatch, status, answer)
    spec_path = write_spec(workspace)
    before = spec_path.read_bytes()

    # When
    result = _invoke("push", workspace, "--workflow", str(_workflow_file(workspace)))

    # Then
    assert result.exit_code == 1, result.output
    assert envelope(result)["error"]["code"] == code
    assert _moves(client) == []
    assert spec_path.read_bytes() == before


def test_push_checks_a_covered_workflow_before_the_first_upload(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    client = _install(monkeypatch)
    write_spec(workspace)
    path = _workflow_file(workspace, UI_WORKFLOW)

    # When
    result = _invoke("push", workspace, "--workflow", str(path))

    # Then
    assert result.exit_code == 0, result.output
    methods = [call["method"] for call in client.calls]
    assert methods.index("validate_workflow") < methods.index("create_blob")
    assert "create_build" in methods
    check = client.calls[methods.index("validate_workflow")]
    assert check["workflow"] == UI_WORKFLOW
    assert [node.get("blobId") for node in check["definition"]["customNodes"]] == ["pending-upload"]
    data = envelope(result)["data"]
    assert data["workflow_coverage"] == {"path": str(path), "format": "api", "complete": True, "covered": 1}
    jsonschema.Draft202012Validator(_schema("build_push.json")).validate(data)


def test_push_rejects_a_workflow_with_dry_run_before_any_http(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    transport = CoverageTransport()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", transport)
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))
    write_spec(workspace)

    # When
    result = _invoke("push", workspace, "--dry-run", "--workflow", str(_workflow_file(workspace)), token=None)

    # Then
    assert result.exit_code == 1, result.output
    error = envelope(result)["error"]
    assert error["code"] == "build_missing_input"
    assert error["details"] == {"conflict": ["--workflow", "--dry-run"]}
    assert transport.calls == []


def test_push_refuses_an_unreadable_workflow_before_signing_in(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("Builder client constructed"))
    write_spec(workspace)

    # When
    result = _invoke("push", workspace, "--workflow", str(workspace / "absent.json"))

    # Then
    assert result.exit_code == 1, result.output
    assert envelope(result)["error"]["code"] == "build_workflow_invalid"


def test_push_without_a_workflow_checks_nothing(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    client = _install(monkeypatch)
    write_spec(workspace)

    # When
    result = _invoke("push", workspace)

    # Then
    assert result.exit_code == 0, result.output
    assert "validate_workflow" not in [call["method"] for call in client.calls]
    data = envelope(result)["data"]
    assert "workflow_coverage" not in data
    jsonschema.Draft202012Validator(_schema("build_push.json")).validate(data)


def test_validate_with_a_workflow_still_says_it_did_not_check_a_folders_case(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only --remote reads the folder list; signing in for the workflow check does not."""
    # Given
    monkeypatch.setattr("comfy_cli.builder_api.request_json", CoverageTransport())
    write_spec(workspace, models=[{"type": "loras", "sourceUri": "https://h.example/a.safetensors"}], nodes=[])
    path = str(_workflow_file(workspace))

    # When
    pretty = _invoke("validate", workspace, "--workflow", path, output="pretty")
    agent = _invoke("validate", workspace, "--workflow", path)

    # Then
    assert pretty.exit_code == 0, pretty.output
    assert "a folder's case (`Loras` for `loras`) was not checked" in _said(pretty)
    assert envelope(agent)["data"]["folder_case_checked"] is False
