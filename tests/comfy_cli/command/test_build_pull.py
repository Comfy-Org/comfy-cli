from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from build_pull_support import PullBuilder, invoke_pull, serve
from build_push_support import envelope, invoke_push, make_workspace, reloaded, write_spec

from comfy_cli.caller import Caller
from comfy_cli.command import build
from comfy_cli.command.build_spec import read_build_spec
from comfy_cli.discovery import COMMAND_SCHEMAS

SCHEMAS_DIR = Path(__file__).parents[3] / "comfy_cli" / "schemas"


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


def install_client(monkeypatch: pytest.MonkeyPatch, client: PullBuilder) -> None:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)


def remote_definition() -> dict:
    return {"schema": "distribution-definition/0", "models": [], "customNodes": []}


def test_agentic_pull_needs_an_id_when_the_spec_has_none(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_spec(workspace, models=[], nodes=[])
    client = PullBuilder()
    install_client(monkeypatch, client)

    result = invoke_pull(workspace, "-y")

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_id_unknown"
    assert client.calls == []


def test_agentic_pull_without_yes_fetches_then_refuses_without_writing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_spec(workspace, build_id="build-a", models=[], nodes=[])
    before = path.read_bytes()
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)

    result = invoke_pull(workspace)

    assert result.exit_code == 1
    assert envelope(result)["error"]["code"] == "build_pull_needs_confirm"
    assert [call["method"] for call in client.calls] == ["get_build"]
    assert path.read_bytes() == before


def test_tty_decline_leaves_the_spec_byte_identical(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_spec(workspace, build_id="build-a", models=[], nodes=[])
    before = path.read_bytes()
    client = PullBuilder()
    serve(client, "build-a", remote_definition())
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: False)

    result = invoke_pull(workspace, agentic=False)

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before


def test_tty_without_an_id_uses_the_build_picker(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_spec(workspace, models=[], nodes=[])
    client = PullBuilder()
    serve(client, "build-picked", remote_definition())
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.ui.prompt_select", lambda *args, **kwargs: "build-picked")
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: True)

    result = invoke_pull(workspace, agentic=False)

    assert result.exit_code == 0, result.output
    assert reloaded(workspace)["id"] == "build-picked"
    assert [call["method"] for call in client.calls] == ["list_builds", "get_build"]


def test_cross_id_pull_rebinds_and_the_next_plain_push_targets_the_new_build(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_spec(
        workspace,
        build_id="build-a",
        revision="revision-a",
        models=[],
        nodes=[],
        definition_extra={"localOnly": {"kept": True}, "conflict": "local"},
    )
    client = PullBuilder()
    serve(client, "build-b", {**remote_definition(), "serverOnly": {"kept": True}, "conflict": "server"})
    install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda question: pytest.fail("-y prompted"))

    pulled = invoke_pull(workspace, "--id", "build-b", "-y")
    after_pull = read_build_spec(workspace / "comfy-build.yaml")
    pushed = invoke_push(workspace)

    assert pulled.exit_code == 0, pulled.stderr
    assert pushed.exit_code == 0, pushed.stderr
    assert (after_pull["id"], after_pull["syncedRevision"]) == ("build-b", "revision-build-b")
    assert envelope(pulled)["data"]["syncedRevision"] == "revision-build-b"
    spec = read_build_spec(workspace / "comfy-build.yaml")
    assert (spec["id"], spec["syncedRevision"], spec["name"], spec["description"]) == (
        "build-b",
        "revision-1",
        "Remote build-b",
        "Description for build-b",
    )
    assert spec["definition"]["localOnly"] == {"kept": True}
    assert spec["definition"]["serverOnly"] == {"kept": True}
    assert spec["definition"]["conflict"] == "server"
    assert [call["id"] for call in client.calls if call["method"] == "update_build"] == ["build-b"]
    assert COMMAND_SCHEMAS["comfy build pull"] == "build_pull"
    schema = json.loads((SCHEMAS_DIR / "build_pull.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(envelope(pulled)["data"])
