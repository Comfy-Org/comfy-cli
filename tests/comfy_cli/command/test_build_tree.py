from __future__ import annotations

from pathlib import Path

import pytest
import typer
from build_push_support import make_workspace, write_spec
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject


class RemoteBuilder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_builds(self) -> list[JsonObject]:
        self.calls.append({"method": "list_builds"})
        return [{"id": "picked-id", "name": "Picked"}]

    def get_build(self, build_id: str) -> JsonObject:
        self.calls.append({"method": "get_build", "id": build_id})
        return {"id": build_id, "name": "Selected", "definition": {}}

    def delete_build(self, build_id: str) -> None:
        self.calls.append({"method": "delete_build", "id": build_id})

    def create_release(self, build_id: str, targets: list[JsonObject]) -> tuple[str, str]:
        self.calls.append({"method": "create_release", "id": build_id, "targets": targets})
        return "release-1", "https://builder.test/v1/releases/release-1"

    def list_releases(self, build_id: str) -> list[JsonObject]:
        self.calls.append({"method": "list_releases", "id": build_id})
        return [{"id": "release-1", "version": 1}]

    def get_release(self, release_id: str) -> JsonObject:
        self.calls.append({"method": "get_release", "id": release_id})
        return {"id": release_id, "status": "complete", "artifactCounts": {"failed": 0}}

    def get_release_logs(self, release_id: str, *, os: str, gpu: str) -> JsonObject:
        self.calls.append({"method": "get_release_logs", "id": release_id, "os": os, "gpu": gpu})
        return {"versionId": release_id, "log": "built", "truncated": False}

    def get_release_manifest(self, release_id: str) -> JsonObject:
        self.calls.append({"method": "get_release_manifest", "id": release_id})
        return {"versionId": release_id, "models": []}


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = make_workspace(tmp_path / "install")
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> RemoteBuilder:
    recorder = RemoteBuilder()
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: recorder)
    return recorder


def invoke_build(*args: str, agentic: bool = True):
    return CliRunner().invoke(
        cli_app,
        ["build", *args],
        env={
            "AI_AGENT": "1" if agentic else None,
            "COMFY_OUTPUT": "json" if agentic else "pretty",
            "NO_COLOR": "1",
        },
    )


def test_complete_visible_build_command_tree_is_exact() -> None:
    # Given
    command = typer.main.get_command(build.app)

    # When
    visible = sorted(name for name, child in command.commands.items() if not child.hidden)

    # Then
    assert visible == [
        "delete",
        "init",
        "ls",
        "pull",
        "push",
        "refs",
        "release",
        "show",
        "status",
        "update",
        "validate",
    ]
    assert command.commands["blob"].hidden is True


@pytest.mark.parametrize(
    ("tier", "spec_id", "options", "expected_id"),
    [
        pytest.param("explicit", "spec-id", ("--id", "explicit-id"), "explicit-id", id="explicit"),
        pytest.param("spec", "spec-id", (), "spec-id", id="spec"),
        pytest.param("picker", None, (), "picked-id", id="picker"),
    ],
)
@pytest.mark.parametrize(
    ("command", "fixed_options", "method"),
    [
        pytest.param("show", (), "get_build", id="show"),
        pytest.param("delete", ("--yes",), "delete_build", id="delete"),
    ],
)
def test_build_command_resolves_id_by_explicit_spec_then_picker(
    workspace: Path,
    client: RemoteBuilder,
    monkeypatch: pytest.MonkeyPatch,
    tier: str,
    spec_id: str | None,
    options: tuple[str, ...],
    expected_id: str,
    command: str,
    fixed_options: tuple[str, ...],
    method: str,
) -> None:
    # Given
    write_spec(workspace, build_id=spec_id, models=[], nodes=[])
    if tier == "picker":
        monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
        monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
        monkeypatch.setattr("comfy_cli.ui.prompt_select", lambda *args, **kwargs: "picked-id")

    # When
    result = invoke_build(command, *fixed_options, *options, agentic=tier != "picker")

    # Then
    assert result.exit_code == 0, result.stderr
    selected = [call for call in client.calls if call["method"] == method]
    assert selected == [{"method": method, "id": expected_id}]


@pytest.mark.parametrize(
    ("tier", "spec_id", "options", "expected_id"),
    [
        pytest.param("explicit", "spec-id", ("--id", "explicit-id"), "explicit-id", id="explicit"),
        pytest.param("spec", "spec-id", (), "spec-id", id="spec"),
        pytest.param("picker", None, (), "picked-id", id="picker"),
    ],
)
@pytest.mark.parametrize(
    ("subcommand", "fixed_options", "resolution_method"),
    [
        pytest.param("create", ("--target", "linux/nvidia"), "create_release", id="create"),
        pytest.param("ls", (), "list_releases", id="ls"),
        pytest.param("show", (), "list_releases", id="show"),
        pytest.param("logs", ("--target", "linux/nvidia"), "list_releases", id="logs"),
        pytest.param("manifest", (), "list_releases", id="manifest"),
    ],
)
def test_release_command_resolves_id_by_explicit_spec_then_picker(
    workspace: Path,
    client: RemoteBuilder,
    monkeypatch: pytest.MonkeyPatch,
    tier: str,
    spec_id: str | None,
    options: tuple[str, ...],
    expected_id: str,
    subcommand: str,
    fixed_options: tuple[str, ...],
    resolution_method: str,
) -> None:
    # Given
    write_spec(workspace, build_id=spec_id, models=[], nodes=[])
    if tier == "picker":
        monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
        monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
        monkeypatch.setattr("comfy_cli.ui.prompt_select", lambda *args, **kwargs: "picked-id")

    # When
    result = invoke_build("release", subcommand, *fixed_options, *options, agentic=tier != "picker")

    # Then
    assert result.exit_code == 0, result.stderr
    resolved = [call for call in client.calls if call["method"] == resolution_method]
    assert len(resolved) == 1
    assert resolved[0]["id"] == expected_id
