from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject


class ReferenceBuilder:
    def list_base_images(self) -> list[JsonObject]:
        return [{"id": "cuda"}]

    def list_build_targets(self) -> list[JsonObject]:
        return [{"os": "linux", "gpu": "nvidia"}]

    def list_model_directories(self) -> list[str]:
        return ["checkpoints", "vae"]

    def resolve_models(self, filenames: list[str]) -> list[JsonObject]:
        return [{"filename": filename, "candidates": []} for filename in filenames]

    def list_blobs(self, kind: str | None = None) -> list[JsonObject]:
        return [{"blobId": "blob-1", "kind": kind or "model"}]


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


def invoke_build(*args: str):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        ["build", *args],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )


def payload(result) -> JsonObject:
    body = result.stdout.strip().splitlines()
    assert len(body) == 1, result.stdout
    envelope = json.loads(body[0])
    data = envelope["data"]
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("command", "args", "expected_key"),
    [
        pytest.param("base-images", (), "baseImages", id="base-images"),
        pytest.param("build-targets", (), "targets", id="build-targets"),
        pytest.param("model-dirs", (), "directories", id="model-dirs"),
        pytest.param("resolve", ("model.safetensors",), "results", id="resolve"),
    ],
)
def test_refs_command_responds_under_the_refs_group(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    args: tuple[str, ...],
    expected_key: str,
) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: ReferenceBuilder())

    # When
    result = invoke_build("refs", command, *args)

    # Then
    assert result.exit_code == 0, result.stderr
    assert expected_key in payload(result)


def test_refs_group_contains_only_the_reference_catalog_commands() -> None:
    # Given
    command = typer.main.get_command(build.app)

    # When
    refs = command.commands["refs"]

    # Then
    assert sorted(refs.commands) == ["base-images", "build-targets", "model-dirs", "resolve"]


def test_hidden_blob_ls_remains_invocable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: ReferenceBuilder())

    # When
    result = invoke_build("blob", "ls")

    # Then
    assert result.exit_code == 0, result.stderr
    assert payload(result)["blobs"] == [{"blobId": "blob-1", "kind": "model"}]


def test_blob_upload_command_is_gone() -> None:
    # Given
    command = typer.main.get_command(build.app)

    # When
    blob_commands = set(command.commands["blob"].commands)

    # Then
    assert blob_commands == {"ls"}


def test_blob_client_methods_survive_for_push() -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    # When
    methods = (BuilderClient.create_blob, BuilderClient.upload_blob)

    # Then
    assert all(callable(method) for method in methods)


def test_build_ls_follows_all_cursor_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    calls: list[str] = []
    pages = {
        None: {"builds": [{"id": "build-1"}], "nextCursor": "page-2"},
        "page-2": {"builds": [{"id": "build-2"}], "nextCursor": "page-3"},
        "page-3": {"builds": [{"id": "build-3", "name": "Third"}]},
    }

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        calls.append(url)
        cursor = parse_qs(urlsplit(url).query).get("cursor", [None])[0]
        return 200, pages[cursor]

    client = BuilderClient("https://builder.test", "token")
    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: client)

    # When
    result = invoke_build("ls")

    # Then
    assert result.exit_code == 0, result.stderr
    builds = payload(result)["builds"]
    assert isinstance(builds, list)
    assert [item["id"] for item in builds if isinstance(item, dict)] == ["build-1", "build-2", "build-3"]
    assert [parse_qs(urlsplit(url).query).get("cursor", [None])[0] for url in calls] == [None, "page-2", "page-3"]


def test_todo_19_command_paths_are_atomically_registered() -> None:
    # Given
    from comfy_cli.discovery import COMMAND_SCHEMAS

    schemas_dir = Path(__file__).parents[3] / "comfy_cli" / "schemas"
    expected = {
        "comfy build blob ls": "build_blob_ls",
        "comfy build ls": "build_ls",
        "comfy build refs base-images": "build_refs_base_images",
        "comfy build refs build-targets": "build_refs_build_targets",
        "comfy build refs model-dirs": "build_refs_model_dirs",
        "comfy build refs resolve": "build_refs_resolve",
        "comfy build show": "build_show",
    }

    # When
    registered = {command: COMMAND_SCHEMAS.get(command) for command in expected}

    # Then
    assert registered == expected
    for command, schema_name in expected.items():
        schema = json.loads((schemas_dir / f"{schema_name}.json").read_text(encoding="utf-8"))
        assert schema["$id"] == f"https://comfy.org/schemas/{schema_name}.json"
        assert schema["title"] == f"{command} --json data payload"

    retired = {
        "comfy build base-images",
        "comfy build blob list",
        "comfy build blob upload",
        "comfy build build-targets",
        "comfy build get",
        "comfy build list",
        "comfy build model-dirs",
        "comfy build resolve",
    }
    assert retired.isdisjoint(COMMAND_SCHEMAS)
