from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import typer
from build_push_support import envelope, make_workspace, write_spec
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject


class ReleaseBuilder:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.releases: list[JsonObject] = []
        self.statuses: list[JsonObject] = []

    def list_build_targets(self) -> list[JsonObject]:
        self.calls.append({"method": "list_build_targets"})
        return [{"target": {"os": "linux", "gpu": "nvidia"}}]

    def create_release(self, build_id: str, targets: list[JsonObject] | None = None) -> tuple[str, str]:
        if not targets:
            raise ValueError("create_release requires a non-empty list of targets")
        self.calls.append({"method": "create_release", "id": build_id, "targets": targets})
        return "release-created", "https://builder.test/v1/releases/release-created"

    def list_releases(self, build_id: str) -> list[JsonObject]:
        self.calls.append({"method": "list_releases", "id": build_id})
        return self.releases

    def get_release(self, release_id: str) -> JsonObject:
        self.calls.append({"method": "get_release", "id": release_id})
        return self.statuses.pop(0)

    def get_release_logs(self, release_id: str, *, os: str, gpu: str) -> JsonObject:
        self.calls.append({"method": "get_release_logs", "id": release_id, "os": os, "gpu": gpu})
        return {"versionId": release_id, "os": os, "gpu": gpu, "log": "built", "truncated": False}

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
    write_spec(root, build_id="build-1", revision="revision-1", models=[], nodes=[])
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ReleaseBuilder:
    recorder = ReleaseBuilder()
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: recorder)
    return recorder


def invoke_release(*args: str, agentic: bool = True):
    return CliRunner().invoke(
        cli_app,
        ["build", "release", *args],
        env={
            "AI_AGENT": "1" if agentic else None,
            "COMFY_OUTPUT": "json" if agentic else "pretty",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": None,
        },
    )


def test_release_surface_replaces_version_and_uses_one_target_spelling() -> None:
    # Given
    command = typer.main.get_command(build.app)

    # When
    release = command.commands["release"]
    logs = release.commands["logs"]
    options = {name for parameter in logs.params for name in getattr(parameter, "opts", ())}

    # Then
    assert "version" not in command.commands
    assert set(release.commands) == {"create", "ls", "show", "logs", "manifest"}
    assert {"--target", "--follow", "-f"} <= options
    assert options.isdisjoint({"--os", "--gpu"})


def test_list_releases_follows_three_cursor_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    calls: list[str] = []
    pages = {
        None: {"versions": [{"id": "release-1", "version": 1}], "nextCursor": "page-2"},
        "page-2": {"versions": [{"id": "release-2", "version": 2}], "nextCursor": "page-3"},
        "page-3": {"versions": [{"id": "release-3", "version": 3}]},
    }

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        calls.append(url)
        cursor = parse_qs(urlsplit(url).query).get("cursor", [None])[0]
        return 200, pages[cursor]

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)

    # When
    releases = BuilderClient("https://builder.test", "token").list_releases("build-1")

    # Then
    assert [release["id"] for release in releases] == ["release-1", "release-2", "release-3"]
    assert len(calls) == 3
    assert [parse_qs(urlsplit(url).query).get("cursor", [None])[0] for url in calls] == [None, "page-2", "page-3"]


def test_omitted_release_selects_newest_from_page_three(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    requested: list[str] = []
    pages = {
        None: {"releases": [{"id": "release-1", "version": 1}], "nextCursor": "page-2"},
        "page-2": {"releases": [{"id": "release-2", "version": 2}], "nextCursor": "page-3"},
        "page-3": {"releases": [{"id": "release-9", "version": 9, "createdAt": "2026-08-23T00:00:00Z"}]},
    }

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        requested.append(url)
        # `/v1/releases/{id}` is the single-release read; the paged list lives at
        # `/v1/builds/{id}/releases`, so anchoring on `/v1/` keeps them apart.
        if "/v1/releases/" in url:
            return 200, {"id": "release-9", "version": 9, "status": "complete", "artifactCounts": {"failed": 0}}
        cursor = parse_qs(urlsplit(url).query).get("cursor", [None])[0]
        return 200, pages[cursor]

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("show")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert isinstance(data, dict)
    assert data["id"] == "release-9"
    assert requested[-1].endswith("/v1/releases/release-9")


def test_empty_release_list_uses_registered_not_found_error(workspace: Path, client: ReleaseBuilder) -> None:
    # Given / When
    result = invoke_release("show")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_release_not_found"


def test_create_without_target_refuses_agent_before_builder(workspace: Path, client: ReleaseBuilder) -> None:
    # Given / When
    result = invoke_release("create")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    details = error["details"]
    assert isinstance(details, dict)
    assert details["missing"] == ["--target"]
    assert client.calls == []


def test_create_without_target_prompts_human_from_catalog(
    workspace: Path, client: ReleaseBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    monkeypatch.setattr("comfy_cli.ui.prompt_multi_select", lambda prompt, choices: ["linux/nvidia"])

    # When
    result = invoke_release("create", agentic=False)

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.calls[-1] == {
        "method": "create_release",
        "id": "build-1",
        "targets": [{"os": "linux", "gpu": "nvidia"}],
    }


def test_logs_target_reaches_unchanged_query_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    calls: list[str] = []

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        calls.append(url)
        return 200, {"versionId": "release-1", "log": "built", "truncated": False}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("logs", "release-1", "--target", "linux/nvidia")

    # Then
    assert result.exit_code == 0, result.stderr
    assert calls == ["https://builder.test/v1/releases/release-1/logs?os=linux&gpu=nvidia"]


def test_logs_envelope_says_releaseid_even_when_the_builder_says_versionid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The emitted payload is the builder's own log body, and a server predating
    the version-to-release rename keys the id `versionId`. `build_release_logs`
    declares `releaseId` required and no longer declares `versionId` at all, so
    passing the body through untouched would publish a payload that fails this
    CLI's own shipped schema against exactly the servers the client still
    supports elsewhere."""
    # Given
    import json

    import jsonschema

    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        return 200, {"versionId": "release-1", "log": "built", "truncated": False}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("logs", "release-1", "--target", "linux/nvidia")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data["releaseId"] == "release-1"
    assert "versionId" not in data
    schemas_dir = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"
    schema = json.loads((schemas_dir / "build_release_logs.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def test_logs_short_follow_flag_polls_until_complete(client: ReleaseBuilder, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    client.statuses = [
        {"id": "release-1", "status": "building", "artifactCounts": {"failed": 0}},
        {"id": "release-1", "status": "complete", "artifactCounts": {"failed": 0}},
    ]
    monkeypatch.setattr(build, "_RELEASE_POLL_SECONDS", 0)

    # When
    result = invoke_release("logs", "release-1", "--target", "linux/nvidia", "-f")

    # Then
    assert result.exit_code == 0, result.stderr
    assert len([call for call in client.calls if call["method"] == "get_release_logs"]) == 2
    assert len([call for call in client.calls if call["method"] == "get_release"]) == 2


@pytest.mark.parametrize("failed, expected_exit", [(0, 0), (1, 1)])
def test_create_watch_polls_to_complete_and_reflects_target_failure(
    workspace: Path,
    client: ReleaseBuilder,
    monkeypatch: pytest.MonkeyPatch,
    failed: int,
    expected_exit: int,
) -> None:
    # Given
    client.statuses = [
        {"id": "release-created", "status": "queued", "artifactCounts": {"failed": 0}},
        {"id": "release-created", "status": "building", "artifactCounts": {"failed": 0}},
        {"id": "release-created", "status": "complete", "artifactCounts": {"failed": failed}},
    ]
    monkeypatch.setattr(build, "_RELEASE_POLL_SECONDS", 0, raising=False)

    # When
    result = invoke_release("create", "--target", "linux/nvidia", "--watch")

    # Then
    assert result.exit_code == expected_exit, result.stderr
    assert len([call for call in client.calls if call["method"] == "get_release"]) == 3


def test_same_version_releases_break_the_tie_on_the_instant_not_the_spelling() -> None:
    """Compared as text, the whole-second stamp wins this pair: ``.`` precedes
    ``Z``, so the strictly later fractional one sorts below it."""
    # Given
    whole_second = {"id": "release-earlier", "version": 1, "createdAt": "2026-08-28T03:26:09Z"}
    fractional = {"id": "release-later", "version": 1, "createdAt": "2026-08-28T03:26:09.5Z"}

    # When
    newest = max([whole_second, fractional], key=build._release_order)

    # Then
    assert newest["id"] == "release-later"


@pytest.mark.parametrize(
    "created_at",
    [
        pytest.param("2026-08-28T03:26:09.43745Z", id="zero-trimmed"),
        pytest.param(None, id="missing"),
        pytest.param("not-a-timestamp", id="unparsable"),
    ],
)
def test_a_release_order_key_stays_comparable_for_every_created_at(created_at: object) -> None:
    """The key is a sort key before it is anything else: one row the builder
    dated oddly must not make ``max`` raise across the whole list."""
    # Given
    dated = {"id": "release-dated", "version": 1, "createdAt": "2026-08-28T03:26:09.5Z"}
    odd = {"id": "release-odd", "version": 1, "createdAt": created_at}

    # When
    newest = max([odd, dated], key=build._release_order)

    # Then
    assert newest["id"] == "release-dated"
