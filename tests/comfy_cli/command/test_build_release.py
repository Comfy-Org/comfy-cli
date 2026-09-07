from __future__ import annotations

import io
import json
import urllib.error
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

#: The builder's own refusal prose for a workspace holding every release its
#: limit allows. Carried verbatim so the envelope's message is the one a user
#: would be shown.
AT_THE_LIMIT = (
    "this workspace already holds 20 releases, which is its limit; delete a release, or delete a build to "
    "give up every release it holds, and cut again"
)


def refusal(status: int, body: dict | bytes, url: str) -> urllib.error.HTTPError:
    """The builder's `{error, message}` body, raised the way the HTTP layer raises it.

    Bytes are sent through untouched, for the bodies a hostile endpoint can send
    that `json.dumps` cannot produce."""
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError(url, status, "Conflict", {}, io.BytesIO(raw))


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


def invoke_build(*args: str):
    """`comfy build ...` — the refusal table below is shared with `build delete`,
    which is not a release verb."""
    return CliRunner().invoke(
        cli_app,
        ["build", *args],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1", "COMFY_BUILDER_TOKEN": None},
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
    assert set(release.commands) == {"create", "ls", "show", "logs", "manifest", "delete"}
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


def test_release_limit_refusal_gets_its_own_code(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A workspace at its release limit is a state the caller can clear itself, so
    it reaches an agent as a code it can branch on rather than folded into the one
    envelope every builder failure shared."""
    # Given
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(409, {"error": "RELEASE_LIMIT", "message": AT_THE_LIMIT}, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("create", "--target", "linux/nvidia")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    # `details.buildId` is what the registered remediation sends the agent back
    # to, so it is part of the code's contract and not decoration.
    assert (error["code"], error["details"]["buildId"]) == ("build_release_limit", "build-1")


#: Every deployment the builder found blocking, named in one sentence. Long on
#: purpose: this is the shape that made the old envelope useless, because the ids
#: sat in a body truncated at 1000 bytes.
BLOCKED_BY = (
    "these deployments still reference this release: "
    + ", ".join(f"dep-{index:02d}a4f7c1e9b3" for index in range(80))
    + "; each stops blocking once it has been deleted and its teardown has released its compute"
)

#: The last id the builder named, and the first thing a capped body loses.
LAST_BLOCKER = "dep-79a4f7c1e9b3"


def delete_refused(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
    """A `RELEASE_IN_USE` body whose long `message` precedes its `error`, so the
    capped copy in `details.body` no longer contains the code at all."""
    raise refusal(409, {"message": BLOCKED_BY, "error": "RELEASE_IN_USE"}, url)


@pytest.mark.parametrize(
    "builder_error, expected_code",
    [
        pytest.param("RELEASE_LIMIT", "build_release_limit", id="release-limit"),
        pytest.param("RELEASE_IN_USE", "build_release_in_use", id="release-in-use"),
        pytest.param("BUILD_IN_USE", "build_in_use", id="build-in-use"),
        pytest.param("STALE", "build_builder_error", id="unmapped-409-is-unchanged"),
    ],
)
def test_each_mapped_refusal_reaches_the_agent_under_its_own_code(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, builder_error: str, expected_code: str
) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(409, {"error": builder_error, "message": AT_THE_LIMIT}, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("create", "--target", "linux/nvidia")

    # Then
    assert envelope(result)["error"]["code"] == expected_code


@pytest.mark.parametrize(
    "status",
    [pytest.param(400, id="bad-request"), pytest.param(500, id="server-error")],
)
def test_a_mapped_code_under_another_status_keeps_the_generic_envelope(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """All three refusals are 409 in the builder's contract, so a mapped code
    arriving under any other status came from something that is not the builder
    -- a proxy, a WAF page, a service that changed -- and must not hand an agent
    remediation for a limit that may not exist."""
    # Given
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(status, {"error": "RELEASE_IN_USE", "message": BLOCKED_BY}, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("create", "--target", "linux/nvidia")

    # Then
    error = envelope(result)["error"]
    assert (error["code"], error["details"]["status"]) == ("build_builder_error", status)


def test_the_blocking_deployment_ids_survive_the_body_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ids are the whole reason an agent reads this refusal, and the builder
    puts them in prose. Matching a substring of `details.body` would have found
    nothing here, and reading the ids back out of it would have found only some."""
    # Given
    from comfy_cli.builder_api import BuilderClient

    monkeypatch.setattr("comfy_cli.builder_api.request_json", delete_refused)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    error = envelope(result)["error"]
    assert (
        error["code"],
        error["message"],
        error["details"]["releaseId"],
        LAST_BLOCKER in error["details"]["body"],
        "RELEASE_IN_USE" in error["details"]["body"],
    ) == ("build_release_in_use", BLOCKED_BY, "release-9", False, False)


#: The builder's `BUILD_IN_USE` prose, naming what blocks a build delete.
BLOCKS_THE_BUILD = (
    "these deployments still reference releases of this build: dep-01a4f7c1e9b3, dep-02a4f7c1e9b3; each stops "
    "blocking once it has been deleted and its teardown has released its compute"
)


def test_a_refused_build_delete_names_the_build_and_the_deployments(monkeypatch: pytest.MonkeyPatch) -> None:
    """`comfy build delete` is the only call the builder answers `BUILD_IN_USE`
    on, so it is the only place the code and its `details.buildId` are real."""
    # Given
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(409, {"error": "BUILD_IN_USE", "message": BLOCKS_THE_BUILD}, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_build("delete", "--id", "build-1", "--yes")

    # Then
    error = envelope(result)["error"]
    assert (result.exit_code, error["code"], error["message"], error["details"]["buildId"]) == (
        1,
        "build_in_use",
        BLOCKS_THE_BUILD,
        "build-1",
    )


def test_delete_exits_non_zero_when_a_deployment_still_references_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    monkeypatch.setattr("comfy_cli.builder_api.request_json", delete_refused)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "already_deleted",
    [pytest.param(False, id="deleted-now"), pytest.param(True, id="deleted-before")],
)
def test_delete_reports_the_same_payload_whether_or_not_the_release_was_already_gone(
    monkeypatch: pytest.MonkeyPatch, already_deleted: bool
) -> None:
    """The builder answers 204 both times -- delete is idempotent -- so a retry
    after a dropped connection must not read as a different outcome."""
    # Given
    import jsonschema

    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        return 204, None

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    if already_deleted:
        invoke_release("delete", "release-9", "--yes")
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    assert result.exit_code == 0, result.stderr
    data = envelope(result)["data"]
    assert data == {"releaseId": "release-9", "deleted": True}
    schemas_dir = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"
    schema = json.loads((schemas_dir / "build_release_delete.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def test_deleting_a_release_the_workspace_does_not_have_stays_a_builder_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 is not a refusal the caller clears by deleting something, so it keeps
    the generic code rather than joining the table."""
    # Given
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(404, {"error": "NOT_FOUND", "message": "release not found"}, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    error = envelope(result)["error"]
    assert (error["code"], error["details"]["status"]) == ("build_builder_error", 404)


def test_delete_reaches_the_release_route_as_a_DELETE(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    from comfy_cli.builder_api import BuilderClient

    calls: list[tuple[str, str]] = []

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        calls.append((method, url))
        return 204, None

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )

    # When
    invoke_release("delete", "release-9", "--yes")

    # Then
    assert calls == [("DELETE", "https://builder.test/v1/releases/release-9")]


def test_delete_without_yes_refuses_an_agent_before_the_builder(
    client: ReleaseBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete is the one release verb that destroys something, so an agent that
    cannot answer a prompt is refused rather than having the prompt skipped."""
    # Given
    deleted: list[str] = []
    monkeypatch.setattr(client, "delete_release", deleted.append, raising=False)

    # When
    result = invoke_release("delete", "release-9")

    # Then
    error = envelope(result)["error"]
    assert (error["code"], error["details"]["releaseId"], deleted) == (
        "build_release_delete_needs_confirm",
        "release-9",
        [],
    )


def _refusing_builder(monkeypatch: pytest.MonkeyPatch, status: int, refused_body: dict | bytes) -> None:
    """A live `BuilderClient` whose transport answers every call with one refusal."""
    from comfy_cli.builder_api import BuilderClient

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        raise refusal(status, refused_body, url)

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )


def test_a_deeply_nested_error_body_still_reaches_the_agent_as_an_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """`json.loads` answers a nest this deep with `RecursionError`, a `RuntimeError`
    rather than a `ValueError`, so it escapes every clause between here and the CLI
    and leaves exit 1 with nothing on stdout -- on every builder HTTP error path,
    not just this one, so an agent cannot even tell whether the delete happened.
    20 000 is deterministic on this interpreter; 5 000 parses fine."""
    # Given
    _refusing_builder(monkeypatch, 409, b"[" * 20_000)

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    error = envelope(result)["error"]
    assert (error["code"], error["details"]["status"]) == ("build_builder_error", 409)


def test_a_lone_surrogate_in_the_builder_message_still_reaches_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone surrogate is legal in a JSON string and legal in a Python `str`, but
    encoding it raises `UnicodeEncodeError` -- a `ValueError`, which the JSON
    writer's own except swallows, so the whole envelope silently never lands."""
    # Given
    _refusing_builder(
        monkeypatch, 409, json.dumps({"error": "RELEASE_IN_USE", "message": "\ud800"}).encode("utf-8", "surrogatepass")
    )

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    error = envelope(result)["error"]
    assert (error["code"], error["details"]["releaseId"]) == ("build_release_in_use", "release-9")


def test_the_carried_message_cannot_forge_a_line_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The carried message is now the envelope's authoritative `message`, and the
    skill page sends an agent to act on the deployments it names. `json.dumps`
    escapes C0 but emits C1 raw, and the pretty path prints the message, so an
    erase-line plus a carriage return would let the builder's prose overwrite what
    the CLI already printed with a success that never happened."""
    # Given
    _refusing_builder(
        monkeypatch,
        409,
        {
            "error": "RELEASE_IN_USE",
            "message": "these deployments still reference this release: dep-01a4f7c1e9b3, dep-02a4f7c1e9b3"
            "\x1b[2K\rDeleted release release-9",
        },
    )

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    assert envelope(result)["error"]["message"] == (
        "these deployments still reference this release: dep-01a4f7c1e9b3, dep-02a4f7c1e9b3Deleted release release-9"
    )


def test_a_long_message_under_an_unmapped_status_obeys_the_message_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The block comment beside the caps says nothing an endpoint sends reaches the
    envelope unbounded, but only the refusal branch three lines below it applied
    the message cap: a 400 carrying a 60 000-character message produced an
    `error.message` detail of 60 006 beside a `details.body` of exactly 1 000."""
    # Given
    _refusing_builder(monkeypatch, 400, {"error": "NOPE", "message": "x" * 60_000})

    # When
    result = invoke_release("delete", "release-9", "--yes")

    # Then
    message = envelope(result)["error"]["message"]
    assert len(message.removeprefix("builder call failed (400): ")) == build._BUILDER_MESSAGE_CAP


def _recording_builder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A live `BuilderClient` whose transport records the URLs it is handed."""
    from comfy_cli.builder_api import BuilderClient

    calls: list[str] = []

    def request_json(url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        calls.append(url)
        return 204, None

    monkeypatch.setattr("comfy_cli.builder_api.request_json", request_json)
    monkeypatch.setattr(
        build, "_builder_client", lambda renderer, builder_url: BuilderClient("https://builder.test", "token")
    )
    return calls


def test_delete_keeps_a_traversing_release_id_inside_one_path_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Target.url` only strips slashes, so an unencoded id would send this DELETE
    -- through any normalizing proxy -- at a resource the prompt never named."""
    # Given
    calls = _recording_builder(monkeypatch)

    # When
    invoke_release("delete", "../builds/abc", "--yes")

    # Then
    assert calls == ["https://builder.test/v1/releases/..%2Fbuilds%2Fabc"]


@pytest.mark.parametrize(
    "release_id",
    [
        pytest.param("   ", id="blank"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="dot-dot"),
    ],
)
def test_delete_refuses_an_id_that_names_no_release_before_it_reaches_the_builder(
    monkeypatch: pytest.MonkeyPatch, release_id: str
) -> None:
    """An empty part drops out of the path entirely, and `quote(safe="")` leaves a
    dot segment alone because RFC 3986 calls it unreserved -- so all three aim the
    DELETE at the collection or at its parent rather than at a release. `.` is a
    plausible argument because every other `comfy build` verb takes a path
    defaulting to it. The empty `calls` is the load-bearing half: nothing reached
    the wire, rather than something reaching it encoded."""
    # Given
    calls = _recording_builder(monkeypatch)

    # When
    result = invoke_release("delete", release_id, "--yes")

    # Then
    assert (result.exit_code, envelope(result)["error"]["code"], calls) == (1, "build_missing_input", [])


def test_delete_uses_one_stripped_release_id_for_the_prompt_the_url_and_the_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A padded id otherwise splits into three different strings: the prompt echoes
    the padding, the client strips it for the URL, and the payload reports the
    padding back -- so the id confirmed is not the id deleted is not the id
    reported."""
    # Given
    calls = _recording_builder(monkeypatch)

    # When
    refused = invoke_release("delete", "  release-9  ")
    deleted = invoke_release("delete", "  release-9  ", "--yes")

    # Then
    refusal_details = envelope(refused)["error"]["details"]
    assert (refusal_details["question"], refusal_details["releaseId"], calls, envelope(deleted)["data"]) == (
        "Delete release release-9?",
        "release-9",
        ["https://builder.test/v1/releases/release-9"],
        {"releaseId": "release-9", "deleted": True},
    )
