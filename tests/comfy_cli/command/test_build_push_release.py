"""``comfy build push --release`` — targets are always explicit, never defaulted.

A target belongs to a release and spends build minutes, so every path here is
about refusing to invent one: a malformed spelling, an absent ``--target``, an
aborted picker and a failed push must all end with **zero** release cuts.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from build_push_support import (
    RecordingBuilder,
    envelope,
    invoke_push,
    make_workspace,
    write_spec,
)

from comfy_cli.caller import Caller
from comfy_cli.command import build
from comfy_cli.command.build_spec import JsonObject

SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas"

HUMAN = Caller(kind="user", agentic=False, source_env=None)

LINUX_NVIDIA = {"os": "linux", "gpu": "nvidia"}
LINUX_CPU = {"os": "linux", "gpu": "cpu"}


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return make_workspace(tmp_path / "install")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> RecordingBuilder:
    recorder = RecordingBuilder()
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: recorder)
    return recorder


@pytest.fixture
def refusing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """No builder at all: any HTTP-bearing step fails the test outright."""
    monkeypatch.setattr(build, "_builder_client", lambda *args, **kwargs: pytest.fail("constructed Builder client"))


def _calls(client: RecordingBuilder, method: str) -> list[JsonObject]:
    return [call for call in client.calls if call["method"] == method]


def _methods(client: RecordingBuilder) -> list[str]:
    return [str(call["method"]) for call in client.calls]


def _become_human(monkeypatch: pytest.MonkeyPatch) -> None:
    """A human at a TTY who has not opted out of prompts (interaction tenet 1)."""
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: HUMAN)
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)


# --- the happy path ----------------------------------------------------------


def test_release_cuts_one_release_carrying_every_target_after_the_push_lands(
    workspace: Path, client: RecordingBuilder
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    client.remote_revisions["build-1"] = "revision-0"

    # When
    result = invoke_push(workspace, "--release", "--target", "linux/nvidia", "--target", "linux/cpu")

    # Then
    assert result.exit_code == 0, result.stderr
    releases = _calls(client, "create_release")
    assert len(releases) == 1
    assert releases[0]["targets"] == [LINUX_NVIDIA, LINUX_CPU]
    assert releases[0]["id"] == "build-1"
    methods = _methods(client)
    assert methods.index("update_build") < methods.index("create_release")


def test_the_release_envelope_reports_the_release_and_matches_the_push_schema(
    workspace: Path, client: RecordingBuilder
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    client.remote_revisions["build-1"] = "revision-0"

    # When
    result = invoke_push(workspace, "--release", "--target", "linux/nvidia")

    # Then
    data = envelope(result)["data"]
    assert isinstance(data, dict)
    assert data["release"] == {
        "releaseId": "release-1",
        "statusUrl": "https://builder.test/v1/releases/release-1",
    }
    assert data["targets"] == [LINUX_NVIDIA]
    schema = json.loads((SCHEMAS_DIR / "build_push.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(data)


def test_plain_push_cuts_no_release(workspace: Path, client: RecordingBuilder) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    client.remote_revisions["build-1"] = "revision-0"

    # When
    result = invoke_push(workspace)

    # Then
    assert result.exit_code == 0, result.stderr
    assert _calls(client, "create_release") == []
    data = envelope(result)["data"]
    assert isinstance(data, dict)
    assert "release" not in data
    assert "targets" not in data


# --- target syntax -----------------------------------------------------------


@pytest.mark.parametrize("value", ["linux", "linux/", "/nvidia", "linux/nvidia/extra"])
@pytest.mark.usefixtures("refusing_client")
def test_a_malformed_target_names_the_expected_form_and_reaches_no_builder(workspace: Path, value: str) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--release", "--target", value)

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_missing_input"
    assert "<os>/<gpu>" in str(error["message"])
    details = error["details"]
    assert isinstance(details, dict)
    assert details["invalid"] == [value]


@pytest.mark.usefixtures("refusing_client")
def test_every_malformed_target_is_named_in_one_refusal(workspace: Path) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--release", "--target", "linux", "--target", "linux/nvidia", "--target", "/amd")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    details = error["details"]
    assert isinstance(details, dict)
    assert details["invalid"] == ["linux", "/amd"]


# --- a missing --target ------------------------------------------------------


def test_missing_target_refuses_an_agentic_caller_before_any_builder_call(
    workspace: Path, client: RecordingBuilder
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--release")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_missing_input"
    details = error["details"]
    assert isinstance(details, dict)
    assert details["missing"] == ["--target"]
    assert client.calls == []


def test_missing_target_prompts_a_human_from_the_build_targets_catalog(
    workspace: Path, client: RecordingBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    client.remote_revisions["build-1"] = "revision-0"
    _become_human(monkeypatch)
    offered: list[list[str]] = []

    def pick(prompt: str, choices: list[str]) -> list[str]:
        offered.append(list(choices))
        return ["linux/cpu"]

    monkeypatch.setattr("comfy_cli.ui.prompt_multi_select", pick)

    # When
    result = invoke_push(workspace, "--release", agentic=False)

    # Then
    assert result.exit_code == 0, result.stderr
    assert offered == [["linux/nvidia", "linux/cpu"]]
    assert len(_calls(client, "list_build_targets")) == 1
    releases = _calls(client, "create_release")
    assert len(releases) == 1
    assert releases[0]["targets"] == [LINUX_CPU]


def test_a_json_release_without_a_target_refuses_before_reaching_the_picker(
    workspace: Path, client: RecordingBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--json` caller is never prompted, so `--release` with no `--target`
    is a structured refusal rather than a picker it could not answer. The
    abandoned-picker fallthrough itself is pinned in `test_interaction.py`."""
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    _become_human(monkeypatch)
    monkeypatch.setattr(
        "comfy_cli.ui.prompt_multi_select",
        lambda prompt, choices: pytest.fail("prompted a --json caller"),
    )

    # When
    result = invoke_push(workspace, "--release")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_missing_input"
    assert _calls(client, "create_release") == []
    assert _calls(client, "update_build") == []


def test_a_human_abandoning_the_target_picker_refuses_instead_of_defaulting(
    workspace: Path, client: RecordingBuilder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])
    _become_human(monkeypatch)
    offered: list[list[str]] = []

    def abandon(prompt: str, choices: list[str]) -> list[str]:
        offered.append(list(choices))
        return []

    monkeypatch.setattr("comfy_cli.ui.prompt_multi_select", abandon)

    # When
    result = invoke_push(workspace, "--release", agentic=False)

    # Then
    assert result.exit_code != 0
    assert offered, "the picker was never reached"
    assert "build_missing_input" in result.stdout
    assert _calls(client, "create_release") == []
    assert _calls(client, "update_build") == []


# --- sequencing and flag gating ----------------------------------------------


def test_a_stale_push_cuts_no_release(workspace: Path, client: RecordingBuilder) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-stale", models=[], nodes=[])
    client.remote_revisions["build-1"] = "revision-remote"

    # When
    result = invoke_push(workspace, "--release", "--target", "linux/nvidia")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_spec_stale"
    assert len(_calls(client, "update_build")) == 1
    assert _calls(client, "create_release") == []


@pytest.mark.usefixtures("refusing_client")
def test_target_without_release_is_refused_rather_than_ignored(workspace: Path) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--target", "linux/nvidia")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_missing_input"
    details = error["details"]
    assert isinstance(details, dict)
    assert details["missing"] == ["--release"]


@pytest.mark.usefixtures("refusing_client")
def test_release_with_dry_run_is_refused_rather_than_silently_skipped(workspace: Path) -> None:
    # Given
    write_spec(workspace, build_id="build-1", revision="revision-0", models=[], nodes=[])

    # When
    result = invoke_push(workspace, "--release", "--target", "linux/nvidia", "--dry-run")

    # Then
    assert result.exit_code != 0
    error = envelope(result)["error"]
    assert isinstance(error, dict)
    assert error["code"] == "build_missing_input"
    details = error["details"]
    assert isinstance(details, dict)
    assert details["conflict"] == ["--release", "--dry-run"]
