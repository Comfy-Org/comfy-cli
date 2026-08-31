"""The binding per-command auth contract for the whole `comfy build` tree.

One row per command (and per auth-relevant flag variant): signed out it must
refuse with ``build_not_signed_in`` before touching the Builder, and signed in it
must reach the Builder exactly as often as the contract says. The tree walk at
the bottom is what makes the matrix *complete* — a new build command with no row
fails there rather than shipping unproven.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from build_auth_support import BUILD_ID, RecordingTransport, write_snapshot
from build_push_support import make_workspace, write_spec
from build_tree_support import leaf_commands
from typer.testing import CliRunner
from typing_extensions import assert_never

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build


class BuilderCallExpectation(Enum):
    EXACTLY_ZERO = "exactly 0"
    AT_LEAST_ONE = "at least 1"
    AT_LEAST_ZERO = "at least 0"


class FixtureKind(Enum):
    INIT = "init"
    UPDATE = "update"
    VALIDATE = "validate"
    INIT_SNAPSHOT = "init --from-snapshot"
    UPDATE_SNAPSHOT = "update --from-snapshot"
    VALIDATE_REMOTE_ZERO = "validate --remote (zero eligible)"
    VALIDATE_REMOTE_LOOKUP = "validate --remote (lookupable)"
    PUSH = "push"
    PUSH_DRY_RUN = "push --dry-run"
    PULL = "pull"
    STATUS = "status"
    LS = "ls"
    SHOW = "show"
    DELETE = "delete"
    RELEASE_CREATE = "release create"
    RELEASE_LS = "release ls"
    RELEASE_SHOW = "release show"
    RELEASE_LOGS = "release logs"
    RELEASE_MANIFEST = "release manifest"
    REFS_RESOLVE = "refs resolve"
    REFS_BASE_IMAGES = "refs base-images"
    REFS_BUILD_TARGETS = "refs build-targets"
    REFS_MODEL_DIRS = "refs model-dirs"
    BLOB_LS = "blob ls"


@dataclass(frozen=True, slots=True)
class BuildAuthCase:
    fixture: FixtureKind
    command: str
    requires_session: bool
    calls: BuilderCallExpectation
    exercised_calls: int | None = None


_ZERO = BuilderCallExpectation.EXACTLY_ZERO
_ONE_PLUS = BuilderCallExpectation.AT_LEAST_ONE
_ZERO_PLUS = BuilderCallExpectation.AT_LEAST_ZERO

BUILD_AUTH_CASES = (
    BuildAuthCase(FixtureKind.INIT, "init", False, _ZERO),
    BuildAuthCase(FixtureKind.UPDATE, "update", False, _ZERO),
    BuildAuthCase(FixtureKind.VALIDATE, "validate", False, _ZERO),
    BuildAuthCase(FixtureKind.INIT_SNAPSHOT, "init", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.UPDATE_SNAPSHOT, "update", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.VALIDATE_REMOTE_ZERO, "validate", True, _ZERO_PLUS, 0),
    BuildAuthCase(FixtureKind.VALIDATE_REMOTE_LOOKUP, "validate", True, _ZERO_PLUS, 1),
    BuildAuthCase(FixtureKind.PUSH, "push", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.PUSH_DRY_RUN, "push", False, _ZERO),
    BuildAuthCase(FixtureKind.PULL, "pull", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.STATUS, "status", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.LS, "ls", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.SHOW, "show", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.DELETE, "delete", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.RELEASE_CREATE, "release create", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.RELEASE_LS, "release ls", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.RELEASE_SHOW, "release show", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.RELEASE_LOGS, "release logs", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.RELEASE_MANIFEST, "release manifest", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.REFS_RESOLVE, "refs resolve", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.REFS_BASE_IMAGES, "refs base-images", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.REFS_BUILD_TARGETS, "refs build-targets", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.REFS_MODEL_DIRS, "refs model-dirs", True, _ONE_PLUS),
    BuildAuthCase(FixtureKind.BLOB_LS, "blob ls", True, _ONE_PLUS),
)


@pytest.fixture(autouse=True)
def stable_command_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build,
        "capture_pip_provenance",
        lambda python: {
            "pipDependencies": "example==1.0.0\n",
            "environment": {"os": "Linux", "arch": "x86_64", "pythonVersion": "3.12.0", "torch": None},
        },
    )


def _scan_options() -> list[str]:
    """Pin the scan's two ambient inputs so no case reaches the network."""
    return ["--python", sys.executable, "--comfy-version", "0.3"]


def _prepare(kind: FixtureKind, root: Path) -> list[str]:
    make_workspace(root)
    match kind:
        case FixtureKind.INIT:
            return ["build", "init", "--name", "Matrix", *_scan_options(), str(root)]
        case FixtureKind.UPDATE:
            write_spec(root)
            return ["build", "update", "-y", *_scan_options(), str(root)]
        case FixtureKind.VALIDATE:
            write_spec(root)
            return ["build", "validate", str(root)]
        case FixtureKind.INIT_SNAPSHOT:
            return ["build", "init", "--name", "Matrix", "--from-snapshot", str(write_snapshot(root)), str(root)]
        case FixtureKind.UPDATE_SNAPSHOT:
            write_spec(root)
            return ["build", "update", "-y", "--from-snapshot", str(write_snapshot(root)), str(root)]
        case FixtureKind.VALIDATE_REMOTE_ZERO:
            write_spec(root, models=[{"type": "checkpoints", "blobId": "blob-private"}])
            return ["build", "validate", "--remote", str(root)]
        case FixtureKind.VALIDATE_REMOTE_LOOKUP:
            write_spec(
                root, models=[{"type": "checkpoints", "filename": "local-only.safetensors", "blobId": "blob-private"}]
            )
            return ["build", "validate", "--remote", str(root)]
        case FixtureKind.PUSH:
            write_spec(root, models=[], nodes=[])
            return ["build", "push", str(root)]
        case FixtureKind.PUSH_DRY_RUN:
            write_spec(root, models=[], nodes=[])
            return ["build", "push", "--dry-run", str(root)]
        case FixtureKind.PULL:
            write_spec(root, models=[], nodes=[])
            return ["build", "pull", "-y", "--id", BUILD_ID, str(root)]
        case FixtureKind.STATUS:
            write_spec(root, models=[], nodes=[])
            return ["build", "status", "--id", BUILD_ID, *_scan_options(), str(root)]
        case FixtureKind.LS:
            return ["build", "ls"]
        case FixtureKind.SHOW:
            return ["build", "show", "--id", BUILD_ID]
        case FixtureKind.DELETE:
            return ["build", "delete", "-y", "--id", BUILD_ID]
        case FixtureKind.RELEASE_CREATE:
            return ["build", "release", "create", "--id", BUILD_ID, "--target", "linux/nvidia"]
        case FixtureKind.RELEASE_LS:
            return ["build", "release", "ls", "--id", BUILD_ID]
        case FixtureKind.RELEASE_SHOW:
            return ["build", "release", "show", "--id", BUILD_ID]
        case FixtureKind.RELEASE_LOGS:
            return ["build", "release", "logs", "--id", BUILD_ID, "--target", "linux/nvidia"]
        case FixtureKind.RELEASE_MANIFEST:
            return ["build", "release", "manifest", "--id", BUILD_ID]
        case FixtureKind.REFS_RESOLVE:
            return ["build", "refs", "resolve", "base.safetensors"]
        case FixtureKind.REFS_BASE_IMAGES:
            return ["build", "refs", "base-images"]
        case FixtureKind.REFS_BUILD_TARGETS:
            return ["build", "refs", "build-targets"]
        case FixtureKind.REFS_MODEL_DIRS:
            return ["build", "refs", "model-dirs"]
        case FixtureKind.BLOB_LS:
            return ["build", "blob", "ls"]
        case unreachable:
            assert_never(unreachable)


def _invoke(args: list[str], token: str | None):
    return CliRunner(mix_stderr=False).invoke(
        cli_app,
        args,
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": "json",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": token,
            "COMFY_BUILDER_URL": "https://builder.test",
        },
    )


def _error_code(result) -> str:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])["error"]["code"]


def _assert_call_expectation(expectation: BuilderCallExpectation, count: int) -> None:
    match expectation:
        case BuilderCallExpectation.EXACTLY_ZERO:
            assert count == 0
        case BuilderCallExpectation.AT_LEAST_ONE:
            assert count >= 1
        case BuilderCallExpectation.AT_LEAST_ZERO:
            assert count >= 0
        case unreachable:
            assert_never(unreachable)


@pytest.mark.parametrize("case", BUILD_AUTH_CASES, ids=lambda case: case.fixture.value)
def test_build_auth_contract(case: BuildAuthCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    recorder = RecordingTransport()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", recorder)

    if case.requires_session:
        # When
        signed_out = _invoke(_prepare(case.fixture, tmp_path / "signed-out"), None)

        # Then
        assert signed_out.exit_code == 1
        assert _error_code(signed_out) == "build_not_signed_in"
        assert recorder.calls == []

    # When
    result = _invoke(_prepare(case.fixture, tmp_path / "signed-in"), "tok_test" if case.requires_session else None)

    # Then
    assert result.exit_code == 0, result.stderr
    _assert_call_expectation(case.calls, len(recorder.calls))
    if case.exercised_calls is not None:
        assert len(recorder.calls) == case.exercised_calls


def test_auth_matrix_covers_every_build_command() -> None:
    """The matrix is only binding if it is complete.

    Equality (not containment) in both directions: a new build command with no
    row fails here, and so does a row naming a command that no longer exists.
    """
    # Given / When
    covered = {case.command for case in BUILD_AUTH_CASES}

    # Then
    assert covered == leaf_commands()
