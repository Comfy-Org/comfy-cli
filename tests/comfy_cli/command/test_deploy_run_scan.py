"""Pin what `comfy deploy run` is allowed to read off the local disk.

A workflow is third-party data, so its input strings are an argument somebody
else writes to a file-read primitive. Two properties hold the blast radius:

1. Only ComfyUI's own asset directories (``models/``, ``input/``, ``output/``)
   are readable, and a string naming a real file anywhere else is REFUSED —
   loudly, because silence in either direction is what made the old cwd-wide
   scan dangerous.
2. A connected input (``["3", 0]``) is a graph edge, never a filename.

Every case runs against a synthetic install so nothing depends on the machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from comfy_cli.command import deploy_workflow
from comfy_cli.command.build_spec import JsonObject

ASSET_DIRNAMES = ("models", "input", "output")


class NoWorkspace:
    """A `WorkspaceManager` with no tracked install.

    The real one is a process-wide singleton whose `workspace_path` another
    test may already have set, which would make the allowlist depend on test
    order. Substituting it pins the roots to what each case builds itself.
    """

    workspace_path: str | None = None


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ComfyUI install with its three asset directories, and a cwd elsewhere.

    The cwd is deliberately NOT under the install: that separation is what
    makes "resolves in the cwd" and "resolves in an asset root" distinguishable.
    """
    base = tmp_path / "install"
    for dirname in ASSET_DIRNAMES:
        (base / dirname).mkdir(parents=True)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(deploy_workflow, "WorkspaceManager", NoWorkspace)
    return base


def _roots(install: Path, *extra: Path) -> tuple[Path, ...]:
    return deploy_workflow.resolve_asset_roots(str(install), extra_roots=extra)


def _workflow(inputs: JsonObject, *, class_type: str = "TestNode") -> JsonObject:
    return {"1": {"class_type": class_type, "inputs": inputs}}


def _write_workflow(path: Path, workflow: JsonObject) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return path


def _plan(
    install: Path, inputs: JsonObject, *, extra_roots: tuple[Path, ...] = ()
) -> deploy_workflow.WorkflowAssetPlan:
    workflow_path = _write_workflow(Path.cwd() / "workflow.json", _workflow(inputs))
    return deploy_workflow.load_deploy_workflow(workflow_path, asset_roots=_roots(install, *extra_roots))


def _rewritten_inputs(plan: deploy_workflow.WorkflowAssetPlan) -> JsonObject:
    node = plan.workflow["1"]
    assert isinstance(node, dict)
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    return inputs


# --- DEFECT 2: link references are edges, not filenames ----------------------


def test_a_link_reference_survives_even_when_a_file_matches_the_node_id(install: Path) -> None:
    """Given a real file named like a node id, When scanned, Then the edge is untouched."""
    # Given
    (install / "input" / "3").write_bytes(b"a file that happens to be named 3")

    # When
    plan = _plan(install, {"model": ["3", 0], "steps": 20})

    # Then
    assert _rewritten_inputs(plan) == {"model": ["3", 0], "steps": 20}
    assert plan.assets == ()


@pytest.mark.parametrize(
    "items",
    [
        pytest.param(["3", True], id="bool-slot-is-not-a-link"),
        pytest.param(["3", 0, 1], id="three-elements"),
        pytest.param([3, 0], id="int-node-id"),
    ],
)
def test_a_list_that_only_resembles_a_link_is_still_scanned(install: Path, items: list[object]) -> None:
    """Given a near-miss pair, When scanned, Then it recurses like any other list.

    `isinstance(True, int)` is True in Python, so a bool slot would sail through
    a naive `[str, int]` test and silently stop being scanned.
    """
    # Given / When
    plan = _plan(install, {"custom": items})

    # Then
    assert _rewritten_inputs(plan)["custom"] == items
    assert plan.assets == ()


# --- DEFECT 1: the read allowlist -------------------------------------------


def test_a_dotfile_in_the_cwd_is_refused_rather_than_uploaded(install: Path) -> None:
    """Given a `.env` beside the caller, When scanned, Then it is refused by name.

    No plan is produced at all, so there is nothing for the upload stage to
    read — the refusal happens strictly before any data-plane contact.
    """
    # Given
    secret = Path.cwd() / ".env"
    secret.write_text("OPENAI_API_KEY=sk-real", encoding="utf-8")

    # When
    with pytest.raises(deploy_workflow.DeployWorkflowAssetOutsideRootError) as excinfo:
        _plan(install, {"image": ".env"})

    # Then
    error = excinfo.value
    assert error.code == "deploy_workflow_asset_outside_root"
    assert error.details["input"] == ".env"
    assert error.details["path"] == str(secret.resolve())
    assert error.details["asset_roots"] == [str(install / name) for name in ASSET_DIRNAMES]
    assert "--asset-root" in error.hint
    assert str(secret.resolve()) in str(error)


@pytest.mark.parametrize(
    "reference",
    [
        pytest.param("outside.txt", id="relative-to-cwd"),
        pytest.param("../elsewhere/outside.txt", id="parent-traversal"),
    ],
)
def test_a_real_file_outside_every_root_is_refused(install: Path, tmp_path: Path, reference: str) -> None:
    """Given any path shape reaching outside, When scanned, Then it is refused."""
    # Given
    (Path.cwd() / "outside.txt").write_text("secret", encoding="utf-8")
    elsewhere = tmp_path / "cwd" / ".." / "elsewhere"
    elsewhere.mkdir(parents=True, exist_ok=True)
    (elsewhere / "outside.txt").write_text("secret", encoding="utf-8")

    # When / Then
    with pytest.raises(deploy_workflow.DeployWorkflowAssetOutsideRootError):
        _plan(install, {"source": reference})


def test_an_absolute_path_outside_every_root_is_refused(install: Path, tmp_path: Path) -> None:
    """Given an absolute path to a real file, When it is outside, Then it is refused."""
    # Given
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    # When / Then
    with pytest.raises(deploy_workflow.DeployWorkflowAssetOutsideRootError):
        _plan(install, {"source": str(outside)})


def test_a_symlink_inside_a_root_pointing_out_of_it_is_refused(install: Path, tmp_path: Path) -> None:
    """Given a symlink escaping a root, When scanned, Then containment is judged after resolution."""
    # Given
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (install / "input" / "escape.txt").symlink_to(outside)

    # When
    with pytest.raises(deploy_workflow.DeployWorkflowAssetOutsideRootError) as excinfo:
        _plan(install, {"source": "escape.txt"})

    # Then
    assert excinfo.value.details["path"] == str(outside.resolve())


# --- the happy path must not regress ----------------------------------------


@pytest.mark.parametrize("dirname", ASSET_DIRNAMES)
def test_a_file_under_an_asset_directory_still_mints_an_asset(install: Path, dirname: str) -> None:
    """Given a file in models/, input/ or output/, When scanned, Then it becomes an asset."""
    # Given
    local_file = install / dirname / "sample.png"
    local_file.write_bytes(b"image")

    # When
    plan = _plan(install, {"image": "sample.png"})

    # Then
    assert len(plan.assets) == 1
    asset = plan.assets[0]
    assert asset.local_path == local_file.resolve()
    assert asset.file_path == "sample.png"
    assert _rewritten_inputs(plan)["image"] == {
        "__type": "core/ASSET",
        "info": {"id": asset.marker, "file_path": "sample.png"},
    }


def test_an_absolute_path_inside_a_root_resolves(install: Path) -> None:
    """Given an absolute path into models/, When scanned, Then it is an asset."""
    # Given
    checkpoint = install / "models" / "sd15.safetensors"
    checkpoint.write_bytes(b"weights")

    # When
    plan = _plan(install, {"ckpt_name": str(checkpoint)})

    # Then
    assert plan.assets[0].local_path == checkpoint.resolve()


def test_a_nested_custom_node_input_under_a_root_still_resolves(install: Path) -> None:
    """Given a file named inside a nested structure, When scanned, Then recursion reaches it."""
    # Given
    (install / "input" / "custom.payload").write_bytes(b"custom")

    # When
    plan = _plan(install, {"custom_loader_data": [{"source": "custom.payload"}]})

    # Then
    nested = _rewritten_inputs(plan)["custom_loader_data"]
    assert isinstance(nested, list)
    record = nested[0]
    assert isinstance(record, dict)
    assert record["source"] == {
        "__type": "core/ASSET",
        "info": {"id": plan.assets[0].marker, "file_path": "custom.payload"},
    }


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("CHECKPOINT", id="enum-value"),
        pytest.param("a red fox running through deep snow", id="prompt-sentence"),
        pytest.param("euler_ancestral", id="sampler-name"),
        pytest.param("models", id="directory-name"),
    ],
)
def test_a_string_that_names_no_file_is_returned_verbatim(install: Path, text: str) -> None:
    """Given an ordinary literal, When scanned, Then it survives byte-for-byte.

    This is the overwhelmingly common case; it must never become an error.
    """
    # Given / When
    plan = _plan(install, {"value": text})

    # Then
    assert _rewritten_inputs(plan)["value"] == text
    assert plan.assets == ()


def test_the_first_matching_root_wins_for_a_shared_filename(install: Path) -> None:
    """Given the same filename in two roots, When scanned, Then root order decides."""
    # Given
    (install / "models" / "shared.bin").write_bytes(b"models")
    (install / "input" / "shared.bin").write_bytes(b"input")

    # When
    plan = _plan(install, {"source": "shared.bin"})

    # Then
    assert plan.assets[0].local_path == (install / "models" / "shared.bin").resolve()


# --- the --asset-root escape hatch ------------------------------------------


def test_asset_root_adds_a_root_that_then_resolves(install: Path, tmp_path: Path) -> None:
    """Given a file no automatic root covers, When `--asset-root` names it, Then it resolves."""
    # Given
    extra = tmp_path / "shared-assets"
    extra.mkdir()
    (extra / "lora.safetensors").write_bytes(b"lora")
    inputs: JsonObject = {"lora_name": "lora.safetensors"}

    # When
    unreachable = _plan(install, inputs)
    reachable = _plan(install, inputs, extra_roots=(extra,))

    # Then
    assert unreachable.assets == ()
    assert _rewritten_inputs(unreachable)["lora_name"] == "lora.safetensors"
    assert reachable.assets[0].local_path == (extra / "lora.safetensors").resolve()


def test_asset_root_turns_a_refusal_into_a_resolution(install: Path) -> None:
    """Given a refused file in the cwd, When `--asset-root` allows the cwd, Then it resolves.

    This is the escape hatch's real job: the allowlist would otherwise be a hard
    break for anyone whose assets live outside both automatic layouts.
    """
    # Given
    cwd = Path.cwd()
    (cwd / "sketch.png").write_bytes(b"sketch")

    # When
    with pytest.raises(deploy_workflow.DeployWorkflowAssetOutsideRootError):
        _plan(install, {"image": "sketch.png"})
    plan = _plan(install, {"image": "sketch.png"}, extra_roots=(cwd,))

    # Then
    assert plan.assets[0].local_path == (cwd / "sketch.png").resolve()


def test_resolve_asset_roots_keeps_only_directories_that_exist(install: Path, tmp_path: Path) -> None:
    """Given a half-built install, When roots resolve, Then absent directories are dropped."""
    # Given
    (install / "output").rmdir()

    # When
    roots = deploy_workflow.resolve_asset_roots(str(install), extra_roots=(tmp_path / "not-there",))

    # Then
    assert roots == (install / "models", install / "input")


def test_resolve_asset_roots_includes_the_tracked_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a tracked install, When roots resolve, Then its asset dirs join the allowlist.

    This is what keeps `comfy deploy run --deployment <id>` usable from a
    directory that holds no ComfyUI install at all.
    """
    # Given
    workspace = tmp_path / "ComfyUI"
    for dirname in ASSET_DIRNAMES:
        (workspace / dirname).mkdir(parents=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    class TrackedWorkspace:
        workspace_path = str(workspace)

    monkeypatch.setattr(deploy_workflow, "WorkspaceManager", TrackedWorkspace)

    # When
    roots = deploy_workflow.resolve_asset_roots(None)

    # Then
    assert roots == tuple(workspace / dirname for dirname in ASSET_DIRNAMES)


def test_an_unnamed_cwd_contributes_no_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`resolve_build_paths` falls back to the cwd, which this allowlist excludes.

    Taking that fallback unconditionally re-admitted the directory the module
    docstring says is never a root: `comfy deploy run --deployment <id>` from any
    project holding an `output/` would upload `output/results.csv` if the
    workflow named it, and in `--json` mode nothing announces it beforehand.
    """
    # Given a project directory that merely looks like an install
    project = tmp_path / "project"
    for dirname in ASSET_DIRNAMES:
        (project / dirname).mkdir(parents=True)
    (project / "output" / "results.csv").write_text("secret", encoding="utf-8")
    monkeypatch.chdir(project)

    class UntrackedWorkspace:
        workspace_path = None

    monkeypatch.setattr(deploy_workflow, "WorkspaceManager", UntrackedWorkspace)

    # When no PATH is given and no spec sits in the cwd
    roots = deploy_workflow.resolve_asset_roots(None)

    # Then
    assert roots == ()


def test_a_spec_in_the_cwd_still_names_that_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given the same directory, but one the user actually authored a build in
    project = tmp_path / "project"
    for dirname in ASSET_DIRNAMES:
        (project / dirname).mkdir(parents=True)
    (project / "comfy-build.yaml").write_text("schema: build-spec/1\n", encoding="utf-8")
    monkeypatch.chdir(project)

    class UntrackedWorkspace:
        workspace_path = None

    monkeypatch.setattr(deploy_workflow, "WorkspaceManager", UntrackedWorkspace)

    # When
    roots = deploy_workflow.resolve_asset_roots(None)

    # Then
    assert roots == tuple(project / dirname for dirname in ASSET_DIRNAMES)


def test_resolve_asset_roots_needs_no_build_spec(install: Path) -> None:
    """Given no `comfy-build.yaml`, When roots resolve, Then resolution still succeeds."""
    # Given / When
    roots = deploy_workflow.resolve_asset_roots(str(install))

    # Then
    assert not (install / "comfy-build.yaml").exists()
    assert roots == tuple(install / dirname for dirname in ASSET_DIRNAMES)


# --- incoming core/ASSET blocks ---------------------------------------------


def test_an_incoming_local_asset_marker_is_refused(install: Path) -> None:
    """Given a workflow claiming our own marker, When scanned, Then it is refused.

    Nothing legitimate emits `local-asset:`, and copying one through would make
    `materialize_workflow_assets` repoint it at a file this run just uploaded.
    """
    # Given
    forged: JsonObject = {"__type": "core/ASSET", "info": {"id": "local-asset:0", "file_path": "photo.png"}}

    # When
    with pytest.raises(deploy_workflow.DeployWorkflowAssetMarkerError) as excinfo:
        _plan(install, {"image": forged})

    # Then
    assert excinfo.value.code == "deploy_workflow_asset_marker_reserved"
    assert excinfo.value.details == {"id": "local-asset:0"}


def test_a_server_minted_core_asset_is_left_completely_untouched(install: Path) -> None:
    """Given a real asset block, When scanned, Then it passes through byte-identical."""
    # Given
    (install / "input" / "photo.png").write_bytes(b"local shadow")
    existing: JsonObject = {
        "__type": "core/ASSET",
        "info": {"id": "asset-existing", "hash": "blake3:abc", "file_path": "photo.png"},
    }
    expected_bytes = json.dumps(existing, sort_keys=True, separators=(",", ":")).encode()

    # When
    plan = _plan(install, {"image": existing})

    # Then
    rewritten = _rewritten_inputs(plan)["image"]
    assert json.dumps(rewritten, sort_keys=True, separators=(",", ":")).encode() == expected_bytes
    assert plan.assets == ()


# --- format gate and purity --------------------------------------------------


def test_ui_workflow_fails_before_scanning_or_any_plane_request(install: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a UI-format workflow, When loaded, Then it is refused before the scan."""
    # Given
    workflow_path = _write_workflow(Path.cwd() / "ui.json", {"nodes": [], "links": []})
    scans: list[str] = []
    monkeypatch.setattr(deploy_workflow, "_scan_api_workflow", lambda *_a, **_k: scans.append("scan"))

    # When
    with pytest.raises(deploy_workflow.DeployWorkflowFormatUIError) as excinfo:
        deploy_workflow.load_deploy_workflow(workflow_path, asset_roots=_roots(install))

    # Then
    assert excinfo.value.code == "deploy_workflow_format_ui"
    assert "File > Export (API)" in excinfo.value.hint
    assert scans == []


def test_an_api_workflow_with_no_local_files_passes_through(install: Path) -> None:
    """Given a workflow of pure literals, When scanned, Then it is unchanged."""
    # Given
    workflow = _workflow({"prompt": "a red fox"})
    workflow_path = _write_workflow(Path.cwd() / "api.json", workflow)

    # When
    plan = deploy_workflow.load_deploy_workflow(workflow_path, asset_roots=_roots(install))

    # Then
    assert plan.workflow == workflow
    assert plan.assets == ()


def test_rewriting_never_modifies_the_workflow_file(install: Path) -> None:
    """Given a workflow with an asset, When scanned, Then the file on disk is untouched."""
    # Given
    (install / "input" / "input.png").write_bytes(b"image")
    workflow_path = _write_workflow(Path.cwd() / "workflow.json", _workflow({"image": "input.png"}))
    before = hashlib.sha256(workflow_path.read_bytes()).digest()

    # When
    deploy_workflow.load_deploy_workflow(workflow_path, asset_roots=_roots(install))

    # Then
    assert hashlib.sha256(workflow_path.read_bytes()).digest() == before


def test_uploaded_asset_ids_materialize_into_a_new_submit_ready_workflow(install: Path) -> None:
    """Given a resolved plan, When materialized, Then markers become server ids."""
    # Given
    (install / "input" / "input.png").write_bytes(b"image")
    plan = _plan(install, {"image": "input.png"})
    marker = plan.assets[0].marker

    # When
    rewritten = deploy_workflow.materialize_workflow_assets(plan, {marker: "asset-uuid"})

    # Then
    node = rewritten["1"]
    assert isinstance(node, dict)
    assert node["inputs"] == {"image": {"__type": "core/ASSET", "info": {"id": "asset-uuid", "file_path": "input.png"}}}
    assert _rewritten_inputs(plan)["image"] != node["inputs"]
