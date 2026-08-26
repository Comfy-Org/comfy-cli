"""Prepare API-format workflows for deployment job submission.

A workflow is third-party data. Scanning its node inputs for local filenames
and uploading whatever they name is therefore a read primitive somebody else
writes the argument to, and the only thing standing between it and the user's
private files is the set of directories it is allowed to read from.

That set is ComfyUI's own asset layout — ``models/``, ``input/``, ``output/``
— resolved from the install this command is already acting on, never "anywhere
under the cwd". The narrower rule is the whole containment mechanism: with the
cwd allowed, an input of ``.env`` uploaded the caller's secrets whenever the
CLI ran from a project root or ``$HOME``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from typing_extensions import assert_never

from comfy_cli.command.build_paths import resolve_build_paths
from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.command.run.loader import _load_workflow_file, is_ui_workflow
from comfy_cli.constants import (
    DEFAULT_COMFY_INPUT_PATH,
    DEFAULT_COMFY_MODEL_PATH,
    DEFAULT_COMFY_OUTPUT_PATH,
)
from comfy_cli.workspace_manager import WorkspaceManager

_ASSET_TYPE: Final = "core/ASSET"
_LOCAL_ASSET_MARKER_PREFIX: Final = "local-asset:"
_ASSET_DIRNAMES: Final = (DEFAULT_COMFY_MODEL_PATH, DEFAULT_COMFY_INPUT_PATH, DEFAULT_COMFY_OUTPUT_PATH)

#: Directories a workflow input may name a file inside. Every member is
#: absolute, symlink-resolved and known to be a directory — ``resolve_asset_roots``
#: is the only producer, so the scanner never re-probes them.
AssetRoots = tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class WorkflowAsset:
    marker: str
    local_path: Path
    file_path: str


@dataclass(frozen=True, slots=True)
class WorkflowAssetPlan:
    workflow: JsonObject
    assets: tuple[WorkflowAsset, ...]


class DeployWorkflowFormatUIError(Exception):
    code = "deploy_workflow_format_ui"
    hint = (
        "use ComfyUI's 'File > Export (API)' to save as API format, or convert locally with "
        "`comfy run` against a running ComfyUI instance"
    )

    def __init__(self) -> None:
        super().__init__("deploy run accepts API-format workflows only")


class DeployWorkflowAssetError(Exception):
    """Base for every refusal to carry a workflow's asset reference to the cloud.

    Carries the ``code``/``hint``/``details`` trio ``run_deploy`` renders through
    ``renderer.error``, so a single ``except`` clause covers the family and no
    member can reach the user as a traceback.
    """

    code: str
    hint: str

    def __init__(self, message: str, details: JsonObject) -> None:
        self.details = details
        super().__init__(message)


class DeployWorkflowAssetOutsideRootError(DeployWorkflowAssetError):
    """An input named a real local file no allowed asset root contains.

    Refusing is deliberate, and it is the *opposite* of the defect that
    motivated the allowlist. The caller plainly meant that file; passing the
    literal through instead would submit a job against a name the server cannot
    resolve and say nothing — the same silence that let ``.env`` leave the
    machine, only pointed the other way.
    """

    code = "deploy_workflow_asset_outside_root"
    hint = (
        "move the file under the install's models/, input/ or output/ directory, "
        "or pass `--asset-root <dir>` to allow the directory that holds it"
    )

    def __init__(self, value: str, resolved: Path, roots: AssetRoots) -> None:
        listed = ", ".join(str(root) for root in roots) if roots else "(no asset directory exists)"
        super().__init__(
            f"workflow input {value!r} resolves to {resolved}, outside every allowed asset root: {listed}",
            {"input": value, "path": str(resolved), "asset_roots": [str(root) for root in roots]},
        )


class DeployWorkflowAssetMarkerError(DeployWorkflowAssetError):
    """An incoming ``core/ASSET`` block already claims a ``local-asset:`` id.

    That prefix is minted here and consumed by ``materialize_workflow_assets``,
    so no legitimate producer emits one. Copying it through would let an
    authored workflow claim an id this run is about to bind to one of the
    caller's own uploaded files.
    """

    code = "deploy_workflow_asset_marker_reserved"
    hint = "remove the `local-asset:` asset id from the workflow and reference the local file by its path instead"

    def __init__(self, marker: str) -> None:
        super().__init__(
            f"workflow asset id {marker!r} uses the reserved {_LOCAL_ASSET_MARKER_PREFIX!r} prefix",
            {"id": marker},
        )


def _existing_dirs(candidates: Iterable[Path]) -> AssetRoots:
    """Keep the candidates that are directories on disk, resolved and deduplicated.

    Insertion order survives, so the caller's precedence decides which root
    wins when two of them hold the same bare filename.
    """
    kept: dict[Path, None] = {}
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_dir():
            kept.setdefault(resolved, None)
    return tuple(kept)


def resolve_asset_roots(path: str | Path | None = None, *, extra_roots: Sequence[Path] = ()) -> AssetRoots:
    """Return every ComfyUI asset directory this run is allowed to read from.

    Two automatic sources, because neither alone covers the field: the build
    PATH positional names the install the command is already acting on, and the
    tracked workspace covers ``comfy deploy run --deployment <id>`` fired from
    an unrelated directory. ``require_spec=False`` is what keeps that second
    case working — a deployment id needs no build spec.

    ``extra_roots`` is ``--asset-root``. Without it the allowlist would be a
    hard usability break for anyone who keeps assets outside either layout.
    """
    paths = resolve_build_paths(path, require_spec=False)
    candidates = [paths.models_dir, paths.input_dir, paths.output_dir]
    workspace = WorkspaceManager().workspace_path
    if workspace:
        workspace_root = Path(workspace)
        candidates.extend(workspace_root / dirname for dirname in _ASSET_DIRNAMES)
    candidates.extend(extra_roots)
    return _existing_dirs(candidates)


def _candidate_paths(value: str, roots: AssetRoots) -> Iterator[Path]:
    """Every location one input string could name, allowed roots first.

    The cwd is deliberately probed last and is *not* a root: it exists only so
    a string that names a real file nobody may send is refused out loud rather
    than passed through as a literal.
    """
    input_path = Path(value)
    if input_path.is_absolute():
        yield input_path
        return
    for root in roots:
        yield root / input_path
    yield Path.cwd() / input_path


def _resolve_local_file(value: str, roots: AssetRoots) -> Path | None:
    """Map one input string to the asset file it names, or ``None`` for a literal.

    ``None`` is the overwhelmingly common answer — prompts, sampler names and
    class names all land here — so it stays a plain return rather than an error.

    Raises:
        DeployWorkflowAssetOutsideRootError: the string names a real file that
            no allowed root contains.
    """
    outside: Path | None = None
    for candidate in _candidate_paths(value, roots):
        try:
            resolved = candidate.resolve()
            if not resolved.is_file():
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        if any(resolved.is_relative_to(root) for root in roots):
            return resolved
        if outside is None:
            outside = resolved
    if outside is not None:
        raise DeployWorkflowAssetOutsideRootError(value, outside, roots)
    return None


def _is_link_reference(items: Sequence[JsonValue]) -> bool:
    """True for an API-format link — ``["<node id>", <output slot>]``.

    A connected input is a graph edge, never a filename. Rewriting its node id
    into an asset placeholder severs the edge and submits a broken graph in
    silence, so the pair survives the scan untouched.
    """
    match items:
        # `isinstance(True, int)` is True, so bools are ruled out before ints.
        case [str(), bool()]:
            return False
        case [str(), int()]:
            return True
        case _:
            return False


def _reject_reserved_marker(mapping: Mapping[str, JsonValue]) -> None:
    info = mapping.get("info")
    if not isinstance(info, dict):
        return
    marker = info.get("id")
    if isinstance(marker, str) and marker.startswith(_LOCAL_ASSET_MARKER_PREFIX):
        raise DeployWorkflowAssetMarkerError(marker)


def _rewrite_input(
    value: JsonValue,
    roots: AssetRoots,
    discovered: dict[Path, WorkflowAsset],
) -> JsonValue:
    match value:
        case str() as text:
            local_path = _resolve_local_file(text, roots)
            if local_path is None:
                return text
            asset = discovered.get(local_path)
            if asset is None:
                asset = WorkflowAsset(
                    marker=f"{_LOCAL_ASSET_MARKER_PREFIX}{len(discovered)}",
                    local_path=local_path,
                    file_path=local_path.name,
                )
                discovered[local_path] = asset
            return {
                "__type": _ASSET_TYPE,
                "info": {"id": asset.marker, "file_path": asset.file_path},
            }
        case dict() as mapping:
            if mapping.get("__type") == _ASSET_TYPE:
                _reject_reserved_marker(mapping)
                return deepcopy(mapping)
            return {key: _rewrite_input(item, roots, discovered) for key, item in mapping.items()}
        case list() as items:
            if _is_link_reference(items):
                return list(items)
            return [_rewrite_input(item, roots, discovered) for item in items]
        case None | bool() | int() | float():
            return value
        case unreachable:
            assert_never(unreachable)


def _scan_api_workflow(workflow: JsonObject, roots: AssetRoots) -> WorkflowAssetPlan:
    discovered: dict[Path, WorkflowAsset] = {}
    rewritten = deepcopy(workflow)
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        rewritten_node = deepcopy(node)
        rewritten_node["inputs"] = _rewrite_input(inputs, roots, discovered)
        rewritten[node_id] = rewritten_node
    return WorkflowAssetPlan(workflow=rewritten, assets=tuple(discovered.values()))


def scan_workflow_inputs(workflow: JsonObject, *, asset_roots: AssetRoots) -> WorkflowAssetPlan:
    """Return a pure placeholder rewrite plan for local files in node inputs."""
    if is_ui_workflow(workflow):
        raise DeployWorkflowFormatUIError
    return _scan_api_workflow(workflow, asset_roots)


def load_deploy_workflow(workflow_file: Path, *, asset_roots: AssetRoots) -> WorkflowAssetPlan:
    """Load an API workflow and return its contained local-file rewrite plan."""
    raw_workflow, _absolute_path, _is_ui = _load_workflow_file(str(workflow_file))
    return scan_workflow_inputs(raw_workflow, asset_roots=asset_roots)


def _materialize_value(value: JsonValue, markers: frozenset[str], asset_ids: Mapping[str, str]) -> JsonValue:
    match value:
        case dict() as mapping:
            if mapping.get("__type") == _ASSET_TYPE:
                info = mapping.get("info")
                if isinstance(info, dict):
                    marker = info.get("id")
                    if isinstance(marker, str) and marker in markers:
                        rewritten = deepcopy(mapping)
                        rewritten_info = deepcopy(info)
                        rewritten_info["id"] = asset_ids[marker]
                        rewritten["info"] = rewritten_info
                        return rewritten
                return deepcopy(mapping)
            return {key: _materialize_value(item, markers, asset_ids) for key, item in mapping.items()}
        case list() as items:
            return [_materialize_value(item, markers, asset_ids) for item in items]
        case None | bool() | int() | float() | str():
            return value
        case unreachable:
            assert_never(unreachable)


def materialize_workflow_assets(plan: WorkflowAssetPlan, asset_ids: Mapping[str, str]) -> JsonObject:
    """Replace every local marker with the uploaded asset id in a new workflow."""
    markers = frozenset(asset.marker for asset in plan.assets)
    return {key: _materialize_value(value, markers, asset_ids) for key, value in plan.workflow.items()}
