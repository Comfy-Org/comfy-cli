"""``comfy workflow check <workflow> --where cloud``: can Comfy Cloud run this workflow as it is?

Signed in, the check reads Cloud's node list (``object_info``) and, for a model
Cloud's dropdowns do not list, the person's uploaded models (``/api/assets``). The
comparing lives in :mod:`comfy_cli.workflow_coverage`; this module fetches, writes
the remapped copy, and reports. Mounted on ``comfy workflow`` in ``workflow.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer

_ASSET_PAGE = 200
_ASSET_PAGES = 50  # 10,000 rows: a server that never says it is done cannot loop forever

_SUMMARY = {
    "covered": "Runs on Comfy Cloud as it is.",
    "covered_with_remaps": "Runs on Comfy Cloud once {remaps} pointed at Cloud's copies.",
    "covered_needs_check": (
        "Nothing is missing on Comfy Cloud, but {unproven} not proven the same file as Cloud's copy."
    ),
    "needs_build": "Needs a build to run on Comfy Cloud: {blocking} below.",
}


def _plural(count: int, word: str, verb: str = "") -> str:
    noun = word if count == 1 else word + "s"
    if verb:
        verb = {"is": "is" if count == 1 else "are"}.get(verb, verb)
        return f"{count} {noun} {verb}"
    return f"{count} {noun}"


def _load(renderer, workflow: str) -> tuple[Path, Any]:
    from comfy_cli.workflow_coverage import workflow_format

    path = Path(workflow).expanduser()
    if not path.is_file():
        renderer.error(
            code="workflow_not_found",
            message=f"Workflow file not found: {workflow}",
            hint="check the path",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        renderer.error(
            code="workflow_read_error",
            message=f"Unable to read workflow file: {e}",
            hint="check file permissions and encoding",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as e:
        renderer.error(
            code="workflow_invalid_json",
            message=f"Workflow file is not valid JSON: {e}",
            hint="save the workflow again from ComfyUI",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e
    if workflow_format(data) is None:
        renderer.error(
            code="workflow_format_unknown",
            message="The file is JSON but not a ComfyUI workflow: no `nodes[]` and not an API export.",
            hint="pass a workflow saved from ComfyUI (Workflow > Save) or its API export (Workflow > Export (API))",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1)
    return path.resolve(), data


def _fetch_model_assets(target, name: str, *, include_public: bool) -> list[dict[str, Any]]:
    """Every model asset whose name contains ``name``, all pages of it.

    ``name_contains`` is a substring match, so the exact file can sit past the
    first page; an answer from part of the list could call a model missing.
    Raises on any failure, a list that never ends included; callers decide what
    it means.
    """
    import urllib.parse

    from comfy_cli.command.cloud_http import http_request

    found: list[dict[str, Any]] = []
    offset = 0
    for _ in range(_ASSET_PAGES):
        query = urllib.parse.urlencode(
            {
                "include_tags": "models",
                "include_public": "true" if include_public else "false",
                "name_contains": name,
                "limit": _ASSET_PAGE,
                "offset": offset,
            }
        )
        _, body = http_request(target.url("assets") + "?" + query, target, strict_json=True)
        rows = body.get("assets") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ValueError("unexpected response shape from /api/assets (expected an object with assets[])")
        found.extend(row for row in rows if isinstance(row, dict) and "models" in (row.get("tags") or []))
        if not rows or not body.get("has_more"):
            return found
        offset += len(rows)
    raise ValueError(f"/api/assets kept paging past {_ASSET_PAGES * _ASSET_PAGE} models named like {name!r}")


def _own_models(renderer, target):
    """``own_models`` for :func:`resolve_models`. A failed read ends the command:
    without it a model the person uploaded would be reported missing."""
    import http.client
    import urllib.error

    from comfy_cli.command._cloud_errors import handle_cloud_http_error
    from comfy_cli.command.cloud_http import ResponseUnparseable

    def lookup(name: str) -> list[dict[str, Any]]:
        try:
            return _fetch_model_assets(target, name, include_public=False)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise handle_cloud_http_error(
                renderer,
                e,
                operation="model lookup",
                not_found_code="cloud_http_error",
                not_found_message="Comfy Cloud has no model list at /api/assets",
                not_found_hint="check the Cloud base URL with `comfy cloud status`",
                id_label="model",
                resource_id=name,
            ) from e
        except (ResponseUnparseable, ValueError) as e:
            renderer.error(
                code="cloud_http_error",
                message=f"unexpected response from /api/assets while looking up {name!r}: {e}",
                hint="check whether a proxy or captive portal is intercepting the request",
                details={"operation": "model lookup", "model": name},
            )
            raise typer.Exit(code=1) from e

    return lookup


def _cloud_file(target, cloud_value: str, warnings: list[dict]) -> tuple[int | None, str | None]:
    """Size and blake3 hash of Cloud's model file named ``cloud_value``, or ``(None, None)``."""
    import http.client
    import urllib.error

    from comfy_cli.command.cloud_http import ResponseUnparseable
    from comfy_cli.workflow_coverage import content_hash

    name = cloud_value.replace("\\", "/").rsplit("/", 1)[-1]
    try:
        rows = _fetch_model_assets(target, name, include_public=True)
    except (
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        ResponseUnparseable,
        ValueError,
    ) as e:
        warnings.append(
            {"code": "cloud_model_lookup_failed", "message": f"could not read Cloud's file {cloud_value!r}: {e}"}
        )
        return None, None
    rows = [row for row in rows if row.get("name") in (cloud_value, name)]
    sizes = {row.get("size") for row in rows}
    hashes = {content_hash(row.get("hash")) for row in rows}
    size = sizes.pop() if len(sizes) == 1 else None
    return (size if isinstance(size, int) else None), (hashes.pop() if len(hashes) == 1 else None)


def _print_summary(renderer, payload: dict[str, Any]) -> None:
    models = payload["models"]
    remaps = [m for m in models if m["status"] == "remap"]
    unproven = [m for m in models if m["identity"] not in (None, "same")]
    renderer.info(
        _SUMMARY[payload["verdict"]].format(
            remaps=_plural(len(remaps), "model", "is"),
            unproven=_plural(len(unproven), "model", "is"),
            blocking=_plural(len(payload["blocking"]), "item"),
        )
    )
    for item in payload["blocking"]:
        if item["kind"] == "node":
            renderer.warn(f"  node type {item['class_type']} (node {', '.join(item['node_ids'])}): not on Comfy Cloud")
        else:
            reason = {
                "not_on_cloud": "not on Comfy Cloud",
                "ambiguous": "several Cloud files share this name",
                "different_file": "Cloud's file of this name is a different file",
            }[item["reason"]]
            where = f" (belongs in models/{item['folder']})" if item["folder"] else ""
            renderer.warn(f"  model {item['value']} (node {item['node_id']}): {reason}{where}")
            for local in item["local_paths"]:
                renderer.info(f"    local copy: {local}")
    for m in remaps:
        renderer.info(f"  remap node {m['node_id']}: {m['value']} -> {m['cloud_value']}")
    for m in unproven:
        renderer.info(f"  {m['value']}: {m['identity']}")
    if payload["out"]:
        renderer.success(f"Wrote the copy with {_plural(payload['remaps_applied'], 'remap')} to {payload['out']}")
    for warning in payload["warnings"]:
        renderer.warn(warning["message"])


@tracking.track_command("workflow")
def check_cmd(
    workflow: Annotated[
        str,
        typer.Argument(help="Workflow JSON file: a ComfyUI save (with nodes[]) or an API export."),
    ],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Where to check. Only 'cloud' today; needs sign-in."),
    ] = None,
    models_dir: Annotated[
        list[str] | None,
        typer.Option(
            "--models-dir",
            show_default=False,
            help="A local models folder, repeatable. A model found there is hashed and compared with Cloud's copy; "
            "without one, a model Cloud has is `unverified`.",
        ),
    ] = None,
    apply_remaps: Annotated[
        bool,
        typer.Option(
            "--apply-remaps", help="Write a copy with each remap applied to --out. The original is not changed."
        ),
    ] = False,
    out: Annotated[
        str | None,
        typer.Option("--out", show_default=False, help="Where --apply-remaps writes the remapped copy."),
    ] = None,
):
    """Report whether Comfy Cloud runs a workflow as it is, with model remaps, or only with a build."""
    from comfy_cli import workflow_coverage as cov
    from comfy_cli.command.cloud_http import cloud_target_or_local_error
    from comfy_cli.cql.engine import LoadError
    from comfy_cli.cql.loader import resilient_load_object_info
    from comfy_cli.file_utils import atomic_write_text
    from comfy_cli.hashing import blake3_file
    from comfy_cli.workflow_to_api import WorkflowConversionError

    renderer = get_renderer()
    if apply_remaps != (out is not None):
        renderer.error(
            code="missing_argument",
            message="--apply-remaps and --out go together: --out names the file the remapped copy is written to.",
            hint="pass both, e.g. `--apply-remaps --out remapped.json`, or neither",
            details={"missing": "--out" if apply_remaps else "--apply-remaps"},
        )
        raise typer.Exit(code=1)
    path, data = _load(renderer, workflow)
    out_path = Path(out).expanduser().resolve() if out is not None else None
    if out_path is not None and out_path == path:
        renderer.error(
            code="workflow_out_is_input",
            message="--out names the workflow being checked; the check never changes its input.",
            hint="pass a new file name to --out",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1)

    target = cloud_target_or_local_error(where, renderer)
    renderer.where = "cloud"
    if not (target.auth_token or target.api_key):
        renderer.error(
            code="auth_not_signed_in",
            message="`comfy workflow check --where cloud` reads your models on Comfy Cloud and needs a sign-in.",
            hint="run `comfy cloud login`",
        )
        raise typer.Exit(code=1)
    stale: dict[str, str] = {}
    try:
        object_info = resilient_load_object_info(
            mode="cloud", on_stale=lambda source, reason: stale.update(source=source, reason=reason)
        )
    except LoadError as e:
        renderer.error(
            code="cql_no_graph",
            message=f"Could not read Comfy Cloud's node list: {e}",
            hint="check `comfy cloud status`, then run the check again",
        )
        raise typer.Exit(code=1) from e

    try:
        uses = cov.find_uses(data, object_info)
    except WorkflowConversionError as e:
        renderer.error(
            code="workflow_conversion_failed",
            message=f"Could not read the workflow the way ComfyUI submits it: {e}",
            hint="open it in ComfyUI and save it again, or check its API export (Workflow > Export (API))",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e
    models = cov.resolve_models(uses, object_info, _own_models(renderer, target))
    warnings: list[dict[str, Any]] = []
    if stale:
        # Cloud's node list could not be refreshed: an older copy answered, and a
        # node removed since may still read as available.
        warnings.append(
            {
                "code": "object_info_stale",
                "message": "Cloud's node list could not be refreshed; a cached copy answered "
                f"({stale['source']}): {stale['reason']}",
                "details": dict(stale),
            }
        )
    roots = list(models_dir or [])
    for root in roots:
        if not Path(root).expanduser().is_dir():
            warnings.append({"code": "models_dir_not_found", "message": f"--models-dir is not a folder: {root}"})
    index = cov.local_index(roots) if roots else None
    cloud_files: dict[str, tuple[int | None, str | None]] = {}
    digests: dict[str, str | None] = {}

    def hash_file(file: str) -> str | None:
        if file not in digests:
            try:
                digests[file] = blake3_file(Path(file))
            except OSError as e:  # unreadable, or gone since the folder was listed
                digests[file] = None
                warnings.append(
                    {"code": "local_model_unreadable", "message": f"could not read {file} to compare it: {e}"}
                )
        return digests[file]

    cov.add_sources(models, data, index)
    for model in models:
        model["identity"] = None
        if model["status"] not in ("present", "remap"):
            continue
        local = None if index is None else index.get(model["filename"], [])
        size, digest = model.get("cloud_size"), model.get("cloud_hash")
        if model["source"] == "cloud" and local:
            if model["cloud_value"] not in cloud_files:
                cloud_files[model["cloud_value"]] = _cloud_file(target, model["cloud_value"], warnings)
            size, digest = cloud_files[model["cloud_value"]]
        model["identity"], _ = cov.identity(local, size, digest, hash_file)

    blocked = cov.blocking(uses, object_info, models)
    remaps_applied = 0
    if out_path is not None:
        copy_, remaps_applied = cov.apply_remaps(data, models)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(out_path, json.dumps(copy_, indent=2, ensure_ascii=False) + "\n")
        except OSError as e:
            renderer.error(
                code="workflow_write_error",
                message=f"could not write the remapped workflow to {out_path}: {e}",
                hint="check the --out path is writable and the disk has space",
                details={"path": str(out_path)},
            )
            raise typer.Exit(code=1) from e

    for model in models:
        for key in ("cloud_size", "cloud_hash"):
            model.pop(key, None)
    payload: dict[str, Any] = {
        "workflow": str(path),
        "format": uses["format"],
        "verdict": cov.verdict(blocked, models),
        "stale": bool(stale),
        "blocking": blocked,
        "models": models,
        "nodes": [
            {"class_type": class_type, "node_ids": node_ids, "on_cloud": class_type in object_info}
            for class_type, node_ids in sorted(uses["nodes"].items())
        ],
        "skipped": uses["skipped"],
        "out": str(out_path) if out_path is not None else None,
        "remaps_applied": remaps_applied,
        "warnings": warnings,
    }
    if renderer.is_pretty():
        _print_summary(renderer, payload)
    renderer.emit(payload, command="workflow check", where="cloud", changed=out_path is not None)
