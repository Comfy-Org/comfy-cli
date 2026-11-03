"""Explicit, content-bound imports into the authenticated Cloud model library."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, BinaryIO

import blake3
import typer

from comfy_cli import tracking
from comfy_cli.command.cloud_http import cloud_target_or_local_error
from comfy_cli.http import ResponseTooLarge, no_redirect_urlopen, read_capped, request_json, target_auth_headers
from comfy_cli.output import get_renderer
from comfy_cli.workflow_inventory import MODEL_EXTENSIONS

app = typer.Typer()
_CHUNK_BYTES = 1024 * 1024


def _content_hash(value) -> str | None:
    # Older Cloud versions return the BLAKE3 storage filename, newer ones the
    # algorithm-prefixed digest. Both forms describe the same asset bytes.
    match = re.fullmatch(r"(?:blake3:)?([a-f0-9]{64})(?:\.[A-Za-z0-9]+)?", value) if isinstance(value, str) else None
    return "blake3:" + match[1] if match else None


def _refuse(renderer, reason: str, message: str):
    renderer.error(
        code="model_upload_refused",
        message=message,
        hint="inspect `comfy model upload --help`; preview with --dry-run and approve the exact file and workspace",
        details={"reason": reason, "uploaded": False},
    )
    raise typer.Exit(code=1)


class _Body:
    def __init__(self, file: BinaryIO, prefix: bytes, suffix: bytes, size: int):
        self.file, self.prefix, self.suffix, self.size = file, prefix, suffix, size
        self.content_length = len(prefix) + size + len(suffix)

    def __iter__(self):
        yield self.prefix
        self.file.seek(0)
        while chunk := self.file.read(_CHUNK_BYTES):
            yield chunk
        yield self.suffix


def _snapshot(source: Path, snapshot: BinaryIO) -> tuple[str, str, int]:
    sha = hashlib.sha256()
    content_hash = blake3.blake3()
    size = 0
    with source.open("rb") as file:
        while chunk := file.read(_CHUNK_BYTES):
            snapshot.write(chunk)
            sha.update(chunk)
            content_hash.update(chunk)
            size += len(chunk)
    snapshot.seek(0)
    return sha.hexdigest(), "blake3:" + content_hash.hexdigest(), size


def _request(target, file, size: int, name: str, model_type: str):
    boundary = uuid.uuid4().hex
    fields = {"tags": json.dumps(["models", model_type]), "name": name}
    prefix = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        for key, value in fields.items()
    )
    prefix += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body = _Body(file, prefix, f"\r\n--{boundary}--\r\n".encode(), size)
    request = urllib.request.Request(target.url("assets"), data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Content-Length", str(body.content_length))
    for key, value in target_auth_headers(target).items():
        request.add_header(key, value)
    return request


@app.command(
    "upload", help="Import an approved local model into the standard Cloud model library; preview with --dry-run."
)
@tracking.track_command("model")
def upload_cmd(
    file: Annotated[Path, typer.Argument(help="The exact local model file to import.")],
    model_type: Annotated[
        str, typer.Option("--type", help="Exact model folder advertised by `model list-folders --where cloud`.")
    ],
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace", help="Approved destination workspace ID; must match current authenticated workspace."
        ),
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Read capabilities and prepare a file plan; never upload.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm approval for this named file, digest and destination.")
    ] = False,
    expected_sha256: Annotated[
        str | None, typer.Option("--expected-sha256", help="Approved file SHA256, required for an upload.")
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", help="Approved destination filename; defaults to the source basename.")
    ] = None,
    where: Annotated[str | None, typer.Option("--where")] = None,
):
    renderer = get_renderer()
    destination_name = name if name is not None else file.name
    if (
        not destination_name
        or any(c in destination_name for c in '/\\"\r\n\x00')
        or destination_name in (".", "..")
        or not destination_name.lower().endswith(MODEL_EXTENSIONS)
    ):
        _refuse(renderer, "invalid_name", "Use a model filename without directory or multipart framing characters.")
    if not file.is_file():
        _refuse(renderer, "file_not_found", f"Model file not found: {file}")
    if not dry_run and (not yes or not expected_sha256 or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256)):
        _refuse(renderer, "approval_required", "Uploading requires --yes and the exact approved --expected-sha256.")
    target = cloud_target_or_local_error(where, renderer)
    try:
        _, current = request_json(target.url("workspaces", "current"), target, max_bytes=1024 * 1024)
        if not isinstance(current, dict) or current.get("id") != workspace:
            _refuse(
                renderer, "workspace_mismatch", "The approved workspace does not match the authenticated destination."
            )
        _, features = request_json(target.url("features"), target, max_bytes=1024 * 1024)
        if not isinstance(features, dict) or features.get("private_models_enabled") is not True:
            _refuse(
                renderer, "unavailable", "Private model import is unavailable or unverified for this account/workspace."
            )
        _, folders = request_json(target.url("experiment", "models"), target, max_bytes=1024 * 1024)
        names = (
            [item.get("name") if isinstance(item, dict) else item for item in folders]
            if isinstance(folders, list)
            else []
        )
        supported = {item for item in names if isinstance(item, str)}
        if model_type not in supported or model_type.lower().strip() == "nlf":
            _refuse(
                renderer,
                "unsupported_type",
                f"The selected environment does not advertise a supported import folder: {model_type}",
            )
        # Snapshot before comparing the approved digest. Upload only this private
        # handle, so source edits during the network request cannot change bytes.
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            digest, content_hash, size = _snapshot(file, snapshot)
            if expected_sha256 and digest != expected_sha256.lower():
                _refuse(
                    renderer, "file_changed", "The file bytes differ from the approved SHA256; review the new file."
                )
            query = urllib.parse.urlencode(
                {"include_tags": "models", "name_contains": destination_name, "include_public": "true", "limit": 500}
            )
            status, existing = request_json(target.url("assets") + "?" + query, target, max_bytes=5 * 1024 * 1024)
            if (
                status != 200
                or not isinstance(existing, dict)
                or not isinstance(existing.get("assets"), list)
                or any(not isinstance(row, dict) for row in existing["assets"])
                or existing.get("has_more") is not False
            ):
                _refuse(
                    renderer,
                    "collision_unverified",
                    "Could not verify destination-name collisions; no file was uploaded.",
                )
            matches = [
                row for row in existing["assets"] if isinstance(row, dict) and row.get("name") == destination_name
            ]
            if any(_content_hash(row.get("hash")) != content_hash for row in matches):
                _refuse(
                    renderer,
                    "name_collision",
                    "That model name already identifies different or unverified bytes; approve a unique --name.",
                )
            owned_matches = []
            if matches:
                # Public visibility establishes a collision, not ownership.
                # Cloud's from-hash endpoint requires this workspace to own
                # the bytes, so verify that with a separate scoped listing.
                owned_query = urllib.parse.urlencode(
                    {
                        "include_tags": "models",
                        "name_contains": destination_name,
                        "include_public": "false",
                        "limit": 500,
                    }
                )
                status, owned = request_json(
                    target.url("assets") + "?" + owned_query, target, max_bytes=5 * 1024 * 1024
                )
                if (
                    status != 200
                    or not isinstance(owned, dict)
                    or not isinstance(owned.get("assets"), list)
                    or any(not isinstance(row, dict) for row in owned["assets"])
                    or owned.get("has_more") is not False
                ):
                    _refuse(
                        renderer,
                        "ownership_unverified",
                        "Could not verify workspace-owned model assets; no file was uploaded.",
                    )
                owned_matches = [row for row in owned["assets"] if row.get("name") == destination_name]
                if any(_content_hash(row.get("hash")) != content_hash for row in owned_matches):
                    _refuse(
                        renderer,
                        "name_collision",
                        "That model name already identifies different or unverified bytes; approve a unique --name.",
                    )
            plan = {
                "file": str(file.resolve()),
                "name": destination_name,
                "model_type": model_type,
                "size_bytes": size,
                "sha256": digest,
                "hash": content_hash,
                "workspace_id": workspace,
                "workspace_name": current.get("name"),
                "destination": target.url("assets"),
                "environment": "standard",
                "uploaded": False,
                "deduped": False,
                "dry_run": dry_run,
                "license": "unknown",
                "nsfw": "unknown",
                "estimated_upload_seconds": None,
                "upload_cost": None,
                "storage_cost": None,
                "compute_cost": None,
                "execution_state": "prepared, not run",
            }
            if dry_run:
                renderer.emit(plan, command="model upload", where="cloud", changed=False)
                return
            if owned_matches:
                _, body = request_json(
                    target.url("assets", "from-hash"),
                    target,
                    method="POST",
                    body={"hash": content_hash, "tags": ["models", model_type]},
                    max_bytes=1024 * 1024,
                )
            else:
                request = _request(target, snapshot, size, destination_name, model_type)
                with no_redirect_urlopen(request, timeout=3600) as response:
                    raw = read_capped(response, target.url("assets"), max_bytes=1024 * 1024)
                body = json.loads(raw)
            if not isinstance(body, dict) or _content_hash(body.get("hash")) != content_hash or not body.get("id"):
                renderer.error(
                    code="model_upload_failed",
                    message="Upload response did not verify the exact approved model bytes.",
                    hint="inspect the asset library before retrying; the server may have received the file",
                    details={"uploaded": "unknown", "workspace_id": workspace},
                )
                raise typer.Exit(code=1)
            tags = body.get("tags")
            loader_path = body.get("loader_path")
            file_path = body.get("file_path")
            folder_verified = (isinstance(file_path, str) and file_path.startswith(f"models/{model_type}/")) or (
                isinstance(tags, list) and "/" not in model_type and model_type.lower() in tags
            )
            # Before loader_path was served, the inference model resolver used
            # Asset.Name verbatim. Derive only after verifying its folder and
            # exact bytes, never from the requested local filename alone.
            if loader_path is None and folder_verified:
                loader_path = body.get("name")
            if (
                not isinstance(tags, list)
                or "models" not in tags
                or not folder_verified
                or not isinstance(loader_path, str)
                or not loader_path
                or "\\" in loader_path
                or loader_path.startswith("/")
                or any(part in (".", "..", "") for part in loader_path.split("/"))
            ):
                renderer.error(
                    code="model_upload_failed",
                    message="Model bytes were stored but a usable model-library path was not verified.",
                    hint="inspect the returned asset in the selected workspace before running",
                    details={"uploaded": True, "asset_id": body["id"], "workspace_id": workspace},
                )
                raise typer.Exit(code=1)
            renderer.emit(
                {
                    **plan,
                    "uploaded": not bool(owned_matches),
                    "deduped": bool(owned_matches),
                    "asset_id": body["id"],
                    "loader_path": loader_path,
                    "created_new": body.get("created_new"),
                },
                command="model upload",
                where="cloud",
                changed=True,
            )
    except (OSError, ValueError, RecursionError, ResponseTooLarge, urllib.error.URLError) as error:
        renderer.error(
            code="model_upload_failed",
            message=f"Model import could not be completed: {type(error).__name__}",
            hint="check account/network and the asset library before retrying; do not assume a failed request stored nothing",
            details={"status": getattr(error, "code", None), "uploaded": "unknown"},
        )
        raise typer.Exit(code=1) from error
