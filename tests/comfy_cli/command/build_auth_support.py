"""A Builder that answers every route the build tree can reach, recording each call.

This fake sits at the *HTTP* seam (``builder_api.request_json``), not at the
client seam the other build fixtures patch. That is the whole point of the auth
matrix: a signed-out command must be refused before a single request is issued,
which is only observable one layer below the client object.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from comfy_cli.command.build_spec import JsonObject, JsonValue

BUILD_ID = "build-matrix"
RELEASE_ID = "release-matrix"
REVISION = "revision-matrix"
STATUS_URL = f"https://builder.test/v1/releases/{RELEASE_ID}"

_EMPTY_DEFINITION: JsonObject = {"models": [], "customNodes": []}

# Keyed on the URL *path* so a route carrying query parameters (`?cursor=`,
# `?os=&gpu=`) still matches.
_ROUTES: dict[tuple[str, str], tuple[int, JsonObject]] = {
    ("POST", "/v1/snapshots/resolve"): (200, {"definition": _EMPTY_DEFINITION, "report": {}}),
    ("POST", "/v1/builds"): (201, {"id": BUILD_ID, "updatedAt": REVISION}),
    ("GET", "/v1/builds"): (200, {"builds": [{"id": BUILD_ID, "name": "Matrix"}]}),
    ("GET", f"/v1/builds/{BUILD_ID}"): (
        200,
        {
            "id": BUILD_ID,
            "name": "Matrix",
            "description": "",
            "updatedAt": REVISION,
            "definition": _EMPTY_DEFINITION,
        },
    ),
    ("DELETE", f"/v1/builds/{BUILD_ID}"): (204, {}),
    ("POST", f"/v1/builds/{BUILD_ID}/releases"): (201, {"releaseId": RELEASE_ID, "statusUrl": STATUS_URL}),
    ("GET", f"/v1/builds/{BUILD_ID}/releases"): (
        200,
        {
            "releases": [
                {
                    "id": RELEASE_ID,
                    "status": "complete",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "buildId": BUILD_ID,
                    "version": 1,
                    "deployable": True,
                }
            ]
        },
    ),
    ("GET", f"/v1/releases/{RELEASE_ID}"): (
        200,
        {"id": RELEASE_ID, "status": "complete", "buildId": BUILD_ID, "version": 1, "deployable": True},
    ),
    ("GET", f"/v1/releases/{RELEASE_ID}/logs"): (
        200,
        {
            "versionId": RELEASE_ID,
            "releaseId": RELEASE_ID,
            "os": "linux",
            "gpu": "nvidia",
            "log": "built\n",
            "truncated": False,
        },
    ),
    ("GET", f"/v1/releases/{RELEASE_ID}/manifest"): (200, {"models": []}),
    ("GET", "/v1/base-images"): (200, {"baseImages": []}),
    ("GET", "/v1/build-targets"): (200, {"targets": []}),
    ("GET", "/v1/model-directories"): (200, {"directories": []}),
    ("GET", "/v1/blobs"): (200, {"blobs": []}),
}


class RecordingTransport:
    def __init__(self) -> None:
        self.calls: list[JsonObject] = []

    def __call__(
        self,
        url: str,
        target: object,
        *,
        method: str = "GET",
        body: JsonObject | None = None,
        timeout: float = 30.0,
        max_bytes: int,
    ) -> tuple[int, JsonObject]:
        self.calls.append({"url": url, "method": method, "body": body})
        path = urlsplit(url).path
        if (method, path) == ("POST", "/v1/models/resolve"):
            return 200, {"results": _resolved_filenames(body)}
        route = _ROUTES.get((method, path))
        if route is None:
            pytest.fail(f"unexpected Builder call: {method} {url}")
        status, payload = route
        return status, copy.deepcopy(payload)


def _resolved_filenames(body: JsonObject | None) -> list[JsonValue]:
    assert body is not None
    filenames = body["filenames"]
    assert isinstance(filenames, list)
    return [{"filename": filename, "candidates": []} for filename in filenames]


def write_snapshot(root: Path) -> Path:
    path = root / "snapshot.json"
    path.write_text(json.dumps({"customNodes": [], "pipPackages": {}, "pythonVersion": "3.12"}), encoding="utf-8")
    return path
