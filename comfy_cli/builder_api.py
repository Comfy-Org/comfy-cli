"""HTTP client for the comfy-builder build API (``/v1``).

Authenticates with the OAuth Cloud JWT the CLI already holds after
``comfy cloud login`` — the builder validates it by signature (issuer
``cloud.comfy.org``, audience ``comfy-cloud``, ``workspace_id`` claim), so no
server-side changes are needed. Blob bytes are PUT straight to a presigned GCS
URL and never transit the builder.

Field names match services/comfy-builder/openapi.yaml exactly:
  POST /v1/blobs {kind, filename, sizeBytes, sha256}     -> {blobId, uploadUrl, expiresAt}
  POST /v1/builds {name, definition}              -> {id, ...}
  POST /v1/builds/from-workflow {name, workflow}  -> {build, report}
  POST /v1/builds/{id}/releases {targets?}        -> {releaseId, statusUrl}
  GET  /v1/releases/{id}                          -> {status, ...}
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import requests

from comfy_cli import credentials
from comfy_cli.http import request_json
from comfy_cli.target import Target

# Builder JSON responses are small (ids, status, a manifest pointer); cap well
# above that but far below anything that could OOM the CLI.
_MAX_JSON = 5 * 1024 * 1024
# Build logs are the exception: the builder caps a served log at 8 MiB, and the
# JSON envelope escapes it on top, so the logs call needs a cap well above 8 MiB
# or a large log is rejected client-side. Generous headroom for a future raise.
_MAX_LOG_JSON = 32 * 1024 * 1024
# Presigned model PUTs can be many GB: generous read timeout, sane connect.
_UPLOAD_TIMEOUT = (10, 600)
# The shared HTTP layer's own default, named here so one call can raise it.
_POST_TIMEOUT = 30.0
# Importing a workflow looks every distinct node class up in the registry behind
# a 20 second budget of the builder's own, then matches models and writes the
# build row, so the default leaves a large graph nothing to spare.
_WORKFLOW_IMPORT_TIMEOUT = 90.0


class BuilderAuthError(Exception):
    """No usable Cloud JWT — the user needs to run `comfy cloud login`."""


class BuilderClient:
    """Thin authed client over the builder's /v1 API. One OAuth JWT, attached
    as Bearer via the shared authed HTTP path (HTTPS-enforced, no credential
    replay on redirect, response-size capped)."""

    def __init__(self, base_url: str, token: str):
        # kind="cloud" is what makes the shared http layer attach the Bearer;
        # path_prefix carries the /v1 base so target.url("builds") is right.
        self.target = Target(
            kind="cloud",
            base_url=base_url.rstrip("/"),
            path_prefix="/v1",
            auth_token=token,
        )

    @classmethod
    def from_session(cls, base_url: str) -> BuilderClient:
        """Build a client from the CLI's OAuth session, refreshing the JWT if it
        is near expiry (the CLI's existing rotation machinery)."""
        session = credentials.get_session(refresh=True)
        if not session or not session.access_token:
            raise BuilderAuthError("not signed in — run `comfy cloud login`")
        return cls(base_url, session.access_token)

    def _post(self, parts: tuple[str, ...], body: dict, *, timeout: float = _POST_TIMEOUT) -> dict:
        _, parsed = request_json(
            self.target.url(*parts), self.target, method="POST", body=body, max_bytes=_MAX_JSON, timeout=timeout
        )
        return parsed or {}

    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str]:
        """Mint a presigned upload. Returns (blobId, uploadUrl)."""
        r = self._post(("blobs",), {"kind": kind, "filename": filename, "sha256": sha256, "sizeBytes": size_bytes})
        return r["blobId"], r["uploadUrl"]

    def upload_blob(self, upload_url: str, path: Path) -> None:
        """Stream a file to its presigned PUT URL (bytes go straight to storage).

        The builder signs the URL with ``x-goog-if-generation-match: 0`` (create-
        only, so re-using a blob id can't clobber bytes), and that header is part
        of the signature — GCS rejects the PUT with 400 unless the client sends it.
        """
        with path.open("rb") as f:
            resp = requests.put(
                upload_url,
                data=f,
                headers={"x-goog-if-generation-match": "0"},
                timeout=_UPLOAD_TIMEOUT,
                # A presigned PUT targets one exact object; a 3xx would divert the file
                # stream to another host. Never follow it (raise_for_status ignores 3xx).
                allow_redirects=False,
            )
        if 300 <= resp.status_code < 400:
            raise requests.HTTPError(
                f"presigned upload was redirected ({resp.status_code}); refusing to follow", response=resp
            )
        resp.raise_for_status()

    def create_build(self, name: str, definition: dict, description: str | None = None) -> str:
        """Create a build from a definition. Returns its id."""
        body: dict = {"name": name, "definition": definition}
        if description:
            body["description"] = description
        return self._post(("builds",), body)["id"]

    def create_build_from_workflow(self, name: str, workflow: dict, *, description: str | None = None) -> dict:
        """POST /v1/builds/from-workflow: read a workflow and store what it
        maps to, in one call. Returns ``{build, report}``."""
        body: dict = {"name": name, "workflow": workflow}
        if description:
            body["description"] = description
        return self._post(("builds", "from-workflow"), body, timeout=_WORKFLOW_IMPORT_TIMEOUT)

    def cut_version(self, build_id: str, targets: list[dict] | None = None) -> tuple[str, str]:
        """POST /v1/builds/{id}/releases: freeze the definition and enqueue a
        build. Returns (releaseId, statusUrl). A server that predates the
        version-to-release rename still answers with ``buildVersionId``, so
        that key is the fallback and the CLI works against either generation."""
        body: dict = {}
        if targets:
            body["targets"] = targets
        r = self._post(("builds", build_id, "releases"), body)
        return r.get("releaseId") or r["buildVersionId"], r["statusUrl"]

    def get_version(self, version_id: str) -> dict:
        """GET /v1/releases/{id}: poll a release's build status."""
        _, parsed = request_json(
            self.target.url("releases", version_id), self.target, method="GET", max_bytes=_MAX_JSON
        )
        return parsed or {}

    def resolve_models(self, filenames: list[str]) -> list[dict]:
        """POST /v1/models/resolve — bare filenames -> one result per filename
        (each ``{filename, candidates: [{sourceUri, sha256?, type?, ...}], error?}``),
        searching HuggingFace/CivitAI. At most 32 filenames per call."""
        r = self._post(("models", "resolve"), {"filenames": filenames})
        return r.get("results", [])

    def resolve_snapshot(self, snapshot: dict) -> dict:
        """POST /v1/snapshots/resolve — read a captured environment as a definition.

        The builder's importer is the one place that knows what the Comfy Registry
        actually publishes, which curated base image a Python fits, and how a pin
        normalizes. Returns ``{definition, report, ...}``; the report names what
        did not translate rather than leaving it to fail inside a build."""
        return self._post(("snapshots", "resolve"), {"snapshot": snapshot})

    def create_build_from_snapshot(
        self, name: str, snapshot: dict, *, description: str | None = None, base_image_id: str | None = None
    ) -> dict:
        """POST /v1/builds/from-snapshot — read a snapshot and store what it
        maps to, in one call. Returns ``{build, report}``."""
        body: dict = {"name": name, "snapshot": snapshot}
        if description:
            body["description"] = description
        if base_image_id:
            body["baseImageId"] = base_image_id
        return self._post(("builds", "from-snapshot"), body)

    def _get(self, parts: tuple[str, ...], params: dict | None = None, *, max_bytes: int = _MAX_JSON) -> dict:
        url = self.target.url(*parts)
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
            if query:
                url = f"{url}?{query}"
        _, parsed = request_json(url, self.target, method="GET", max_bytes=max_bytes)
        return parsed or {}

    def list_builds(self) -> list[dict]:
        """GET /v1/builds -> the caller's builds (summaries)."""
        return self._get(("builds",)).get("builds", [])

    def get_build(self, build_id: str) -> dict:
        """GET /v1/builds/{id} -> a build with its full definition."""
        return self._get(("builds", build_id))

    def list_releases(self, build_id: str) -> list[dict]:
        """GET /v1/builds/{id}/releases -> the build's releases. A server
        that predates the version-to-release rename keys the list ``versions``,
        so both spellings parse."""
        body = self._get(("builds", build_id, "releases"))
        return body.get("releases") or body.get("versions") or []

    def get_version_logs(self, version_id: str, *, os: str | None = None, gpu: str | None = None) -> dict:
        """GET /v1/releases/{id}/logs -> the build log for one target
        (``os``/``gpu`` select which; the builder picks a target when omitted).
        Returns ``{versionId, os?, gpu?, log, truncated}``. Uses a larger response
        cap than other reads because a build log can be several MiB."""
        return self._get(("releases", version_id, "logs"), {"os": os, "gpu": gpu}, max_bytes=_MAX_LOG_JSON)

    def delete_build(self, build_id: str) -> None:
        """DELETE /v1/builds/{id} -> soft-delete. Idempotent (204 even when
        already gone); the builder returns 409 while a deployment still runs one of
        its versions."""
        request_json(self.target.url("builds", build_id), self.target, method="DELETE", max_bytes=_MAX_JSON)

    def validate_build(self, build_id: str) -> dict:
        """POST /v1/builds/{id}/validate -> dry-run resolve the stored
        definition (no build). 200 with a ValidateResult when resolvable; the
        builder returns 400 with the issues when the definition has problems."""
        _, parsed = request_json(
            self.target.url("builds", build_id, "validate"),
            self.target,
            method="POST",
            max_bytes=_MAX_JSON,
        )
        return parsed or {}

    def update_build(self, build_id: str, definition: dict, expected_updated_at: str | None) -> dict:
        """PATCH /v1/builds/{id} -> replace the stored definition. Returns
        the updated build. ``expected_updated_at`` is the ``updatedAt`` the
        caller last saw (optimistic concurrency); the builder rejects the save with
        409 STALE if it is stale or missing, so pass the value from a fresh GET."""
        body = {"definition": definition, "expectedUpdatedAt": expected_updated_at}
        _, parsed = request_json(
            self.target.url("builds", build_id),
            self.target,
            method="PATCH",
            body=body,
            max_bytes=_MAX_JSON,
        )
        return parsed or {}

    def get_version_manifest(self, version_id: str) -> dict:
        """GET /v1/releases/{id}/manifest -> the release's models and
        runtime policies."""
        return self._get(("releases", version_id, "manifest"))

    def get_artifact_download(self, artifact_id: str) -> dict:
        """GET /v1/build-artifacts/{id}/download -> ``{downloadUrl, expiresAt?}``."""
        return self._get(("build-artifacts", artifact_id, "download"))

    def list_blobs(self, kind: str | None = None) -> list[dict]:
        """GET /v1/blobs -> the workspace's private uploaded content (summaries),
        optionally filtered by ``kind`` (model | node_zip)."""
        return self._get(("blobs",), {"kind": kind}).get("blobs", [])

    def list_base_images(self) -> list[dict]:
        """GET /v1/base-images -> the curated base images a build may be based on."""
        return self._get(("base-images",)).get("baseImages", [])

    def list_build_targets(self) -> list[dict]:
        """GET /v1/build-targets -> the build targets a version can be cut for."""
        return self._get(("build-targets",)).get("targets", [])

    def list_model_directories(self) -> list[str]:
        """GET /v1/model-directories -> the model directories a model may land in."""
        return self._get(("model-directories",)).get("directories", [])
