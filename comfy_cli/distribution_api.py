"""HTTP client for the comfy-builder distribution API (``/v1``).

Authenticates with the OAuth Cloud JWT the CLI already holds after
``comfy cloud login`` — the builder validates it by signature (issuer
``cloud.comfy.org``, audience ``comfy-cloud``, ``workspace_id`` claim), so no
server-side changes are needed. Blob bytes are PUT straight to a presigned GCS
URL and never transit the builder.

Field names match services/comfy-builder/openapi.yaml exactly:
  POST /v1/blobs {kind, filename, sizeBytes, sha256}     -> {blobId, uploadUrl, expiresAt}
  POST /v1/distributions {name, definition}              -> {id, ...}
  POST /v1/distributions/{id}/versions {targets?}        -> {distributionVersionId, statusUrl}
  GET  /v1/distribution-versions/{id}                    -> {status, ...}
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
# Presigned model PUTs can be many GB: generous read timeout, sane connect.
_UPLOAD_TIMEOUT = (10, 600)


class BuilderAuthError(Exception):
    """No usable Cloud JWT — the user needs to run `comfy cloud login`."""


class BuilderClient:
    """Thin authed client over the builder's /v1 API. One OAuth JWT, attached
    as Bearer via the shared authed HTTP path (HTTPS-enforced, no credential
    replay on redirect, response-size capped)."""

    def __init__(self, base_url: str, token: str):
        # kind="cloud" is what makes the shared http layer attach the Bearer;
        # path_prefix carries the /v1 base so target.url("distributions") is right.
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

    def _post(self, parts: tuple[str, ...], body: dict) -> dict:
        _, parsed = request_json(self.target.url(*parts), self.target, method="POST", body=body, max_bytes=_MAX_JSON)
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
            )
        resp.raise_for_status()

    def create_distribution(self, name: str, definition: dict, description: str | None = None) -> str:
        """Create a distribution from a definition. Returns its id."""
        body: dict = {"name": name, "definition": definition}
        if description:
            body["description"] = description
        return self._post(("distributions",), body)["id"]

    def cut_version(self, distribution_id: str, targets: list[dict] | None = None) -> tuple[str, str]:
        """Freeze the definition and enqueue a build. Returns (versionId, statusUrl)."""
        body: dict = {}
        if targets:
            body["targets"] = targets
        r = self._post(("distributions", distribution_id, "versions"), body)
        return r["distributionVersionId"], r["statusUrl"]

    def get_version(self, version_id: str) -> dict:
        """Poll a version's build status."""
        _, parsed = request_json(
            self.target.url("distribution-versions", version_id), self.target, method="GET", max_bytes=_MAX_JSON
        )
        return parsed or {}

    def resolve_models(self, filenames: list[str]) -> list[dict]:
        """POST /v1/models/resolve — bare filenames -> one result per filename
        (each ``{filename, candidates: [{sourceUri, sha256?, type?, ...}], error?}``),
        searching HuggingFace/CivitAI. At most 32 filenames per call."""
        r = self._post(("models", "resolve"), {"filenames": filenames})
        return r.get("results", [])

    def _get(self, parts: tuple[str, ...], params: dict | None = None) -> dict:
        url = self.target.url(*parts)
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
            if query:
                url = f"{url}?{query}"
        _, parsed = request_json(url, self.target, method="GET", max_bytes=_MAX_JSON)
        return parsed or {}

    def list_distributions(self) -> list[dict]:
        """GET /v1/distributions -> the caller's distributions (summaries)."""
        return self._get(("distributions",)).get("distributions", [])

    def get_distribution(self, distribution_id: str) -> dict:
        """GET /v1/distributions/{id} -> a distribution with its full definition."""
        return self._get(("distributions", distribution_id))

    def list_distribution_versions(self, distribution_id: str) -> list[dict]:
        """GET /v1/distributions/{id}/versions -> the distribution's versions."""
        return self._get(("distributions", distribution_id, "versions")).get("versions", [])

    def get_version_logs(self, version_id: str, *, os: str | None = None, gpu: str | None = None) -> dict:
        """GET /v1/distribution-versions/{id}/logs -> the build log for one target
        (``os``/``gpu`` select which; the builder picks a target when omitted).
        Returns ``{versionId, os?, gpu?, log, truncated}``."""
        return self._get(("distribution-versions", version_id, "logs"), {"os": os, "gpu": gpu})

    def delete_distribution(self, distribution_id: str) -> None:
        """DELETE /v1/distributions/{id} -> soft-delete. Idempotent (204 even when
        already gone); the builder returns 409 while a deployment still runs one of
        its versions."""
        request_json(
            self.target.url("distributions", distribution_id), self.target, method="DELETE", max_bytes=_MAX_JSON
        )

    def validate_distribution(self, distribution_id: str) -> dict:
        """POST /v1/distributions/{id}/validate -> dry-run resolve the stored
        definition (no build). 200 with a ValidateResult when resolvable; the
        builder returns 400 with the issues when the definition has problems."""
        _, parsed = request_json(
            self.target.url("distributions", distribution_id, "validate"),
            self.target,
            method="POST",
            max_bytes=_MAX_JSON,
        )
        return parsed or {}
