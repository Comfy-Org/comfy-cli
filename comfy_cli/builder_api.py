"""HTTP client for the comfy-builder build API (``/v1``).

Authenticates with the OAuth Cloud JWT the CLI already holds after
``comfy cloud login`` — the builder validates it by signature (issuer
``cloud.comfy.org``, audience ``comfy-cloud``, ``workspace_id`` claim), so no
server-side changes are needed. Blob bytes are PUT straight to a presigned GCS
URL and never transit the builder.

Field names match services/comfy-builder/openapi.yaml exactly:
  POST /v1/blobs {kind, filename, sizeBytes, sha256}     -> {blobId, uploadUrl, expiresAt}
  POST /v1/builds {name, definition}              -> {id, ...}
  POST /v1/snapshots/resolve {snapshot}           -> {definition, report}
  POST /v1/workflows/resolve {workflow}           -> {definition, report, format}
  POST /v1/builds/{id}/releases {targets?}        -> {releaseId, statusUrl}
  GET  /v1/releases/{id}                          -> {status, ...}
"""

from __future__ import annotations

import urllib.error
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import requests

from comfy_cli import credentials
from comfy_cli.builder_pagination import PAGE_LIMIT, cursor_pages
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
# Reading a workflow looks every distinct node class up in the registry behind a
# 20 second budget of the builder's own, then matches models, so the shared
# default leaves a large graph nothing to spare.
_WORKFLOW_IMPORT_TIMEOUT = 90.0


# How much `_CountingReader` hands over per step when something iterates it
# instead of calling `read`. Only the chunked path does, and only for an empty
# file, so the size is not a tuning knob.
_UPLOAD_ITER_BYTES = 1024 * 1024


class _CountingReader:
    """A file as ``requests`` sees it, reporting each block the transport pulls.

    ``requests`` sizes a streamed body with ``os.fstat(body.fileno())`` and sends
    it by calling ``read`` until it runs dry, so passing ``fileno`` and ``tell``
    through keeps the request byte-identical to one handed the bare file: same
    ``Content-Length``, no chunked transfer, which a presigned GCS PUT would
    refuse. ``__iter__`` is there because its presence is how ``requests``
    decides a body streams at all.

    ``progress`` receives the size of each block as it is read, not a running
    total. It runs once per transport block (16 KiB), so it must stay cheap.
    """

    def __init__(self, raw: BinaryIO, progress: Callable[[int], None]) -> None:
        self._raw = raw
        self._progress = progress

    def read(self, size: int = -1) -> bytes:
        block = self._raw.read(size)
        if block:
            self._progress(len(block))
        return block

    @property
    def mode(self) -> str:
        # `requests` reads this straight after `fileno` to warn about text mode.
        return self._raw.mode

    def fileno(self) -> int:
        return self._raw.fileno()

    def tell(self) -> int:
        return self._raw.tell()

    def __iter__(self) -> Iterator[bytes]:
        return iter(lambda: self.read(_UPLOAD_ITER_BYTES), b"")


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
        # Only a client built from the stored sign-in may swap its token; one
        # handed a token directly (COMFY_BUILDER_TOKEN) keeps it and lets a 401 surface.
        self._refreshes_on_401 = False

    @classmethod
    def from_session(cls, base_url: str) -> BuilderClient:
        """Build a client from the CLI's OAuth session, refreshing the JWT if it
        is near expiry (the CLI's existing rotation machinery)."""
        session = credentials.get_session(refresh=True)
        if not session or not session.access_token:
            raise BuilderAuthError("not signed in — run `comfy cloud login`")
        client = cls(base_url, session.access_token)
        client._refreshes_on_401 = True
        return client

    def _send(self, url: str, **kwargs) -> tuple[int, dict | list | None]:
        """Every builder request goes through here, so a long wait survives its token expiring.

        The access token lasts fifteen minutes. After a 401 a client built from
        the stored sign-in forces the shared refresh and sends the request once
        more; the server refuses before acting, so that is safe for every method.
        """
        try:
            return request_json(url, self.target, **kwargs)
        except urllib.error.HTTPError as error:
            if error.code != 401 or not self._refreshes_on_401:
                raise
            token = credentials.refreshed_access_token(self.target.auth_token)
            if token is None:
                raise
            self.target = replace(self.target, auth_token=token)
            return request_json(url, self.target, **kwargs)

    def _post(self, parts: tuple[str, ...], body: dict, *, timeout: float = _POST_TIMEOUT) -> dict:
        _, parsed = self._send(self.target.url(*parts), method="POST", body=body, max_bytes=_MAX_JSON, timeout=timeout)
        return parsed or {}

    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str | None]:
        """Claim a blob id for these bytes. Returns ``(blobId, uploadUrl)``.

        ``uploadUrl`` is ``None`` when the workspace already held these exact
        bytes: the builder returns the id it already has and there is nothing
        to transfer, which is what makes an interrupted multi-GB push cheap to
        retry. Branch on the server's ``deduplicated`` flag rather than on the
        URL being absent — a presigned PUT is signed create-only, so replaying
        one against content that already exists fails rather than harmlessly
        re-uploading.
        """
        r = self._post(("blobs",), {"kind": kind, "filename": filename, "sha256": sha256, "sizeBytes": size_bytes})
        blob_id = r["blobId"]
        if r.get("deduplicated") is True:
            return blob_id, None
        return blob_id, r["uploadUrl"]

    def upload_blob(self, upload_url: str, path: Path, progress: Callable[[int], None] | None = None) -> None:
        """Stream a file to its presigned PUT URL (bytes go straight to storage).

        The builder signs the URL with ``x-goog-if-generation-match: 0`` (create-
        only, so re-using a blob id can't clobber bytes), and that header is part
        of the signature — GCS rejects the PUT with 400 unless the client sends it.

        ``progress`` is called with the size of each block as the transport reads
        it off disk, which is as close to "bytes sent" as a streamed PUT exposes.
        """
        with path.open("rb") as f:
            resp = requests.put(
                upload_url,
                data=f if progress is None else _CountingReader(f, progress),
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

    def create_build_response(self, name: str, definition: dict, description: str | None = None) -> dict:
        """Create a build from a definition. Returns the created build, carrying
        the ``warnings`` the save earned, which a read never returns."""
        body: dict = {"name": name, "definition": definition}
        if description:
            body["description"] = description
        return self._post(("builds",), body)

    def create_release(self, build_id: str, targets: list[dict] | None = None) -> tuple[str, str]:
        """POST /v1/builds/{id}/releases: freeze the definition and enqueue a
        build for ``targets``. Returns (releaseId, statusUrl).

        ``targets`` is required and must be non-empty: an implicit target builds
        an artifact the caller never asked for, so a missing or empty list is a
        caller error raised here, before any request is issued.

        The cut is idempotent. The builder dedupes on the definition's content hash
        scoped to the build, so retrying this call after an ambiguous failure
        returns the same release rather than cutting a second one.

        A server that predates the version-to-release rename still answers with
        ``buildVersionId``, so that key is the fallback and the CLI works against
        either generation."""
        if not isinstance(targets, list) or not targets:
            raise ValueError("create_release requires a non-empty list of targets (e.g. [{'os': ..., 'gpu': ...}])")
        body: dict = {"targets": targets}
        r = self._post(("builds", build_id, "releases"), body)
        return r.get("releaseId") or r["buildVersionId"], r["statusUrl"]

    def get_release(self, release_id: str) -> dict:
        """GET /v1/releases/{id}: poll a release's build status."""
        _, parsed = self._send(self.target.url("releases", release_id), method="GET", max_bytes=_MAX_JSON)
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

    def resolve_workflow(self, workflow: dict) -> dict:
        """POST /v1/workflows/resolve — read a workflow graph as a definition,
        writing nothing. Returns ``{definition, report, format}``.

        The sibling of :meth:`resolve_snapshot`, and the reason both stay
        resolves rather than the builder's one-call ``from-workflow`` creator:
        the spec `build init` writes is a local file, and `build update` amends
        a build that already exists — neither has anywhere to put a build row
        the server minted on its own.

        A workflow is a poorer input than a snapshot: it names no versions, no
        Python and no model sources, so every distinct node class is resolved
        against the registry and every model comes back in the report instead of
        the definition. That registry sweep is what earns the raised timeout."""
        return self._post(("workflows", "resolve"), {"workflow": workflow}, timeout=_WORKFLOW_IMPORT_TIMEOUT)

    def _get(self, parts: tuple[str, ...], params: dict | None = None, *, max_bytes: int = _MAX_JSON) -> dict:
        url = self.target.url(*parts)
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
            if query:
                url = f"{url}?{query}"
        _, parsed = self._send(url, method="GET", max_bytes=max_bytes)
        return parsed or {}

    def list_builds(self) -> list[dict]:
        """GET every cursor page from /v1/builds -> the caller's builds (summaries).

        The builder pages this read, so a workspace past one page silently lost
        its tail when only the first was taken."""
        builds: list[dict] = []
        for page in cursor_pages(
            lambda cursor: self._get(("builds",), {"cursor": cursor, "limit": PAGE_LIMIT}), "builds"
        ):
            builds.extend(page.get("builds", []))
        return builds

    def get_build(self, build_id: str) -> dict:
        """GET /v1/builds/{id} -> a build with its full definition."""
        return self._get(("builds", build_id))

    def list_releases(self, build_id: str) -> list[dict]:
        """GET every cursor page from /v1/builds/{id}/releases -> the build's
        releases. A server that predates the version-to-release rename keys the
        list ``versions``, so both spellings parse."""
        releases: list[dict] = []
        pages = cursor_pages(
            lambda cursor: self._get(("builds", build_id, "releases"), {"cursor": cursor, "limit": PAGE_LIMIT}),
            "releases",
        )
        for page in pages:
            releases.extend(page.get("releases") or page.get("versions") or [])
        return releases

    def get_release_logs(self, release_id: str, *, os: str | None = None, gpu: str | None = None) -> dict:
        """GET /v1/releases/{id}/logs -> the build log for one target
        (``os``/``gpu`` select which; the builder picks a target when omitted).
        Returns ``{releaseId, os?, gpu?, log, truncated}``. Uses a larger response
        cap than other reads because a build log can be several MiB."""
        return self._get(("releases", release_id, "logs"), {"os": os, "gpu": gpu}, max_bytes=_MAX_LOG_JSON)

    def delete_build(self, build_id: str) -> None:
        """DELETE /v1/builds/{id} -> soft-delete. Idempotent (204 even when
        already gone); the builder returns 409 while a deployment still runs one of
        its releases."""
        self._send(self.target.url("builds", build_id), method="DELETE", max_bytes=_MAX_JSON)

    def delete_release(self, release_id: str) -> None:
        """DELETE /v1/releases/{id} -> stamp the release deleted, freeing the slot
        it held against the workspace's release limit. Idempotent (204 even when
        already gone); the builder returns 409 while any deployment in the
        workspace still references it, a stopped one included."""
        # Keep the id a single terminal path segment: ``Target.url`` only
        # ``strip('/')``s each part, so an id carrying ``/`` or ``..`` would aim
        # this DELETE, through any normalizing proxy, at a resource the caller
        # never confirmed. Percent-encoding belongs here because only the client
        # knows it is building a URL -- but it is not enough on its own:
        # ``quote(safe="")`` leaves ``.`` and ``..`` untouched, RFC 3986 calling
        # both unreserved, so those two are refused rather than encoded.
        #
        # This is the guarantee, and it holds for whoever calls the method.
        # ``release_delete`` refuses the same ids above its own prompt, and that
        # check is the one a person or an agent actually reads: it names the
        # command's argument and points at `comfy build release ls`. This one
        # only has to keep the sentence above true.
        release_id = (release_id or "").strip()
        if not release_id or set(release_id) == {"."}:
            raise ValueError("release_id must name one release, not an empty or dot-only path segment")
        encoded_id = urllib.parse.quote(release_id, safe="")
        self._send(self.target.url("releases", encoded_id), method="DELETE", max_bytes=_MAX_JSON)

    def validate_build(self, build_id: str) -> dict:
        """POST /v1/builds/{id}/validate -> dry-run resolve the stored
        definition (no build). 200 with a ValidateResult when resolvable; the
        builder returns 400 with the issues when the definition has problems."""
        _, parsed = self._send(
            self.target.url("builds", build_id, "validate"),
            method="POST",
            max_bytes=_MAX_JSON,
        )
        return parsed or {}

    def update_build(
        self,
        build_id: str,
        definition: dict,
        expected_updated_at: str | None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> dict:
        """PATCH /v1/builds/{id} -> replace the stored definition. Returns
        the updated build. ``expected_updated_at`` is the ``updatedAt`` the
        caller last saw (optimistic concurrency); the builder rejects the save with
        409 STALE if it is stale or missing, so pass the value from a fresh GET.

        ``name`` and ``description`` are the server's own existing fields: a
        local spec owns them too, so a sync that could not carry them would leave
        the two copies disagreeing with no way to reconcile."""
        body = {"definition": definition, "expectedUpdatedAt": expected_updated_at}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        _, parsed = self._send(
            self.target.url("builds", build_id),
            method="PATCH",
            body=body,
            max_bytes=_MAX_JSON,
        )
        return parsed or {}

    def get_release_manifest(self, release_id: str) -> dict:
        """GET /v1/releases/{id}/manifest -> the release's models and
        runtime policies."""
        return self._get(("releases", release_id, "manifest"))

    def get_artifact_download(self, artifact_id: str) -> dict:
        """GET /v1/build-artifacts/{id}/download -> ``{downloadUrl, expiresAt?}``."""
        return self._get(("build-artifacts", artifact_id, "download"))

    def list_blobs(self, kind: str | None = None) -> list[dict]:
        """GET /v1/blobs -> the workspace's private uploaded content (summaries),
        optionally filtered by ``kind`` (model | node_zip).

        One page, no walk: this endpoint mints no cursor, so there is nothing to
        follow. Pass the result through
        :func:`comfy_cli.builder_pagination.blob_listing_is_clamped` to tell a
        complete listing from one the server cut off at its page limit."""
        return self._get(("blobs",), {"kind": kind}).get("blobs", [])

    def list_base_images(self) -> list[dict]:
        """GET /v1/base-images -> the curated base images a build may be based on."""
        return self._get(("base-images",)).get("baseImages", [])

    def list_build_targets(self) -> list[dict]:
        """GET /v1/build-targets -> the build targets a release can be cut for."""
        return self._get(("build-targets",)).get("targets", [])

    def list_model_directories(self) -> list[str]:
        """GET /v1/model-directories -> the model directories a model may land in."""
        return self._get(("model-directories",)).get("directories", [])
