"""Trusted-origin validation and contained downloads for deployment outputs."""

from __future__ import annotations

import ipaddress
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.transfer import (
    _AUTH_HEADERS_TO_STRIP,
    _MAX_REDIRECTS,
    _collision_safe_path,
    _stream_http_one,
)
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import build_http_only_opener
from comfy_cli.output.renderer import Renderer

_HOST_SUFFIXES_ENV: Final = "COMFY_DEPLOY_HOST_SUFFIXES"
_STORAGE_ORIGINS_ENV: Final = "COMFY_DEPLOY_STORAGE_ORIGINS"
_DEFAULT_HOST_SUFFIXES: Final = ".run.comfy.app,.stg.run.comfy.app"
_DEFAULT_STORAGE_ORIGINS: Final = "https://storage.googleapis.com"
_LDH_LABEL: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


class DeployEndpointUnknownError(DeployAPIError):
    code = "deploy_endpoint_unknown"

    def __init__(self, message: str, *, details: JsonObject | None = None) -> None:
        super().__init__(self.code, message, details=details)


def _normalized_dns_name(hostname: str) -> str:
    try:
        labels = tuple(label.encode("idna").decode("ascii").lower() for label in hostname.rstrip(".").split("."))
    except UnicodeError as error:
        raise DeployEndpointUnknownError(
            "the deployment service returned an invalid internationalized hostname"
        ) from error
    if len(labels) < 2 or any(not _LDH_LABEL.fullmatch(label) for label in labels):
        raise DeployEndpointUnknownError("the deployment service returned an invalid DNS hostname")
    normalized = ".".join(labels)
    if len(normalized) > 253:
        raise DeployEndpointUnknownError("the deployment service returned an overlong DNS hostname")
    return normalized


def _normalized_host(hostname: str) -> str:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return _normalized_dns_name(hostname)
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _parsed_port(parsed: urllib.parse.SplitResult) -> int | None:
    try:
        return parsed.port
    except ValueError as error:
        raise DeployEndpointUnknownError("the server returned a URL with an invalid port") from error


def _canonical_origin(url: str, *, allow_port: bool) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https" or parsed.hostname is None:
        raise DeployEndpointUnknownError("the server returned a non-HTTPS origin")
    if parsed.username is not None or parsed.password is not None:
        raise DeployEndpointUnknownError("the server returned an origin containing userinfo")
    port = _parsed_port(parsed)
    if not allow_port and port not in {None, 443}:
        raise DeployEndpointUnknownError("the deployment endpoint uses an untrusted explicit port")
    host = _normalized_host(parsed.hostname)
    return f"https://{host}" if port in {None, 443} else f"https://{host}:{port}"


def _configured_values(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name)
    selected = default if raw is None else raw
    values = tuple(item.strip() for item in selected.split(",") if item.strip())
    if not values:
        raise DeployEndpointUnknownError(f"{name} must contain at least one trusted value", details={"variable": name})
    return values


def _configured_host_suffixes() -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in _configured_values(_HOST_SUFFIXES_ENV, _DEFAULT_HOST_SUFFIXES):
        if not raw.startswith("."):
            raise DeployEndpointUnknownError(
                f"{_HOST_SUFFIXES_ENV} contains an invalid DNS suffix",
                details={"variable": _HOST_SUFFIXES_ENV, "suffix": raw},
            )
        normalized.append(f".{_normalized_dns_name(raw[1:])}")
    return tuple(normalized)


def validate_endpoint_origin(deployment_id: str, endpoint_url: str | None) -> str:
    if endpoint_url is None:
        raise DeployEndpointUnknownError("the deployment has no endpointUrl", details={"deployment_id": deployment_id})
    parsed = urllib.parse.urlsplit(endpoint_url)
    origin = _canonical_origin(endpoint_url, allow_port=False)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise DeployEndpointUnknownError("the deployment endpoint must be an origin without a path, query, or fragment")
    normalized_id = _normalized_dns_name(f"{deployment_id}.invalid").removesuffix(".invalid")
    hostname = urllib.parse.urlsplit(origin).hostname
    if hostname is None:
        raise DeployEndpointUnknownError("the deployment endpoint has no hostname")
    trusted = False
    for suffix in _configured_host_suffixes():
        if hostname.endswith(suffix):
            prefix = hostname.removesuffix(suffix)
            trusted = prefix == normalized_id and "." not in prefix
            if trusted:
                break
    if not trusted:
        raise DeployEndpointUnknownError(
            "the server-returned endpointUrl is outside the trusted deployment host suffixes",
            details={"deployment_id": deployment_id, "endpoint_origin": origin, "variable": _HOST_SUFFIXES_ENV},
        )
    return origin


def configured_storage_origins() -> frozenset[str]:
    origins: set[str] = set()
    for raw in _configured_values(_STORAGE_ORIGINS_ENV, _DEFAULT_STORAGE_ORIGINS):
        parsed = urllib.parse.urlsplit(raw)
        if parsed.path or parsed.query or parsed.fragment:
            raise DeployEndpointUnknownError(
                f"{_STORAGE_ORIGINS_ENV} accepts exact origins without paths, queries, or fragments",
                details={"variable": _STORAGE_ORIGINS_ENV, "origin": raw},
            )
        origins.add(_canonical_origin(raw, allow_port=True))
    return frozenset(origins)


def validate_output_url(url: str, endpoint_origin: str, storage_origins: frozenset[str]) -> str:
    origin = _canonical_origin(url, allow_port=True)
    if origin != endpoint_origin and origin not in storage_origins:
        raise DeployEndpointUnknownError(
            f"output origin {origin} is not trusted; extend {_STORAGE_ORIGINS_ENV} to allow it",
            details={"origin": origin, "variable": _STORAGE_ORIGINS_ENV},
        )
    return origin


def resolve_endpoint_link(endpoint_origin: str, reference: str) -> str:
    resolved = urllib.parse.urljoin(f"{endpoint_origin}/", reference)
    if _canonical_origin(resolved, allow_port=False) != endpoint_origin:
        raise DeployEndpointUnknownError(
            "a job follow-up link points outside the deployment endpoint origin",
            details={"endpoint_origin": endpoint_origin, "rejected_url": resolved},
        )
    return resolved


class DeploymentRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = _MAX_REDIRECTS

    def __init__(self, allowed_origins: frozenset[str]) -> None:
        super().__init__()
        self._allowed_origins = allowed_origins

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            origin = _canonical_origin(newurl, allow_port=True)
            if origin not in self._allowed_origins:
                raise DeployEndpointUnknownError(
                    f"redirect origin {origin} is not trusted; extend {_STORAGE_ORIGINS_ENV} to allow it"
                )
        except DeployEndpointUnknownError as error:
            raise urllib.error.URLError(str(error)) from error
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        for source in (redirected.headers, redirected.unredirected_hdrs):
            for key in tuple(source):
                if key.lower() in _AUTH_HEADERS_TO_STRIP:
                    del source[key]
        return redirected


def safe_output_path(output_dir: Path, original_name: str) -> Path:
    posix_name = PurePosixPath(original_name)
    windows_name = PureWindowsPath(original_name)
    if (
        not original_name
        or posix_name.is_absolute()
        or windows_name.is_absolute()
        or windows_name.drive
        or ".." in posix_name.parts
        or ".." in windows_name.parts
    ):
        raise DeployAPIError(
            "deploy_server_error",
            "the data-plane job returned an unsafe output name",
            details={"name": original_name},
        )
    basename = PureWindowsPath(posix_name.name).name
    if not basename:
        raise DeployAPIError("deploy_server_error", "the data-plane job returned an empty output name")
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / basename).resolve()
    if not candidate.is_relative_to(root):
        raise DeployAPIError(
            "deploy_server_error",
            "the data-plane output path escapes the output directory",
            details={"name": original_name},
        )
    return _collision_safe_path(candidate)


@dataclass(frozen=True, slots=True)
class OutputDownloadRequest:
    outputs: tuple[JsonObject, ...]
    endpoint_origin: str
    token: str
    output_dir: Path


def _output_string(output: JsonObject, field: str) -> str:
    value = output.get(field)
    if not isinstance(value, str) or not value:
        raise DeployAPIError(
            "deploy_server_error",
            f"the data-plane job output has no {field}",
            details={"field": field},
        )
    return value


def download_job_outputs(request: OutputDownloadRequest, renderer: Renderer) -> list[JsonObject]:
    storage_origins = configured_storage_origins()
    allowed_origins = frozenset({request.endpoint_origin, *storage_origins})
    opener = build_http_only_opener(DeploymentRedirectHandler(allowed_origins))
    downloaded: list[JsonObject] = []
    for index, output in enumerate(request.outputs):
        url = _output_string(output, "url")
        initial_origin = validate_output_url(url, request.endpoint_origin, storage_origins)
        name = _output_string(output, "name")
        # Every field the manifest row needs is proved present before the first
        # byte lands. Validating afterwards left files in `outputs/` that the
        # envelope then never accounted for, followed by exit 1 and no manifest.
        node_id = _output_string(output, "node_id")
        output_type = _output_string(output, "type")
        output_id = _output_string(output, "id")
        destination = safe_output_path(request.output_dir, name)
        auth_headers = {"Authorization": f"Bearer {request.token}"} if initial_origin == request.endpoint_origin else {}
        # transfer._assert_download_url checks only the scheme; the allowlist above
        # and DeploymentRedirectHandler supply the destination guard it lacks.
        _stream_http_one(url, index, destination, auth_headers, renderer, opener=opener)
        downloaded.append(
            {
                "node_id": node_id,
                "name": name,
                "type": output_type,
                "id": output_id,
                "path": str(request.output_dir / destination.name),
            }
        )
    return downloaded
