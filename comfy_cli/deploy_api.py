from __future__ import annotations

import os
import urllib.error
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

from comfy_cli import credentials
from comfy_cli.deploy_api_errors import (
    DeployAPIError,
    assert_safe_deploy_url,
    bad_request,
    mapped_error,
    transport_error,
)
from comfy_cli.http import request_json
from comfy_cli.target import Target

DEFAULT_DEPLOY_URL: Final = "https://platformapi.comfy.org/deploy"
_MAX_JSON: Final = 5 * 1024 * 1024
_MAX_LOG_JSON: Final = 32 * 1024 * 1024
_MAX_WORKERS: Final = 20
_MAX_IDEMPOTENCY_KEY_BYTES: Final = 255
#: Ceiling on cursor pages one listing may walk. A server that always answers
#: with a fresh cursor would otherwise spin forever on an ever-growing list.
_MAX_PAGES: Final = 1000


class DeployAuthError(DeployAPIError):
    code = "deploy_not_signed_in"

    def __init__(self) -> None:
        super().__init__(self.code, "not signed in — run `comfy cloud login`")


@dataclass(frozen=True, slots=True)
class _Request:
    operation: str
    parts: tuple[str, ...]
    method: str = "GET"
    body: dict | None = None
    params: dict | None = None
    headers: dict[str, str] | None = None
    max_bytes: int = _MAX_JSON


def _resolved_base_url(base_url: str | None) -> str:
    return base_url if base_url is not None else os.getenv("COMFY_DEPLOY_URL") or DEFAULT_DEPLOY_URL


def _base_url_source(base_url: str | None) -> str:
    """Name whichever setting produced the URL, for the refusal message.

    ``DEFAULT_DEPLOY_URL`` is https, so a refusal can only ever be traced to
    one of the two overrides.
    """
    return "the base_url argument" if base_url is not None else "COMFY_DEPLOY_URL"


def _validate_compute_config(compute_config: dict) -> None:
    minimum = compute_config.get("min", 0)
    maximum = compute_config.get("max", 1)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or not 0 <= minimum <= _MAX_WORKERS:
        raise bad_request(f"computeConfig.min must be an integer from 0 to {_MAX_WORKERS}")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= _MAX_WORKERS:
        raise bad_request(f"computeConfig.max must be an integer from 1 to {_MAX_WORKERS}")
    # Only the bounds actually supplied can be compared. `run_scale` merges in
    # whichever of min/max the flags and the stored config provide and omits the
    # rest, so a deployment holding min without max would otherwise see `--min 5`
    # refused against the placeholder max of 1 above — a ceiling nobody set.
    if "min" in compute_config and "max" in compute_config and minimum > maximum:
        raise bad_request("computeConfig.min must not exceed computeConfig.max")


class DeployClient:
    def __init__(self, base_url: str | None, token: str):
        resolved_url = _resolved_base_url(base_url).rstrip("/")
        assert_safe_deploy_url(resolved_url, source=_base_url_source(base_url))
        self.target = Target(kind="cloud", base_url=resolved_url, path_prefix="/v1", auth_token=token)

    @classmethod
    def from_session(cls, base_url: str | None = None) -> DeployClient:
        resolved_url = _resolved_base_url(base_url).rstrip("/")
        assert_safe_deploy_url(resolved_url, source=_base_url_source(base_url))
        session = credentials.get_session(refresh=True)
        if not session or not session.access_token:
            raise DeployAuthError
        return cls(resolved_url, session.access_token)

    def _request(self, request: _Request) -> dict:
        url = self.target.url(*request.parts)
        if request.params:
            query = urllib.parse.urlencode({key: value for key, value in request.params.items() if value is not None})
            if query:
                url = f"{url}?{query}"
        try:
            _, parsed = request_json(
                url,
                self.target,
                method=request.method,
                body=request.body,
                headers=request.headers,
                max_bytes=request.max_bytes,
            )
        except urllib.error.HTTPError as error:
            raise mapped_error(request.operation, error, url) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise transport_error(request.operation, error) from error
        return parsed if isinstance(parsed, dict) else {}

    def _get(self, operation: str, parts: tuple[str, ...], params: dict | None = None) -> dict:
        return self._request(_Request(operation=operation, parts=parts, params=params))

    def _post(self, operation: str, parts: tuple[str, ...], body: dict | None = None) -> dict:
        return self._request(_Request(operation=operation, parts=parts, method="POST", body=body))

    def _patch(self, operation: str, parts: tuple[str, ...], body: dict) -> dict:
        return self._request(_Request(operation=operation, parts=parts, method="PATCH", body=body))

    def _delete(self, operation: str, parts: tuple[str, ...]) -> None:
        self._request(_Request(operation=operation, parts=parts, method="DELETE"))

    def create_deployment(
        self,
        release_id: str,
        compute_config: dict,
        *,
        idempotency_key: str | None = None,
    ) -> dict:
        if not release_id.strip():
            raise bad_request("releaseId is required")
        _validate_compute_config(compute_config)
        if idempotency_key is not None and len(idempotency_key.encode("utf-8")) > _MAX_IDEMPOTENCY_KEY_BYTES:
            raise bad_request(f"Idempotency-Key must be at most {_MAX_IDEMPOTENCY_KEY_BYTES} bytes")
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key is not None else None
        return self._request(
            _Request(
                operation="create",
                parts=("deployments",),
                method="POST",
                body={"releaseId": release_id, "computeConfig": compute_config},
                headers=headers,
            )
        )

    def list_deployments(self, *, status: str | None = None, limit: int | None = None) -> dict:
        return self._get("list", ("deployments",), {"status": status, "limit": limit})

    def iter_deployments(self, *, status: str | None = None, limit: int | None = None) -> Iterator[dict]:
        after: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            page = self._get("list", ("deployments",), {"status": status, "limit": limit, "after": after})
            yield page
            next_cursor = page.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return
            if next_cursor in seen:
                raise DeployAPIError(
                    "deploy_server_error",
                    "the deploy service repeated a pagination cursor",
                    details={"operation": "list", "cursor": next_cursor},
                )
            seen.add(next_cursor)
            after = next_cursor
        raise DeployAPIError(
            "deploy_server_error",
            f"the deploy service paginated past {_MAX_PAGES} pages",
            details={"operation": "list", "pages": _MAX_PAGES},
        )

    def list_all_deployments(self, *, status: str | None = None, limit: int | None = None) -> list[dict]:
        deployments: list[dict] = []
        for page in self.iter_deployments(status=status, limit=limit):
            deployments.extend(page.get("deployments", []))
        return deployments

    def get_deployment(self, deployment_id: str) -> dict:
        return self._get("get", ("deployments", deployment_id))

    def update_deployment(self, deployment_id: str, compute_config: dict) -> dict:
        _validate_compute_config(compute_config)
        return self._patch("scale", ("deployments", deployment_id), {"computeConfig": compute_config})

    def delete_deployment(self, deployment_id: str) -> None:
        self._delete("delete", ("deployments", deployment_id))

    def start_deployment(self, deployment_id: str) -> dict:
        return self._post("start", ("deployments", deployment_id, "start"))

    def stop_deployment(self, deployment_id: str) -> dict:
        return self._post("stop", ("deployments", deployment_id, "stop"))

    def get_deployment_events(self, deployment_id: str) -> dict:
        return self._get("events", ("deployments", deployment_id, "events"))

    def get_deployment_logs(self, deployment_id: str) -> dict:
        return self._request(
            _Request(operation="logs", parts=("deployments", deployment_id, "logs"), max_bytes=_MAX_LOG_JSON)
        )

    def get_compute_catalog(self) -> dict:
        return self._get("compute", ("compute-catalog",))
