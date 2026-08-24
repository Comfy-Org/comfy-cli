"""Wire-contract proof for ``BuilderClient``.

Todo 13 renamed eleven client methods from distribution/version vocabulary to
build/release vocabulary. That rename is **CLI-side only**: every REST path,
HTTP verb, request body and response-size cap must be byte-identical to what
the pre-rename method produced. Todo 14 deliberately extends ``update_build``
with the server's existing ``name`` and ``description`` fields; that row pins
the extended body while every other row remains the rename-equivalence proof.

The oracle here is deliberately *independent of the code under test*: every
expected verb / URL / body / cap below is transcribed from the pre-rename
implementation (``git show HEAD:comfy_cli/distribution_api.py``, the file this
module was renamed from in Todo 3), not recomputed from today's source. So a
method that quietly moved to a different path, changed a JSON field name, or
lost its raised log cap fails here even though its Python name is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from comfy_cli.builder_api import BuilderClient

# Transcribed from the pre-rename source, NOT imported from builder_api — an
# imported constant would make the cap assertions recompute themselves.
_MAX_JSON = 5 * 1024 * 1024
_MAX_LOG_JSON = 32 * 1024 * 1024

# The trailing slash is deliberate: the client rstrips it, and the pre-rename
# client did too, so every expected URL below carries exactly one.
_BASE_URL = "https://builder.test/"
_BASE = "https://builder.test"


class _Recorder:
    """Stands in for the shared ``request_json`` seam, recording each request.

    Answers with one superset envelope so every method's own response parsing
    still runs (``create_build`` reads ``id``, ``create_release`` reads
    ``buildVersionId``/``statusUrl``, the list reads take their own key).
    """

    _ENVELOPE = {
        "id": "d1",
        "buildVersionId": "v1",
        "statusUrl": "https://status.example/v1",
        "builds": [],
        "versions": [],
        "blobs": [],
        "results": [],
        "downloadUrl": "https://dl.example/a1",
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        self.calls.append({"method": method, "url": url, "body": body, "max_bytes": max_bytes, "target": target})
        return 200, dict(self._ENVELOPE)


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr("comfy_cli.builder_api.request_json", rec)
    return rec


@dataclass(frozen=True)
class Wire:
    """One method call and its exact current request contract."""

    new_name: str
    legacy_name: str | None
    http_method: str
    url: str
    body: dict | None = None
    max_bytes: int = _MAX_JSON
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


# Every row's expected verb/url/body/cap comes from the pre-rename method of the
# same index — see the module docstring. `legacy_name` is the identifier that
# produced it, and is asserted GONE so this table cannot pass un-renamed code.
_WIRE = [
    Wire(
        new_name="create_build",
        legacy_name="create_distribution",
        args=("n", {"models": [], "customNodes": []}),
        http_method="POST",
        url=f"{_BASE}/v1/builds",
        body={"name": "n", "definition": {"models": [], "customNodes": []}},
    ),
    Wire(
        new_name="create_build",
        legacy_name="create_distribution",
        args=("n", {"models": []}, "a description"),
        http_method="POST",
        url=f"{_BASE}/v1/builds",
        body={"name": "n", "definition": {"models": []}, "description": "a description"},
    ),
    Wire(
        new_name="create_release",
        legacy_name="cut_version",
        args=("d1", [{"os": "linux", "gpu": "nvidia"}]),
        http_method="POST",
        url=f"{_BASE}/v1/builds/d1/versions",
        body={"targets": [{"os": "linux", "gpu": "nvidia"}]},
    ),
    Wire(
        new_name="get_release",
        legacy_name="get_version",
        args=("v1",),
        http_method="GET",
        url=f"{_BASE}/v1/build-versions/v1",
    ),
    Wire(
        new_name="list_builds",
        legacy_name="list_distributions",
        http_method="GET",
        url=f"{_BASE}/v1/builds",
    ),
    Wire(
        new_name="get_build",
        legacy_name="get_distribution",
        args=("d1",),
        http_method="GET",
        url=f"{_BASE}/v1/builds/d1",
    ),
    Wire(
        new_name="list_releases",
        legacy_name="list_distribution_versions",
        args=("d1",),
        http_method="GET",
        url=f"{_BASE}/v1/builds/d1/versions",
    ),
    Wire(
        new_name="get_release_logs",
        legacy_name="get_version_logs",
        args=("v1",),
        kwargs={"os": "linux", "gpu": "nvidia"},
        http_method="GET",
        url=f"{_BASE}/v1/build-versions/v1/logs?os=linux&gpu=nvidia",
        max_bytes=_MAX_LOG_JSON,
    ),
    Wire(
        # No selector: the pre-rename `_get` dropped empty params, leaving a bare
        # /logs URL with no `?`.
        new_name="get_release_logs",
        legacy_name="get_version_logs",
        args=("v1",),
        http_method="GET",
        url=f"{_BASE}/v1/build-versions/v1/logs",
        max_bytes=_MAX_LOG_JSON,
    ),
    Wire(
        new_name="delete_build",
        legacy_name="delete_distribution",
        args=("d1",),
        http_method="DELETE",
        url=f"{_BASE}/v1/builds/d1",
    ),
    Wire(
        # The pre-rename validate sent no body at all — only the verb and path.
        new_name="validate_build",
        legacy_name="validate_distribution",
        args=("d1",),
        http_method="POST",
        url=f"{_BASE}/v1/builds/d1/validate",
    ),
    Wire(
        new_name="update_build",
        legacy_name="update_distribution",
        args=("d1", {"models": []}, "2026-08-01T00:00:00Z"),
        kwargs={"name": "Renamed", "description": "Synced description"},
        http_method="PATCH",
        url=f"{_BASE}/v1/builds/d1",
        body={
            "definition": {"models": []},
            "expectedUpdatedAt": "2026-08-01T00:00:00Z",
            "name": "Renamed",
            "description": "Synced description",
        },
    ),
    Wire(
        new_name="get_release_manifest",
        legacy_name="get_version_manifest",
        args=("v1",),
        http_method="GET",
        url=f"{_BASE}/v1/build-versions/v1/manifest",
    ),
    Wire(
        # NOT renamed (its command is gone, the API stays available) — pinned so a
        # future sweep that renames it by association is caught here.
        new_name="get_artifact_download",
        legacy_name=None,
        args=("a1",),
        http_method="GET",
        url=f"{_BASE}/v1/build-artifacts/a1/download",
    ),
]

_IDS = [f"{w.new_name}{'-' + '-'.join(w.kwargs) if w.kwargs else ''}-{i}" for i, w in enumerate(_WIRE)]


@pytest.mark.parametrize("wire", _WIRE, ids=_IDS)
def test_renamed_method_emits_the_pre_rename_request(recorder, wire: Wire):
    """Given a renamed client method, When it is called, Then the request it hands
    the HTTP layer is byte-identical to the pre-rename method's."""
    client = BuilderClient(_BASE_URL, "jwt-token")

    getattr(client, wire.new_name)(*wire.args, **wire.kwargs)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["method"] == wire.http_method
    assert call["url"] == wire.url
    assert call["body"] == wire.body
    assert call["max_bytes"] == wire.max_bytes


@pytest.mark.parametrize("wire", _WIRE, ids=_IDS)
def test_request_rides_the_clients_own_cloud_target(recorder, wire: Wire):
    """The recording seam also captures the Target that built the URL: it must
    stay the client's own /v1-prefixed cloud target carrying the Bearer."""
    client = BuilderClient(_BASE_URL, "jwt-token")

    getattr(client, wire.new_name)(*wire.args, **wire.kwargs)

    target = recorder.calls[0]["target"]
    assert target is client.target
    assert target.is_cloud and target.path_prefix == "/v1"
    assert target.base_url == _BASE and target.auth_token == "jwt-token"


@pytest.mark.parametrize("legacy_name", sorted({w.legacy_name for w in _WIRE if w.legacy_name}))
def test_the_legacy_method_name_is_gone(legacy_name: str):
    """Without this the wire table above would pass just as happily against the
    un-renamed client, since it only ever calls the NEW names."""
    assert not hasattr(BuilderClient, legacy_name)


def test_no_client_method_keeps_distribution_or_version_vocabulary():
    stale = [
        name
        for name in vars(BuilderClient)
        if not name.startswith("_") and ("distribution" in name or "version" in name)
    ]
    assert stale == []


def test_get_artifact_download_survives_the_rename():
    """Its command was removed in Todo 3, but the underlying API stays available."""
    assert callable(BuilderClient.get_artifact_download)


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(None, id="explicit-none"),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param({"os": "linux", "gpu": "nvidia"}, id="single-mapping-not-a-list"),
        pytest.param("linux/nvidia", id="string-not-a-list"),
    ],
)
def test_create_release_refuses_a_missing_or_empty_target_list(recorder, targets):
    """An implicit target spends build minutes nobody asked for, so the refusal
    happens locally — before a single request leaves the client."""
    client = BuilderClient(_BASE_URL, "jwt-token")

    with pytest.raises(ValueError, match="non-empty list of targets"):
        client.create_release("d1", targets)

    assert recorder.calls == []


def test_create_release_refuses_when_targets_is_omitted_entirely(recorder):
    client = BuilderClient(_BASE_URL, "jwt-token")

    with pytest.raises(ValueError, match="non-empty list of targets"):
        client.create_release("d1")

    assert recorder.calls == []
