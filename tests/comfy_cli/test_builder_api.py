"""Wire-contract proof for ``BuilderClient``.

Todo 13 renamed eleven client methods from distribution/version vocabulary to
build/release vocabulary. That rename is **CLI-side only**: every REST path,
HTTP verb, request body and response-size cap must be byte-identical to what
the pre-rename method produced — with one deliberate exception. PR #770 moved
the builder's own release endpoints server-side (``POST /v1/builds/{id}/versions``
-> ``.../releases``; ``GET /v1/build-versions/{id}``, ``/logs`` and ``/manifest``
-> ``GET /v1/releases/{id}...``), so the five release rows below are transcribed
from that migration rather than from the pre-rename source. Todo 14 deliberately
extends ``update_build`` with the server's existing ``name`` and ``description``
fields; that row pins the extended body while every other row remains the
rename-equivalence proof.

The oracle here is deliberately *independent of the code under test*: every
expected verb / URL / body / cap below is transcribed from the pre-rename
implementation (``git show HEAD:comfy_cli/distribution_api.py``, the file this
module was renamed from in Todo 3) — or, for the release rows, from #770's
post-migration paths — not recomputed from today's source. So a method that
quietly moved to a different path, changed a JSON field name, or lost its raised
log cap fails here even though its Python name is right.
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
    ``releaseId``/``statusUrl``, the list reads take their own key).

    ``releaseId``/``releases`` are the post-#770 spellings and ``buildVersionId``/
    ``versions`` the pre-rename ones. Both are present so the superset stays
    honest, but the dedicated fallback tests below are what pin the older
    spellings — this envelope alone would let a client that ignored them pass.
    """

    _ENVELOPE = {
        "id": "d1",
        "releaseId": "v1",
        "buildVersionId": "v1",
        "statusUrl": "https://status.example/v1",
        "builds": [],
        "releases": [],
        "versions": [],
        "blobs": [],
        "results": [],
        "downloadUrl": "https://dl.example/a1",
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, target, *, method="GET", body=None, timeout=30.0, max_bytes):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "max_bytes": max_bytes,
                "target": target,
                "timeout": timeout,
            }
        )
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
        url=f"{_BASE}/v1/builds/d1/releases",
        body={"targets": [{"os": "linux", "gpu": "nvidia"}]},
    ),
    Wire(
        new_name="get_release",
        legacy_name="get_version",
        args=("v1",),
        http_method="GET",
        url=f"{_BASE}/v1/releases/v1",
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
        url=f"{_BASE}/v1/builds/d1/releases",
    ),
    Wire(
        new_name="get_release_logs",
        legacy_name="get_version_logs",
        args=("v1",),
        kwargs={"os": "linux", "gpu": "nvidia"},
        http_method="GET",
        url=f"{_BASE}/v1/releases/v1/logs?os=linux&gpu=nvidia",
        max_bytes=_MAX_LOG_JSON,
    ),
    Wire(
        # No selector: the pre-rename `_get` dropped empty params, leaving a bare
        # /logs URL with no `?`.
        new_name="get_release_logs",
        legacy_name="get_version_logs",
        args=("v1",),
        http_method="GET",
        url=f"{_BASE}/v1/releases/v1/logs",
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
        url=f"{_BASE}/v1/releases/v1/manifest",
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


# --- reading a workflow -------------------------------------------------------
#
# `resolve_workflow` is the sibling of `resolve_snapshot`, not a rename, so it
# is pinned here rather than in the equivalence table above.


def test_resolve_workflow_posts_the_graph_to_the_resolve_endpoint(recorder):
    """`build init/update --from-workflow` reads a graph without writing a build
    row, so it must ride /v1/workflows/resolve — never /v1/builds/from-workflow,
    which mints a build the local spec has nowhere to put."""
    client = BuilderClient(_BASE_URL, "jwt-token")
    graph = {"nodes": [{"type": "KSampler"}]}

    client.resolve_workflow(graph)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == f"{_BASE}/v1/workflows/resolve"
    assert call["body"] == {"workflow": graph}
    assert call["max_bytes"] == _MAX_JSON


def test_resolve_workflow_outlasts_the_registry_sweep(recorder):
    """The builder looks every distinct node class up in the registry behind a
    budget of its own, so the shared POST default would cut a large graph off
    mid-import. Pinned against a sibling POST so a raised *default* — which would
    silently relax every other call — cannot satisfy this."""
    client = BuilderClient(_BASE_URL, "jwt-token")

    client.resolve_workflow({"nodes": []})
    client.resolve_snapshot({"comfyui": "x"})

    workflow_timeout, snapshot_timeout = (call["timeout"] for call in recorder.calls)
    assert workflow_timeout == 90.0
    assert snapshot_timeout == 30.0


# --- paged list reads ---------------------------------------------------------
#
# The builder pages both list endpoints. Taking only the first page loses the
# tail of any workspace or build past one page, silently and without an error,
# so both reads follow ``nextCursor`` to exhaustion.


@pytest.mark.parametrize(
    ("method", "args", "path", "key"),
    [
        pytest.param("list_builds", (), "/v1/builds", "builds", id="list_builds"),
        pytest.param("list_releases", ("d1",), "/v1/builds/d1/releases", "releases", id="list_releases"),
    ],
)
def test_a_paged_list_read_follows_the_cursor_to_the_end(monkeypatch, method, args, path, key):
    pages = [
        {key: [{"id": "a"}], "nextCursor": "c1"},
        {key: [{"id": "b"}], "nextCursor": "c2"},
        {key: [{"id": "c"}]},
    ]
    seen: list[str] = []

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        assert url.startswith(f"{_BASE}{path}")
        seen.append(url)
        return 200, pages[len(seen) - 1]

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    client = BuilderClient(_BASE_URL, "jwt-token")

    assert getattr(client, method)(*args) == [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert seen == [f"{_BASE}{path}", f"{_BASE}{path}?cursor=c1", f"{_BASE}{path}?cursor=c2"]


# --- older-generation builder fallbacks (pre-#770 response spellings) ---------
#
# #770 renamed the builder's own response keys along with its paths. The client
# reads the new spelling first and the old one second, so a builder that has not
# been upgraded yet still works. The superset envelope in ``_Recorder`` carries
# both keys and therefore cannot prove the second half — these two do.


def test_create_release_falls_back_to_buildversionid(monkeypatch):
    """A not-yet-upgraded builder answers the cut with ``buildVersionId`` instead
    of ``releaseId``; the client still parses the id out of it."""

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        assert url == f"{_BASE}/v1/builds/d1/releases"
        return 202, {"buildVersionId": "v1", "statusUrl": "https://s"}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    client = BuilderClient(_BASE_URL, "jwt-token")

    assert client.create_release("d1", [{"os": "linux", "gpu": "nvidia"}]) == ("v1", "https://s")


def test_list_releases_falls_back_to_the_versions_key(monkeypatch):
    """The same generation gap on the list read: an older builder keys the page
    ``versions``. A single page (no ``nextCursor``) keeps this pinned on the key
    rather than on the cursor loop."""

    def fake_request_json(url, target, *, method="GET", body=None, max_bytes, timeout=30.0):
        assert url.startswith(f"{_BASE}/v1/builds/d1/releases")
        return 200, {"versions": [{"id": "v1"}]}

    monkeypatch.setattr("comfy_cli.builder_api.request_json", fake_request_json)
    client = BuilderClient(_BASE_URL, "jwt-token")

    assert client.list_releases("d1") == [{"id": "v1"}]
