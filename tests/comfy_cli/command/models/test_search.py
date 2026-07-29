"""Tests for ``comfy models`` — live discovery against /api/assets + /api/experiment/models.

All HTTP is mocked: tests own a small set of fixture payloads modeled on real
cloud responses. The asset fixtures intentionally exercise both metadata bags
(``user_metadata`` and ``metadata``), the tag conventions (``models`` +
type-tag), and the sparse-field pattern (``base_model`` populated on only some
entries).
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command.models import search as search_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(search_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    return json.loads(captured.strip().splitlines()[-1])


def _fake_resp(body: bytes, status: int = 200):
    """Build a minimal urlopen-compatible response object."""

    class _Resp:
        def __init__(self):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n: int | None = None):
            return body if n is None else body[:n]

    return _Resp()


# Cloud /api/experiment/models — list-of-dicts shape.
_CLOUD_FOLDERS = [
    {"name": "checkpoints", "folders": ["checkpoints"]},
    {"name": "loras", "folders": ["loras"]},
    {"name": "vae", "folders": ["vae"]},
]

# Local /models — flat-string-list shape (older ComfyUI servers also serve dicts;
# both shapes are accepted by the normalizer).
_LOCAL_FOLDERS = ["checkpoints", "loras", "vae"]


_CLOUD_FILES_LORAS = [
    {"name": "wan2.2_t2v_lightx2v.safetensors", "pathIndex": 0},
    {"name": "flux1-redux-dev.safetensors", "pathIndex": 0},
    {"name": "z-image-turbo-rank64.safetensors", "pathIndex": 0},
]


# A local install as BE-4733 found it: the interesting models (flux, ltx) live
# OUTSIDE `checkpoints`, spread across diffusion_models / loras / vae.
_LOCAL_SEARCH_FOLDERS = ["checkpoints", "diffusion_models", "loras", "vae"]

_LOCAL_FILES_BY_FOLDER: dict[str, list[dict[str, Any]]] = {
    "checkpoints": [{"name": "sd_xl_base_1.0.safetensors", "pathIndex": 0}],
    "diffusion_models": [
        {"name": "flux1-dev.safetensors", "pathIndex": 0},
        {"name": "ltx-video-2b-v0.9.safetensors", "pathIndex": 0},
        {"name": "ltxv-13b-0.9.7-dev.safetensors", "pathIndex": 0},
    ],
    "loras": [{"name": "ltx-lora-detail.safetensors", "pathIndex": 0}],
    "vae": [{"name": "ltx-vae.safetensors", "pathIndex": 0}],
}


def _local_routes() -> dict[str, Any]:
    """Routes for a local `models search`: the per-folder listings, then `/models`.

    Order matters — `_patch_urlopen` matches URL substrings first-wins, and the
    bare `/models` needle would otherwise swallow every `/models/<folder>` hit.
    """
    routes: dict[str, Any] = {f"127.0.0.1:8188/models/{f}": files for f, files in _LOCAL_FILES_BY_FOLDER.items()}
    routes["127.0.0.1:8188/models"] = _LOCAL_SEARCH_FOLDERS
    return routes


_ASSETS_RESPONSE = {
    "assets": [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "name": "wan2.2_t2v_lightx2v.safetensors",
            "display_name": "Wan 2.2 LightX2V",
            "size": 295_146_208,
            "tags": ["models", "loras"],
            "user_metadata": {"filename": "wan2.2_t2v_lightx2v.safetensors"},
            "metadata": {
                "repo_url": "https://huggingface.co/example/wan",
                "preview_url": "https://example.com/preview.webp",
                # base_model deliberately omitted — exercises the sparse path.
            },
            "preview_url": "https://example.com/preview.webp",
            "is_immutable": True,
            "created_at": "2026-05-10T00:00:00Z",
            "updated_at": "2026-05-10T00:00:00Z",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "name": "flux1-redux-dev.safetensors",
            "display_name": "Flux Redux",
            "size": 800_000_000,
            "tags": ["models", "style_models"],
            "user_metadata": {},
            "metadata": {
                "base_model": "Flux.1 D",
                "repo_url": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev",
                "trained_words": ["redux", "blend"],
            },
            "preview_url": None,
            "is_immutable": False,
            "created_at": "2026-05-11T00:00:00Z",
            "updated_at": "2026-05-11T00:00:00Z",
        },
    ],
    "total": 2,
    "has_more": False,
}


@pytest.fixture
def cloud_target(monkeypatch: pytest.MonkeyPatch):
    """Pin ``resolve_target(where='cloud')`` to a known cloud target with an API key."""
    from comfy_cli.target import Target

    fake = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-api-key",
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    monkeypatch.setattr("comfy_cli.command.models.search.resolve_target", lambda **kw: fake, raising=False)
    return fake


@pytest.fixture
def local_target(monkeypatch: pytest.MonkeyPatch):
    from comfy_cli.target import Target

    fake = Target(
        kind="local",
        base_url="http://127.0.0.1:8188",
        path_prefix="",
        history_path="history",
        host="127.0.0.1",
        port=8188,
    )
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: fake)
    monkeypatch.setattr("comfy_cli.command.models.search.resolve_target", lambda **kw: fake, raising=False)
    return fake


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, routes: dict[str, Any]):
    """Wire urlopen to a URL→body lookup. Body is JSON-encoded.

    Substring matching: the first registered URL substring that matches wins.
    Unknown URLs raise so we never silently pass on a typo'd path.
    """
    calls = []

    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append({"url": url, "headers": dict(req.headers) if hasattr(req, "headers") else {}})
        for needle, payload in routes.items():
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                body = json.dumps(payload).encode()
                return _fake_resp(body)
        raise AssertionError(f"unexpected URL hit by mock: {url}")

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    return calls


# ---------------------------------------------------------------------------
# list-folders
# ---------------------------------------------------------------------------


class TestListFolders:
    def test_cloud_happy_path(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/experiment/models": _CLOUD_FOLDERS})
        env = _run(["list-folders", "--where", "cloud"], capsys)
        assert env["ok"] is True, env
        assert env["data"]["mode"] == "cloud"
        assert env["data"]["count"] == 3
        names = [f["name"] for f in env["data"]["folders"]]
        assert names == ["checkpoints", "loras", "vae"]
        # Auth header is set on cloud.
        assert any(h.get("X-api-key") or h.get("X-Api-Key") for h in [c["headers"] for c in calls])

    def test_local_happy_path(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"127.0.0.1:8188/models": _LOCAL_FOLDERS})
        env = _run(["list-folders", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["mode"] == "local"
        # The string-list shape normalizes to [{name, subfolders=[]}].
        assert env["data"]["folders"][0] == {"name": "checkpoints", "subfolders": []}

    def test_cloud_http_error_decodes_body(self, cloud_target, monkeypatch, capsys):
        # Shared `_emit_http_error` path: cloud code + truncated/decoded body in details.
        err = urllib.error.HTTPError("https://x", 502, "Bad Gateway", {}, io.BytesIO(b'{"error": "upstream"}'))
        _patch_urlopen(monkeypatch, {"/api/experiment/models": err})
        env = _run(["list-folders", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert env["error"]["details"]["status"] == 502
        assert env["error"]["details"]["body"] == '{"error": "upstream"}'

    def test_local_http_error_uses_server_not_running(self, local_target, monkeypatch, capsys):
        # Same shared path, local branch: code flips to server_not_running.
        err = urllib.error.HTTPError("http://x", 500, "Server Error", {}, io.BytesIO(b"boom"))
        _patch_urlopen(monkeypatch, {"127.0.0.1:8188/models": err})
        env = _run(["list-folders", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"
        assert env["error"]["details"]["status"] == 500
        assert env["error"]["details"]["body"] == "boom"


# ---------------------------------------------------------------------------
# list-folder
# ---------------------------------------------------------------------------


class TestListFolder:
    def test_cloud_lists_files(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/api/experiment/models/loras": _CLOUD_FILES_LORAS})
        env = _run(["list-folder", "loras", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["folder"] == "loras"
        assert env["data"]["total"] == 3
        names = [f["name"] for f in env["data"]["files"]]
        assert "wan2.2_t2v_lightx2v.safetensors" in names

    def test_limit_caps_results(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/api/experiment/models/loras": _CLOUD_FILES_LORAS})
        env = _run(["list-folder", "loras", "--where", "cloud", "--limit", "1"], capsys)
        assert env["data"]["total"] == 3
        assert env["data"]["shown"] == 1
        assert len(env["data"]["files"]) == 1

    def test_404_surfaces_folder_not_found(self, cloud_target, monkeypatch, capsys):
        err = urllib.error.HTTPError("https://x/folder", 404, "Not Found", {}, io.BytesIO(b'{"error": "nope"}'))
        _patch_urlopen(monkeypatch, {"/api/experiment/models/ghost": err})
        env = _run(["list-folder", "ghost", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "folder_not_found"

    def test_folder_name_with_space_is_accepted_and_encoded(self, local_target, monkeypatch, capsys):
        """A user-configured folder like `my loras` lists, matching what `search --where local` finds."""
        routes = {"127.0.0.1:8188/models/my%20loras": [{"name": "ltx-custom.safetensors", "pathIndex": 0}]}
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["list-folder", "my loras", "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert [c["url"] for c in calls] == ["http://127.0.0.1:8188/models/my%20loras"]
        # The payload echoes the decoded name the user typed, not the wire form.
        assert env["data"]["folder"] == "my loras"
        assert [f["name"] for f in env["data"]["files"]] == ["ltx-custom.safetensors"]

    @pytest.mark.parametrize(
        "folder",
        [
            "SDXL (base)",  # parentheses
            "モデル",  # non-ASCII
            "_hidden",  # leading underscore — rejected by the old strict regex
            "a?b#c",  # URL-significant characters, neutralized by percent-encoding
        ],
    )
    def test_non_traversal_folder_names_are_accepted(self, local_target, monkeypatch, capsys, folder):
        segment = urllib.parse.quote(folder, safe="")
        calls = _patch_urlopen(monkeypatch, {f"127.0.0.1:8188/models/{segment}": []})
        env = _run(["list-folder", folder, "--where", "local"], capsys)
        assert env["ok"] is True, env
        # Every character that could re-shape the request is encoded away.
        assert [c["url"] for c in calls] == [f"http://127.0.0.1:8188/models/{segment}"]
        assert env["data"]["folder"] == folder

    @pytest.mark.parametrize("folder", ["../../etc", "..", "a/b", "a\\b", ""])
    def test_traversal_shapes_still_rejected(self, local_target, monkeypatch, capsys, folder):
        """Relaxing the charset must not relax the traversal guard — and no request is issued."""
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["list-folder", folder, "--where", "local"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    @pytest.mark.parametrize("folder", ["%2e%2e%2fetc", "%2E%2E%2Fetc", "a%00b", "a\r\nX-Evil: 1"])
    def test_encoded_traversal_and_injection_shapes_are_neutralized(self, local_target, monkeypatch, capsys, folder):
        """`%` and control characters are no longer charset-rejected, so encoding must defuse them.

        These are the shapes the old strict regex blocked incidentally. The
        traversal guard alone lets them through — `..` and `/` are not
        *literally* present — so the percent-encoding in `list_folder_cmd` is
        what keeps them inside one path segment: `%` becomes `%25`, so an
        encoded `../` can never be decoded back into one by the server.
        """
        segment = urllib.parse.quote(folder, safe="")
        calls = _patch_urlopen(monkeypatch, {f"127.0.0.1:8188/models/{segment}": []})
        env = _run(["list-folder", folder, "--where", "local"], capsys)
        assert env["ok"] is True, env
        url = calls[0]["url"]
        assert url == f"http://127.0.0.1:8188/models/{segment}"
        # Exactly one layer of encoding: the server decodes back to the literal
        # folder name the user asked for, never to a traversal or a new header.
        assert urllib.parse.unquote(segment) == folder
        # And nothing escaped the segment on the wire.
        assert url.count("/") == 4  # http:// + /127.0.0.1:8188 + /models + /<segment>
        assert not any(c in url for c in "\r\n\x00")

    @pytest.mark.parametrize("folder", ["model..v2", "my..folder", "..hidden", "trailing.."])
    def test_dots_inside_a_name_are_not_traversal(self, local_target, monkeypatch, capsys, folder):
        """Only the exact segments `.`/`..` are dot-segments; `..` mid-name is an ordinary name.

        A substring `".." not in value` guard silently skipped these on the
        `search --where local` walk and rejected them outright here, even though
        `/models/model..v2` resolves to exactly that folder — no resolver rewrites it.
        """
        segment = urllib.parse.quote(folder, safe="")
        calls = _patch_urlopen(monkeypatch, {f"127.0.0.1:8188/models/{segment}": []})
        env = _run(["list-folder", folder, "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert [c["url"] for c in calls] == [f"http://127.0.0.1:8188/models/{segment}"]
        assert env["data"]["folder"] == folder

    @pytest.mark.parametrize("folder", [".", ".."])
    def test_bare_dot_segments_are_rejected(self, local_target, monkeypatch, capsys, folder):
        """`.` and `..` are rewritten by a URL resolver, so they can't stay one segment.

        `quote(..., safe="")` leaves `.` untouched, so a bare `.` would reach the
        server as `/models/.` and normalize back to the `/models` *collection* —
        rendering the folder list as if it were a file listing.
        """
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["list-folder", folder, "--where", "local"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_undecodable_argv_bytes_error_cleanly(self, local_target, monkeypatch, capsys):
        """A non-UTF-8 filename from argv must be `invalid_argument`, not a traceback.

        Python decodes undecodable argv bytes into lone surrogates (PEP 383
        `surrogateescape`). `quote(..., safe="")` raises `UnicodeEncodeError` on
        those, and it runs *before* `list_folder_cmd`'s try block — so without a
        guard the CLI dies with an uncaught stack trace.
        """
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["list-folder", "a\udcffb", "--where", "local"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_cloud_returns_enriched_rows(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/assets": _ASSETS_RESPONSE})
        env = _run(["search", "--text", "flux", "--limit", "5", "--where", "cloud"], capsys)
        assert env["ok"] is True
        assert env["data"]["mode"] == "cloud"
        assert env["data"]["total"] == 2
        rows = env["data"]["rows"]
        # The `metadata.base_model` field was populated for flux, sparse for wan.
        flux = next(r for r in rows if r["name"] == "flux1-redux-dev.safetensors")
        wan = next(r for r in rows if r["name"] == "wan2.2_t2v_lightx2v.safetensors")
        assert flux["base_model"] == "Flux.1 D"
        assert wan["base_model"] is None  # sparse — mirrors real cloud data
        # Source URL falls back from repo_url
        assert flux["source_url"].startswith("https://huggingface.co/")
        # The type is derived from the first non-"models" tag.
        assert wan["type"] == "loras"
        assert flux["type"] == "style_models"
        # The query string carried name_contains + include_tags=models.
        url = calls[0]["url"]
        assert "name_contains=flux" in url
        # include_tags is comma-separated (URL-encoded) per the cloud OpenAPI
        # spec — exploded form is rejected by /api/assets with HTTP 400.
        assert "include_tags=models" in url

    def test_cloud_type_filter_appends_tag(self, cloud_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {"/api/assets": _ASSETS_RESPONSE})
        _run(["search", "--type", "lora", "--where", "cloud"], capsys)
        url = calls[0]["url"]
        # Comma-separated form: include_tags=models,loras (URL-encoded as %2C).
        assert "include_tags=models%2Cloras" in url

    def test_local_falls_back_to_folder_listing(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "sd_xl", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["mode"] == "local"
        rows = env["data"]["rows"]
        assert len(rows) == 1
        assert rows[0]["name"] == "sd_xl_base_1.0.safetensors"
        assert rows[0]["type"] == "checkpoints"
        # Local has no enrichment.
        assert rows[0]["base_model"] is None
        assert rows[0]["source_url"] is None

    def test_local_text_filter_is_client_side(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "flux", "--where", "local"], capsys)
        names = [r["name"] for r in env["data"]["rows"]]
        # BE-4733: flux lives in diffusion_models, not checkpoints — it must still be found.
        assert names == ["flux1-dev.safetensors"]
        assert env["data"]["rows"][0]["type"] == "diffusion_models"

    def test_local_text_matches_across_folders(self, local_target, monkeypatch, capsys):
        """BE-4733: every on-disk ltx* file is returned, whatever folder it lives in."""
        _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        rows = env["data"]["rows"]
        by_name = {r["name"]: r["type"] for r in rows}
        assert by_name == {
            "ltx-video-2b-v0.9.safetensors": "diffusion_models",
            "ltxv-13b-0.9.7-dev.safetensors": "diffusion_models",
            "ltx-lora-detail.safetensors": "loras",
            "ltx-vae.safetensors": "vae",
        }
        # `tags` mirrors the source folder, and `total` counts every cross-folder match.
        assert all(r["tags"] == [r["type"]] for r in rows)
        assert env["data"]["total"] == 4
        assert env["data"]["shown"] == 4

    def test_local_type_still_scopes_to_one_folder(self, local_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "ltx", "--type", "lora", "--where", "local"], capsys)
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-lora-detail.safetensors"]
        # Exactly one fetch, of /models/loras — no folder-list walk.
        assert [c["url"] for c in calls] == ["http://127.0.0.1:8188/models/loras"]

    def test_local_folder_fetch_error_is_skipped(self, local_target, monkeypatch, capsys):
        routes = _local_routes()
        routes["127.0.0.1:8188/models/vae"] = urllib.error.HTTPError(
            "http://127.0.0.1:8188/models/vae", 404, "Not Found", {}, io.BytesIO(b"nope")
        )
        _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True
        # The vae hit is gone; the other folders' matches still come back.
        assert [r["name"] for r in env["data"]["rows"]] == [
            "ltx-video-2b-v0.9.safetensors",
            "ltxv-13b-0.9.7-dev.safetensors",
            "ltx-lora-detail.safetensors",
        ]
        assert env["data"]["total"] == 3

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(urllib.error.URLError("connection refused"), id="urlerror"),
            pytest.param(OSError("socket hung up"), id="oserror"),
            pytest.param(json.JSONDecodeError("Expecting value", "<html>", 0), id="html-proxy-page"),
            pytest.param(ValueError("response exceeds 67108864 byte cap"), id="oversize-cap"),
        ],
    )
    def test_local_folder_transport_errors_are_skipped(self, local_target, monkeypatch, capsys, failure):
        """Every way `_http_get_json` can fail on ONE folder is tolerated, not fatal.

        The walk used to catch only `HTTPError`, so a hung folder or a proxy
        serving an HTML error page aborted the entire multi-folder search.
        """
        routes = _local_routes()
        routes["127.0.0.1:8188/models/vae"] = failure
        _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert [r["name"] for r in env["data"]["rows"]] == [
            "ltx-video-2b-v0.9.safetensors",
            "ltxv-13b-0.9.7-dev.safetensors",
            "ltx-lora-detail.safetensors",
        ]
        assert env["data"]["total"] == 3

    def test_local_folder_name_with_space_is_walked_and_encoded(self, local_target, monkeypatch, capsys):
        """A user-configured folder like `my loras` is searched, not silently skipped."""
        # Built explicitly (not via `_local_routes`) so the per-folder needle is
        # registered before the bare `/models` one — first-wins substring match.
        routes = {
            "127.0.0.1:8188/models/my%20loras": [{"name": "ltx-custom.safetensors", "pathIndex": 0}],
            "127.0.0.1:8188/models": ["my loras"],
        }
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-custom.safetensors"]
        # The segment is percent-encoded on the wire but decoded in the payload.
        assert [c["url"] for c in calls[1:]] == ["http://127.0.0.1:8188/models/my%20loras"]
        assert env["data"]["rows"][0]["type"] == "my loras"
        assert env["data"]["rows"][0]["tags"] == ["my loras"]

    def test_local_type_with_space_scopes_to_that_folder(self, local_target, monkeypatch, capsys):
        """`--type "my loras"` is an unmapped passthrough — it must scope, not error."""
        routes = {"127.0.0.1:8188/models/my%20loras": [{"name": "ltx-custom.safetensors", "pathIndex": 0}]}
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--type", "my loras", "--where", "local"], capsys)
        assert env["ok"] is True, env
        # Scoped: the bare `/models` folder listing is never fetched.
        assert [c["url"] for c in calls] == ["http://127.0.0.1:8188/models/my%20loras"]
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-custom.safetensors"]
        assert env["data"]["rows"][0]["type"] == "my loras"

    def test_local_type_traversal_still_rejected(self, local_target, monkeypatch, capsys):
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["search", "--text", "ltx", "--type", "../../etc", "--where", "local"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_local_type_undecodable_argv_bytes_error_cleanly(self, local_target, monkeypatch, capsys):
        """`--type` reaches `quote` via `_local_folder_matches`, whose handler misses this.

        `search`'s except clause catches `json.JSONDecodeError` but not bare
        `ValueError`, so the `UnicodeEncodeError` a surrogate raises would escape
        uncaught. The guard rejects it up front instead.
        """
        calls = _patch_urlopen(monkeypatch, {})
        env = _run(["search", "--text", "ltx", "--type", "a\udcffb", "--where", "local"], capsys)
        assert env["ok"] is False, env
        assert env["error"]["code"] == "invalid_argument"
        assert calls == []

    def test_local_walk_skips_undecodable_server_folder_name(self, local_target, monkeypatch, capsys):
        """A server-advertised name carrying a lone surrogate is skipped, not fatal.

        `json.loads('"\\udcff"')` yields a lone surrogate, so this shape is
        reachable from a malicious or buggy backend. The walk must drop just that
        folder and still return every other folder's models.
        """
        routes = {
            "127.0.0.1:8188/models/loras": [{"name": "ltx-lora-detail.safetensors", "pathIndex": 0}],
            "127.0.0.1:8188/models": ["loras", "bad\udcffname"],
        }
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-lora-detail.safetensors"]
        # The undecodable folder is never fetched at all.
        assert [c["url"] for c in calls] == [
            "http://127.0.0.1:8188/models",
            "http://127.0.0.1:8188/models/loras",
        ]

    def test_local_non_string_entry_name_is_skipped(self, local_target, monkeypatch, capsys):
        """A server sending a non-string `name` must not crash the walk or the sort."""
        routes = _local_routes()
        routes["127.0.0.1:8188/models/vae"] = [{"name": 1234, "pathIndex": 0}, {"name": "ltx-vae.safetensors"}]
        _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True, env
        assert "ltx-vae.safetensors" in [r["name"] for r in env["data"]["rows"]]
        assert all(isinstance(r["name"], str) for r in env["data"]["rows"])

    def test_local_limit_caps_rows_but_not_total(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "ltx", "--limit", "2", "--where", "local"], capsys)
        assert env["data"]["shown"] == 2
        assert len(env["data"]["rows"]) == 2
        assert env["data"]["total"] == 4

    @pytest.mark.parametrize("extra", [[], ["--type", "lora"]])
    def test_local_negative_limit_yields_no_rows_not_a_negative_slice(self, local_target, monkeypatch, capsys, extra):
        """`--limit -1` must not silently drop the *last* row via a negative slice."""
        _patch_urlopen(monkeypatch, _local_routes())
        env = _run(["search", "--text", "ltx", "--limit", "-1", "--where", "local", *extra], capsys)
        assert env["ok"] is True, env
        assert env["data"]["rows"] == []
        assert env["data"]["shown"] == 0
        # `total` still reports every match behind the cap.
        assert env["data"]["total"] == (1 if extra else 4)

    def test_local_unsafe_folder_name_is_skipped(self, local_target, monkeypatch, capsys):
        """A server-advertised folder that can't be a URL path segment is skipped, not fatal."""
        routes = _local_routes()
        routes["127.0.0.1:8188/models"] = ["../../etc", "loras"]
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is True
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-lora-detail.safetensors"]
        assert not any("etc" in c["url"] for c in calls)

    def test_local_duplicate_folder_is_fetched_once(self, local_target, monkeypatch, capsys):
        """A folder listed twice by the server must not double-count its files."""
        routes = _local_routes()
        routes["127.0.0.1:8188/models"] = ["loras", "loras", "vae"]
        calls = _patch_urlopen(monkeypatch, routes)
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert [r["name"] for r in env["data"]["rows"]] == ["ltx-lora-detail.safetensors", "ltx-vae.safetensors"]
        assert env["data"]["total"] == 2
        assert sum(1 for c in calls if c["url"].endswith("/models/loras")) == 1

    def test_local_folder_list_error_surfaces_server_not_running(self, local_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"127.0.0.1:8188/models": urllib.error.URLError("connection refused")})
        env = _run(["search", "--text", "ltx", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "server_not_running"

    def test_cloud_http_error_decodes_body(self, cloud_target, monkeypatch, capsys):
        # Shared `_emit_http_error` path via the search handler.
        err = urllib.error.HTTPError("https://x", 400, "Bad Request", {}, io.BytesIO(b'{"detail": "bad tag"}'))
        _patch_urlopen(monkeypatch, {"/api/assets": err})
        env = _run(["search", "--text", "flux", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "cloud_http_error"
        assert env["error"]["details"]["status"] == 400
        assert env["error"]["details"]["body"] == '{"detail": "bad tag"}'


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestShow:
    def test_exact_match_returns_full_asset(self, cloud_target, monkeypatch, capsys):
        _patch_urlopen(monkeypatch, {"/api/assets": _ASSETS_RESPONSE})
        env = _run(["show", "flux1-redux-dev.safetensors", "--where", "cloud"], capsys)
        assert env["ok"] is True
        # Both the projected row and the raw asset ride along.
        assert env["data"]["row"]["name"] == "flux1-redux-dev.safetensors"
        assert env["data"]["asset"]["id"] == "22222222-2222-2222-2222-222222222222"
        assert env["data"]["row"]["base_model"] == "Flux.1 D"
        assert env["data"]["row"]["trained_words"] == ["redux", "blend"]

    def test_no_exact_match_returns_close_matches(self, cloud_target, monkeypatch, capsys):
        # Substring hits but no exact name match.
        _patch_urlopen(monkeypatch, {"/api/assets": _ASSETS_RESPONSE})
        env = _run(["show", "flux-DOES-NOT-EXIST.safetensors", "--where", "cloud"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "model_not_found"
        # The 5-or-fewer close_matches affordance helps the agent self-correct.
        assert "close_matches" in env["error"]["details"]

    def test_local_is_explicitly_unsupported(self, local_target, monkeypatch, capsys):
        # urlopen should never be called for `show --where local`.
        _patch_urlopen(monkeypatch, {})
        env = _run(["show", "anything.safetensors", "--where", "local"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "models_show_local_unsupported"
