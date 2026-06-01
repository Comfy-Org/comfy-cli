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
        _patch_urlopen(
            monkeypatch,
            {"127.0.0.1:8188/models/checkpoints": [{"name": "sd_xl_base_1.0.safetensors", "pathIndex": 0}]},
        )
        env = _run(["search", "--text", "sd_xl", "--where", "local"], capsys)
        assert env["ok"] is True
        assert env["data"]["mode"] == "local"
        rows = env["data"]["rows"]
        assert len(rows) == 1
        assert rows[0]["name"] == "sd_xl_base_1.0.safetensors"
        # Local has no enrichment.
        assert rows[0]["base_model"] is None
        assert rows[0]["source_url"] is None

    def test_local_text_filter_is_client_side(self, local_target, monkeypatch, capsys):
        _patch_urlopen(
            monkeypatch,
            {
                "127.0.0.1:8188/models/checkpoints": [
                    {"name": "sd_xl_base_1.0.safetensors", "pathIndex": 0},
                    {"name": "flux1-dev.safetensors", "pathIndex": 0},
                ]
            },
        )
        env = _run(["search", "--text", "flux", "--where", "local"], capsys)
        names = [r["name"] for r in env["data"]["rows"]]
        assert names == ["flux1-dev.safetensors"]


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
