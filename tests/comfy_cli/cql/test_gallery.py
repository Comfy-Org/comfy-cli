"""Tests for the CQL gallery-search engine (cql/gallery.py).

These exercise the engine directly (load → flatten → predicates), independent
of the Typer command shell in ``command/templates.py``.
"""

from __future__ import annotations

import json
import time

import pytest

from comfy_cli.cql import gallery

CATEGORIES = [
    {
        "title": "Image",
        "type": "image",
        "category": "GENERATION TYPE",
        "templates": [
            {
                "name": "image_flux2",
                "title": " Flux 2 ",
                "description": "Text-to-image via BFL.",
                "tags": ["API", "Text to Image"],
                "models": ["Flux 2"],
                "logos": [{"provider": ["Black Forest Labs"]}],
                "openSource": False,
                "usage": 42,
                "mediaSubtype": "webp",
            },
            {
                "name": "image_sd15",
                "title": "SD 1.5",
                "tags": ["Text to Image"],
                "models": ["SD1.5"],
                "logos": [{"provider": "Stability"}],
            },
        ],
    },
    {
        "title": "Video",
        "type": "video",
        "templates": [
            {"name": "video_kling", "title": "Kling", "tags": ["API"], "logos": [{"provider": "Kling"}]},
        ],
    },
    "not-a-dict",  # tolerated/skipped
]


def test_flatten_walks_categories_and_adds_extras():
    rows = gallery.flatten_templates(CATEGORIES)
    assert [r["name"] for r in rows] == ["image_flux2", "image_sd15", "video_kling"]
    flux = rows[0]
    assert flux["output_type"] == "image"  # from parent category type, not mediaSubtype
    assert flux["category_title"] == "Image"
    assert flux["title"] == "Flux 2"  # stripped
    assert flux["providers"] == ["Black Forest Labs"]
    assert flux["open_source"] is False
    assert flux["usage"] == 42


def test_flatten_providers_handles_scalar_and_array_and_dedupes():
    assert gallery.flatten_providers([{"provider": "Kling"}]) == ["Kling"]
    assert gallery.flatten_providers([{"provider": ["A", "B", "A"]}]) == ["A", "B"]
    assert gallery.flatten_providers(["junk", {"no": "provider"}]) == []


def test_matches_predicates():
    rows = gallery.flatten_templates(CATEGORIES)
    flux = rows[0]
    assert gallery.matches(flux, type_="image") is True
    assert gallery.matches(flux, type_="video") is False
    assert gallery.matches(flux, tag="api") is True  # case-insensitive exact
    assert gallery.matches(flux, tag="vid") is False  # not a substring match
    assert gallery.matches(flux, model="flux") is True  # substring
    assert gallery.matches(flux, provider="forest") is True  # substring
    assert gallery.matches(flux, name_sub="flux2") is True


def test_filter_rows_applies_all_predicates():
    rows = gallery.flatten_templates(CATEGORIES)
    out = gallery.filter_rows(rows, type_="image", tag="API")
    assert [r["name"] for r in out] == ["image_flux2"]
    assert gallery.filter_rows(rows, type_="video") == [r for r in rows if r["name"] == "video_kling"]
    assert gallery.filter_rows(rows) == rows  # no predicates → identity


def test_load_gallery_from_explicit_path(tmp_path):
    p = tmp_path / "index.json"
    p.write_text(json.dumps(CATEGORIES[:1]))
    cats = gallery.load_gallery(str(p))
    assert cats[0]["title"] == "Image"


def test_load_gallery_fetches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "cache_path", lambda: tmp_path / "g" / "index.json")
    payload = json.dumps(CATEGORIES[:1]).encode()
    monkeypatch.setattr(gallery, "fetch_gallery", lambda *a, **k: payload)
    cats = gallery.load_gallery(None, refresh=True)
    assert cats[0]["title"] == "Image"
    # Cached to disk; a second non-refresh load reads cache without fetching.
    monkeypatch.setattr(gallery, "fetch_gallery", lambda *a, **k: pytest.fail("should use cache"))
    again = gallery.load_gallery(None)
    assert again[0]["title"] == "Image"


def test_load_gallery_refetches_when_cache_stale(tmp_path, monkeypatch):
    cache = tmp_path / "g" / "index.json"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(json.dumps([{"title": "OldImage", "type": "image", "templates": []}]).encode())
    # Age the cache past the TTL.
    import os

    old = time.time() - gallery._CACHE_TTL_SECONDS - 100
    os.utime(cache, (old, old))
    monkeypatch.setattr(gallery, "cache_path", lambda: cache)
    monkeypatch.setattr(gallery, "fetch_gallery", lambda *a, **k: json.dumps(CATEGORIES[:1]).encode())

    cats = gallery.load_gallery(None)
    assert cats[0]["title"] == "Image"  # fetched fresh, not the stale "OldImage"


def test_load_gallery_falls_back_to_stale_cache_when_offline(tmp_path, monkeypatch):
    cache = tmp_path / "g" / "index.json"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(json.dumps([{"title": "StaleImage", "type": "image", "templates": []}]).encode())
    old = time.time() - gallery._CACHE_TTL_SECONDS - 100
    import os

    os.utime(cache, (old, old))
    monkeypatch.setattr(gallery, "cache_path", lambda: cache)

    def offline(*a, **k):
        raise gallery.GalleryError("network down")

    monkeypatch.setattr(gallery, "fetch_gallery", offline)
    cats = gallery.load_gallery(None)
    assert cats[0]["title"] == "StaleImage"  # degraded gracefully to stale cache


def test_load_gallery_errors_when_offline_and_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, "cache_path", lambda: tmp_path / "g" / "index.json")

    def offline(*a, **k):
        raise gallery.GalleryError("network down")

    monkeypatch.setattr(gallery, "fetch_gallery", offline)
    with pytest.raises(gallery.GalleryError, match="no cache present"):
        gallery.load_gallery(None)
