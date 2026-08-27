"""Unit tests for :func:`comfy_cli.knowledge.attach` and its matching helpers.

Contract under test:
  * keyed lookup only: normalized alias keys (most specific prefix first) and
    capability ids/aliases; template ids and node classes via the reverse index.
  * the block is capped (row count, list items, total bytes) by dropping whole
    entries, never by trimming inside one.
  * a row or pick whose ids do not resolve in the command's catalog is dropped
    silently (version skew is normal).
  * fail-open: no bundle, or any exception inside ``attach``, leaves the payload
    untouched and prints nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli import knowledge, tracking

FIXTURE_KNOWLEDGE = Path(__file__).parent / "fixtures" / "knowledge" / "knowledge.json"

# A fixture template id the reverse index maps to a model row, so a block can
# carry rows while the query strings themselves resolve to nothing.
_REVERSE_TEMPLATE = "api_lipco_lip_sync_video"


def _network_guard(url: str) -> bytes:
    raise AssertionError(f"network touched: {url}")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(knowledge.ENV_FILE, str(FIXTURE_KNOWLEDGE))
    for var in (knowledge.ENV_URL, knowledge.ENV_TTL, knowledge.ENV_DISABLE):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(knowledge, "_http_get", _network_guard)
    knowledge._reset_for_testing()
    yield
    knowledge._reset_for_testing()


def _attach(**kwargs) -> dict:
    p: dict = {}
    knowledge.attach(p, **kwargs)
    return p


def _walk(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_walk(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_walk(v, key) for v in obj)
    return False


class TestNormalization:
    def test_norm_strips_to_alnum(self):
        assert knowledge._normalize("Lip Sync") == "lipsync"
        assert knowledge._normalize("Image to Video") == "imagetovideo"


class TestLookup:
    def test_keyed_alias_hits_its_row(self):
        p = _attach(queries=["Testvid 3.0"])
        k = p["knowledge"]
        entry = k["models"][0]
        assert entry["id"] == "testvid"
        assert entry["matched_on"] == "testvid 3.0"
        assert entry["tier"] == "law"
        assert isinstance(entry["pitfalls"], list) and all(isinstance(t, str) for t in entry["pitfalls"])
        assert not _walk(k, "source")
        assert k["hit_ids"] == ["testvid"]
        assert k["zero_hit"] is False
        assert k["bundle_version"] == "0.1.0-fixture"
        assert k["stale"] is False

    def test_unkeyed_variant_gets_no_family_row(self):
        """A prefix walk used to hand 'testvid-lipsync' the whole 'testvid' row: a
        green-light law-tier answer for a variant the bundle never keyed, while
        `comfy knowledge resolve` called the same string unknown."""
        bundle = knowledge.load_bundle()
        for q in ("testvid-lipsync", "testvid-avatar", "Testvid Avatar", "flux-kontext-max"):
            assert knowledge.resolve_id(bundle, q) is None, q
            # The guard is on the MODEL row. A capability the wording names may
            # still answer with its ranked picks — 'testvid-lipsync' asking about
            # lipsync is a question the bundle can answer.
            assert _attach(queries=[q]).get("knowledge", {}).get("models", []) == [], q

    def test_an_intent_phrase_reaches_the_capability_it_names(self):
        """Callers ask in sentences; the bundle is keyed on tags and ids. Every
        word of a key appearing in the query is enough to resolve it."""
        for q in ("make a talking head video", "lip sync this footage", "audio generation for my clip"):
            k = _attach(queries=[q])["knowledge"]
            assert k["picks"], q
            assert k["hit_ids"] in (["cap:lipsync"], ["cap:audio-generation"]), q

    def test_a_phrase_naming_no_capability_resolves_to_nothing(self):
        bundle = knowledge.load_bundle()
        for q in ("how do I load a checkpoint", "a video", "zzzz", "head"):
            assert knowledge._resolve_tokens(bundle, q) is None, q

    def test_a_tie_between_capabilities_resolves_to_neither(self):
        """Same rule as the normalized map: two answers are not an answer. The
        keys differ as strings, so they survive capability_keys' first-wins
        dedupe and collide only once reduced to word sets."""
        data = {
            "models": {},
            "capabilities": {"alpha": {"aliases": ["video upscale"]}, "beta": {"aliases": ["upscale video"]}},
        }
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._resolve_tokens(b, "please upscale this video") is None

    def test_the_longest_key_wins_over_a_broader_one(self):
        data = {
            "models": {},
            "capabilities": {"broad": {"aliases": ["video"]}, "narrow": {"aliases": ["video upscale"]}},
        }
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._resolve_tokens(b, "please video upscale this") == "narrow"
        assert knowledge._resolve_tokens(b, "just a video") == "broad"

    def test_non_english_input_never_resolves_by_accident(self):
        """The agent answers in the user's language, so it searches in it too.
        Non-ASCII is dropped by the character class (as :func:`_normalize`
        already does), leaving whatever ASCII the query carried. That must not
        add up to a capability on its own."""
        bundle = knowledge.load_bundle()
        for q in ("トーキングヘッド動画", "обработка изображения", "vídeo de una persona hablando"):
            assert knowledge._resolve_tokens(bundle, q) is None, q
        # A mixed query still resolves on the English term it contains, which is
        # the common case: capability names are English whatever the user writes.
        assert knowledge._resolve_tokens(bundle, "この動画に lip sync したい") == "lipsync"

    def test_a_short_single_word_key_never_matches_alone(self):
        """Guards a future alias like '3d' from joining every query that says it."""
        data = {"models": {}, "capabilities": {"cap": {"aliases": ["3d"]}}}
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._resolve_tokens(b, "a 3d mesh of my cat") is None

    def test_lookup_agrees_with_the_resolve_verb(self):
        bundle = knowledge.load_bundle()
        for q in ("testvid", "Testvid 3.0", "HALO-03", "Acme H3", "testvid-lipsync", "zzz"):
            hits, _ = knowledge._lookup(bundle, [q])
            resolved = knowledge.resolve_id(bundle, q)
            assert [mid for mid, _ in hits] == ([resolved] if resolved else []), q

    def test_deprecated_variant_keeps_its_own_row(self):
        k = _attach(queries=["Testvid Avatar 2"])["knowledge"]
        entry = k["models"][0]
        assert entry["id"] == "testvid-avatar-2"
        assert entry["status"] == "deprecated"
        assert entry["superseded_by"] == "lipco-3"

    def test_capability_by_id_alias_and_gallery_tag(self):
        for q in ("lipsync", "Lip Sync", "talking head"):
            k = _attach(queries=[q])["knowledge"]
            assert k["picks"], q
            ranks = [p["rank"] for p in k["picks"]]
            assert ranks == sorted(ranks)
            assert k["picks"][0]["capability"] == "lipsync"
            assert "cap:lipsync" in k["hit_ids"]

    def test_pick_entry_shape(self):
        k = _attach(queries=["lipsync"])["knowledge"]
        by_model = {p["model"]: p for p in k["picks"]}
        assert set(by_model["testvid"]) == {
            "capability",
            "rank",
            "model",
            "route",
            "template",
            "caveat",
            "status",
            "superseded_by",
        }
        assert by_model["testvid"]["status"] == "available"
        assert by_model["testvid"]["template"] is None
        # A pick naming a model the trimmed fixture lacks still ships, with nulls.
        assert by_model["testlx"]["status"] is None

    def test_reverse_index_templates_and_nodes(self):
        p = _attach(templates=["video_acme_h3_i2v"])
        assert p["knowledge"]["models"][0]["id"] == "acme-h3"
        assert p["knowledge"]["models"][0]["matched_on"] == "video_acme_h3_i2v"
        p = _attach(nodes=["TestvidImage2VideoNode"])
        assert p["knowledge"]["models"][0]["id"] == "testvid"
        assert p["knowledge"]["models"][0]["matched_on"] == "TestvidImage2VideoNode"

    def test_capability_id_beats_a_foreign_alias(self):
        data = {"models": {}, "capabilities": {"lipsync": {"aliases": ["upscale", "---"]}, "upscale": {}}}
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._lookup(b, ["upscale"]) == ([], ["upscale"])
        assert knowledge._lookup(b, ["!!!"]) == ([], [])

    def test_direct_alias_hit_keeps_its_matched_on_over_reverse_hit(self):
        p = _attach(queries=["testvid"], nodes=["TestvidImage2VideoNode"])
        models = p["knowledge"]["models"]
        assert [m["id"] for m in models] == ["testvid"]
        assert models[0]["matched_on"] == "testvid"

    def test_a_query_can_hit_model_and_capability_and_both_dedupe(self):
        k = _attach(queries=["testvid", "Testvid 3.0", "lipsync", "Lip Sync"])["knowledge"]
        assert [m["id"] for m in k["models"]] == ["testvid"]
        assert [p["capability"] for p in k["picks"]].count("lipsync") == len(k["picks"])
        assert k["hit_ids"] == ["testvid", "cap:lipsync"]

    def test_deprecation_fields_come_from_the_deprecations_list(self):
        entry = _attach(queries=["Testvid Avatar 2.0"])["knowledge"]["models"][0]
        assert entry["matched_on"] == "testvid avatar 2.0"
        assert entry["status"] == "deprecated"
        assert entry["superseded_by"] == "lipco-3"
        assert entry["deprecated_on"] == "2031-01-09"

    def test_full_entry_shape_and_list_reshaping(self):
        entry = _attach(queries=["lipco-3"])["knowledge"]["models"][0]
        assert list(entry)[:5] == ["id", "matched_on", "status", "tier", "route"]
        assert all(set(r) == {"when", "use"} for r in entry["routing"])
        for key in ("pitfalls", "corrections", "warnings"):
            assert all(isinstance(t, str) for t in entry.get(key, []))
        assert "resolves" not in entry
        assert "verified_at" not in entry
        assert "owner" not in entry

    def test_tier_is_opaque(self):
        entry = _attach(queries=["lipco-3"])["knowledge"]["models"][0]
        assert entry["tier"] == "canon"


def _phrased_bundle() -> knowledge.Bundle:
    """The real bundle's capability keys and descriptions, minus the picks."""
    capabilities = {
        "audio-generation": {
            "aliases": ["Text to Audio", "Music", "Audio"],
            "description": "Music, text-to-speech and sound effects. Neither is the native audio track of a clip.",
        },
        "image-edit": {"description": "Change an existing still by instruction (instruct-edit)."},
        "inpaint": {
            "description": "Remove or replace part of an image. Prompt-remove and mask-fill are different fetches."
        },
        "lipsync": {"description": "A still plus audio becomes a talking clip, or dubbing existing footage."},
        "text-in-image": {
            "description": "Stills where in-image text must be readable: posters, signs, UI, typography."
        },
        "text-to-image": {
            "description": "Make a still image from a prompt with no source image. The photorealism row."
        },
        "text-to-video": {"description": "Make a clip from a prompt with no source image."},
        "upscale": {
            "aliases": ["Video Upscale"],
            "description": "Add resolution or restore detail after generation or editing.",
        },
    }
    return knowledge._index(
        {"models": {}, "capabilities": capabilities}, None, source="env", stale=False, path="x", mtime=0.0
    )


class TestPhrasedQueries:
    """The three defects behind ``knowledge pick``'s miss rate, and the misses that must stay misses."""

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            # A tie on {text, image} breaks on the key spelled out literally, or on description words.
            ("fast cheap text to image drafts for A/B testing", "text-to-image"),
            ("fast local text to image with accurate object binding", "text-to-image"),
            ("image with accurate text and typography", "text-in-image"),
            ("generate image with readable Chinese text local", "text-in-image"),
            ("text image", None),
            # lipsync versus the alias "Audio": the longer literal key wins.
            ("lipsync talking portrait from photo and audio", "lipsync"),
            ("make a portrait speak my audio (lipsync talking avatar)", "lipsync"),
            # text-to-video versus "Text to Audio": same rule.
            ("text to video with audio / sound", "text-to-video"),
            # A partial key match counts when the description supplies more words than the key lacks.
            ("poster with readable text", "text-in-image"),
            ("poster with lots of small readable text typography", "text-in-image"),
            ("sound effects generation", "audio-generation"),
            ("4K video generation", None),
            ("generate image keeping multiple reference photos consistent", None),
            ("a video", None),
            # Stemming and joined adjacent words.
            ("keep the same character while editing an image", "image-edit"),
            # "removal" is only in inpaint's description, never a key, so this stays a canon gap.
            ("remove the background from this photo", None),
            ("lip sync video to new speech", "lipsync"),
            ("dub video into another language with lip sync", "lipsync"),
            # Nothing in the table names these; they must stay misses, not resolve to the closest row.
            ("restore old damaged photo keep faces", None),
            ("camera control video", None),
            ("how do I load a checkpoint", None),
        ],
    )
    def test_resolves_the_capability_the_phrase_names(self, query, expected):
        assert knowledge._resolve_tokens(_phrased_bundle(), query) == expected

    def test_every_id_still_resolves_to_itself(self):
        b = _phrased_bundle()
        for cid in b.capabilities:
            assert knowledge.pick(b, cid)["id"] == cid

    def test_a_partial_match_needs_its_description_to_out_vote_the_missing_words(self):
        data = {"models": {}, "capabilities": {"cap": {"aliases": ["alpha beta"], "description": "gamma delta"}}}
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._resolve_tokens(b, "alpha gamma") is None
        assert knowledge._resolve_tokens(b, "alpha gamma delta") == "cap"
        assert knowledge._resolve_tokens(b, "gamma delta") is None

    def test_an_inflected_key_word_still_matches(self):
        data = {"models": {}, "capabilities": {"cap": {"aliases": ["Background Removal"]}}}
        b = knowledge._index(data, None, source="env", stale=False, path="x", mtime=0.0)
        assert knowledge._resolve_tokens(b, "remove the background from this photo") == "cap"

    def test_stem_agrees_across_inflections(self):
        for a, b in (("editing", "edit"), ("edits", "edit"), ("removal", "remove"), ("images", "image")):
            assert knowledge._stem(a) == knowledge._stem(b), (a, b)
        assert knowledge._stem("3d") == "3d"
        assert knowledge._query_tokens("lip sync") >= {"lip", "sync", "lipsync"}


class TestScalarGuards:
    def test_non_string_routing_values_are_emitted_as_null(self):
        """Routing is copied straight out of the bundle, and the block schema allows
        only string or null there. A number reaching the wire fails a consumer's
        validation on data we chose to forward."""
        data = {
            "models": {
                "m": {
                    "id": "m",
                    "routing": [{"when": 1, "use": ["not", "a", "string"]}, {"when": "ok", "use": "t"}],
                }
            }
        }
        b = knowledge._index(data, None, source="env", stale=False, path="p", mtime=0.0)
        entry = knowledge._model_entry(b, "m", b.models["m"], matched_on="m", brief=False)
        assert entry["routing"] == [{"when": None, "use": None}, {"when": "ok", "use": "t"}]


class TestSkewFilter:
    """A row this install cannot run is annotated, never hidden. Dropping it made
    a curated answer look like no answer, which is the confusion the bundle exists
    to remove."""

    def test_row_with_templates_is_annotated_not_dropped(self):
        k = _attach(templates=["video_acme_h3_i2v"], catalog_templates={"something_else"})["knowledge"]
        entry = k["models"][0]
        assert entry["id"] == "acme-h3"
        assert entry["available_locally"] is False
        assert entry["unavailable_reason"] == knowledge.UNAVAILABLE_LOCALLY
        k = _attach(templates=["video_acme_h3_i2v"], catalog_templates={"video_acme_h3_i2v"})["knowledge"]
        assert k["models"][0]["id"] == "acme-h3"
        assert "available_locally" not in k["models"][0]

    def test_row_without_templates_survives_a_template_catalog(self):
        entry = _attach(queries=["testvid"], catalog_templates={"x"})["knowledge"]["models"][0]
        assert entry["id"] == "testvid"
        assert "available_locally" not in entry

    def test_row_with_nodes_is_annotated_not_dropped(self):
        entry = _attach(queries=["testvid"], catalog_nodes={"KSampler"})["knowledge"]["models"][0]
        assert entry["id"] == "testvid"
        assert entry["available_locally"] is False
        p = _attach(queries=["testvid"], catalog_nodes={"KSampler", "TestvidLipSyncTextToVideoNode"})
        assert p["knowledge"]["models"][0]["id"] == "testvid"
        assert "available_locally" not in p["knowledge"]["models"][0]

    def test_either_catalog_resolving_the_row_is_enough(self):
        """acme-h3 names both templates and nodes; one catalog carrying it settles it."""
        k = _attach(
            queries=["acme-h3"],
            catalog_templates={"video_acme_h3_i2v"},
            catalog_nodes={"KSampler"},
        )["knowledge"]
        assert "available_locally" not in k["models"][0]

    def test_none_catalog_is_not_consulted(self):
        entry = _attach(queries=["testvid"], catalog_templates=None, catalog_nodes=None)["knowledge"]["models"][0]
        assert "available_locally" not in entry

    def test_skew_is_not_reported_as_a_miss(self):
        """`nodes search testvid` on an install without the classes used to answer
        `zero_hit: true, no curated knowledge for 'testvid'` — the opposite of true."""
        k = _attach(queries=["testvid"], catalog_nodes={"KSampler"}, thin=True)["knowledge"]
        assert k["zero_hit"] is False
        assert "nudge" not in k
        # The borrowed capability counts as returned, so the miss log sees it too.
        assert k["hit_ids"] == ["testvid", "cap:lipsync"]

    def test_unavailable_rows_sort_after_available_ones_in_their_tier(self):
        k = _attach(queries=["acme-h3", "testvid"], catalog_nodes={"AcmeH3ImageToVideo"})["knowledge"]
        assert [m["id"] for m in k["models"]] == ["acme-h3", "testvid"]
        assert k["models"][1]["available_locally"] is False

    def test_an_unavailable_row_borrows_its_capability_picks(self):
        """Matched by model name, so no capability matched and picks would be empty.
        The row alone is a dead end: curated, unrunnable, nothing to reach for."""
        k = _attach(queries=["testvid"], catalog_nodes={"KSampler"}, thin=True)["knowledge"]
        assert k["models"][0]["available_locally"] is False
        assert {p["capability"] for p in k["picks"]} == {"lipsync"}
        assert "cap:lipsync" in k["hit_ids"]

    def test_an_available_row_borrows_nothing(self):
        k = _attach(queries=["testvid"], catalog_nodes={"TestvidImage2VideoNode"})["knowledge"]
        assert "available_locally" not in k["models"][0]
        assert k["picks"] == []

    def test_borrowed_picks_still_respect_a_template_catalog(self):
        k = _attach(
            queries=["testvid"],
            catalog_nodes={"KSampler"},
            catalog_templates={"api_lipco_lip_sync_video"},
        )["knowledge"]
        by_model = {p["model"]: p for p in k["picks"]}
        assert "available_locally" not in by_model["lipco-3"]
        assert by_model["testlx"]["available_locally"] is False

    def test_picks_are_annotated_in_rank_order(self):
        k = _attach(queries=["lipsync"], catalog_templates={"api_lipco_lip_sync_video"})["knowledge"]
        ranks = [p["rank"] for p in k["picks"]]
        assert ranks == sorted(ranks)
        by_model = {p["model"]: p for p in k["picks"]}
        assert "available_locally" not in by_model["lipco-3"]
        assert "available_locally" not in by_model["testvid"]  # rank 6 names no template
        assert by_model["testlx"]["available_locally"] is False
        assert by_model["testlx"]["unavailable_reason"] == knowledge.UNAVAILABLE_LOCALLY


class TestCaps:
    def test_max_models_keeps_law_first(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_MODELS", 1)
        k = _attach(queries=["lipco-3", "test-audio-1-5", "testvid"])["knowledge"]
        assert [m["id"] for m in k["models"]] == ["testvid"]
        assert k["hit_ids"] == ["testvid"]

    def test_law_rows_sort_first_then_input_order(self):
        k = _attach(queries=["lipco-3", "test-audio-1-5", "testvid"])["knowledge"]
        assert [(m["id"], m["tier"]) for m in k["models"]] == [
            ("testvid", "law"),
            ("lipco-3", "canon"),
            ("test-audio-1-5", "canon"),
        ]

    def test_byte_ceiling_drops_whole_entries(self, monkeypatch):
        full = _attach(queries=["testvid", "Lip Sync"])["knowledge"]
        assert full["models"] and full["picks"]
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 600)
        k = _attach(queries=["testvid", "Lip Sync"])["knowledge"]
        assert knowledge._block_bytes(k) <= 600
        assert k["models"] == []
        assert 0 < len(k["picks"]) < len(full["picks"])
        assert k["hit_ids"] == ["cap:lipsync"]
        json.loads(json.dumps(k))

    def test_byte_ceiling_can_empty_the_block(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 10)
        assert "knowledge" not in _attach(queries=["testvid"])
        assert "knowledge" not in _attach(queries=["testvid"], thin=True)

    def test_the_ceiling_counts_wire_bytes_not_escaped_characters(self, monkeypatch):
        """The renderer emits with ``ensure_ascii=False``, so a curly quote costs
        three bytes on the wire and six characters once escaped. Measuring the
        escaped form trims blocks that would have fit."""
        prose = "don\u2019t use the \u201cfast\u201d route"
        block = {"models": [{"id": "m", "pitfalls": [prose]}], "picks": [], "hit_ids": ["m"]}
        wire = len(json.dumps(block, ensure_ascii=False).encode())
        assert wire < len(json.dumps(block))

        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", wire)
        knowledge._fit(block)
        assert block["models"] == [{"id": "m", "pitfalls": [prose]}]

    def test_max_list_items(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_LIST_ITEMS", 2)
        entry = _attach(queries=["testvid"])["knowledge"]["models"][0]
        assert len(entry["pitfalls"]) <= 2
        entry = _attach(queries=["test-audio-1-5"])["knowledge"]["models"][0]
        assert len(entry["routing"]) <= 2
        entry = _attach(queries=["acme-h3"])["knowledge"]["models"][0]
        assert len(entry["best_for"]) == 2

    def test_max_picks_bounds_the_whole_block_not_one_capability(self):
        k = _attach(queries=["lipsync", "audio-generation"])["knowledge"]
        assert len(k["picks"]) == knowledge.MAX_PICKS

    def test_max_picks(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_PICKS", 2)
        k = _attach(queries=["lipsync"])["knowledge"]
        assert [p["rank"] for p in k["picks"]] == [1, 2]

    def test_brief_entries(self):
        entry = _attach(queries=["testvid"], brief=True)["knowledge"]["models"][0]
        assert "pitfalls" not in entry
        assert "warnings" not in entry
        assert entry["best_for"]
        assert entry["tier"] == "law"

    def test_brief_uses_its_own_row_cap(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_MODELS", 1)
        k = _attach(queries=["lipco-3", "test-audio-1-5", "testvid"], brief=True)["knowledge"]
        assert len(k["models"]) == 3


class TestCapabilitiesAvailable:
    def test_every_enriched_block_carries_the_vocabulary(self):
        for kwargs in (
            {"queries": ["kling"]},
            {"queries": ["lipsync"]},
            {"templates": [_REVERSE_TEMPLATE]},
            {"queries": ["faceswap"], "thin": True},
        ):
            k = _attach(**kwargs)["knowledge"]
            assert k["capabilities_available"] == ["audio-generation", "lipsync"]

    def test_the_vocabulary_is_the_bundle_capability_keys_sorted(self):
        bundle = knowledge.load_bundle(cache_only=True)
        k = _attach(queries=["kling"])["knowledge"]
        assert k["capabilities_available"] == sorted(bundle.capabilities)

    def test_the_vocabulary_outlives_models_and_picks_under_pressure(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 600)
        k = _attach(queries=["kling", "Lip Sync"])["knowledge"]
        assert k["models"] == [] and 0 < len(k["picks"])
        assert k["capabilities_available"] == ["audio-generation", "lipsync"]

    def test_fit_drops_the_vocabulary_only_after_models_and_picks(self, monkeypatch):
        def block() -> dict:
            return {
                "models": [{"id": "kling"}],
                "picks": [{"capability": "lipsync"}],
                "capabilities_available": ["audio-generation", "lipsync"],
                "hit_ids": [],
            }

        emptied = block()
        emptied["models"] = []
        emptied["picks"] = []
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", knowledge._block_bytes(emptied))
        k = block()
        knowledge._fit(k)
        assert k["models"] == [] and k["picks"] == []
        assert k["capabilities_available"] == ["audio-generation", "lipsync"]

        bare = dict(emptied)
        del bare["capabilities_available"]
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", knowledge._block_bytes(bare))
        k = block()
        knowledge._fit(k)
        assert "capabilities_available" not in k
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES

    def test_no_bundle_still_adds_no_key(self, monkeypatch):
        monkeypatch.setattr(knowledge, "load_bundle", lambda **_: None)
        assert "knowledge" not in _attach(queries=["kling"])


class TestNudge:
    def test_thin_zero_hit_gets_a_nudge(self):
        k = _attach(queries=["faceswap"], thin=True)["knowledge"]
        assert k["zero_hit"] is True
        assert k["models"] == [] and k["picks"] == [] and k["hit_ids"] == []
        assert "'faceswap'" in k["nudge"]
        assert "see capabilities_available" in k["nudge"]
        assert k["capabilities_available"] == ["audio-generation", "lipsync"]

    def test_nudge_uses_first_non_blank_query(self):
        k = _attach(queries=["  ", "faceswap", "other"], thin=True)["knowledge"]
        assert "'faceswap'" in k["nudge"]

    def test_nudge_stays_under_byte_ceiling_for_a_huge_query(self):
        k = _attach(queries=["x-" * 4500], thin=True)["knowledge"]
        assert k["zero_hit"] is True
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES
        assert "see capabilities_available" in k["nudge"]

    def test_zero_hit_nudge_sheds_the_pointer_then_the_vocabulary(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 260)
        k = _attach(queries=["faceswap"], thin=True)["knowledge"]
        assert k["nudge"] == "no curated knowledge for 'faceswap'"
        assert "capabilities_available" not in k
        assert k["uncurated_queries"] == ["faceswap"]
        assert knowledge._block_bytes(k) <= 260

    def test_unresolved_query_on_a_non_empty_block_gets_a_nudge(self):
        k = _attach(queries=["FLF2V"], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["models"]
        assert k["zero_hit"] is False
        assert "'FLF2V'" in k["nudge"]
        assert "see capabilities_available" in k["nudge"]

    def test_query_resolving_to_a_model_gets_no_nudge(self):
        k = _attach(queries=["testvid"], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["models"]
        assert "nudge" not in k

    def test_query_the_reverse_index_resolves_gets_no_nudge(self):
        # `nodes search <ClassName>` and `templates ls --name-sub <template id>`
        # pass the same string as a query and as a reverse-index key.
        k = _attach(queries=[_REVERSE_TEMPLATE], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert [e["matched_on"] for e in k["models"]] == [_REVERSE_TEMPLATE]
        assert "nudge" not in k

    def test_query_resolving_to_a_capability_gets_no_nudge(self):
        k = _attach(queries=["lipsync"])["knowledge"]
        assert k["picks"]
        assert "nudge" not in k

    def test_listed_rows_enrich_but_do_not_answer_the_query(self):
        # `generate list --query FLF2V` passes its own result rows. They belong in
        # the block, and they must not make an unanswered query look answered.
        k = _attach(queries=["FLF2V"], models=["testvid"])["knowledge"]
        assert [e["id"] for e in k["models"]] == ["testvid"]
        assert "'FLF2V'" in k["nudge"]

    def test_unresolved_query_nudge_degrades_then_disappears(self, monkeypatch):
        args = {"queries": ["FLF2V"], "templates": [_REVERSE_TEMPLATE]}
        full = _attach(**args)["knowledge"]
        head = "no curated knowledge for 'FLF2V'"
        head_only = dict(full, nudge=head)
        bare = {key: value for key, value in full.items() if key != "nudge"}

        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", knowledge._block_bytes(head_only))
        k = _attach(**args)["knowledge"]
        assert k["nudge"] == head
        assert k["models"]
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES

        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", knowledge._block_bytes(bare))
        k = _attach(**args)["knowledge"]
        assert "nudge" not in k
        assert k["models"]
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES

    def test_not_thin_gets_a_marker_rather_than_a_nudge(self):
        """The command answered, so there is nothing to advise the caller about.
        The miss is still recorded, as data with no prose."""
        k = _attach(queries=["faceswap"], thin=False)["knowledge"]
        assert k["hit_ids"] == [] and k["zero_hit"] is False
        assert "nudge" not in k
        assert "capabilities_available" not in k

    def test_thin_without_queries_means_no_key(self):
        assert "knowledge" not in _attach(queries=[], thin=True)
        assert "knowledge" not in _attach(templates=["nope"], thin=True)


class TestMissMarker:
    """A query nothing curated answered is recorded in the envelope, not just in
    the consent-gated event. Cloud has no CLI-side telemetry client, so the
    envelope is the only copy it can read."""

    def test_a_search_that_returned_rows_records_its_miss(self):
        k = _attach(command="nodes search", queries=["faceswap"])["knowledge"]
        assert k == {
            "bundle_version": "0.1.0-fixture",
            "stale": False,
            "as_of": k["as_of"],
            "hit_ids": [],
            "zero_hit": False,
            "uncurated_queries": ["faceswap"],
        }
        assert "models" not in k and "picks" not in k and "nudge" not in k

    def test_the_marker_dates_the_bundle_it_missed_against(self):
        """A gap found against a stale cache may already be covered upstream, so
        a curator ranking gaps has to be able to discount it."""
        k = _attach(queries=["faceswap"])["knowledge"]
        full = _attach(queries=["testvid"])["knowledge"]
        assert k["as_of"] == full["as_of"]
        assert k["stale"] == full["stale"] is False

    def test_zero_hit_separates_a_search_that_returned_nothing(self):
        """The pair is the whole signal: `hit_ids: []` is the miss, `zero_hit`
        says whether the caller walked away with nothing at all."""
        answered = _attach(queries=["faceswap"])["knowledge"]
        empty_handed = _attach(queries=["faceswap"], thin=True)["knowledge"]
        assert answered["hit_ids"] == empty_handed["hit_ids"] == []
        assert answered["zero_hit"] is False
        assert empty_handed["zero_hit"] is True

    def test_a_block_shed_to_nothing_reports_what_it_shed(self, monkeypatch):
        """`_fit` dropping every entry to make the ceiling used to report as
        silence, the one shape indistinguishable from an unenriched envelope. It
        is not a miss: the bundle answered, and the answer did not fit."""
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 300)
        k = _attach(command="nodes search", queries=["testvid", "Lip Sync"])["knowledge"]
        assert "models" not in k and "picks" not in k
        assert k["hit_ids"] == ["testvid", "cap:lipsync"]
        assert "uncurated_queries" not in k

    def test_a_shed_block_is_not_filed_as_a_gap(self, monkeypatch):
        """Every term resolved, so neither surface may name one. A curated term
        in the inbox costs a curator the same look as a real gap."""
        seen: list[tuple[str, dict]] = []
        monkeypatch.setattr(tracking, "track_event", lambda name, props=None, **kw: seen.append((name, props)))
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 300)
        _attach(command="nodes search", queries=["testvid", "Lip Sync"])
        assert seen[0][1]["hit_ids"] == ["testvid", "cap:lipsync"]
        assert "uncurated_queries" not in seen[0][1]

    def test_a_marker_too_big_for_the_ceiling_is_not_attached(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 80)
        assert "knowledge" not in _attach(queries=["faceswap"])

    def test_a_clipped_query_keeps_the_marker_under_the_ceiling(self):
        k = _attach(queries=["x-" * 4500], thin=False)["knowledge"]
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES
        assert len(k["uncurated_queries"][0]) == knowledge.MAX_QUERY_CHARS

    def test_a_hit_is_not_given_marker_fields(self):
        """The enrichment payload is unchanged wherever the bundle answered."""
        k = _attach(command="nodes search", queries=["testvid"])["knowledge"]
        assert k["models"][0]["id"] == "testvid"
        assert k["hit_ids"] == ["testvid"] and k["zero_hit"] is False
        assert "uncurated_queries" not in k

    def test_rows_borrowed_by_the_reverse_index_still_name_the_missed_term(self):
        """The query matched nothing and the reverse index supplied the rows, so
        `hit_ids` reads as a hit and `zero_hit` is false. Only the term list
        carries the gap, which otherwise survives as prose in the nudge."""
        k = _attach(queries=["FLF2V"], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["models"] and k["hit_ids"] == ["lipco-3"]
        assert k["zero_hit"] is False
        assert k["uncurated_queries"] == ["FLF2V"]
        assert "'FLF2V'" in k["nudge"]

    def test_one_term_landing_does_not_cover_the_one_beside_it(self):
        k = _attach(command="templates ls", queries=["testvid", "zzzz"])["knowledge"]
        assert [e["id"] for e in k["models"]] == ["testvid"]
        assert k["uncurated_queries"] == ["zzzz"]

    def test_a_term_the_reverse_index_answers_by_name_is_not_a_miss(self):
        """`nodes search <ClassName>` passes one string as both a query and a
        reverse-index key. The block answers it on the same line it arrived on."""
        k = _attach(queries=[_REVERSE_TEMPLATE], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert "uncurated_queries" not in k

    def test_a_nudge_that_does_not_fit_is_dropped(self, monkeypatch):
        args = {"queries": ["FLF2V"], "templates": [_REVERSE_TEMPLATE]}
        full = _attach(**args)["knowledge"]
        bare = {key: value for key, value in full.items() if key != "nudge"}
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", knowledge._block_bytes(bare))
        k = _attach(**args)["knowledge"]
        assert "nudge" not in k
        assert k["uncurated_queries"] == ["FLF2V"]

    def test_the_term_list_outlives_the_prose_that_restates_it(self, monkeypatch):
        """The nudge writes one term out as a sentence. The list is every term
        and the only form a consumer can count, so the prose goes first."""
        terms = ["A" * 200, "B" * 200, "C" * 200]
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 2000)
        k = _attach(queries=terms, templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["uncurated_queries"] == terms
        assert k["models"], "a row this size still fits, only the prose had to go"
        assert "nudge" not in k
        assert knowledge._block_bytes(k) <= 2000

    def test_the_term_list_outlives_the_rows_as_well(self, monkeypatch):
        """_fit sheds to make room rather than leaving the gap unrecorded. The
        rows answer one query, the list is every term that missed."""
        terms = ["A" * 200, "B" * 200, "C" * 200]
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 1800)
        k = _attach(queries=terms, templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["uncurated_queries"] == terms
        assert "models" not in k and "capabilities_available" not in k
        assert knowledge._block_bytes(k) <= 1800

    def test_terms_too_long_for_the_ceiling_are_given_up_last(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 700)
        assert "knowledge" not in _attach(queries=["A" * 200, "B" * 200, "C" * 200])

    def test_the_envelope_and_the_event_agree_under_byte_pressure(self, monkeypatch):
        seen: list[tuple[str, dict]] = []
        monkeypatch.setattr(tracking, "track_event", lambda name, props=None, **kw: seen.append((name, props)))
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 2000)
        terms = ["A" * 200, "B" * 200, "C" * 200]
        k = _attach(command="templates ls", queries=terms, templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["uncurated_queries"] == seen[0][1]["uncurated_queries"] == terms

    def test_a_ceiling_nothing_fits_under_still_reaches_the_miss_log(self, monkeypatch):
        """The fail-open `except` turns any slip here into a silent loss of the
        whole payload's enrichment and of the event with it."""
        seen: list[tuple[str, dict]] = []
        monkeypatch.setattr(tracking, "track_event", lambda name, props=None, **kw: seen.append((name, props)))
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 10)
        assert "knowledge" not in _attach(command="nodes search", queries=["testvid"])
        assert [name for name, _ in seen] == ["knowledge_query"]

    def test_a_term_the_cap_dropped_the_row_for_is_not_a_gap(self, monkeypatch):
        """The bundle covers the term either way. Which rows won a place in the
        block is a byte budget, not a statement about what is curated."""
        monkeypatch.setattr(knowledge, "MAX_MODELS", 0)
        k = _attach(queries=[_REVERSE_TEMPLATE], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["models"] == []
        assert "uncurated_queries" not in k
        assert "nudge" not in k

    def test_a_loosely_spelled_class_name_is_not_a_gap(self):
        """`nodes search` matches a class name without regard to case or spacing,
        so the term that comes back enriched must not also read as missing."""
        for spelling in (_REVERSE_TEMPLATE.upper(), _REVERSE_TEMPLATE.replace("_", " ")):
            k = _attach(queries=[spelling], templates=[_REVERSE_TEMPLATE])["knowledge"]
            assert "uncurated_queries" not in k, spelling
            assert "nudge" not in k, spelling

    def test_a_call_that_asked_nothing_still_attaches_nothing(self):
        assert "knowledge" not in _attach(templates=["nope"])
        assert "knowledge" not in _attach(command="nodes ls", queries=["zzzz"], qualified=False)

    def test_the_disable_flag_suppresses_the_marker(self, monkeypatch):
        monkeypatch.setenv(knowledge.ENV_DISABLE, "1")
        assert "knowledge" not in _attach(command="nodes search", queries=["faceswap"])


class TestCacheOnly:
    @pytest.fixture
    def http_calls(self, monkeypatch):
        calls: list[str] = []

        def _record(url: str) -> bytes:
            calls.append(url)
            raise AssertionError(f"network touched: {url}")

        monkeypatch.delenv(knowledge.ENV_FILE)
        monkeypatch.setenv(knowledge.ENV_URL, "https://example.invalid/knowledge.json")
        monkeypatch.setattr(knowledge, "_http_get", _record)
        knowledge._reset_for_testing()
        return calls

    def test_url_and_empty_cache_never_fetches(self, http_calls):
        p: dict = {"count": 0}
        knowledge.attach(p, queries=["testvid"], thin=True)
        assert http_calls == []
        assert p == {"count": 0}

    def test_stale_cache_still_enriches_without_fetching(self, http_calls, monkeypatch):
        import os
        import shutil
        import time

        k_path, m_path = knowledge.cache_paths()
        k_path.parent.mkdir(parents=True)
        shutil.copy(FIXTURE_KNOWLEDGE, k_path)
        shutil.copy(FIXTURE_KNOWLEDGE.parent / "manifest.json", m_path)
        old = time.time() - 2 * knowledge.DEFAULT_TTL_SECONDS
        os.utime(k_path, (old, old))
        p = _attach(queries=["testvid"])
        assert http_calls == []
        assert p["knowledge"]["stale"] is True
        assert p["knowledge"]["models"][0]["id"] == "testvid"

    def test_default_load_still_fetches(self, http_calls):
        knowledge.load_bundle()
        assert http_calls == ["https://example.invalid/knowledge.json"]

    def test_cache_only_miss_does_not_poison_a_later_full_load(self, http_calls):
        assert knowledge.load_bundle(cache_only=True) is None
        assert http_calls == []
        knowledge.load_bundle()
        assert http_calls == ["https://example.invalid/knowledge.json"]


class TestDisableSwitch:
    """``COMFY_KNOWLEDGE_DISABLE`` is the only off switch.

    Clearing ``COMFY_KNOWLEDGE_URL`` does not stop enrichment: once a bundle is
    cached, ``_load`` keeps reading it whether or not it is stale.
    """

    def test_a_set_flag_suppresses_the_block(self, monkeypatch, capsys):
        assert "knowledge" in _attach(queries=["testvid"])

        monkeypatch.setenv(knowledge.ENV_DISABLE, "1")
        p = {"rows": [], "count": 0}
        knowledge.attach(p, queries=["testvid"], thin=True)
        assert p == {"rows": [], "count": 0}
        assert capsys.readouterr() == ("", "")

    @pytest.mark.parametrize("value", ["1", "true", "yes", "please", "0.0"])
    def test_any_value_but_empty_or_zero_disables(self, monkeypatch, value):
        monkeypatch.setenv(knowledge.ENV_DISABLE, value)
        assert "knowledge" not in _attach(queries=["testvid"])

    @pytest.mark.parametrize("value", ["", "0"])
    def test_empty_and_zero_leave_it_on(self, monkeypatch, value):
        monkeypatch.setenv(knowledge.ENV_DISABLE, value)
        assert "knowledge" in _attach(queries=["testvid"])

    def test_the_explicit_verbs_still_resolve(self, monkeypatch):
        """The flag governs envelope enrichment, not the bundle itself, so
        ``comfy knowledge resolve|pick|status`` keep working under it."""
        monkeypatch.setenv(knowledge.ENV_DISABLE, "1")
        bundle = knowledge.load_bundle()
        assert bundle is not None
        assert "testvid" in bundle.models


class TestFailOpen:
    def test_no_bundle_adds_no_key(self, monkeypatch, capsys):
        monkeypatch.delenv(knowledge.ENV_FILE)
        knowledge._reset_for_testing()
        p = {"rows": [], "count": 0}
        knowledge.attach(p, queries=["testvid"], thin=True)
        assert p == {"rows": [], "count": 0}
        assert capsys.readouterr() == ("", "")

    def test_exception_inside_is_swallowed(self, monkeypatch, capsys):
        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(knowledge, "_lookup", _boom)
        p = {"rows": [{"name": "x"}], "count": 1}
        knowledge.attach(p, queries=["testvid"])
        assert p == {"rows": [{"name": "x"}], "count": 1}
        assert capsys.readouterr() == ("", "")

    def test_bad_inputs_are_swallowed(self, capsys):
        p = {"count": 1}
        knowledge.attach(p, queries=[None, 3, "testvid"], templates=[None], nodes=[7])  # type: ignore[list-item]
        assert p["knowledge"]["models"][0]["id"] == "testvid"
        knowledge.attach(p, queries=object())  # type: ignore[arg-type]
        assert capsys.readouterr() == ("", "")

    def test_existing_keys_untouched_and_knowledge_is_last(self):
        p = {"rows": [{"name": "a"}], "count": 1}
        before = json.dumps(p)
        knowledge.attach(p, queries=["testvid"])
        assert list(p) == ["rows", "count", "knowledge"]
        del p["knowledge"]
        assert json.dumps(p) == before


class TestIndex:
    def test_bundle_carries_normalized_maps(self):
        b = knowledge.load_bundle()
        assert b.normalized_aliases["testvid30"] == "testvid"
        assert b.normalized_aliases["halo3"] == "acme-h3"
        assert b.normalized_capabilities["lipsync"] == "lipsync"
        assert b.normalized_capabilities["talkinghead"] == "lipsync"
        assert b.normalized_capabilities["audiogeneration"] == "audio-generation"

    def test_malformed_capability_aliases_are_skipped(self):
        b = knowledge._index(
            {
                "models": {},
                "capabilities": {"c": {"aliases": ["Ok One", 3, None, ""]}, "d": {"aliases": "nope"}},
            },
            None,
            source="env",
            stale=False,
            path="x",
            mtime=0.0,
        )
        assert b.normalized_capabilities == {"c": "c", "okone": "c", "d": "d"}

    def test_capability_alias_shared_by_two_ids_is_dropped(self):
        b = knowledge._index(
            {"models": {}, "capabilities": {"c": {"aliases": ["Same"]}, "d": {"aliases": ["same"]}}},
            None,
            source="env",
            stale=False,
            path="x",
            mtime=0.0,
        )
        assert b.normalized_capabilities == {"c": "c", "d": "d"}


class TestQueryLog:
    """One event per enriched command, carrying every subject it was asked about.
    Without it the capture loop has no feed from local installs, and what gets
    curated stays a matter of intuition."""

    @pytest.fixture
    def events(self, monkeypatch):
        seen: list[tuple[str, dict]] = []
        monkeypatch.setattr(tracking, "track_event", lambda name, props=None, **kw: seen.append((name, props)))
        return seen

    def test_a_hit_is_logged_with_its_ids(self, events):
        _attach(command="nodes search", queries=["testvid"])
        assert [name for name, _ in events] == ["knowledge_query"]
        props = events[0][1]
        assert props["command"] == "nodes search"
        assert props["queries"] == ["testvid"]
        assert props["hit_ids"] == ["testvid"]
        assert props["zero_hit"] is False
        assert props["bundle_version"] == "0.1.0-fixture"

    def test_a_miss_reaches_the_event_and_the_envelope_alike(self, events):
        """The two surfaces are one record. A consumer holding only the envelope
        recovers what the consent-gated event was told."""
        k = _attach(command="nodes search", queries=["zzzz"])["knowledge"]
        assert [name for name, _ in events] == ["knowledge_query"]
        props = events[0][1]
        assert props["hit_ids"] == k["hit_ids"] == []
        assert props["uncurated_queries"] == k["uncurated_queries"] == ["zzzz"]
        assert props["zero_hit"] == k["zero_hit"] is False
        assert props["bundle_version"] == k["bundle_version"]

    def test_a_call_that_missed_nothing_sends_no_term_list(self, events):
        """Absent on the block, absent on the event. An empty list on the event
        would read as a shape the envelope never emits for the same call."""
        _attach(command="nodes search", queries=["testvid"])
        assert "uncurated_queries" not in events[0][1]

    def test_the_event_names_the_terms_that_missed(self, events):
        _attach(command="templates ls", queries=["testvid", "zzzz"])
        props = events[0][1]
        assert props["queries"] == ["testvid", "zzzz"]
        assert props["uncurated_queries"] == ["zzzz"]
        assert props["hit_ids"] == ["testvid"]

    def test_every_query_is_logged_not_just_the_first(self, events):
        _attach(command="templates ls", queries=["testvid", "zzzz", "lipsync"])
        assert events[0][1]["queries"] == ["testvid", "zzzz", "lipsync"]

    def test_rows_the_command_listed_never_enter_the_log(self, events):
        _attach(command="generate list", queries=["zzzz"], models=["testvid", "acme-h3"])
        assert events[0][1]["queries"] == ["zzzz"]

    def test_a_listing_with_no_query_logs_nothing(self, events):
        _attach(command="generate list", models=["testvid"])
        assert events == []

    def test_a_zero_hit_is_logged_as_one(self, events):
        _attach(command="nodes search", queries=["zzzz"], thin=True)
        assert events[0][1]["zero_hit"] is True

    def test_a_call_with_no_query_logs_nothing(self, events):
        _attach(command="templates show", templates=["video_acme_h3_i2v"])
        assert events == []

    def test_an_unqualified_listing_logs_nothing(self, events):
        _attach(command="nodes ls", queries=["testvid"], qualified=False)
        assert events == []

    def test_a_telemetry_failure_never_breaks_the_payload(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("posthog down")

        monkeypatch.setattr(tracking, "track_event", boom)
        assert _attach(command="nodes search", queries=["testvid"])["knowledge"]["models"]
