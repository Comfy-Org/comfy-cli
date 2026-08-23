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
            assert "knowledge" not in _attach(queries=[q]), q

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
        assert k["hit_ids"] == ["testvid"]

    def test_unavailable_rows_sort_after_available_ones_in_their_tier(self):
        k = _attach(queries=["acme-h3", "testvid"], catalog_nodes={"AcmeH3ImageToVideo"})["knowledge"]
        assert [m["id"] for m in k["models"]] == ["acme-h3", "testvid"]
        assert k["models"][1]["available_locally"] is False

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


class TestNudge:
    def test_thin_zero_hit_gets_a_nudge(self):
        k = _attach(queries=["faceswap"], thin=True)["knowledge"]
        assert k["zero_hit"] is True
        assert k["models"] == [] and k["picks"] == [] and k["hit_ids"] == []
        assert "'faceswap'" in k["nudge"]
        assert "lipsync" in k["nudge"]
        assert "audio-generation" in k["nudge"]

    def test_nudge_uses_first_non_blank_query(self):
        k = _attach(queries=["  ", "faceswap", "other"], thin=True)["knowledge"]
        assert "'faceswap'" in k["nudge"]

    def test_nudge_stays_under_byte_ceiling_for_a_huge_query(self):
        k = _attach(queries=["x-" * 4500], thin=True)["knowledge"]
        assert k["zero_hit"] is True
        assert knowledge._block_bytes(k) <= knowledge.MAX_BLOCK_BYTES
        assert "lipsync" in k["nudge"]

    def test_nudge_drops_the_capability_list_rather_than_overrun(self, monkeypatch):
        monkeypatch.setattr(knowledge, "MAX_BLOCK_BYTES", 220)
        k = _attach(queries=["faceswap"], thin=True)["knowledge"]
        assert k["nudge"] == "no curated knowledge for 'faceswap'"
        assert knowledge._block_bytes(k) <= 220

    def test_unresolved_query_on_a_non_empty_block_gets_a_nudge(self):
        k = _attach(queries=["FLF2V"], templates=[_REVERSE_TEMPLATE])["knowledge"]
        assert k["models"]
        assert k["zero_hit"] is False
        assert "'FLF2V'" in k["nudge"]
        assert "lipsync" in k["nudge"]

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

    def test_not_thin_means_no_key(self):
        assert "knowledge" not in _attach(queries=["faceswap"], thin=False)

    def test_thin_without_queries_means_no_key(self):
        assert "knowledge" not in _attach(queries=[], thin=True)
        assert "knowledge" not in _attach(templates=["nope"], thin=True)


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
    """One event per query, hit or miss. Without it the capture loop has no feed
    from local installs, and what gets curated stays a matter of intuition."""

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
        assert props["query"] == "testvid"
        assert props["hit_ids"] == ["testvid"]
        assert props["zero_hit"] is False
        assert props["bundle_version"] == "0.1.0-fixture"

    def test_a_miss_is_logged_even_though_nothing_attaches(self, events):
        assert "knowledge" not in _attach(command="nodes search", queries=["zzzz"])
        assert [name for name, _ in events] == ["knowledge_query"]
        assert events[0][1]["hit_ids"] == []
        assert events[0][1]["query"] == "zzzz"

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
