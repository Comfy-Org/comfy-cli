"""Tests for ``comfy nodes`` — agent-facing node-class introspection.

The commands are thin wrappers over the CQL stub's loader, so the heavy
lifting is in the wiring (filter precedence, error codes, envelope shape).
Tests use a hand-rolled graph fixture instead of hitting a live server.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    """Pin the renderer to JSON so tests can read envelopes off stdout."""
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _fake_object_info() -> dict[str, Any]:
    """A small object_info dict covering the cases the tests assert on."""
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {}},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "category": "loaders",
            "display_name": "Load Checkpoint",
            "description": "Loads a diffusion model checkpoint.",
            "output_node": False,
            "python_module": "nodes",
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "sampler_name": [["euler", "heun", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras", "simple"], {"default": "normal"}],
                },
            },
            "input_order": {"required": ["model", "positive", "steps", "sampler_name", "scheduler"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise the latent via the provided model.",
            "output_node": False,
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "clip": ["CLIP"],
                    "text": ["STRING", {"multiline": True}],
                },
            },
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode (Prompt)",
            "description": "Encode prompt text to conditioning.",
            "output_node": False,
            "python_module": "nodes",
        },
        "SaveImage": {
            "input": {"required": {}},
            "output": [],
            "category": "image",
            "display_name": "Save Image",
            "description": "Save image to disk.",
            "output_node": True,
            "python_module": "nodes",
        },
    }


def _search_object_info() -> dict[str, Any]:
    """`_fake_object_info` plus the node names the search tiers discriminate on.

    Kept separate so the `ls` tests keep their exact-count assertions.
    """
    info = _fake_object_info()
    info["KSamplerAdvanced"] = {
        "input": {"required": {"model": ["MODEL"]}},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
        "display_name": "KSampler (Advanced)",
        "description": "Denoise the latent with extra knobs.",
        "output_node": False,
        "python_module": "nodes",
    }
    info["VAEDecode"] = {
        "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "latent",
        "display_name": "VAE Decode",
        "description": "Turn a latent back into pixels.",
        "output_node": False,
        "python_module": "nodes",
    }
    return info


def _fake_graph():
    """Build a Graph from the fake object_info."""
    from comfy_cli.cql.engine import Graph

    return Graph.from_object_info(_fake_object_info())


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    """Bypass network/file loading; serve the fake graph straight to the command."""
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _fake_graph())


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(nodes_cmd.app, args, standalone_mode=False)
    # The renderer writes to its own stream; capsys captures stdout.
    captured = capsys.readouterr().out
    if not captured.strip():
        # Renderer wrote to its bound stream (sys.stdout in JSON mode); the
        # CliRunner may have stolen it. Fall back to result.stdout.
        captured = result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code})"
    return json.loads(captured.strip().splitlines()[-1])


class TestLs:
    def test_no_filter_returns_all(self, patched_loader, capsys):
        env = _run(["ls"], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 4

    def test_produces_filter(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "MODEL"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "CheckpointLoaderSimple"

    def test_produces_filter_case_insensitive(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "model"], capsys)
        assert env["data"]["count"] == 1

    def test_accepts_filter(self, patched_loader, capsys):
        env = _run(["ls", "--accepts", "MODEL"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_category_glob(self, patched_loader, capsys):
        env = _run(["ls", "--category", "sampling*"], capsys)
        assert env["data"]["count"] == 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_category_sql_percent_pattern(self, patched_loader, capsys):
        """Agents from the SQL-CQL grammar might still send `%`."""
        env = _run(["ls", "--category", "samp%"], capsys)
        assert env["data"]["count"] == 1

    def test_limit(self, patched_loader, capsys):
        env = _run(["ls", "--limit", "2"], capsys)
        assert env["data"]["count"] == 2

    def test_filter_block_present_in_envelope(self, patched_loader, capsys):
        env = _run(["ls", "--produces", "LATENT", "--category", "samp*"], capsys)
        f = env["data"]["filter"]
        assert f["produces"] == "LATENT"
        assert f["accepts"] is None
        assert f["category"] == "samp*"


class TestShow:
    def test_basic_envelope(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        assert env["ok"] is True
        d = env["data"]
        assert d["name"] == "KSampler"
        assert d["category"] == "sampling"
        assert d["output_types"] == ["LATENT"]
        # Inputs include section + type + options.
        inputs = {i["name"]: i for i in d["inputs"]}
        assert "model" in inputs and inputs["model"]["type"] == "MODEL"
        assert inputs["steps"]["options"]["default"] == 20

    def test_inputs_sorted_required_first(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        sections = [i["section"] for i in env["data"]["inputs"]]
        # No `optional` in the fixture; just verify all required come before any non-required.
        first_optional = next((i for i, s in enumerate(sections) if s != "required"), len(sections))
        last_required = max((i for i, s in enumerate(sections) if s == "required"), default=-1)
        assert last_required < first_optional

    def test_node_not_found_emits_structured_error(self, patched_loader, capsys):
        env = _run(["show", "Nonexistent"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "node_not_found"
        assert "close_matches" in env["error"]["details"]

    def test_node_not_found_suggests_close_matches(self, patched_loader, capsys):
        # Lowercase typo should still surface KSampler as a close match.
        env = _run(["show", "ksampler"], capsys)
        close = env["error"]["details"]["close_matches"]
        assert "KSampler" in close

    def test_choices_populated_for_local_enum(self, patched_loader, capsys):
        env = _run(["show", "KSampler"], capsys)
        inputs = {i["name"]: i for i in env["data"]["inputs"]}
        assert inputs["sampler_name"]["choices"] == ["euler", "heun", "dpmpp_2m"]

    def test_choices_populated_for_cloud_combo(self, patched_loader, capsys):
        """Cloud-API COMBO inputs nest their choices at options.options.
        `nodes show` should normalize them into the same `choices` array
        as local ENUM inputs — agents shouldn't need to know which shape
        the graph happens to use."""
        env = _run(["show", "KSampler"], capsys)
        inputs = {i["name"]: i for i in env["data"]["inputs"]}
        assert inputs["scheduler"]["choices"] == ["normal", "karras", "simple"]
        # And the raw options block is still passed through for callers
        # that need the default / min / max metadata.
        assert inputs["scheduler"]["options"]["default"] == "normal"


class TestSearch:
    @pytest.fixture
    def patched_loader(self, monkeypatch: pytest.MonkeyPatch):
        """Class-local override: search asserts on tiering, so it needs the
        richer node set (`KSamplerAdvanced`, `VAEDecode`)."""
        from comfy_cli.cql.engine import Graph

        graph = Graph.from_object_info(_search_object_info())
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

    def test_exact_name_wins(self, patched_loader, capsys):
        env = _run(["search", "KSampler"], capsys)
        assert env["data"]["count"] >= 1
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_substring_match(self, patched_loader, capsys):
        env = _run(["search", "checkpoint"], capsys)
        assert env["data"]["count"] >= 1
        assert env["data"]["rows"][0]["name"] == "CheckpointLoaderSimple"

    def test_description_match(self, patched_loader, capsys):
        env = _run(["search", "Denoise"], capsys)
        # KSampler's description contains "Denoise"; should match.
        names = [r["name"] for r in env["data"]["rows"]]
        assert "KSampler" in names

    def test_no_match_returns_empty(self, patched_loader, capsys):
        env = _run(["search", "xyzzy_nothing"], capsys)
        assert env["data"]["count"] == 0
        assert env["data"]["rows"] == []

    def test_limit_caps_results(self, patched_loader, capsys):
        env = _run(["search", "e", "--limit", "2"], capsys)
        assert env["data"]["count"] <= 2

    def test_multi_word_query_hits_camelcase_class(self, patched_loader, capsys):
        """'ksampler advanced' must find KSamplerAdvanced — the display name's
        parens used to defeat the contiguous-substring match."""
        env = _run(["search", "ksampler advanced"], capsys)
        assert env["data"]["total"] >= 1
        assert env["data"]["rows"][0]["name"] == "KSamplerAdvanced"

    def test_multi_word_query_is_word_order_independent(self, patched_loader, capsys):
        env = _run(["search", "advanced ksampler"], capsys)
        assert env["data"]["total"] >= 1
        assert env["data"]["rows"][0]["name"] == "KSamplerAdvanced"

    @pytest.mark.parametrize("query", ["vae decode", "decode vae"])
    def test_spaced_query_matches_either_order(self, query, patched_loader, capsys):
        env = _run(["search", query], capsys)
        names = [r["name"] for r in env["data"]["rows"]]
        assert "VAEDecode" in names

    def test_category_is_searched(self, patched_loader, capsys):
        """`sampling` is only in the category field of the KSampler nodes."""
        env = _run(["search", "sampling"], capsys)
        names = [r["name"] for r in env["data"]["rows"]]
        assert "KSampler" in names
        assert "KSamplerAdvanced" in names

    def test_exact_name_outranks_prefix_sibling(self, patched_loader, capsys):
        """Regression: tier 0 (exact) must beat KSamplerAdvanced's tier 1."""
        env = _run(["search", "KSampler"], capsys)
        assert env["data"]["rows"][0]["name"] == "KSampler"

    def test_typo_falls_back_to_close_matches(self, patched_loader, capsys):
        env = _run(["search", "KSampeler"], capsys)
        rows = env["data"]["rows"]
        assert rows, "close-match fallback should surface something for a typo"
        assert rows[0]["name"] == "KSampler"
        assert all(r["close_match"] is True for r in rows)
        assert env["data"]["total"] == env["data"]["count"]

    def test_exact_matches_carry_no_close_match_flag(self, patched_loader, capsys):
        env = _run(["search", "KSampler"], capsys)
        assert all("close_match" not in r for r in env["data"]["rows"])

    def test_no_match_and_no_close_match_returns_empty(self, patched_loader, capsys):
        """Nothing is within difflib's 0.6 cutoff of this, so the fallback is
        empty too — an agent gets a clean zero, not noise."""
        env = _run(["search", "xyzzy_nothing"], capsys)
        assert env["data"]["total"] == 0
        assert env["data"]["rows"] == []

    def test_zero_limit_does_not_crash_the_fallback(self, patched_loader, capsys):
        """`difflib.get_close_matches` rejects n <= 0; the fallback must not
        pass the raw limit straight through."""
        env = _run(["search", "KSampeler", "--limit", "0"], capsys)
        assert env["ok"] is True
        assert env["data"]["rows"] == []

    @pytest.mark.parametrize("query", [" ", "   ", "\t ", ""])
    def test_blank_query_matches_nothing(self, query, patched_loader, capsys):
        """A query with no tokens must not match the whole catalog.

        `" "` is a substring of every blob (the fields are joined with spaces)
        and an empty token list makes `all(...)` vacuously true, so both used to
        return every node as a "match".
        """
        env = _run(["search", query], capsys)
        assert env["ok"] is True
        assert env["data"]["total"] == 0
        assert env["data"]["rows"] == []
        assert env["data"]["close_match"] is False

    def test_close_match_flag_is_always_present_at_top_level(self, patched_loader, capsys):
        """A caller gating on `count == 0` needs one stable place to learn the
        search is guessing, without inspecting every row."""
        hit = _run(["search", "KSampler"], capsys)
        assert hit["data"]["close_match"] is False
        guess = _run(["search", "KSampeler"], capsys)
        assert guess["data"]["close_match"] is True

    def test_zero_limit_still_reports_the_close_match_total(self, patched_loader, capsys):
        """`total` counts before the `--limit` slice on the fallback path too,
        so `--limit 0` doesn't erase the fact that a close match exists."""
        env = _run(["search", "KSampeler", "--limit", "0"], capsys)
        assert env["data"]["total"] >= 1
        assert env["data"]["count"] == 0
        assert env["data"]["close_match"] is True

    def test_zero_limit_keeps_the_scored_total(self, patched_loader, capsys):
        """Same invariant on the normal path: rows are capped, `total` isn't."""
        env = _run(["search", "KSampler", "--limit", "0"], capsys)
        assert env["data"]["total"] >= 1
        assert env["data"]["rows"] == []

    def test_case_colliding_ids_are_both_suggested(self, monkeypatch, capsys):
        """A pack may register both `LoadImage` and `loadimage`; bucketing ids by
        their lowered form for difflib must not drop one of them."""
        from comfy_cli.cql.engine import Graph

        info = _search_object_info()
        for node_id in ("LoadImage", "loadimage"):
            info[node_id] = {
                "input": {"required": {}},
                "output": ["IMAGE"],
                "output_name": ["IMAGE"],
                "category": "image",
                "display_name": node_id,
                "description": "Load an image.",
                "python_module": "nodes",
            }
        graph = Graph.from_object_info(info)
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

        env = _run(["search", "loadimgae"], capsys)
        names = [r["name"] for r in env["data"]["rows"]]
        assert env["data"]["close_match"] is True
        assert "LoadImage" in names
        assert "loadimage" in names

    def test_pretty_empty_state_agrees_with_the_json_total(self, patched_loader):
        """The two sinks must not contradict each other.

        Keying the empty state on the post-slice list made `--limit 0` print
        "No nodes match" while the envelope reported `total > 0`; and the
        fallback's finds are guesses, so the limit-dropped-everything line must
        not call them matches either.
        """
        import io

        from comfy_cli.command.nodes import search_cmd

        def pretty_run(**kwargs) -> str:
            stream = io.StringIO()
            r = Renderer.resolve(is_stdout_tty=True, env={}, caller=None)
            r.mode = OutputMode.PRETTY
            r.pretty_stream = stream
            set_renderer(r)
            search_cmd(input_path=None, host=None, port=None, where=None, **kwargs)
            return stream.getvalue()

        hit = pretty_run(query="KSampler", limit=0)
        assert "No nodes match" not in hit
        assert "--limit 0 returned none" in hit

        guess = pretty_run(query="KSampeler", limit=0)
        assert "No nodes match" in guess, "a fallback find is not a match"
        assert "close name match" in guess

        # The ordinary fallback still renders its rows rather than falling into
        # the limit-dropped-everything branch.
        shown = pretty_run(query="KSampeler", limit=20)
        assert "showing close name matches" in shown
        assert "KSampler" in shown

    def test_non_string_category_does_not_crash(self, monkeypatch, capsys):
        """/object_info is server-supplied; a custom node can declare a category
        that isn't a string, and `.lower()` on it used to raise a raw
        AttributeError that took down every search."""
        from comfy_cli.cql.engine import Graph

        info = _search_object_info()
        info["HostileNode"] = {
            "input": {"required": {}},
            "output": [],
            "output_name": [],
            "category": 42,
            "display_name": ["not", "a", "string"],
            "description": {"nope": True},
            "python_module": 7,
            "output_node": False,
        }
        graph = Graph.from_object_info(info)
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

        env = _run(["search", "sampling"], capsys)
        assert env["ok"] is True
        assert "KSampler" in [r["name"] for r in env["data"]["rows"]]


class TestIsApiNodeRows:
    """`is_api_node` on `nodes ls` / `nodes search` rows.

    The twin-family trap: a paid partner-API node and a free open-weights one can
    carry the same display name, so an agent picking off a search row needs the
    flag inline rather than a follow-up `nodes show` per candidate.
    """

    @pytest.fixture
    def patched_loader(self, monkeypatch: pytest.MonkeyPatch):
        from comfy_cli.cql.engine import Graph

        info = {
            # Paid partner-API twin: object_info carries `api_node: true`.
            "MinimaxHailuo03ReferenceNode": {
                "input": {"required": {}},
                "output": ["VIDEO"],
                "output_name": ["VIDEO"],
                "category": "partner/video",
                "display_name": "MiniMax H3 Reference to Video",
                "description": "Generate video from a reference image.",
                "output_node": False,
                "api_node": True,
                "python_module": "comfy_api_nodes",
            },
            # Free open-weights twin: no `api_node` key at all.
            "MiniMaxH3ReferenceToVideo": {
                "input": {"required": {}},
                "output": ["VIDEO"],
                "output_name": ["VIDEO"],
                "category": "video",
                "display_name": "MiniMax H3 Reference to Video",
                "description": "Generate video from a reference image.",
                "output_node": False,
                "python_module": "nodes",
            },
        }
        graph = Graph.from_object_info(info)
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

    def test_ls_rows_carry_the_flag(self, patched_loader, capsys):
        env = _run(["ls"], capsys)
        flags = {r["name"]: r["is_api_node"] for r in env["data"]["rows"]}
        assert flags == {
            "MinimaxHailuo03ReferenceNode": True,
            "MiniMaxH3ReferenceToVideo": False,
        }

    def test_search_rows_carry_the_flag(self, patched_loader, capsys):
        """Both twins match the same query and are told apart only by the flag."""
        env = _run(["search", "MiniMax H3 Reference to Video"], capsys)
        flags = {r["name"]: r["is_api_node"] for r in env["data"]["rows"]}
        assert flags == {
            "MinimaxHailuo03ReferenceNode": True,
            "MiniMaxH3ReferenceToVideo": False,
        }

    def test_missing_api_node_key_is_false_not_absent(self, patched_loader, capsys):
        """A node whose object_info entry omits `api_node` reports `false`; a
        caller must never have to distinguish "free" from "field missing"."""
        for args in (["ls"], ["search", "MiniMaxH3ReferenceToVideo"]):
            env = _run(args, capsys)
            row = next(r for r in env["data"]["rows"] if r["name"] == "MiniMaxH3ReferenceToVideo")
            assert row["is_api_node"] is False

    def test_close_match_fallback_rows_carry_the_flag(self, patched_loader, capsys):
        """The typo path builds its rows from the same projection, so the flag
        must survive there too."""
        env = _run(["search", "MinimaxHailou03ReferenceNode"], capsys)
        rows = env["data"]["rows"]
        assert rows and all(r["close_match"] is True for r in rows)
        assert next(r for r in rows if r["name"] == "MinimaxHailuo03ReferenceNode")["is_api_node"] is True

    def test_api_only_filter_is_unchanged(self, patched_loader, capsys):
        """Additive change: `--api-only` still selects exactly the API nodes."""
        env = _run(["ls", "--api-only"], capsys)
        rows = env["data"]["rows"]
        assert [r["name"] for r in rows] == ["MinimaxHailuo03ReferenceNode"]
        assert rows[0]["is_api_node"] is True


class TestPath:
    """`comfy nodes path` — the envelope an agent plans a graph off (BE-6857)."""

    @pytest.fixture
    def patched_loader(self, monkeypatch: pytest.MonkeyPatch):
        import json as _json
        from pathlib import Path

        from comfy_cli.cql.engine import Graph

        fixture = Path(__file__).parent.parent / "fixtures" / "nodes_path_object_info.json"
        graph = Graph.from_object_info(_json.loads(fixture.read_text()))
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

    def test_reachable_route(self, patched_loader, capsys):
        env = _run(["path", "MODEL", "IMAGE", "--max-depth", "4"], capsys)
        data = env["data"]
        assert data["mode"] == "exact"
        assert data["count"] >= 1
        chains = [[s["node"] for s in p["steps"]] for p in data["paths"]]
        assert ["KSampler", "VAEDecode"] in chains
        first = data["paths"][0]["steps"][0]
        assert first["from_type"] == "MODEL"
        assert first["to_type"] == "LATENT"

    def test_unreachable_source_is_an_honest_empty_set(self, patched_loader, capsys):
        env = _run(["path", "AUDIO", "IMAGE", "--max-depth", "4", "--max-paths", "3"], capsys)
        data = env["data"]
        assert data["count"] == 0
        assert data["paths"] == []
        # Nothing was cut short, so the empty answer is genuinely exhaustive.
        assert data["truncated"] is False
        assert data["depth_limited"] is False
        assert data["exact"] is True

    def test_source_type_changes_the_rows(self, patched_loader, capsys):
        model = _run(["path", "MODEL", "IMAGE", "--max-depth", "4", "--max-paths", "3"], capsys)["data"]
        audio = _run(["path", "AUDIO", "IMAGE", "--max-depth", "4", "--max-paths", "3"], capsys)["data"]
        assert model["paths"] != audio["paths"]
        assert model["count"] > 0 and audio["count"] == 0

    def test_partner_api_node_with_a_model_widget_is_not_routed_through(self, patched_loader, capsys):
        env = _run(["path", "MODEL", "IMAGE", "--max-depth", "6"], capsys)
        nodes = {s["node"] for p in env["data"]["paths"] for s in p["steps"]}
        assert "ByteDanceImageNode" not in nodes

    def test_shallow_depth_is_a_subset_and_says_so(self, patched_loader, capsys):
        shallow = _run(["path", "MODEL", "IMAGE", "--max-depth", "1"], capsys)["data"]
        deep = _run(["path", "MODEL", "IMAGE", "--max-depth", "4"], capsys)["data"]
        assert shallow["count"] < deep["count"]
        for p in deep["paths"]:
            assert len(p["steps"]) <= 4
        # An empty result from a search that stopped at the depth bound is not
        # proof of unreachability, so `exact` is withheld.
        assert shallow["depth_limited"] is True
        assert shallow["exact"] is False

    def test_max_paths_truncation_withholds_the_exact_claim(self, patched_loader, capsys):
        env = _run(["path", "LATENT", "IMAGE", "--max-depth", "4", "--max-paths", "1"], capsys)["data"]
        assert env["count"] == 1
        assert env["truncated"] is True
        assert env["truncated_by"] == "max_paths"
        assert env["exact"] is False

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--max-depth", "0"), ("--max-depth", "-1"), ("--max-paths", "0"), ("--max-paths", "-3")],
    )
    def test_non_positive_bounds_are_refused_before_any_graph_load(self, monkeypatch, capsys, flag, value):
        """A bound below 1 admits no path, so the walker returns an empty result
        with every limit flag false — which the envelope would publish as
        `exact: true, count: 0`, a proof that no route exists. That proof would
        come from the typo, not from a walk, so the bound is rejected up front.

        Deliberately *not* using the `patched_loader` fixture: `_get_graph` is
        replaced with a tripwire, so the test fails if the command does any
        object_info I/O before validating the caller's bounds.
        """

        def _tripwire(*a, **kw):
            raise AssertionError("_get_graph was called before the bounds were validated")

        monkeypatch.setattr(nodes_cmd, "_get_graph", _tripwire)

        env = _run(["path", "MODEL", "IMAGE", flag, value], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "path_bounds_invalid"
        assert env["error"]["details"][flag.removeprefix("--").replace("-", "_")] == int(value)
        # Crucially: no envelope claiming an exhaustive empty answer.
        assert "data" not in env or not (env.get("data") or {}).get("exact")

    def test_smallest_valid_bounds_are_searched_not_refused(self, patched_loader, capsys):
        """The rejection is for bounds below 1 only. `1` is a legitimate bound:
        it must run a real (if very shallow) search rather than error out, even
        though nothing is reachable from MODEL in a single hop."""
        env = _run(["path", "MODEL", "IMAGE", "--max-depth", "1", "--max-paths", "1"], capsys)
        assert env["ok"] is True
        data = env["data"]
        assert data["count"] == 0
        # The walk genuinely ran and hit the depth bound — it was not declined.
        assert data["depth_limited"] is True
        assert data["not_searched"] is False
        assert data["exact"] is False

    def test_same_type_query_lists_the_route_it_used_to_decline(self, patched_loader, capsys):
        """`MODEL -> MODEL` is a *reachable* query — the fixture carries
        `LoraLoaderModelOnly`, a stock-shaped node taking a MODEL link input and
        emitting MODEL — and the CLI now answers it.

        It used to decline: the walker's no-op rule dropped any step whose
        output type equalled its input type, which for a same-type query is the
        terminal step, so the command returned `count: 0` with the abstention
        declared. Declining was honest but useless — the route is real, so it is
        listed.
        """
        lora = _run(["show", "LoraLoaderModelOnly"], capsys)["data"]
        assert "MODEL" in {o["type"] for o in lora["outputs"]}, "fixture must offer a real MODEL -> MODEL route"
        assert "MODEL" in {i["type"] for i in lora["inputs"]}

        data = _run(["path", "MODEL", "MODEL"], capsys)["data"]
        assert data["count"] >= 1
        one_step = [p for p in data["paths"] if [s["node"] for s in p["steps"]] == ["LoraLoaderModelOnly"]]
        assert one_step, "the one-step MODEL -> MODEL route must be listed"
        assert one_step[0]["steps"][0] == {
            "node": "LoraLoaderModelOnly",
            "from_type": "MODEL",
            "to_type": "MODEL",
        }
        # The walk ran to completion: no abstention, no bound bit it.
        assert data["not_searched"] is False
        assert data["not_searched_reason"] is None
        assert data["truncated"] is False
        assert data["depth_limited"] is False
        # `exact` is still withheld here, and for the ordinary reason rather
        # than a leftover of the old refusal: reaching MODEL ends a path, so the
        # walk keeps expanding the branches that do not (KSampler -> LATENT),
        # and there both decoders land on the same (IMAGE, {IMAGE, LATENT})
        # state — a collapse. It costs no MODEL -> MODEL route (nothing in this
        # catalog routes IMAGE back to MODEL), but the flag errs toward true by
        # design, so the claim is withheld rather than forged.
        assert data["collapsed"] is True
        assert data["exact"] is False

    def test_loose_mode_never_claims_exactness(self, patched_loader, capsys):
        env = _run(["path", "MODEL", "IMAGE", "--loose", "--max-depth", "4"], capsys)["data"]
        assert env["mode"] == "loose"
        assert env["exact"] is False
        assert env["count"] >= 1

    def test_support_inputs_are_reported(self, patched_loader, capsys):
        env = _run(["path", "MODEL", "IMAGE", "--max-depth", "4"], capsys)["data"]
        path = next(p for p in env["paths"] if [s["node"] for s in p["steps"]] == ["KSampler", "VAEDecode"])
        assert {s["type"] for s in path["support"]} == {"CONDITIONING", "LATENT", "VAE"}

    def test_collapsed_alternate_routes_withhold_the_exact_claim(self, monkeypatch, capsys):
        """A second node offering the same hop is not re-expanded, so its routes
        never reach the output. The envelope has to say so — a silently partial
        list labelled `exact` is the defect this ticket is about."""
        import copy
        import json as _json
        from pathlib import Path

        from comfy_cli.cql.engine import Graph

        fixture = Path(__file__).parent.parent / "fixtures" / "nodes_path_object_info.json"
        info = _json.loads(fixture.read_text())
        info["KSamplerAdvanced"] = copy.deepcopy(info["KSampler"])
        info["KSamplerAdvanced"]["name"] = "KSamplerAdvanced"
        graph = Graph.from_object_info(info)
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

        data = _run(["path", "MODEL", "IMAGE", "--max-depth", "3"], capsys)["data"]
        assert data["count"] > 0
        assert data["collapsed"] is True
        assert data["truncated"] is False
        assert data["depth_limited"] is False
        assert data["exact"] is False


class TestFlattenCategoryTree:
    """Pin the shape contract for the wasm CategoryTree, since the flattener
    has to know the (capital-cased) field names the Go side emits."""

    def test_walks_nested_children(self):
        tree = {
            "Root": {
                "Name": "",
                "FullPath": "",
                "Children": {
                    "loaders": {
                        "FullPath": "loaders",
                        "Count": 22,
                        "Children": {
                            "advanced": {
                                "FullPath": "loaders/advanced",
                                "Count": 4,
                                "Children": {},
                            },
                        },
                    },
                    "sampling": {
                        "FullPath": "sampling",
                        "Count": 8,
                        "Children": {},
                    },
                },
            },
        }
        from comfy_cli.command.nodes import _flatten_category_tree

        flat = _flatten_category_tree(tree)
        flat_dict = dict(flat)
        assert flat_dict["loaders"] == 22
        assert flat_dict["loaders/advanced"] == 4
        assert flat_dict["sampling"] == 8

    def test_empty_or_malformed_returns_empty(self):
        from comfy_cli.command.nodes import _flatten_category_tree

        assert _flatten_category_tree({}) == []
        assert _flatten_category_tree({"Root": None}) == []
        assert _flatten_category_tree("not a dict") == []  # type: ignore[arg-type]
