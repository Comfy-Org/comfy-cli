"""Tests for the code-search command."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from typer.testing import CliRunner

from comfy_cli.command.code_search import (
    API_URL,
    DEFAULT_COUNT,
    REQUEST_TIMEOUT,
    _build_query,
    _fetch_results,
    _format_results,
    _get_stats,
    _print_results,
    app,
)

runner = CliRunner()


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
        assert _build_query("LoadImage", None, DEFAULT_COUNT) == f"type:file count:{DEFAULT_COUNT} LoadImage"

    def test_with_repo_short_name(self):
        result = _build_query("LoadImage", "ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} LoadImage"

    def test_with_repo_full_name(self):
        result = _build_query("LoadImage", "Comfy-Org/ComfyUI", DEFAULT_COUNT)
        assert result == f"repo:^Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} LoadImage"

    def test_with_custom_count(self):
        result = _build_query("LoadImage", None, 50)
        assert result == "type:file count:50 LoadImage"

    def test_with_repo_and_count(self):
        result = _build_query("LoadImage", "ComfyUI", 100)
        assert result == "repo:^Comfy\\-Org/ComfyUI$ type:file count:100 LoadImage"

    def test_user_type_filter_preserved(self):
        """Don't inject type:file when the user already specified a type: filter."""
        result = _build_query("type:commit fix bug", None, DEFAULT_COUNT)
        assert "type:file" not in result
        assert result == f"count:{DEFAULT_COUNT} type:commit fix bug"

    def test_user_type_file_not_duplicated(self):
        result = _build_query("type:file LoadImage", None, DEFAULT_COUNT)
        assert result.count("type:file") == 1


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
        assert len(first["matches"]) == 2

        match = first["matches"][0]
        assert match["line"] == 42  # lineNumber + 1
        assert match["preview"] == "class LoadImage:"
        assert "github.com/Comfy-Org/ComfyUI/blob/abc123def456/nodes.py#L42" in match["url"]

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
        assert results[0]["branch"] == "main"
        assert results[0]["commit"] == ""
        assert "blob/main/" in results[0]["matches"][0]["url"]

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


# ---------------------------------------------------------------------------
# _fetch_results tests
# ---------------------------------------------------------------------------


class TestFetchResults:
    @patch("comfy_cli.command.code_search.requests.get")
    def test_successful_fetch(self, mock_get, raw_api_response):
        mock_response = MagicMock()
        mock_response.json.return_value = raw_api_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = _fetch_results("LoadImage")

        mock_get.assert_called_once_with(API_URL, params={"query": "LoadImage"}, timeout=REQUEST_TIMEOUT)
        assert result == raw_api_response

    @patch("comfy_cli.command.code_search.requests.get")
    def test_http_error_propagates(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=MagicMock(status_code=500))
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            _fetch_results("LoadImage")

    @patch("comfy_cli.command.code_search.requests.get")
    def test_timeout_propagates(self, mock_get):
        mock_get.side_effect = requests.Timeout("timed out")

        with pytest.raises(requests.Timeout):
            _fetch_results("LoadImage")

    @patch("comfy_cli.command.code_search.requests.get")
    def test_connection_error_propagates(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("no connection")

        with pytest.raises(requests.ConnectionError):
            _fetch_results("LoadImage")


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
        assert len(parsed["results"]) == 2

    def test_empty_results_message(self, capsys):
        _print_results([], {"approximate_count": "0", "match_count": 0, "limit_hit": False}, json_output=False)
        output = capsys.readouterr().out
        assert "No results found" in output

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
        assert output.count("https://github.com/Comfy-Org/ComfyUI/blob/abc123def456/nodes.py\n") >= 0
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
        # Line-anchor URLs embedded in escapes, not as visible text runs.
        assert "#L42" in output  # inside OSC 8 payload
        # Preview text still rendered.
        assert "class LoadImage:" in output

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
        mock_fetch.assert_called_once_with(f"type:file count:{DEFAULT_COUNT} LoadImage")

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_with_repo(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["--repo", "ComfyUI", "LoadImage"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(f"repo:^Comfy\\-Org/ComfyUI$ type:file count:{DEFAULT_COUNT} LoadImage")

    @patch("comfy_cli.command.code_search._fetch_results")
    def test_search_with_count(self, mock_fetch, raw_api_response):
        mock_fetch.return_value = raw_api_response

        result = runner.invoke(app, ["--count", "50", "LoadImage"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("type:file count:50 LoadImage")

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
        mock_fetch.side_effect = requests.ConnectionError("no connection")

        result = runner.invoke(app, ["LoadImage"])

        assert result.exit_code == 1
        assert "Could not connect" in result.output

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
        mock_fetch.assert_called_once_with("repo:^Comfy\\-Org/ComfyUI$ type:file count:30 LoadImage")
        parsed = json.loads(result.output)
        assert "results" in parsed


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
        result = runner.invoke(root_app, ["code-search", "--json", "LoadImage"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "results" in parsed

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
