"""Root CLI tests for local docs search and retrieval."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli import docs as docs_index
from comfy_cli.cmdline import app


@pytest.fixture
def invoke_docs():
    def invoke(args: list[str]):
        with (
            patch(
                "comfy_cli.cmdline.workspace_manager.setup_workspace_manager",
                side_effect=AssertionError("workspace init"),
            ),
            patch("comfy_cli.tracking.prompt_tracking_consent", side_effect=AssertionError("tracking prompt")),
        ):
            return CliRunner().invoke(app, ["--json", "docs", *args], env={"DO_NOT_TRACK": "1"})

    return invoke


def _envelope(result):
    assert result.stdout.strip(), result.output
    return json.loads(result.stdout)


def test_search_and_show_work_without_workspace_or_tracking(invoke_docs):
    result = invoke_docs(["search", "install custom nodes"])
    assert result.exit_code == 0, result.output
    envelope = _envelope(result)
    assert envelope["ok"] is True
    assert envelope["command"] == "docs search"
    match = envelope["data"]["results"][0]
    assert match["source"] == "README.md"
    assert "install" in match["excerpt"].casefold()

    shown = invoke_docs(["show", match["id"]])
    assert shown.exit_code == 0, shown.output
    data = _envelope(shown)["data"]
    assert data["id"] == match["id"]
    assert data["content"]
    assert data["corpus_hash"] == envelope["data"]["corpus_hash"]


@pytest.mark.parametrize(
    ("args", "expected_code"),
    [
        (["search"], "usage_error"),
        (["search", ""], "usage_error"),
        (["search", "workflow", "--limit", "0"], "usage_error"),
        (["show", "readme:usage", "--max-chars", "50001"], "usage_error"),
        (["show", "readme:usage", "--offset", "-1"], "usage_error"),
        (["show", "missing-section"], "docs_not_found"),
    ],
)
def test_errors_are_structured_in_the_json_envelope(invoke_docs, args, expected_code):
    result = invoke_docs(args)
    assert result.exit_code != 0
    envelope = _envelope(result)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == expected_code


def test_missing_corpus_has_a_structured_error(invoke_docs, monkeypatch):
    def unavailable():
        raise docs_index.DocsUnavailableError("missing")

    monkeypatch.setattr(docs_index, "load_corpus", unavailable)
    result = invoke_docs(["search", "workflow"])
    assert result.exit_code == 1
    envelope = _envelope(result)
    assert envelope["error"]["code"] == "docs_unavailable"


def test_out_of_range_section_offset_is_a_usage_error(invoke_docs):
    section = docs_index.load_corpus()[0][0]
    invalid_offset = len(section["content"]) + 1
    result = invoke_docs(["show", section["id"], "--offset", str(invalid_offset)])
    assert result.exit_code == 2
    assert _envelope(result)["error"]["code"] == "usage_error"


def test_pretty_output_prints_document_text_literally():
    runner = CliRunner()
    search_result = runner.invoke(app, ["--no-json", "docs", "search", "custom nodes"])
    assert search_result.exit_code == 0, search_result.output
    assert "custom-nodes:" in search_result.output

    section = docs_index.search("custom nodes")["results"][0]
    show_result = runner.invoke(app, ["--no-json", "docs", "show", section["id"]])
    assert show_result.exit_code == 0, show_result.output
    assert section["source"] in show_result.output
