"""Freeze the edit-findings contract.

Two pins:

  * the severity vocabulary — ``cql.engine.FATAL_FINDING_CODES`` and
    ``finding_severity`` — so a code cannot change fatality unnoticed;
  * the language-neutral conformance corpus
    (``tests/data/edit_findings_conformance.json``) replays against the real
    engine, so the executable contract a non-Python consumer inherits cannot
    rot. That file is the portable form of this contract: a Go or TypeScript
    caller replays it instead of re-deriving the semantics.

Plus the outcome rule itself: a fatal finding refuses the edit and leaves the
document untouched, which is the behaviour the ~750 lines of Go in
`services/agent/internal/loop/enumgate.go` existed to synthesize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import (
    FATAL_FINDING_CODES,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    Graph,
    finding_severity,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "edit_findings_conformance.json"

_VALID_SEVERITIES = {SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO}


class TestSeverityVocabulary:
    """The severity table is `FATAL_FINDING_CODES` + `finding_severity`.

    Pinned here rather than against a prose doc: these assertions are what a
    non-Python consumer needs to agree with, and the conformance corpus below is
    their executable form.
    """

    def test_every_fatal_code_reports_error_severity(self):
        for code in FATAL_FINDING_CODES:
            assert finding_severity(code) == SEVERITY_ERROR, code

    def test_severity_values_are_the_declared_three(self):
        assert _VALID_SEVERITIES == {"error", "warning", "info"}

    def test_unlisted_code_is_advisory_never_fatal(self):
        """A finding added later cannot become silently fatal."""
        assert finding_severity("some_future_code") == SEVERITY_WARNING
        assert "some_future_code" not in FATAL_FINDING_CODES

    def test_fatal_set_is_exactly_the_four_run_time_rejections(self):
        """Locked deliberately: each is a value `validate_inputs` rejects, which
        is why the edit path refuses it rather than warning about it."""
        assert set(FATAL_FINDING_CODES) == {
            "unknown_enum_value",
            "no_options_available",
            "below_min",
            "above_max",
        }


# ---------------------------------------------------------------------------
# conformance corpus — the portable form of the contract
# ---------------------------------------------------------------------------


def _load_corpus() -> dict[str, Any]:
    assert CORPUS.is_file(), f"conformance corpus missing: {CORPUS}"
    return json.loads(CORPUS.read_text(encoding="utf-8"))


CORPUS_DATA = _load_corpus()
_GRAPH = Graph.from_object_info(CORPUS_DATA["object_info"])


@pytest.mark.parametrize("case", CORPUS_DATA["cases"], ids=lambda c: c["name"])
def test_conformance_corpus(case: dict[str, Any]):
    expect = case["expect"]
    call = lambda: workflow_ops._validate_widget(  # noqa: E731
        _GRAPH, case["class_type"], case["widget"], case["value"]
    )

    if not expect["fatal"]:
        assert call() == expect.get("findings", []), case["name"]
        return

    with pytest.raises(workflow_ops.FatalFindingError) as ei:
        call()
    f = ei.value.finding

    assert f["code"] == expect["code"]
    assert f["severity"] == expect["severity"]
    # The operand is a FIELD. This is the assertion that makes regexing the
    # message unnecessary, and it is the reason the corpus exists.
    assert f["value"] == expect["value"]
    if "field" in expect:
        assert f["field"] == expect["field"]
    if expect.get("has_valid_options"):
        assert f["valid_options"]
    if "did_you_mean_contains" in expect:
        assert expect["did_you_mean_contains"] in f.get("did_you_mean", [])


class TestOutcomeRule:
    """A fatal finding refuses the edit and leaves the document alone."""

    @staticmethod
    def _workflow() -> dict:
        return {
            "nodes": [{"id": 1, "type": "Loader", "widgets_values": ["v1-5-pruned-emaonly.safetensors", 20, ""]}],
            "links": [],
            "last_node_id": 1,
            "last_link_id": 0,
        }

    def test_fatal_edit_leaves_the_document_byte_identical(self):
        wf = self._workflow()
        before = json.loads(json.dumps(wf))
        with pytest.raises(workflow_ops.FatalFindingError):
            workflow_ops.set_widget(wf, _GRAPH, 1, "ckpt_name", "nope.safetensors")
        assert wf == before

    def test_non_fatal_edit_still_applies(self):
        wf = self._workflow()
        wf2, op = workflow_ops.set_widget(wf, _GRAPH, 1, "ckpt_name", "sd_xl_base_1.0.safetensors")
        assert op["value"] == "sd_xl_base_1.0.safetensors"
        assert wf2["nodes"][0]["widgets_values"][0] == "sd_xl_base_1.0.safetensors"


class TestDocumentCompleteness:
    """A serialized document carries the save-format keys consumers assume."""

    def test_absent_keys_are_filled_and_derived_from_content(self):
        wf = {"nodes": [{"id": 7, "type": "X"}], "links": [[3, 7, 0, 9, 0, "IMAGE"]]}
        workflow_ops.strip_internal(wf)
        assert wf["version"] == workflow_ops.SAVE_FORMAT_VERSION
        assert wf["last_node_id"] == 7
        assert wf["last_link_id"] == 3

    def test_existing_values_are_never_rewritten(self):
        wf = {"nodes": [], "links": [], "version": 0.4, "last_node_id": 999, "last_link_id": 42}
        workflow_ops.strip_internal(wf)
        assert (wf["last_node_id"], wf["last_link_id"]) == (999, 42)
