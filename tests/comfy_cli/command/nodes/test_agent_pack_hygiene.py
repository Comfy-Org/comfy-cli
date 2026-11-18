"""Tests for the advisory agent-pack hygiene check in `comfy node validate`.

The check reads a pack's frontend files and reports what the frontend's
Settings > Extensions switch cannot turn off: code that runs at module top
level, CSS aimed at the whole page, and an extension name without the
`agent.` prefix. It never fails validation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from comfy_cli.command.custom_nodes.agent_pack_hygiene import (
    check_pack,
    is_agent_pack,
    top_level_statements,
)
from comfy_cli.command.custom_nodes.command import app

runner = CliRunner()

GOOD_JS = """\
import { app } from "../../scripts/app.js";

// Comments and strings with braces { do not count }
const STYLE = `.agent-cushions-node { border-radius: 12px; }`;

function addStyle() {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "agent.cushions.style",
  setup() {
    addStyle();
  },
});
"""


def _pack(tmp_path, js: str | None = GOOD_JS, *, web_dir="web", init=None, record=True, files=None):
    pack = tmp_path / "cushions"
    pack.mkdir()
    (pack / "__init__.py").write_text(
        init if init is not None else f'WEB_DIRECTORY = "./{web_dir}"\nNODE_CLASS_MAPPINGS = {{}}\n'
    )
    web = pack / web_dir
    web.mkdir(parents=True)
    if js is not None:
        (web / "main.js").write_text(js)
    for name, content in (files or {}).items():
        target = web / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if record:
        (pack / ".comfy-agent-install").write_text(json.dumps({"v": 1, "pack": "cushions"}))
    return pack


def _kinds(findings):
    return sorted({f.kind for f in findings})


class TestTopLevelStatements:
    def test_splits_on_depth_zero_semicolons_and_newlines(self):
        stmts = [text for _, text in top_level_statements("a();\nfunction f() { b(); c(); }\nd = 1")]
        assert stmts == ["a()", "function f() { b(); c(); }", "d = 1"]

    def test_ignores_braces_in_strings_comments_templates_and_regexes(self):
        src = 'const s = "{";\n// }\n/* { */\nconst t = `${"}"}`;\nconst r = /[{]/g;\nx();'
        stmts = [text for _, text in top_level_statements(src)]
        assert stmts[-1] == "x()"
        assert len(stmts) == 4

    def test_reports_the_line_a_statement_starts_on(self):
        lines = [line for line, _ in top_level_statements("\n\nfoo();\n\nbar();")]
        assert lines == [3, 5]


class TestCheckPack:
    def test_a_clean_agent_pack_has_no_findings(self, tmp_path):
        assert check_pack(_pack(tmp_path)) == []

    def test_top_level_side_effects_are_reported_with_their_line(self, tmp_path):
        js = (
            GOOD_JS
            + '\ndocument.head.appendChild(document.createElement("style"));\nLGraphCanvas.prototype.drawNode = () => {};\n'
        )
        findings = check_pack(_pack(tmp_path, js))
        top = [f for f in findings if f.kind == "top-level-code"]
        assert [f.line for f in top] == [19, 20]
        assert top[0].file == "web/main.js"

    def test_assigning_to_the_modules_own_class_is_allowed(self, tmp_path):
        js = GOOD_JS + "\nclass CushionNode {}\nCushionNode.title = 'Cushion';\nCushionNode['@mode'] = 1;\n"
        assert check_pack(_pack(tmp_path, js)) == []

    def test_assigning_to_a_global_is_reported(self, tmp_path):
        js = GOOD_JS + "\napp.handleFile = () => {};\n"
        assert _kinds(check_pack(_pack(tmp_path, js))) == ["top-level-code"]

    def test_page_wide_css_is_reported(self, tmp_path):
        js = GOOD_JS.replace(".agent-cushions-node {", "body, .litegraph {")
        findings = check_pack(_pack(tmp_path, js, files={"style.css": "* { margin: 0 }\n.mine { color: red }\n"}))
        css = [f for f in findings if f.kind == "page-wide-css"]
        assert {f.file for f in css} == {"web/main.js", "web/style.css"}

    def test_scoped_css_is_not_reported(self, tmp_path):
        files = {"style.css": ".agent-cushions-node body-text { color: red }\n"}
        assert check_pack(_pack(tmp_path, files=files)) == []

    def test_extension_name_without_agent_prefix_is_reported(self, tmp_path):
        findings = check_pack(_pack(tmp_path, GOOD_JS.replace("agent.cushions.style", "Cushions")))
        assert _kinds(findings) == ["extension-name"]

    def test_provenance_stands_in_for_the_prefix(self, tmp_path):
        js = GOOD_JS.replace(
            'name: "agent.cushions.style",', "name: \"Cushions\",\n  provenance: { createdBy: 'comfy-agent' },"
        )
        assert check_pack(_pack(tmp_path, js)) == []

    def test_web_directory_is_read_from_init(self, tmp_path):
        pack = _pack(tmp_path, GOOD_JS + "\nboom();\n", web_dir="js")
        assert [f.file for f in check_pack(pack)] == ["js/main.js"]

    def test_pack_without_frontend_code_has_no_findings(self, tmp_path):
        assert check_pack(_pack(tmp_path, None, init="NODE_CLASS_MAPPINGS = {}\n")) == []

    def test_unreadable_js_is_skipped(self, tmp_path):
        pack = _pack(tmp_path)
        (pack / "web" / "bad.js").write_bytes(b"\xff\xfe\x00")
        assert check_pack(pack) == []


class TestIsAgentPack:
    def test_install_record_marks_an_agent_pack(self, tmp_path):
        assert is_agent_pack(_pack(tmp_path))

    def test_agent_prefixed_extension_name_marks_an_agent_pack(self, tmp_path):
        assert is_agent_pack(_pack(tmp_path, record=False))

    def test_other_packs_are_not_agent_packs(self, tmp_path):
        assert not is_agent_pack(_pack(tmp_path, GOOD_JS.replace("agent.cushions.style", "Cushions"), record=False))


class TestValidateCommand:
    def _invoke(self, monkeypatch, pack, *args):
        monkeypatch.chdir(pack)
        with patch("comfy_cli.command.custom_nodes.command.validate_node_for_publishing") as publish_check:
            result = runner.invoke(app, ["validate", *args])
        return result, publish_check

    def test_agent_pack_findings_are_advisory(self, tmp_path, monkeypatch):
        result, publish_check = self._invoke(monkeypatch, _pack(tmp_path, GOOD_JS + "\nboom();\n"))
        assert result.exit_code == 0, result.output
        assert "top-level" in result.output
        assert "web/main.js:19" in result.output
        publish_check.assert_called_once()

    def test_non_agent_pack_is_not_checked_unless_asked(self, tmp_path, monkeypatch):
        js = GOOD_JS.replace("agent.cushions.style", "Cushions") + "\nboom();\n"
        pack = _pack(tmp_path, js, record=False)
        result, _ = self._invoke(monkeypatch, pack)
        assert "top-level" not in result.output
        result, _ = self._invoke(monkeypatch, pack, "--agent-pack")
        assert "top-level" in result.output
        assert result.exit_code == 0
