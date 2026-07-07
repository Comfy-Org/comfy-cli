"""`$var.<name>` references — project constants from comfy.yaml `vars:`.

Mirror of test_asset_refs.py, same three layers, no network anywhere:

- pure fragments: `$var` with an injected resolver resolves wherever a
  whole-value string can appear (params, foreach item fields, inputs) and
  returns the RAW scalar so non-STRING params keep their types; resolver
  absent is a clear BlueprintError; foreach namespacing must not mangle the
  prefix.
- pure project: resolver behavior is covered in tests/comfy_cli/test_project.py.
- CLI: `comfy workflow compose` inside a project wires the resolver, maps
  VarError onto `var_not_defined`, and snapshots referenced vars into
  `_meta.vars` for provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.fragments import (
    BlueprintError,
    VarError,
    _substitute_item,
    compose_blueprint,
    compose_blueprints,
)
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _text_overlay_fragment() -> dict:
    """STRING + INT params and an IMAGE input — exercises every $var site."""
    return {
        "_fragment": {
            "name": "text_overlay",
            "version": "1",
            "inputs": {"image": {"type": "IMAGE", "binds": "10.image"}},
            "outputs": {"image": {"type": "IMAGE", "from": "10", "port": 0}},
            "params": {
                "label": {"type": "STRING", "binds": "10.label", "default": "x"},
                "size": {"type": "INT", "binds": "10.size", "default": 12},
            },
        },
        "10": {"class_type": "TextOverlay", "inputs": {"image": "P", "label": "x", "size": 12}},
    }


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fragments"
    d.mkdir()
    (d / "text_overlay.json").write_text(json.dumps(_text_overlay_fragment()))
    return d


def _overlay_blueprint(*, inputs: dict | None = None, params: dict | None = None) -> dict:
    return {
        "pipeline": [
            {
                "fragment": "text_overlay",
                "alias": "t",
                "inputs": inputs or {"image": "base.png"},
                "params": params or {},
            }
        ]
    }


def _overlay_node(workflow: dict) -> dict:
    nodes = [n for n in workflow.values() if n.get("class_type") == "TextOverlay"]
    assert len(nodes) == 1
    return nodes[0]


_VARS = {"style": "cinematic, golden hour", "steps": 28, "base": "shared_bg.png"}


# ---------------------------------------------------------------------------
# pure fragments: $var resolution in params, inputs, foreach item fields
# ---------------------------------------------------------------------------


class TestVarResolution:
    def test_string_param_var_ref_resolves(self, lib_dir):
        wf, _ = compose_blueprint(
            _overlay_blueprint(params={"label": "$var.style"}),
            lib_dir=lib_dir,
            var_resolver=_VARS.__getitem__,
        )
        assert _overlay_node(wf)["inputs"]["label"] == "cinematic, golden hour"

    def test_int_param_var_ref_keeps_int_type(self, lib_dir):
        """$var returns the raw scalar, not str()'d — INT widgets stay ints."""
        wf, _ = compose_blueprint(
            _overlay_blueprint(params={"size": "$var.steps"}),
            lib_dir=lib_dir,
            var_resolver=_VARS.__getitem__,
        )
        size = _overlay_node(wf)["inputs"]["size"]
        assert size == 28 and isinstance(size, int) and not isinstance(size, bool)

    def test_var_ref_resolves_in_inputs_too(self, lib_dir):
        """Same shared helper as $asset: an IMAGE input fed `$var.base` gets
        the loader materialization of the resolved filename."""
        wf, _ = compose_blueprint(
            _overlay_blueprint(inputs={"image": "$var.base"}),
            lib_dir=lib_dir,
            var_resolver=_VARS.__getitem__,
        )
        loads = [n for n in wf.values() if n["class_type"] == "LoadImage"]
        assert len(loads) == 1 and loads[0]["inputs"]["image"] == "shared_bg.png"

    def test_embedded_var_ref_mid_string_is_not_a_reference(self, lib_dir):
        """Whole-value only: `$var.` mid-string is plain text, no resolver
        consulted (must not raise even with no resolver)."""
        wf, _ = compose_blueprint(
            _overlay_blueprint(params={"label": "in the $var.style style"}),
            lib_dir=lib_dir,
        )
        assert _overlay_node(wf)["inputs"]["label"] == "in the $var.style style"

    def test_var_ref_without_resolver_is_blueprint_error(self, lib_dir):
        with pytest.raises(BlueprintError) as exc:
            compose_blueprint(_overlay_blueprint(params={"label": "$var.style"}), lib_dir=lib_dir)
        assert "$var.style" in str(exc.value)
        assert "comfy.yaml" in (exc.value.hint or "")
        assert "project" in (exc.value.hint or "")

    def test_resolver_var_error_propagates(self, lib_dir):
        def undefined(name: str):
            raise VarError(f"var {name!r} is not defined", code="var_not_defined", hint="add it under `vars:`")

        with pytest.raises(VarError) as exc:
            compose_blueprint(
                _overlay_blueprint(params={"label": "$var.style"}), lib_dir=lib_dir, var_resolver=undefined
            )
        assert exc.value.code == "var_not_defined"


# ---------------------------------------------------------------------------
# foreach: namespacing guard + per-item resolution through $item fields
# ---------------------------------------------------------------------------


class TestForeachVar:
    def test_substitute_item_leaves_var_refs_unmangled(self):
        """REGRESSION GUARD: without the prefix guard, alias namespacing would
        rewrite `$var.style` into `$i0_s1__var.style` and the ref is lost."""
        assert _substitute_item("$var.style", {"id": "s1"}, ns="i0_s1") == "$var.style"

    def test_foreach_item_field_var_ref_resolves_per_item(self, lib_dir):
        """$item substitution first, then $var resolution on the result."""
        blueprint = {
            "foreach": [
                {"id": "s1", "look": "$var.style"},
                {"id": "s2", "look": "plain"},
            ],
            "pipeline": [
                {
                    "fragment": "text_overlay",
                    "alias": "t",
                    "inputs": {"image": "base.png"},
                    "params": {"label": "$item.look"},
                }
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir, var_resolver=_VARS.__getitem__)
        assert len(graphs) == 1
        workflow = graphs[0][0]
        labels = {n["inputs"]["label"] for n in workflow.values() if n.get("class_type") == "TextOverlay"}
        assert labels == {"cinematic, golden hour", "plain"}
        assert "__var" not in json.dumps(workflow)

    def test_shared_var_across_branches(self, lib_dir):
        blueprint = {
            "foreach": [{"id": "s1"}, {"id": "s2"}],
            "pipeline": [
                {
                    "fragment": "text_overlay",
                    "alias": "t",
                    "inputs": {"image": "base.png"},
                    "params": {"label": "$var.style"},
                }
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir, var_resolver=_VARS.__getitem__)
        labels = [n["inputs"]["label"] for n in graphs[0][0].values() if n.get("class_type") == "TextOverlay"]
        assert labels == ["cinematic, golden hour"] * 2


# ---------------------------------------------------------------------------
# CLI: compose inside a project with a comfy.yaml `vars:` block
# ---------------------------------------------------------------------------


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


def _invoke(args: list[str], capsys) -> tuple[int, dict]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    exit_code = result.return_value if isinstance(result.return_value, int) else result.exit_code
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return exit_code, json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, out={captured[:500]!r})")


@pytest.fixture
def project_tree(tmp_path: Path) -> Path:
    """A project/1 tree whose comfy.yaml carries a `vars:` block."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "comfy.yaml").write_text(
        "schema: project/1\n"
        "defaults:\n"
        "  where: cloud\n"
        "vars:\n"
        "  style: cinematic, golden hour\n"
        "  steps: 28\n"
        "  unused: never referenced\n"
    )
    for d in ("assets", "fragments", "blueprints", "outputs", ".comfy"):
        (root / d).mkdir()
    (root / "fragments" / "text_overlay.json").write_text(json.dumps(_text_overlay_fragment()))
    (root / "blueprints" / "story.yaml").write_text(
        "pipeline:\n"
        "  - fragment: text_overlay\n"
        "    alias: t\n"
        "    inputs:\n"
        "      image: base.png\n"
        "    params:\n"
        "      label: $var.style\n"
        "      size: $var.steps\n"
    )
    return root


class TestComposeCli:
    def test_compose_resolves_vars_from_comfy_yaml(self, project_tree, capsys):
        bp = project_tree / "blueprints" / "story.yaml"
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 0, env
        assert env["ok"] is True
        compiled = json.loads(Path(env["data"]["out"]).read_text())
        node = _overlay_node({k: v for k, v in compiled.items() if isinstance(v, dict) and "class_type" in v})
        assert node["inputs"]["label"] == "cinematic, golden hour"
        assert node["inputs"]["size"] == 28 and isinstance(node["inputs"]["size"], int)

    def test_meta_vars_records_only_referenced_vars(self, project_tree, capsys):
        """Provenance: `_meta.vars` snapshots the values this compilation
        used — referenced names only, `unused` stays out."""
        bp = project_tree / "blueprints" / "story.yaml"
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 0, env
        compiled = json.loads(Path(env["data"]["out"]).read_text())
        assert compiled["_meta"]["vars"] == {"style": "cinematic, golden hour", "steps": 28}

    def test_no_vars_referenced_no_meta_vars_key(self, project_tree, capsys):
        bp = project_tree / "blueprints" / "plain.yaml"
        bp.write_text(
            "pipeline:\n  - fragment: text_overlay\n    alias: t\n    inputs:\n      image: base.png\n",
        )
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 0, env
        compiled = json.loads(Path(env["data"]["out"]).read_text())
        assert "vars" not in compiled["_meta"]

    def test_undefined_var_exits_1_with_var_not_defined(self, project_tree, capsys):
        bp = project_tree / "blueprints" / "bad.yaml"
        bp.write_text(
            "pipeline:\n"
            "  - fragment: text_overlay\n"
            "    alias: t\n"
            "    inputs:\n"
            "      image: base.png\n"
            "    params:\n"
            "      label: $var.missing\n"
        )
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 1
        assert env["ok"] is False
        assert env["error"]["code"] == "var_not_defined"
        assert "vars:" in env["error"]["hint"]

    def test_compose_outside_project_keeps_blueprint_invalid(self, tmp_path, capsys):
        """No governing project → no resolver → BlueprintError with the
        comfy.yaml-requires-a-project hint."""
        lib = tmp_path / "fragments"
        lib.mkdir()
        (lib / "text_overlay.json").write_text(json.dumps(_text_overlay_fragment()))
        bp = tmp_path / "story.yaml"
        bp.write_text(
            "pipeline:\n"
            "  - fragment: text_overlay\n"
            "    alias: t\n"
            "    inputs:\n"
            "      image: base.png\n"
            "    params:\n"
            "      label: $var.style\n"
        )
        rc, env = _invoke(["compose", str(bp), "--lib", str(lib)], capsys)
        assert rc == 1
        assert env["error"]["code"] == "blueprint_invalid"
        assert "comfy.yaml" in env["error"]["hint"]


# ---------------------------------------------------------------------------
# ratchets
# ---------------------------------------------------------------------------


def test_var_error_code_registered():
    from comfy_cli import error_codes

    assert error_codes.is_registered("var_not_defined")
