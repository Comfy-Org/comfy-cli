"""`$asset.<name>` references — compose-time resolution from the push lock.

Three layers, no network anywhere:

- pure fragments: ``$asset`` on an input with an injected resolver lands as
  the SAME loader materialization a literal filename gets; resolver absent
  is a clear BlueprintError; foreach namespacing must not mangle the prefix.
- pure project: resolver behavior is covered in tests/comfy_cli/test_project.py.
- CLI: ``comfy workflow compose`` inside a project wires the resolver and
  maps AssetError onto its error code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.fragments import (
    AssetError,
    BlueprintError,
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
    """STRING + INT params — exercises $asset resolution on the param path."""
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


def _image_blend_fragment() -> dict:
    return {
        "_fragment": {
            "name": "image_blend",
            "version": "1",
            "inputs": {
                "image1": {"type": "IMAGE", "binds": "10.image1"},
                "image2": {"type": "IMAGE", "binds": "10.image2"},
            },
            "outputs": {"image": {"type": "IMAGE", "from": "10", "port": 0}},
            "params": {"blend_factor": {"type": "FLOAT", "binds": "10.blend_factor", "default": 0.5}},
        },
        "10": {"class_type": "ImageBlend", "inputs": {"image1": "P", "image2": "P", "blend_factor": 0.5}},
    }


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fragments"
    d.mkdir()
    (d / "image_blend.json").write_text(json.dumps(_image_blend_fragment()))
    (d / "text_overlay.json").write_text(json.dumps(_text_overlay_fragment()))
    return d


def _blueprint(image1: str, image2: str) -> dict:
    return {
        "pipeline": [
            {
                "fragment": "image_blend",
                "alias": "blend",
                "inputs": {"image1": image1, "image2": image2},
            }
        ]
    }


# ---------------------------------------------------------------------------
# pure fragments: $asset resolution in _resolve_input
# ---------------------------------------------------------------------------


class TestAssetResolution:
    def test_asset_ref_lands_identically_to_literal_filename(self, lib_dir):
        """`$asset.x` with a resolver must produce byte-identical wiring to
        composing the resolved name as a literal — same LoadImage nodes."""
        literal_wf, _ = compose_blueprint(_blueprint("ab12.png", "cd34.png"), lib_dir=lib_dir)

        resolved = {"keyframes/s1.png": "ab12.png", "s2.png": "cd34.png"}
        asset_wf, _ = compose_blueprint(
            _blueprint("$asset.keyframes/s1.png", "$asset.s2.png"),
            lib_dir=lib_dir,
            asset_resolver=resolved.__getitem__,
        )
        assert asset_wf == literal_wf

    def test_asset_ref_without_resolver_is_blueprint_error(self, lib_dir):
        with pytest.raises(BlueprintError) as exc:
            compose_blueprint(_blueprint("$asset.s1.png", "x.png"), lib_dir=lib_dir)
        assert "$asset.s1.png" in str(exc.value)
        assert "comfy project init" in (exc.value.hint or "")
        assert "comfy assets push" in (exc.value.hint or "")

    def test_resolver_asset_error_propagates(self, lib_dir):
        def stale(name: str) -> str:
            raise AssetError(f"asset {name!r} is stale", code="asset_stale", hint="run: comfy assets push")

        with pytest.raises(AssetError) as exc:
            compose_blueprint(_blueprint("$asset.s1.png", "x.png"), lib_dir=lib_dir, asset_resolver=stale)
        assert exc.value.code == "asset_stale"

    def test_compose_blueprints_threads_resolver(self, lib_dir):
        calls: list[str] = []

        def resolver(name: str) -> str:
            calls.append(name)
            return f"srv-{name}"

        graphs = compose_blueprints(
            _blueprint("$asset.a.png", "$asset.b.png"), lib_dir=lib_dir, asset_resolver=resolver
        )
        assert sorted(calls) == ["a.png", "b.png"]
        workflow = graphs[0][0]
        loaded = {n["inputs"]["image"] for n in workflow.values() if n["class_type"] == "LoadImage"}
        assert loaded == {"srv-a.png", "srv-b.png"}


# ---------------------------------------------------------------------------
# params: whole-value $asset refs resolve to widget values too
# ---------------------------------------------------------------------------


def _overlay_blueprint(params: dict) -> dict:
    return {
        "pipeline": [
            {
                "fragment": "text_overlay",
                "alias": "t",
                "inputs": {"image": "base.png"},
                "params": params,
            }
        ]
    }


def _overlay_node(workflow: dict) -> dict:
    nodes = [n for n in workflow.values() if n.get("class_type") == "TextOverlay"]
    assert len(nodes) == 1
    return nodes[0]


class TestAssetParamResolution:
    def test_string_param_asset_ref_resolves_to_cloud_name(self, lib_dir):
        wf, _ = compose_blueprint(
            _overlay_blueprint({"label": "$asset.fonts/x.ttf"}),
            lib_dir=lib_dir,
            asset_resolver=lambda name: f"srv-{name}",
        )
        assert _overlay_node(wf)["inputs"]["label"] == "srv-fonts/x.ttf"

    def test_param_asset_ref_without_resolver_is_blueprint_error(self, lib_dir):
        """Same error + hint as the inputs path — one shared resolution helper."""
        with pytest.raises(BlueprintError) as exc:
            compose_blueprint(_overlay_blueprint({"label": "$asset.x.ttf"}), lib_dir=lib_dir)
        assert "$asset.x.ttf" in str(exc.value)
        assert "comfy project init" in (exc.value.hint or "")
        assert "comfy assets push" in (exc.value.hint or "")

    def test_embedded_asset_ref_mid_string_is_not_a_reference(self, lib_dir):
        """Whole-value only: `$asset.` mid-string is plain text, no resolver
        consulted, value untouched (there is no interpolation/templating)."""
        wf, _ = compose_blueprint(
            _overlay_blueprint({"label": "use $asset.x here"}),
            lib_dir=lib_dir,  # no resolver — must not raise either
        )
        assert _overlay_node(wf)["inputs"]["label"] == "use $asset.x here"

    def test_int_param_asset_ref_resolves_and_lands_as_string(self, lib_dir):
        """Resolution happens regardless of the declared param type; there is
        no client-side widget-type validation, so the resolved STRING lands in
        the INT widget and the server complains at run time. Deterministic and
        documented — the staleness checks still fire identically."""
        wf, _ = compose_blueprint(
            _overlay_blueprint({"size": "$asset.x"}),
            lib_dir=lib_dir,
            asset_resolver=lambda name: f"srv-{name}",
        )
        assert _overlay_node(wf)["inputs"]["size"] == "srv-x"

    def test_param_resolver_asset_error_propagates(self, lib_dir):
        def stale(name: str) -> str:
            raise AssetError(f"asset {name!r} is stale", code="asset_stale", hint="run: comfy assets push")

        with pytest.raises(AssetError) as exc:
            compose_blueprint(_overlay_blueprint({"label": "$asset.x.ttf"}), lib_dir=lib_dir, asset_resolver=stale)
        assert exc.value.code == "asset_stale"

    def test_foreach_item_field_asset_ref_resolves_per_item(self, lib_dir):
        """Order: $item substitution first, then $asset resolution on the
        resulting whole-value string — an item field can carry an asset ref
        that a `$item.<field>` param receives and resolves."""
        blueprint = {
            "foreach": [
                {"id": "s1", "first": "$asset.k/s1.png"},
                {"id": "s2", "first": "$asset.k/s2.png"},
            ],
            "pipeline": [
                {
                    "fragment": "text_overlay",
                    "alias": "t",
                    "inputs": {"image": "base.png"},
                    "params": {"label": "$item.first"},
                }
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir, asset_resolver=lambda name: f"srv-{name}")
        assert len(graphs) == 1
        workflow = graphs[0][0]
        labels = {n["inputs"]["label"] for n in workflow.values() if n.get("class_type") == "TextOverlay"}
        assert labels == {"srv-k/s1.png", "srv-k/s2.png"}
        assert "__asset" not in json.dumps(workflow)


# ---------------------------------------------------------------------------
# foreach: the namespacing substitution must leave $asset.* alone
# ---------------------------------------------------------------------------


class TestForeachAssetGuard:
    def test_substitute_item_leaves_asset_refs_unmangled(self):
        """REGRESSION: without the early guard, alias namespacing rewrites
        `$asset.x` into `$i0_s1__asset.x` and the ref is lost."""
        assert _substitute_item("$asset.x", {"id": "s1"}, ns="i0_s1") == "$asset.x"

    def test_substitute_item_recurses_containers_without_mangling(self):
        value = {"image": "$asset.keyframes/s1.png", "rest": ["$asset.b.png"]}
        out = _substitute_item(value, {"id": "s1"}, ns="i0_s1")
        assert out == value

    def test_foreach_blueprint_resolves_assets_per_branch(self, lib_dir):
        calls: list[str] = []

        def resolver(name: str) -> str:
            calls.append(name)
            return "srv-shared.png"

        blueprint = {
            "foreach": [{"id": "s1"}, {"id": "s2"}],
            "pipeline": [
                {
                    "fragment": "image_blend",
                    "alias": "blend",
                    "inputs": {"image1": "$asset.shared.png", "image2": "$asset.shared.png"},
                }
            ],
        }
        graphs = compose_blueprints(blueprint, lib_dir=lib_dir, asset_resolver=resolver)
        assert len(graphs) == 1
        workflow = graphs[0][0]
        assert calls == ["shared.png"] * 4  # 2 inputs × 2 branches
        loads = [n for n in workflow.values() if n["class_type"] == "LoadImage"]
        assert len(loads) == 4
        assert all(n["inputs"]["image"] == "srv-shared.png" for n in loads)
        # Nothing in the graph carries a mangled `__asset` token.
        assert "__asset" not in json.dumps(workflow)


# ---------------------------------------------------------------------------
# CLI: compose inside a project
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
    """A project/1 tree with a fragment lib, a $asset blueprint, and an asset."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "comfy.yaml").write_text("schema: project/1\ndefaults:\n  where: cloud\n")
    for d in ("assets", "fragments", "blueprints", "outputs", ".comfy"):
        (root / d).mkdir()
    (root / "fragments" / "image_blend.json").write_text(json.dumps(_image_blend_fragment()))
    (root / "blueprints" / "story.yaml").write_text(
        "pipeline:\n"
        "  - fragment: image_blend\n"
        "    alias: blend\n"
        "    inputs:\n"
        "      image1: $asset.s1.png\n"
        "      image2: $asset.s1.png\n"
    )
    (root / "assets" / "s1.png").write_bytes(b"png-v1")
    return root


def _write_lock(root: Path, sha: str) -> None:
    (root / ".comfy" / "assets.lock.json").write_text(
        json.dumps({"schema": "assets-lock/1", "assets": {"s1.png": {"sha256": sha, "cloud_name": "ab12.png"}}})
    )


class TestComposeCli:
    def test_compose_with_pushed_asset_uses_cloud_name(self, project_tree, capsys):
        _write_lock(project_tree, hashlib.sha256(b"png-v1").hexdigest())
        bp = project_tree / "blueprints" / "story.yaml"
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 0, env
        assert env["ok"] is True
        compiled = json.loads(Path(env["data"]["out"]).read_text())
        loads = [n for n in compiled.values() if isinstance(n, dict) and n.get("class_type") == "LoadImage"]
        assert loads and all(n["inputs"]["image"] == "ab12.png" for n in loads)

    def test_compose_stale_asset_exits_1_with_asset_stale(self, project_tree, capsys):
        _write_lock(project_tree, "0" * 64)  # lock sha != on-disk sha
        bp = project_tree / "blueprints" / "story.yaml"
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 1
        assert env["ok"] is False
        assert env["error"]["code"] == "asset_stale"
        assert env["error"]["hint"] == "run: comfy assets push"

    def test_compose_unpushed_asset_exits_1_with_asset_not_pushed(self, project_tree, capsys):
        # No lock written at all.
        bp = project_tree / "blueprints" / "story.yaml"
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 1
        assert env["error"]["code"] == "asset_not_pushed"
        assert env["error"]["hint"] == "run: comfy assets push"

    def test_compose_stale_asset_via_param_exits_1_with_asset_stale(self, project_tree, capsys):
        """Staleness checks fire identically when the ref sits in `params:`."""
        _write_lock(project_tree, "0" * 64)  # lock sha != on-disk sha
        (project_tree / "fragments" / "text_overlay.json").write_text(json.dumps(_text_overlay_fragment()))
        bp = project_tree / "blueprints" / "overlay.yaml"
        bp.write_text(
            "pipeline:\n"
            "  - fragment: text_overlay\n"
            "    alias: t\n"
            "    inputs:\n"
            "      image: base.png\n"
            "    params:\n"
            "      label: $asset.s1.png\n"
        )
        rc, env = _invoke(["compose", str(bp), "--lib", str(project_tree / "fragments")], capsys)
        assert rc == 1
        assert env["error"]["code"] == "asset_stale"
        assert env["error"]["hint"] == "run: comfy assets push"

    def test_compose_outside_project_keeps_blueprint_invalid(self, tmp_path, capsys):
        """No governing project → no resolver → the generic BlueprintError
        path with the init/push hint."""
        lib = tmp_path / "fragments"
        lib.mkdir()
        (lib / "image_blend.json").write_text(json.dumps(_image_blend_fragment()))
        bp = tmp_path / "story.yaml"
        bp.write_text(
            "pipeline:\n"
            "  - fragment: image_blend\n"
            "    alias: blend\n"
            "    inputs:\n"
            "      image1: $asset.s1.png\n"
            "      image2: x.png\n"
        )
        rc, env = _invoke(["compose", str(bp), "--lib", str(lib)], capsys)
        assert rc == 1
        assert env["error"]["code"] == "blueprint_invalid"
        assert "comfy assets push" in env["error"]["hint"]


# ---------------------------------------------------------------------------
# ratchets
# ---------------------------------------------------------------------------


def test_asset_error_codes_registered():
    from comfy_cli import error_codes

    assert error_codes.is_registered("asset_not_pushed")
    assert error_codes.is_registered("asset_stale")
