import logging
import os
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from comfy_cli.extra_model_paths import (
    ExtraPath,
    collect_extra_paths,
    load_extra_paths,
    paths_for_category,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(dedent(content).lstrip())
    return path


# ---------- load_extra_paths ----------


def test_missing_file_returns_empty(tmp_path):
    assert load_extra_paths(tmp_path / "absent.yaml") == []


def test_empty_file_returns_empty(tmp_path):
    p = _write(tmp_path / "x.yaml", "")
    assert load_extra_paths(p) == []


def test_invalid_yaml_propagates(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
          base_path: /foo
            checkpoints: bar
        """,
    )
    with pytest.raises(yaml.YAMLError):
        load_extra_paths(p)


def test_top_level_not_a_mapping_warns_and_returns_empty(tmp_path, caplog):
    p = _write(tmp_path / "x.yaml", "[a, b]\n")
    with caplog.at_level(logging.WARNING, logger="comfy_cli.extra_model_paths"):
        assert load_extra_paths(p) == []
    assert "not a YAML mapping" in caplog.text


def test_absolute_path_no_base_path(tmp_path):
    abs_dir = tmp_path / "external" / "checkpoints"
    p = _write(
        tmp_path / "x.yaml",
        f"""
        comfyui:
            checkpoints: {abs_dir}
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.category == "checkpoints"
    assert entry.path == Path(os.path.normpath(str(abs_dir)))
    assert entry.is_default is False
    assert entry.section == "comfyui"


def test_relative_base_path_resolves_to_yaml_dir(tmp_path):
    yaml_dir = tmp_path / "configs"
    yaml_dir.mkdir()
    p = _write(
        yaml_dir / "x.yaml",
        """
        comfyui:
            base_path: store
            checkpoints: cp
        """,
    )
    [entry] = load_extra_paths(p)
    expected = Path(os.path.normpath(str(yaml_dir / "store" / "cp")))
    assert entry.path == expected


def test_base_path_with_tilde_expansion(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: ~/models
            checkpoints: cp
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.path == Path(os.path.normpath(str(fake_home / "models" / "cp")))


def test_base_path_with_env_var_expansion(tmp_path, monkeypatch):
    target = tmp_path / "var_target"
    monkeypatch.setenv("MODEL_ROOT", str(target))
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: $MODEL_ROOT/models
            loras: l
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.path == Path(os.path.normpath(str(target / "models" / "l")))


def test_multiline_block_scalar_splits_paths(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /base
            text_encoders: |
                models/text_encoders/
                models/clip/
        """,
    )
    paths = [e.path for e in load_extra_paths(p)]
    assert paths == [
        Path(os.path.normpath("/base/models/text_encoders/")),
        Path(os.path.normpath("/base/models/clip/")),
    ]


def test_blank_lines_in_block_scalar_skipped(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /base
            loras: |
                loras

                more_loras
        """,
    )
    paths = [e.path for e in load_extra_paths(p)]
    assert paths == [
        Path(os.path.normpath("/base/loras")),
        Path(os.path.normpath("/base/more_loras")),
    ]


def test_is_default_flag_preserved(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /a
            is_default: true
            checkpoints: cp
        a111:
            base_path: /b
            checkpoints: cp
        """,
    )
    entries = load_extra_paths(p)
    assert entries[0].is_default is True and entries[0].section == "comfyui"
    assert entries[1].is_default is False and entries[1].section == "a111"


def test_legacy_aliases_unet_and_clip(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /a
            unet: u
            clip: c
        """,
    )
    cats = [e.category for e in load_extra_paths(p)]
    assert cats == ["diffusion_models", "text_encoders"]


def test_none_section_skipped(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
        a111:
            base_path: /b
            checkpoints: cp
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.section == "a111"


def test_none_category_value_skipped(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /a
            checkpoints:
            loras: l
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.category == "loras"


def test_non_string_category_value_warns_and_skips(tmp_path, caplog):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /a
            checkpoints: [bad, list]
            loras: ok
        """,
    )
    with caplog.at_level(logging.WARNING, logger="comfy_cli.extra_model_paths"):
        entries = load_extra_paths(p)
    assert [e.category for e in entries] == ["loras"]
    assert "checkpoints" in caplog.text


def test_section_not_a_mapping_warns_and_skips(tmp_path, caplog):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui: just_a_string
        a111:
            base_path: /b
            checkpoints: cp
        """,
    )
    with caplog.at_level(logging.WARNING, logger="comfy_cli.extra_model_paths"):
        entries = load_extra_paths(p)
    assert [e.section for e in entries] == ["a111"]
    assert "comfyui" in caplog.text


def test_normpath_collapses_dot_dot(tmp_path):
    p = _write(
        tmp_path / "x.yaml",
        """
        comfyui:
            base_path: /a/b
            checkpoints: ../cp
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.path == Path(os.path.normpath("/a/b/../cp"))
    assert ".." not in entry.path.parts


def test_relative_path_no_base_path_uses_yaml_dir(tmp_path):
    yaml_dir = tmp_path / "cfg"
    yaml_dir.mkdir()
    p = _write(
        yaml_dir / "x.yaml",
        """
        comfyui:
            checkpoints: rel/cp
        """,
    )
    [entry] = load_extra_paths(p)
    assert entry.path == Path(os.path.normpath(str(yaml_dir / "rel" / "cp")))


# ---------- collect_extra_paths ----------


def test_collect_workspace_yaml_only(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write(
        workspace / "extra_model_paths.yaml",
        """
        comfyui:
            base_path: /a
            checkpoints: cp
        """,
    )
    entries = collect_extra_paths(workspace)
    assert len(entries) == 1
    assert entries[0].section == "comfyui"


def test_collect_no_workspace_yaml_with_extra_config(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    extra = _write(
        tmp_path / "extra.yaml",
        """
        comfyui:
            base_path: /a
            loras: l
        """,
    )
    entries = collect_extra_paths(workspace, [extra])
    assert len(entries) == 1
    assert entries[0].category == "loras"


def test_collect_workspace_then_extras_in_order(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write(
        workspace / "extra_model_paths.yaml",
        """
        ws_section:
            base_path: /a
            checkpoints: cp
        """,
    )
    extra1 = _write(
        tmp_path / "e1.yaml",
        """
        e1_section:
            base_path: /b
            checkpoints: cp
        """,
    )
    extra2 = _write(
        tmp_path / "e2.yaml",
        """
        e2_section:
            base_path: /c
            checkpoints: cp
        """,
    )
    entries = collect_extra_paths(workspace, [extra1, extra2])
    assert [e.section for e in entries] == ["ws_section", "e1_section", "e2_section"]


# ---------- paths_for_category ----------


def test_paths_for_unknown_category_empty():
    assert paths_for_category([], "anything") == []
    extras = [ExtraPath("loras", Path("/a"), False, "s")]
    assert paths_for_category(extras, "checkpoints") == []


def test_paths_preserve_yaml_order_when_no_default():
    extras = [
        ExtraPath("checkpoints", Path("/A"), False, "a"),
        ExtraPath("checkpoints", Path("/B"), False, "b"),
        ExtraPath("checkpoints", Path("/C"), False, "c"),
    ]
    assert paths_for_category(extras, "checkpoints") == [Path("/A"), Path("/B"), Path("/C")]


def test_is_default_paths_come_before_non_default():
    extras = [
        ExtraPath("checkpoints", Path("/X"), False, "x"),
        ExtraPath("checkpoints", Path("/Y"), True, "y"),
    ]
    assert paths_for_category(extras, "checkpoints") == [Path("/Y"), Path("/X")]


def test_two_is_default_later_wins_slot_zero():
    extras = [
        ExtraPath("checkpoints", Path("/A"), True, "a"),
        ExtraPath("checkpoints", Path("/B"), True, "b"),
    ]
    assert paths_for_category(extras, "checkpoints") == [Path("/B"), Path("/A")]


def test_legacy_alias_query_returns_canonical_paths():
    extras = [
        ExtraPath("diffusion_models", Path("/dm"), False, "s"),
        ExtraPath("text_encoders", Path("/te"), False, "s"),
    ]
    assert paths_for_category(extras, "unet") == [Path("/dm")]
    assert paths_for_category(extras, "clip") == [Path("/te")]


def test_duplicate_with_default_moves_to_head():
    extras = [
        ExtraPath("loras", Path("/A"), False, "a"),
        ExtraPath("loras", Path("/B"), False, "b"),
        ExtraPath("loras", Path("/B"), True, "c"),
    ]
    assert paths_for_category(extras, "loras") == [Path("/B"), Path("/A")]


def test_duplicate_without_default_is_noop():
    extras = [
        ExtraPath("loras", Path("/A"), True, "a"),
        ExtraPath("loras", Path("/B"), False, "b"),
        ExtraPath("loras", Path("/A"), False, "c"),
    ]
    assert paths_for_category(extras, "loras") == [Path("/A"), Path("/B")]
