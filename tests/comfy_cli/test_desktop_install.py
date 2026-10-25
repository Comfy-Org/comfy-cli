"""Comfy Desktop 2 detection from the records Desktop keeps on disk.

Every test builds its own fixture layout under ``tmp_path``; none reads the
machine's real Desktop files (the autouse ``_isolate_comfy_desktop`` fixture
points detection at an empty dir unless a test passes its own).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli import desktop_install
from comfy_cli.constants import OS

# Bound at import, before the autouse isolation fixture replaces the module
# attributes, so the per-OS resolution itself can still be tested.
_real_app_data_dir = desktop_install.desktop_app_data_dir
_real_legacy_config_dir = desktop_install.legacy_config_dir


def _checkout(code_dir: Path, version: str | None = None) -> Path:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "main.py").write_text("")
    if version:
        (code_dir / "comfyui_version.py").write_text(f'__version__ = "{version}"\n')
    return code_dir


def _venv(root: Path) -> Path:
    py = root / ".venv" / "bin" / "python3"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    return py


def _write_desktop(app_dir: Path, installs: list, *, models_dirs: list[str] | None = None) -> Path:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "installations.json").write_text(json.dumps(installs))
    (app_dir / "settings.json").write_text(json.dumps({"modelsDirs": models_dirs or []}))
    return app_dir


def test_no_installations_file_means_no_desktop(tmp_path):
    assert desktop_install.detect(tmp_path / "Comfy Desktop", tmp_path / "legacy") is None


def test_adopted_install_reports_its_split_layout(comfy_desktop):
    layout = comfy_desktop

    found = desktop_install.detect(layout["app"], layout["legacy"])

    assert found is not None
    install = found.selected
    assert install is not None
    assert install.install_id == "inst-adopted"
    assert install.code_dir == str(layout["code"])
    assert install.data_dir == str(layout["data"])
    assert install.python == str(layout["python"])
    assert install.comfyui_version == "v0.34.2"
    assert install.comfyui_commit == "b53f77a7"
    # The data dir's own models/ is listed once even though settings.json also
    # names it; the shared root follows.
    assert [(r.path, r.source) for r in install.model_roots] == [
        (str(layout["data"] / "models"), "data"),
        (str(layout["shared"]), "shared"),
    ]
    assert all(r.exists for r in install.model_roots)
    # The Comfy Cloud entry has no local files and is not an install here.
    assert [i.install_id for i in found.installs] == ["inst-adopted"]
    assert found.selected_reason == "the only Comfy Desktop install with a ComfyUI checkout on disk"
    assert found.problems == ()


def test_to_dict_puts_the_selected_install_at_the_top(comfy_desktop):
    layout = comfy_desktop

    block = desktop_install.detect(layout["app"], layout["legacy"]).to_dict()

    assert block["code_dir"] == str(layout["code"])
    assert block["data_dir"] == str(layout["data"])
    assert block["custom_nodes_dir"] == str(layout["data"] / "custom_nodes")
    assert block["python"] == str(layout["python"])
    assert block["comfyui_version"] == "v0.34.2"
    assert [r["path"] for r in block["model_roots"]] == [str(layout["data"] / "models"), str(layout["shared"])]
    assert block["installs"] == [
        {
            "install_id": "inst-adopted",
            "name": "ComfyUI",
            "code_dir": str(layout["code"]),
            "data_dir": str(layout["data"]),
            "comfyui_version": "v0.34.2",
            "last_launched_at": "2026-10-16T22:10:27Z",
            "status": "installed",
            "exists": True,
            "selected": True,
        }
    ]
    assert block["app_data_dir"] == str(layout["app"])


def test_standalone_install_keeps_data_beside_the_code(tmp_path):
    install_path = tmp_path / "ComfyUI-Installs" / "Fresh"
    code = _checkout(install_path / "ComfyUI")
    python = _venv(code)
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [{"id": "inst-fresh", "installPath": str(install_path), "comfyVersion": {"baseTag": "v0.41.1"}}],
    )

    install = desktop_install.detect(app, tmp_path / "legacy").selected

    assert install.code_dir == str(code)
    assert install.data_dir == str(code)
    assert install.python == str(python)
    assert [r.path for r in install.model_roots] == [str(code / "models")]


def test_most_recently_launched_install_is_selected_and_the_reason_says_so(tmp_path):
    older = tmp_path / "Installs" / "Old"
    newer = tmp_path / "Installs" / "New"
    _checkout(older / "ComfyUI")
    _checkout(newer / "ComfyUI")
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [
            {"id": "old", "installPath": str(older), "lastLaunchedAt": 1_700_000_000_000},
            {"id": "new", "installPath": str(newer), "lastLaunchedAt": 1_790_000_000_000},
            {"id": "never", "installPath": str(tmp_path / "Installs" / "Gone")},
        ],
    )

    found = desktop_install.detect(app, tmp_path / "legacy")

    assert found.selected.install_id == "new"
    assert "most recently launched of 2 Comfy Desktop installs" in found.selected_reason
    assert [i.install_id for i in found.installs] == ["old", "new", "never"]
    rows = {row["install_id"]: row for row in found.to_dict()["installs"]}
    assert rows["new"]["selected"] is True
    assert rows["old"]["selected"] is False
    # An install whose checkout is gone is listed but never selected.
    assert rows["never"]["exists"] is False
    assert rows["never"]["selected"] is False


def test_no_launch_history_picks_the_first_listed_install(tmp_path):
    first = tmp_path / "Installs" / "A"
    second = tmp_path / "Installs" / "B"
    _checkout(first / "ComfyUI")
    _checkout(second / "ComfyUI")
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [{"id": "a", "installPath": str(first)}, {"id": "b", "installPath": str(second)}],
    )

    found = desktop_install.detect(app, tmp_path / "legacy")

    assert found.selected.install_id == "a"
    assert "none has been launched" in found.selected_reason


def test_no_install_with_a_checkout_selects_nothing(tmp_path):
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "gone", "installPath": str(tmp_path / "missing")}])

    found = desktop_install.detect(app, tmp_path / "legacy")

    assert found.selected is None
    assert found.selected_reason is None
    assert [i.install_id for i in found.installs] == ["gone"]


def test_opting_out_of_shared_models_drops_the_shared_roots(tmp_path):
    install_path = tmp_path / "Installs" / "Isolated"
    code = _checkout(install_path / "ComfyUI")
    extra = tmp_path / "extra-models"
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [{"id": "iso", "installPath": str(install_path), "useSharedModels": False, "modelDirs": [str(extra)]}],
        models_dirs=[str(tmp_path / "shared")],
    )

    install = desktop_install.detect(app, tmp_path / "legacy").selected

    assert [(r.path, r.source, r.exists) for r in install.model_roots] == [
        (str(code / "models"), "data", False),
        (str(extra), "install", False),
    ]


def test_adopted_record_without_a_base_dir_falls_back_to_desktop_1_config(comfy_desktop):
    layout = comfy_desktop
    record = dict(layout["record"])
    del record["adoptedBaseDir"]
    del record["adoptedPythonPath"]
    _write_desktop(layout["app"], [record])

    install = desktop_install.detect(layout["app"], layout["legacy"]).selected

    assert install.data_dir == str(layout["data"])
    assert install.python == str(layout["python"])


def test_version_falls_back_to_the_checkout_marker(tmp_path):
    install_path = tmp_path / "Installs" / "Marker"
    _checkout(install_path / "ComfyUI", version="0.41.2")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "m", "installPath": str(install_path)}])

    install = desktop_install.detect(app, tmp_path / "legacy").selected

    assert install.comfyui_version == "v0.41.2"


def test_unreadable_installations_file_is_a_problem_not_a_crash(tmp_path):
    app = tmp_path / "Comfy Desktop"
    app.mkdir()
    (app / "installations.json").write_text("{not json")

    found = desktop_install.detect(app, tmp_path / "legacy")

    assert found is not None
    assert found.selected is None
    assert len(found.problems) == 1
    assert "installations.json" in found.problems[0]


def test_unreadable_settings_file_still_detects_the_install(comfy_desktop):
    layout = comfy_desktop
    (layout["app"] / "settings.json").write_text("[broken")

    found = desktop_install.detect(layout["app"], layout["legacy"])

    assert found.selected.install_id == "inst-adopted"
    assert [r.source for r in found.selected.model_roots] == ["data"]
    assert len(found.problems) == 1
    assert "settings.json" in found.problems[0]


def test_install_for_dir_matches_the_code_dir_and_the_data_dir(comfy_desktop, tmp_path):
    layout = comfy_desktop
    found = desktop_install.detect(layout["app"], layout["legacy"])

    assert desktop_install.install_for_dir(layout["data"], found).install_id == "inst-adopted"
    assert desktop_install.install_for_dir(layout["code"], found).install_id == "inst-adopted"
    assert desktop_install.install_for_dir(tmp_path, found) is None


@pytest.mark.parametrize(
    ("os_value", "env", "expected"),
    [
        (OS.MACOS, {}, Path.home() / "Library" / "Application Support" / "Comfy Desktop"),
        (OS.WINDOWS, {"APPDATA": "/appdata"}, Path("/appdata") / "Comfy Desktop"),
        (OS.LINUX, {"XDG_DATA_HOME": "/xdg"}, Path("/xdg") / "Comfy Desktop"),
    ],
)
def test_app_data_dir_per_os(monkeypatch, os_value, env, expected):
    monkeypatch.setattr(desktop_install, "get_os", lambda: os_value)
    for key in ("APPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert _real_app_data_dir() == expected


def test_legacy_config_dir_per_os(monkeypatch):
    monkeypatch.setattr(desktop_install, "get_os", lambda: OS.MACOS)
    assert _real_legacy_config_dir() == Path.home() / "Library" / "Application Support" / "ComfyUI"
    monkeypatch.setattr(desktop_install, "get_os", lambda: OS.LINUX)
    assert _real_legacy_config_dir() is None


@pytest.mark.parametrize("timestamp", [1e30, float("inf")])
def test_invalid_launch_timestamp_does_not_break_discovery(tmp_path, timestamp):
    install_path = tmp_path / "Installs" / "Valid"
    _checkout(install_path / "ComfyUI")
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [{"id": "valid", "installPath": str(install_path), "lastLaunchedAt": timestamp}],
    )

    found = desktop_install.detect(app, tmp_path / "legacy")

    assert found.to_dict()["installs"][0]["last_launched_at"] is None
    assert found.selected.install_id == "valid"


def test_adopted_install_with_unreadable_legacy_config_marks_data_root_uncertain(tmp_path):
    code = _checkout(tmp_path / "checkout")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "adopted", "installPath": str(code), "adopted": True}])
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "config.json").write_text("{broken")

    found = desktop_install.detect(app, legacy)

    assert any("config.json" in problem for problem in found.problems)
    assert any("unverified" in problem and "data" in problem for problem in found.problems)


def test_detect_scans_modern_userdata_and_keeps_each_registry_roots(tmp_path, monkeypatch):
    old = tmp_path / "Comfy Desktop"
    modern = tmp_path / "comfyui-desktop-2"
    for app, name in ((old, "old"), (modern, "modern")):
        code = _checkout(tmp_path / name / "ComfyUI")
        _write_desktop(
            app, [{"id": name, "installPath": str(code.parent)}], models_dirs=[str(tmp_path / name / "shared")]
        )
    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: old)
    found = desktop_install.detect()
    assert {install.install_id for install in found.installs} == {"old", "modern"}
    modern_install = next(install for install in found.installs if install.install_id == "modern")
    assert str(tmp_path / "modern" / "shared") in [root.path for root in modern_install.model_roots]
    assert str(tmp_path / "old" / "shared") not in [root.path for root in modern_install.model_roots]


def test_running_server_selects_matching_instance_instead_of_latest(tmp_path):
    records = []
    for name, launched in (("running", 1), ("latest", 2)):
        code = _checkout(tmp_path / name / "ComfyUI")
        records.append({"id": name, "installPath": str(code.parent), "lastLaunchedAt": launched})
    app = _write_desktop(tmp_path / "Comfy Desktop", records)
    found = desktop_install.detect(
        app, server_stats={"system": {"argv": [str(tmp_path / "running" / "ComfyUI" / "main.py")]}}
    )
    assert found.selected.install_id == "running"
    assert "running server" in found.selected_reason


def test_server_instance_config_matches_install_and_applies_effective_paths(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "inst-live", "installPath": str(code.parent)}])
    config = app / "instance-model-paths" / "inst-live.yaml"
    config.parent.mkdir()
    root = tmp_path / "effective-models"
    config.write_text(f"live:\n  base_path: {root}\n  checkpoints: checkpoints\n")
    data = tmp_path / "effective-data"
    inputs = tmp_path / "private-inputs"
    stats = {
        "system": {
            "argv": [
                "main.py",
                "--extra-model-paths-config",
                str(config),
                "--base-directory",
                str(data),
                "--input-directory",
                str(inputs),
            ]
        }
    }
    found = desktop_install.detect(app, server_stats=stats)
    assert found.selected.install_id == "inst-live"
    assert found.selected.data_dir == str(data)
    assert str(root) in [root.path for root in found.selected.model_roots]
    assert found.to_dict()["input_dir"] == str(inputs)


def test_server_that_does_not_match_records_is_not_silently_assumed(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "candidate", "installPath": str(code.parent)}])
    found = desktop_install.detect(app, server_stats={"system": {"argv": [str(tmp_path / "different" / "main.py")]}})
    assert found.selected is None
    assert any("running server" in problem for problem in found.problems)


def test_server_model_configs_replace_stale_shared_settings(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(
        tmp_path / "Comfy Desktop",
        [{"id": "live", "installPath": str(code.parent)}],
        models_dirs=[str(tmp_path / "stale")],
    )
    first, second = tmp_path / "one.yaml", tmp_path / "two.yaml"
    first.write_text(f"one:\n  base_path: {tmp_path / 'models-one'}\n  checkpoints: checkpoints\n")
    second.write_text(f"two:\n  base_path: {tmp_path / 'models-two'}\n  loras: loras\n")
    stats = {
        "system": {
            "argv": [str(code / "main.py"), "--extra-model-paths-config", str(first), str(second)],
            "comfyui_version": "0.99.0",
        }
    }
    found = desktop_install.detect(app, server_stats=stats)
    paths = [root.path for root in found.selected.model_roots]
    assert str(tmp_path / "stale") not in paths
    assert str(tmp_path / "models-one") in paths
    assert str(tmp_path / "models-two") in paths
    assert found.selected.comfyui_version == "v0.99.0"


def test_relative_server_data_paths_are_unverified(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "live", "installPath": str(code.parent)}])
    found = desktop_install.detect(
        app, server_stats={"system": {"argv": [str(code / "main.py"), "--base-directory", "relative-data"]}}
    )
    assert found.selected is None
    assert any("relative" in problem for problem in found.problems)


def test_server_config_includes_absolute_paths_outside_base_and_no_base_sections(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "live", "installPath": str(code.parent)}])
    config = tmp_path / "models.yaml"
    outside = tmp_path / "private-loras"
    config.write_text(
        f"one:\n  base_path: {tmp_path / 'models'}\n  loras: {outside}\ntwo:\n  checkpoints: relative-checkpoints\n"
    )
    found = desktop_install.detect(
        app, server_stats={"system": {"argv": [str(code / "main.py"), "--extra-model-paths-config", str(config)]}}
    )
    paths = [root.path for root in found.selected.model_roots]
    assert str(outside) in paths
    assert str(tmp_path / "relative-checkpoints") in paths
    assert not found.problems


@pytest.mark.parametrize("explicit", [True, False])
def test_running_server_output_directory_is_reported_with_source(tmp_path, explicit):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "live", "installPath": str(code.parent)}])
    output = tmp_path / "actual-output" if explicit else code / "output"
    argv = [str(code / "main.py")]
    if explicit:
        argv += ["--output-directory", str(output)]
    found = desktop_install.detect(app, server_stats={"system": {"argv": argv}})
    assert found.to_dict()["output_dir"] == str(output)
    assert found.to_dict()["output_dir_source"] == ("server_argv" if explicit else "server_default")


def test_relative_output_directory_requires_process_cwd_verification(tmp_path):
    code = _checkout(tmp_path / "install" / "ComfyUI")
    app = _write_desktop(tmp_path / "Comfy Desktop", [{"id": "live", "installPath": str(code.parent)}])
    found = desktop_install.detect(
        app, server_stats={"system": {"argv": [str(code / "main.py"), "--output-directory", "relative-output"]}}
    )
    assert found.selected is None
    assert "output_dir" not in found.to_dict()
