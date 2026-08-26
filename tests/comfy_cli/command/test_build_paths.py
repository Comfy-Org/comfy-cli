"""Pin the `comfy build` PATH resolution table (build design lines 291-312).

One optional positional answers two questions — where the spec is and where the
install is — so the three rows of that table, and the rule that
``--models-dir``/``--custom-nodes-dir``/``--python`` move only the install half,
are the whole contract. Every test runs from a temp cwd so nothing leaks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_cli import error_codes
from comfy_cli.command.build_paths import (
    SPEC_FILENAME,
    BuildSpecNotFoundError,
    InstallOverrides,
    resolve_build_paths,
)

#: The install-half members. A row names the ones it expects to MOVE; every
#: other member must still equal the no-override baseline — that is what makes
#: "overrides the install half only" an assertion rather than a comment.
INSTALL_MEMBERS = ("models_dir", "custom_nodes_dir", "python")


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty temp cwd. Yields ``Path.cwd()`` — not ``tmp_path`` — because the
    OS resolves symlinks in the working directory and the two can differ."""
    monkeypatch.chdir(tmp_path)
    return Path.cwd()


def _write_spec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("schema: comfy-build/1\n", encoding="utf-8")
    return path


# (positional PATH, spec relative to cwd, install root relative to cwd)
RESOLUTION_ROWS = [
    pytest.param(None, SPEC_FILENAME, ".", id="nothing"),
    pytest.param("", SPEC_FILENAME, ".", id="empty-string"),
    pytest.param(".", SPEC_FILENAME, ".", id="dot"),
    pytest.param("install", f"install/{SPEC_FILENAME}", "install", id="directory"),
    pytest.param("nested/install", f"nested/install/{SPEC_FILENAME}", "nested/install", id="nested-directory"),
    pytest.param(Path("install"), f"install/{SPEC_FILENAME}", "install", id="directory-as-Path"),
    pytest.param("spec.yaml", "spec.yaml", ".", id="yaml-file"),
    pytest.param("cfg/spec.yaml", "cfg/spec.yaml", "cfg", id="yaml-file-nested"),
    pytest.param("spec.json", "spec.json", ".", id="json-file"),
    pytest.param("cfg/spec.json", "cfg/spec.json", "cfg", id="json-file-nested"),
]


@pytest.mark.parametrize(("given", "rel_spec", "rel_root"), RESOLUTION_ROWS)
def test_path_resolves_to_a_spec_file_and_an_install_root(cwd: Path, given, rel_spec: str, rel_root: str):
    """Given a PATH from the design's table, When resolved, Then both halves match the row."""
    _write_spec(cwd / rel_spec)

    paths = resolve_build_paths(given)

    assert paths.spec_file == cwd / rel_spec
    assert paths.install_root == cwd / rel_root
    assert paths.spec_file.is_absolute()
    assert paths.install_root.is_absolute()
    # The spec always sits directly in the install root — true of all three rows.
    assert paths.spec_file.parent == paths.install_root


@pytest.mark.parametrize(("given", "rel_spec", "rel_root"), RESOLUTION_ROWS)
def test_the_install_half_defaults_to_the_install_root(cwd: Path, given, rel_spec: str, rel_root: str):
    """Given no override flags, When resolved, Then models/custom_nodes hang off the root."""
    _write_spec(cwd / rel_spec)

    paths = resolve_build_paths(given)

    assert paths.models_dir == paths.install_root / "models"
    assert paths.custom_nodes_dir == paths.install_root / "custom_nodes"
    # No filesystem-free default exists for the interpreter.
    assert paths.python is None


def test_an_absolute_path_is_used_verbatim(cwd: Path, tmp_path: Path):
    """Given an absolute PATH, When resolved, Then the cwd is not consulted."""
    elsewhere = tmp_path.parent / "absolute-install"
    _write_spec(elsewhere / SPEC_FILENAME)

    paths = resolve_build_paths(str(elsewhere))

    assert paths.install_root == elsewhere
    assert paths.spec_file == elsewhere / SPEC_FILENAME


# (overrides, the install members it may move → their paths relative to cwd)
OVERRIDE_ROWS = [
    pytest.param(
        InstallOverrides.from_options(models_dir="elsewhere/models"),
        # A split layout moves custom_nodes/ with it: the sibling rule `scan`
        # already uses (build.py:783-784).
        {"models_dir": "elsewhere/models", "custom_nodes_dir": "elsewhere/custom_nodes"},
        id="models-dir",
    ),
    pytest.param(
        InstallOverrides.from_options(custom_nodes_dir="elsewhere/nodes"),
        {"custom_nodes_dir": "elsewhere/nodes"},
        id="custom-nodes-dir",
    ),
    pytest.param(
        InstallOverrides.from_options(python="venv/bin/python"),
        {"python": "venv/bin/python"},
        id="python",
    ),
    pytest.param(
        InstallOverrides.from_options(
            models_dir="d/models",
            custom_nodes_dir="n/nodes",
            python="p/bin/python",
        ),
        {"models_dir": "d/models", "custom_nodes_dir": "n/nodes", "python": "p/bin/python"},
        id="all-three",
    ),
]


@pytest.mark.parametrize(("overrides", "moved"), OVERRIDE_ROWS)
def test_overrides_move_the_install_half_only(cwd: Path, overrides: InstallOverrides, moved: dict[str, str]):
    """Given install-half overrides, When resolved, Then the spec half is untouched."""
    _write_spec(cwd / "install" / SPEC_FILENAME)
    baseline = resolve_build_paths("install")

    paths = resolve_build_paths("install", overrides=overrides)

    assert paths.spec_file == baseline.spec_file
    assert paths.install_root == baseline.install_root
    for member in INSTALL_MEMBERS:
        expected = cwd / moved[member] if member in moved else getattr(baseline, member)
        assert getattr(paths, member) == expected, member


MISSING_SPEC_ROWS = [
    pytest.param(None, SPEC_FILENAME, id="nothing"),
    pytest.param("install", f"install/{SPEC_FILENAME}", id="directory"),
    pytest.param("cfg/spec.yaml", "cfg/spec.yaml", id="yaml-file"),
    pytest.param("cfg/spec.json", "cfg/spec.json", id="json-file"),
]


@pytest.mark.parametrize(("given", "probed"), MISSING_SPEC_ROWS)
def test_a_missing_spec_names_the_exact_absolute_path_probed(cwd: Path, given, probed: str):
    """Given an empty cwd, When a non-`init` command resolves, Then it fails naming the probe."""
    with pytest.raises(BuildSpecNotFoundError) as excinfo:
        resolve_build_paths(given)

    error = excinfo.value
    expected = cwd / probed
    assert error.code == "build_spec_not_found"
    assert error.spec_file == expected
    assert error.details == {"path": str(expected)}
    assert Path(error.details["path"]).is_absolute()
    assert str(expected) in str(error)


def test_a_directory_that_does_not_exist_still_reports_the_spec_it_would_hold(cwd: Path):
    """Given a PATH that is not on disk at all, When resolved, Then the probe is still the spec path."""
    with pytest.raises(BuildSpecNotFoundError) as excinfo:
        resolve_build_paths("no/such/dir")

    assert excinfo.value.details == {"path": str(cwd / "no" / "such" / "dir" / SPEC_FILENAME)}


def test_init_may_resolve_a_spec_that_does_not_exist_yet(cwd: Path):
    """Given `require_spec=False`, When the spec is absent, Then resolution succeeds."""
    paths = resolve_build_paths("install", require_spec=False)

    assert paths.spec_file == cwd / "install" / SPEC_FILENAME
    assert paths.install_root == cwd / "install"
    assert not paths.spec_file.exists()


def test_a_directory_named_like_a_spec_file_does_not_hijack_a_bare_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Given a cwd literally named `*.yaml`, When PATH is omitted, Then it is still row one."""
    workdir = tmp_path / "project.yaml"
    _write_spec(workdir / SPEC_FILENAME)
    monkeypatch.chdir(workdir)
    here = Path.cwd()

    for given in (None, "."):
        paths = resolve_build_paths(given)
        assert paths.install_root == here
        assert paths.spec_file == here / SPEC_FILENAME


def test_resolution_does_not_follow_a_symlink_out_of_the_named_root(cwd: Path):
    """Given a symlinked PATH, When resolved, Then the paths stay under the name the caller typed."""
    real = cwd / "real"
    _write_spec(real / SPEC_FILENAME)
    (cwd / "link").symlink_to(real, target_is_directory=True)

    paths = resolve_build_paths("link")

    assert paths.install_root == cwd / "link"
    assert paths.spec_file == cwd / "link" / SPEC_FILENAME
    assert paths.spec_file.parent == paths.install_root
    # Not an accident: following the link WOULD have landed on a path the
    # caller never named, which is exactly what the error payload must not say.
    assert paths.spec_file != real / SPEC_FILENAME
    assert paths.spec_file.resolve() == (real / SPEC_FILENAME).resolve()


def test_a_parent_traversal_is_left_for_the_os_to_walk(cwd: Path):
    """Given a PATH containing `..`, When resolved, Then it is not collapsed lexically.

    Collapsing `link/../x` to `x` would silently retarget the path whenever
    `link` is a symlink, so the component survives into the resolved paths.
    """
    _write_spec(cwd / "other" / SPEC_FILENAME)

    paths = resolve_build_paths("real/../other", require_spec=False)

    assert paths.install_root == cwd / "real" / ".." / "other"
    assert ".." in paths.spec_file.parts


def test_the_error_code_is_registered():
    """The registry is the agent-facing contract; the raised code must be in it."""
    registered = error_codes.get(BuildSpecNotFoundError.code)
    assert registered is not None
    assert registered.hint
