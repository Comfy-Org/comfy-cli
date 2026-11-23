"""The typer requirement must exclude releases whose parameters have no click type.

typer 0.29 dropped ``Parameter.type`` from ``TyperOption``/``TyperArgument``.
``comfy --help --json`` (``comfy_cli/help_json.py``) reads ``param.type`` for every
option, so a fresh ``pip install comfy-cli`` that resolves typer 0.29 crashes
discovery with ``AttributeError: 'TyperOption' object has no attribute 'type'``.
CI installs from ``uv.lock`` and never sees it, so the bound is checked here.
"""

from pathlib import Path

import pytest
import tomlkit
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from comfy_cli import help_json

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
INCOMPATIBLE = [Version(v) for v in ("0.29.0", "0.29.1", "0.29.5", "0.30.0", "1.0.0")]


def _typer_requirements():
    project = tomlkit.parse(PYPROJECT.read_text())["project"]
    groups = [project["dependencies"], *project.get("optional-dependencies", {}).values()]
    reqs = [Requirement(str(dep)) for group in groups for dep in group]
    return [req for req in reqs if canonicalize_name(req.name) == "typer"]


def test_every_typer_requirement_excludes_0_29_and_later():
    reqs = _typer_requirements()
    assert reqs, "pyproject.toml no longer declares typer"
    for req in reqs:
        for version in INCOMPATIBLE:
            assert not req.specifier.contains(version, prereleases=True), (
                f"{req} allows typer {version}, whose parameters have no "
                "click type; help_json.py would crash on a fresh install"
            )


def test_every_locked_typer_satisfies_the_requirement():
    lock = tomlkit.parse((PYPROJECT.parent / "uv.lock").read_text())
    locked = [Version(str(p["version"])) for p in lock["package"] if canonicalize_name(str(p["name"])) == "typer"]
    assert locked, "uv.lock has no typer package"
    for version in locked:
        for req in _typer_requirements():
            assert req.specifier.contains(version), f"uv.lock pins typer {version}, outside {req}"


class _ParamWithoutType:
    """What typer >= 0.29 hands out: a parameter with no click ``type``."""

    name = "x"
    param_type_name = "option"
    required = False
    default = None
    opts = ["--x"]
    secondary_opts: list[str] = []
    envvar = None


def test_param_without_click_type_names_the_fix():
    with pytest.raises(RuntimeError, match=r"typer<0\.29"):
        help_json._param_to_dict(_ParamWithoutType())
