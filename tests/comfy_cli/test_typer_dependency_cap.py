"""The typer requirement must exclude releases whose parameters have no click type.

typer 0.29 dropped ``Parameter.type`` from ``TyperOption``/``TyperArgument``.
``comfy --help --json`` (``comfy_cli/help_json.py``) reads ``param.type`` for every
option, so a fresh ``pip install comfy-cli`` that resolves typer 0.29 crashes
discovery with ``AttributeError: 'TyperOption' object has no attribute 'type'``.
CI installs from ``uv.lock`` and never sees it, so the bound is checked here.
"""

from pathlib import Path

import tomlkit
from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
FIRST_INCOMPATIBLE = Version("0.29.0")


def _typer_requirements():
    project = tomlkit.parse(PYPROJECT.read_text())["project"]
    groups = [project["dependencies"], *project.get("optional-dependencies", {}).values()]
    reqs = [Requirement(str(dep)) for group in groups for dep in group]
    return [req for req in reqs if req.name == "typer"]


def test_every_typer_requirement_excludes_0_29():
    reqs = _typer_requirements()
    assert reqs, "pyproject.toml no longer declares typer"
    for req in reqs:
        assert not req.specifier.contains(FIRST_INCOMPATIBLE, prereleases=True), (
            f"{req} allows typer {FIRST_INCOMPATIBLE}, whose parameters have no "
            "click type; help_json.py would crash on a fresh install"
        )


def test_locked_typer_satisfies_the_requirement():
    lock = tomlkit.parse((PYPROJECT.parent / "uv.lock").read_text())
    locked = next(Version(str(p["version"])) for p in lock["package"] if p["name"] == "typer")
    for req in _typer_requirements():
        assert req.specifier.contains(locked), f"uv.lock pins typer {locked}, outside {req}"
