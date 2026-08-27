"""``comfy_cli.command`` must expose every subcommand module as an attribute
and must import none of them at package import.

Background: the package used to import every submodule eagerly, added during
the agent-shaped-cli rewrite after a startup flake —
``ImportError: cannot import name 'transfer' from 'comfy_cli.command'`` on
the first ``comfy run`` of a session. The eager import cost ~200 ms and ~340
modules on every ``comfy`` invocation. The actual cause was an import cycle
(``launch`` → ``custom_nodes.cm_cli_util`` → ``custom_nodes.command`` →
``bisect_custom_nodes`` → ``launch``) that the eager, ordered import happened
to mask; ``test_each_submodule_imports_on_its_own`` below catches that class
of bug directly.

The package now resolves submodules lazily (PEP 562 ``__getattr__``). Adding a
new subcommand? Add it to ``comfy_cli/command/__init__.py:_SUBMODULES`` AND to
``EXPECTED_SUBMODULES`` here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

EXPECTED_SUBMODULES = frozenset(
    {
        "code_search",
        "custom_nodes",
        "generate",
        "install",
        "job_watcher",
        "jobs",
        "launch",
        "nodes",
        "pr_command",
        "project",
        "run",
        "run_cli",
        "templates",
        "transfer",
        "workflow",
    }
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_command_package_exposes_all_submodules_as_attributes():
    """Attribute access — ``hasattr(comfy_cli.command, 'X')`` — for every
    expected name; lazy resolution must cover the whole set."""
    import comfy_cli.command as cmd

    missing = sorted(name for name in EXPECTED_SUBMODULES if not hasattr(cmd, name))
    assert not missing, (
        f"comfy_cli.command/__init__.py cannot resolve: {missing}. Add them to `_SUBMODULES` (and `__all__`)."
    )
    assert set(cmd.__all__) == EXPECTED_SUBMODULES


def test_fresh_interpreter_can_from_import_each_submodule():
    """In a *fresh* interpreter (no shared module cache), each
    ``from comfy_cli.command import X`` must succeed. This is the scenario
    where the original ImportError flake bit."""
    result = _run(
        f"""
        names = {sorted(EXPECTED_SUBMODULES)!r}
        for n in names:
            mod = __import__("comfy_cli.command", fromlist=[n])
            assert hasattr(mod, n), f"missing {{n!r}} after import"
        """
    )
    assert result.returncode == 0, result.stderr


def test_package_import_pulls_no_submodules():
    """The startup-cost guarantee: importing the package imports none of the
    subcommand modules."""
    result = _run(
        """
        import sys
        import comfy_cli.command
        loaded = sorted(m for m in sys.modules if m.startswith("comfy_cli.command."))
        print(loaded)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"package import loaded submodules: {result.stdout}"


@pytest.mark.parametrize("name", sorted(EXPECTED_SUBMODULES))
def test_each_submodule_imports_on_its_own(name):
    """Every submodule must import first, in a fresh interpreter, with nothing
    else loaded. With lazy resolution the import order is whatever the
    invoked command needs, so a cycle between two submodules is a real bug
    rather than something the package init papers over."""
    result = _run(f"import comfy_cli.command.{name}")
    assert result.returncode == 0, result.stderr
