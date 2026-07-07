"""``comfy_cli.command`` package must eagerly re-export every submodule.

Background: during the agent-shaped-cli rewrite, ``cmdline.py:20`` does

    from comfy_cli.command import transfer as transfer_inner

which Python normally resolves either as an attribute of the ``command``
package or as a lazy submodule import (file ``comfy_cli/command/transfer.py``).
A single startup flake was observed where the lazy path returned
``ImportError: cannot import name 'transfer' from 'comfy_cli.command'``
on the very first invocation despite the file existing on disk.

The fix is defensive: every CLI subcommand module is re-exported from the
package's ``__init__.py``, so ``from comfy_cli.command import X`` always
succeeds via attribute lookup before falling through to the submodule
discovery path. This pins the contract: if a submodule is added to the
filesystem but forgotten in ``__init__``, the test below fails before any
end user sees a broken CLI.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

EXPECTED_SUBMODULES = frozenset(
    {
        "code_search",
        "custom_nodes",
        "install",
        "job_watcher",
        "jobs",
        "launch",
        "nodes",
        "pr_command",
        "run",
        "run_cli",
        "transfer",
        "workflow",
    }
)


def test_command_package_exposes_all_submodules_as_attributes():
    """Direct attribute access — ``hasattr(comfy_cli.command, 'X')`` for
    every ``X`` in the expected set. This catches forgotten re-exports
    without needing a fresh subprocess.
    """
    import comfy_cli.command as cmd

    missing = sorted(name for name in EXPECTED_SUBMODULES if not hasattr(cmd, name))
    assert not missing, (
        f"comfy_cli.command/__init__.py forgot to re-export: {missing}. "
        "Add them to the `from . import …` line and to `__all__`."
    )


def test_fresh_interpreter_can_from_import_each_submodule():
    """Stronger check: in a *fresh* interpreter (no shared module cache),
    each ``from comfy_cli.command import X`` must succeed. This is the
    scenario where the original ImportError flake bit — the very first
    ``comfy run`` in a session.
    """
    code = textwrap.dedent(f"""
        names = {sorted(EXPECTED_SUBMODULES)!r}
        for n in names:
            mod = __import__("comfy_cli.command", fromlist=[n])
            assert hasattr(mod, n), f"missing {{n!r}} after import"
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"fresh-interpreter check failed:\n  stdout={result.stdout}\n  stderr={result.stderr}"
    )


def test_all_lists_every_submodule():
    """``__all__`` is the source of truth for ``from comfy_cli.command
    import *`` consumers. Keep it in sync with the actual exports.
    """
    import comfy_cli.command as cmd

    declared = set(getattr(cmd, "__all__", ()))
    missing = EXPECTED_SUBMODULES - declared
    assert not missing, f"__all__ is missing: {sorted(missing)}. Update __all__ in comfy_cli/command/__init__.py."
