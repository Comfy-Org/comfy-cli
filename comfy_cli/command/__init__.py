"""CLI subcommand modules.

Every submodule is eagerly re-exported so ``from comfy_cli.command import
X`` is resolved via attribute lookup, not via Python's lazy
submodule-discovery path. The lazy path was observed to flake on the
very first ``comfy run`` of a fresh session — ``ImportError: cannot
import name 'transfer' from 'comfy_cli.command'`` — even though the file
existed on disk. Eager imports here pin the contract and make
``cmdline.py``'s many ``from comfy_cli.command import X as Y_inner``
lines robust against that race.

Adding a new subcommand? Add it here AND to
``tests/comfy_cli/command/test_command_init.py:EXPECTED_SUBMODULES``.
"""

from . import (
    code_search,
    custom_nodes,
    generate,
    install,
    job_watcher,
    jobs,
    launch,
    nodes,
    pr_command,
    project,
    run,
    run_cli,
    templates,
    transfer,
    workflow,
)

__all__ = [
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
]
