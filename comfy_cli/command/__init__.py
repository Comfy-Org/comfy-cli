"""CLI subcommand modules.

Submodules are exposed as package attributes but imported on first access
(PEP 562 ``__getattr__``), not at package import. Importing this package used
to import every subcommand module eagerly, which put ~200 ms and ~340 modules
on the startup path of every ``comfy`` invocation, including the ones agents
make on every tool call.

``from comfy_cli.command import X`` stays robust two ways: Python's own
``from``-import falls through to importing the ``comfy_cli.command.X``
submodule when the attribute is missing, and ``__getattr__`` below does the
same for plain attribute access (``comfy_cli.command.X``). The contract is
pinned by ``tests/comfy_cli/command/test_command_init.py``.

Adding a new subcommand? Add it to ``_SUBMODULES`` here AND to
``tests/comfy_cli/command/test_command_init.py:EXPECTED_SUBMODULES``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_SUBMODULES = (
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
)

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
