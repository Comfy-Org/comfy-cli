"""Startup import budget for the ``comfy`` entry point.

Every ``comfy`` invocation pays its import graph before doing any work, and
agents (comfy-agent, the MCP server) pay it on every tool call. This test pins
two things so the floor cannot creep back up unnoticed:

* Heavy libraries and subcommand modules that only some commands need must
  not load for a trivial ``--version`` call. ``psutil``, GitPython, and the
  whole ``comfy_cli.command`` package used to load at startup; see
  ``comfy_cli/_lazy.py`` and ``comfy_cli/command/__init__.py``.
* The total module count stays under a budget. The budget is deliberately
  loose (it tracks Python and dependency versions, not just our code); tighten
  it when the next lazy-import fix lands.

The measurement is ``python -X importtime -m comfy_cli --json --version``, run
in a subprocess so it sees a cold interpreter exactly like a user does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter

# Modules that must stay off the startup path (a bare name covers the whole
# package). Add to this list when a lazy-import fix lands so the fix is guarded.
FORBIDDEN_AT_STARTUP = (
    "psutil",
    "git",
    "gitdb",
    "requests",
    "urllib.request",  # via comfy_cli.http; downloads only
    "comfy_cli.http",
    "comfy_cli.standalone",
    "comfy_cli.command.custom_nodes",
    "comfy_cli.command.install",
    "comfy_cli.command.run",
    "comfy_cli.command.transfer",
    "comfy_cli.command.project",
    "comfy_cli.command.nodes",
    "comfy_cli.command.launch",
    "comfy_cli.cloud.command",
)

# Ceiling with some slack over today's ~460 (it tracks Python and dependency
# versions, not just our code). Tighten when the next lazy-import fix lands.
MODULE_BUDGET = 700

_IMPORT_LINE = re.compile(r"^import time:\s+\d+ \|\s+\d+ \|\s*(\S+)")


def _startup_modules() -> list[str]:
    # pytest-cov's subprocess hook (a .pth file keyed on COV_CORE_* / COVERAGE_*
    # env vars) would load `coverage` and its dependencies into the child before
    # comfy_cli runs, adding a few hundred modules that a user never pays for.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE_", "COVERAGE_"))}
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "comfy_cli", "--json", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, (
        f"`comfy --json --version` failed (rc={proc.returncode}); stderr tail: {proc.stderr[-500:]}"
    )
    modules = []
    for line in proc.stderr.splitlines():
        m = _IMPORT_LINE.match(line)
        if m:
            modules.append(m.group(1))
    assert modules, f"no importtime output captured (rc={proc.returncode}); stderr tail: {proc.stderr[-500:]}"
    return modules


def _breakdown(modules: list[str], top: int = 12) -> str:
    """Module count per top-level package, so a budget miss names the culprit."""
    counts = Counter(name.split(".")[0] for name in modules)
    return ", ".join(f"{pkg}={n}" for pkg, n in counts.most_common(top))


def test_heavy_libraries_stay_lazy():
    modules = _startup_modules()
    loaded = sorted(
        forbidden
        for forbidden in FORBIDDEN_AT_STARTUP
        if any(name == forbidden or name.startswith(forbidden + ".") for name in modules)
    )
    assert not loaded, f"imported at startup but should be lazy: {loaded}"


def test_startup_module_count_within_budget():
    modules = _startup_modules()
    assert len(modules) <= MODULE_BUDGET, (
        f"{len(modules)} modules imported for `comfy --json --version` (budget {MODULE_BUDGET}); "
        f"a new module-level import probably landed on the startup path. By package: {_breakdown(modules)}"
    )
