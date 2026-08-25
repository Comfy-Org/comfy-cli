"""Startup import budget for the ``comfy`` entry point.

Every ``comfy`` invocation pays its import graph before doing any work, and
agents (comfy-agent, the MCP server) pay it on every tool call. This test pins
two things so the floor cannot creep back up unnoticed:

* Heavy libraries that only a few commands need must not load for a trivial
  ``--json`` command. ``psutil`` and GitPython used to be imported at module
  level by ``utils.py`` / ``hardware.py`` / ``command/install.py``; together
  they were the largest single share of startup time.
* The total module count stays under a budget. The budget is deliberately
  loose (it tracks Python and dependency versions, not just our code); tighten
  it when the lazy subcommand groups land.

The measurement is ``python -X importtime``, run in a subprocess so it sees a
cold interpreter exactly like a user does.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Libraries that must stay off the startup path. Add to this list when a
# lazy-import fix lands so the fix is guarded.
FORBIDDEN_AT_STARTUP = ("psutil", "git", "gitdb")

# Loose ceiling; the eager command-group imports keep us near 800 today.
MODULE_BUDGET = 900

_IMPORT_LINE = re.compile(r"^import time:\s+\d+ \|\s+\d+ \|\s*(\S+)")


def _startup_modules() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "comfy_cli", "--json", "version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    modules = []
    for line in proc.stderr.splitlines():
        m = _IMPORT_LINE.match(line)
        if m:
            modules.append(m.group(1))
    assert modules, f"no importtime output captured (rc={proc.returncode}); stderr tail: {proc.stderr[-500:]}"
    return modules


def test_heavy_libraries_stay_lazy():
    modules = _startup_modules()
    top_level = {name.split(".")[0] for name in modules}
    loaded = sorted(top_level & set(FORBIDDEN_AT_STARTUP))
    assert not loaded, f"imported at startup but should be lazy: {loaded}"


def test_startup_module_count_within_budget():
    modules = _startup_modules()
    assert len(modules) <= MODULE_BUDGET, (
        f"{len(modules)} modules imported for `comfy --json version` (budget {MODULE_BUDGET}); "
        "a new module-level import probably landed on the startup path"
    )
