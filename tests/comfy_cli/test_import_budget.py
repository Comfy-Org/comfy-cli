"""Startup import guard for the ``comfy`` entry point.

Every ``comfy`` invocation pays its import graph before doing any work, and
agents (comfy-agent, the MCP server) pay it on every tool call. This test pins
one thing: heavy libraries that only a few commands need must not load for a
trivial ``--version`` call. ``psutil`` and GitPython used to be imported at
module level by ``utils.py`` / ``hardware.py`` / ``command/install.py``;
together they were the largest single share of startup time.

The measurement is ``python -X importtime -m comfy_cli --json --version``, run
in a subprocess so it sees a cold interpreter exactly like a user does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

# Libraries that must stay off the startup path. Add to this list when a
# lazy-import fix lands so the fix is guarded.
FORBIDDEN_AT_STARTUP = ("psutil", "git", "gitdb")

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


def test_heavy_libraries_stay_lazy():
    modules = _startup_modules()
    top_level = {name.split(".")[0] for name in modules}
    loaded = sorted(top_level & set(FORBIDDEN_AT_STARTUP))
    assert not loaded, f"imported at startup but should be lazy: {loaded}"
