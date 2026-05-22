"""Guided realtime walkthrough of the comfy CLI surface.

Invoked as ``comfy run-cli``. Subprocess-executes the CLI itself step by step
so what the user sees is the real CLI, not a recording. For most commands the
demo shows two views of the same call — the pretty terminal output and the
``--json`` envelope an agent would parse. Toward the end the demo fires a
fleet of jobs in parallel via ``--async`` to show background execution and
queue inspection.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from comfy_cli.output import rprint as pprint

# ---------- Demo workflows ---------------------------------------------------

# Tiny workflow — completes in ~30 ms. Used for the single-job demos.
DEMO_WORKFLOW: dict = {
    "1": {
        "inputs": {"width": 512, "height": 512, "batch_size": 1, "color": 16711680},
        "class_type": "EmptyImage",
        "_meta": {"title": "Red canvas"},
    },
    "2": {
        "inputs": {"image": ["1", 0]},
        "class_type": "ImageInvert",
        "_meta": {"title": "Invert"},
    },
    "3": {
        "inputs": {"filename_prefix": "comfy_run_cli_demo", "images": ["2", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save"},
    },
}


def fleet_workflow(idx: int) -> dict:
    """Heavier workflow used for the parallel-fleet demo.

    4096×4096 canvas + a chained invert/scale pipeline so the server takes
    ~1 second per job. With 5 in the queue processed serially, the queue is
    clearly visible in ``jobs ls`` as running + pending. The ``color`` differs
    per index so ComfyUI's per-node cache can't collapse them into one.
    """
    color = 0x110000 * (idx + 1)  # distinct color per job → no cache hits
    return {
        "1": {
            "inputs": {"width": 4096, "height": 4096, "batch_size": 1, "color": color},
            "class_type": "EmptyImage",
            "_meta": {"title": f"Canvas #{idx}"},
        },
        "2": {
            "inputs": {"image": ["1", 0]},
            "class_type": "ImageInvert",
        },
        "3": {
            "inputs": {"scale_by": 0.5, "upscale_method": "lanczos", "image": ["2", 0]},
            "class_type": "ImageScaleBy",
        },
        "4": {
            "inputs": {"scale_by": 2.0, "upscale_method": "lanczos", "image": ["3", 0]},
            "class_type": "ImageScaleBy",
        },
        "5": {
            "inputs": {"image": ["4", 0]},
            "class_type": "ImageInvert",
        },
        "6": {
            "inputs": {"filename_prefix": f"comfy_run_cli_fleet_{idx}", "images": ["5", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save"},
        },
    }


FLEET_SIZE = 5


# ---------- Step model -------------------------------------------------------


@dataclass
class Invocation:
    """One subprocess invocation, labelled for the demo."""

    argv: list[str]
    label: str
    capture: bool = False
    on_output: Callable[[str], None] | None = None
    optional: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class Step:
    title: str
    desc: str = ""  # one-line description, dim, optional
    invocations: list[Invocation] = field(default_factory=list)
    custom: Callable[[_DemoState, float, bool], int] | None = None  # for non-subprocess steps
    skip_if: Callable[[], bool] | None = None


# ---------- Runner -----------------------------------------------------------


def _comfy_argv() -> list[str]:
    return [sys.executable, "-m", "comfy_cli"]


def _print_banner(text: str) -> None:
    bar = "─" * max(8, min(72, len(text) + 4))
    pprint(f"\n[bold cyan]{bar}[/bold cyan]")
    pprint(f"[bold cyan]  {text}[/bold cyan]")
    pprint(f"[bold cyan]{bar}[/bold cyan]")


def _print_subhead(label: str, kind: str) -> None:
    if kind == "human":
        pprint(f"\n[bold green]▶ {label}[/bold green]  [dim](human view)[/dim]")
    elif kind == "agent":
        pprint(f"\n[bold magenta]▶ {label}[/bold magenta]  [dim](agent view, --json)[/dim]")
    else:
        pprint(f"\n[bold blue]▶ {label}[/bold blue]")


def _print_command(argv: list[str]) -> None:
    rendered = " ".join(shlex.quote(a) for a in argv)
    rendered = rendered.replace(shlex.quote(sys.executable) + " -m comfy_cli", "comfy")
    pprint(f"[bold]$[/bold] [yellow]{rendered}[/yellow]")


def _classify(inv: Invocation) -> str:
    if "--json-stream" in inv.argv or "--json" in inv.argv:
        return "agent"
    return "human"


def _run_invocation(inv: Invocation, *, pause_seconds: float, show_header: bool = True) -> int:
    if show_header:
        _print_subhead(inv.label, _classify(inv))
    _print_command(inv.argv)

    env = {**os.environ, "COMFY_OUTPUT": "pretty", **inv.extra_env}

    if inv.capture:
        try:
            result = subprocess.run(
                inv.argv,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            pprint("[bold red]invocation timed out[/bold red]")
            return 1
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
        if inv.on_output is not None and result.returncode == 0:
            inv.on_output(result.stdout)
        rc = result.returncode
    else:
        try:
            rc = subprocess.run(inv.argv, env=env, check=False, timeout=120).returncode
        except subprocess.TimeoutExpired:
            pprint("[bold red]invocation timed out[/bold red]")
            return 1

    if rc != 0 and not inv.optional:
        pprint(f"[bold red]Invocation exited with code {rc}[/bold red]")
    elif rc != 0:
        pprint(f"[dim](optional step exited {rc}, continuing)[/dim]")

    if pause_seconds > 0:
        time.sleep(pause_seconds)
    return rc


def _run_step(step: Step, state: _DemoState, *, pause_seconds: float, show_agent: bool) -> int:
    if step.skip_if is not None and step.skip_if():
        pprint(f"\n[bright_black](skipping: {step.title})[/bright_black]")
        return 0

    _print_banner(step.title)
    if step.desc:
        pprint(f"[dim]{step.desc}[/dim]")
    if pause_seconds > 0:
        time.sleep(min(pause_seconds, 1.5))

    if step.custom is not None:
        rc = step.custom(state, pause_seconds, show_agent)
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        return rc

    first_error = 0
    for inv in step.invocations:
        if not show_agent and _classify(inv) == "agent":
            pprint(f"\n[bright_black](skipping agent view of `{inv.label}` — use --show-agent)[/bright_black]")
            continue
        rc = _run_invocation(inv, pause_seconds=pause_seconds)
        if rc != 0 and not inv.optional and first_error == 0:
            first_error = rc

    if pause_seconds > 0:
        time.sleep(pause_seconds)
    return first_error


# ---------- Parallel fleet step ---------------------------------------------


def _submit_one_async(argv: list[str], idx: int) -> tuple[int, str | None, float]:
    """Submit one workflow with --async, return (idx, prompt_id, elapsed)."""
    env = {**os.environ, "COMFY_OUTPUT": "pretty"}
    t0 = time.time()
    try:
        result = subprocess.run(argv, env=env, check=False, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return (idx, None, time.time() - t0)
    elapsed = time.time() - t0
    if result.returncode != 0:
        return (idx, None, elapsed)
    try:
        doc = json.loads(result.stdout.strip().splitlines()[-1])
        return (idx, doc.get("data", {}).get("prompt_id"), elapsed)
    except (json.JSONDecodeError, IndexError):
        return (idx, None, elapsed)


def _fleet_step(state: _DemoState, pause_seconds: float, show_agent: bool) -> int:
    """Submit FLEET_SIZE workflows concurrently, then watch the queue drain."""
    comfy = _comfy_argv()

    # Write per-job workflow files so we can show distinct prompt_ids and
    # ComfyUI's cache treats them as separate jobs.
    workflow_paths: list[str] = []
    for i in range(FLEET_SIZE):
        fd, path = tempfile.mkstemp(prefix=f"comfy_run_cli_fleet_{i}_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(fleet_workflow(i), f)
        workflow_paths.append(path)
    state.fleet_workflow_paths = workflow_paths

    pprint(
        f"[dim]Submitting {FLEET_SIZE} workflows concurrently with --async. "
        "Each runs a 4096×4096 invert + scale pipeline (≈1s of CPU work). "
        "ComfyUI processes them one at a time, so you'll see the queue with one "
        "running and the rest pending in `jobs ls`.[/dim]"
    )
    _print_subhead(f"submit ×{FLEET_SIZE} in parallel", "agent")
    example_argv = [*comfy, "--json", "run", "--workflow", workflow_paths[0], "--async"]
    _print_command(example_argv)
    pprint(f"[dim]…and {FLEET_SIZE - 1} more like it, in parallel via a thread pool.[/dim]")

    submit_start = time.time()
    prompt_ids: list[str | None] = [None] * FLEET_SIZE
    elapsed: list[float] = [0.0] * FLEET_SIZE
    with concurrent.futures.ThreadPoolExecutor(max_workers=FLEET_SIZE) as ex:
        futures = []
        for i, path in enumerate(workflow_paths):
            argv = [*comfy, "--json", "run", "--workflow", path, "--async"]
            futures.append(ex.submit(_submit_one_async, argv, i))
        for fut in concurrent.futures.as_completed(futures):
            idx, pid, el = fut.result()
            prompt_ids[idx] = pid
            elapsed[idx] = el
            stamp = f"+{(time.time() - submit_start):0.2f}s"
            if pid:
                pprint(
                    f"  [green]✓[/green] job #{idx} → [yellow]{pid}[/yellow]  [dim](submit took {el:0.2f}s, wall {stamp})[/dim]"
                )
            else:
                pprint(f"  [red]✗[/red] job #{idx} submit failed  [dim](wall {stamp})[/dim]")

    state.fleet_prompt_ids = [p for p in prompt_ids if p]
    if not state.fleet_prompt_ids:
        pprint("[bold red]No fleet jobs submitted; skipping queue inspection.[/bold red]")
        return 1

    submit_wall = time.time() - submit_start
    pprint(
        f"\n[bold]Submitted {len(state.fleet_prompt_ids)}/{FLEET_SIZE} jobs in {submit_wall:0.2f}s wall time.[/bold]"
    )

    # Immediate snapshot — server is still processing job 0. Expect one
    # running + others pending (or already-completed if the box is fast).
    _print_subhead("jobs ls — snapshot right after submit", "human")
    _run_invocation(
        Invocation(argv=[*comfy, "jobs", "ls"], label=""),
        pause_seconds=0,
        show_header=False,
    )

    if show_agent:
        _print_subhead("jobs ls (--json) — same data, agent view", "agent")
        _run_invocation(
            Invocation(argv=[*comfy, "--json", "jobs", "ls"], label=""),
            pause_seconds=0,
            show_header=False,
        )

    # Parallel status check — fan out across all prompt_ids at once. This
    # demonstrates the CLI is safe to invoke concurrently from agent code.
    def _check_one(pid: str) -> tuple[str, str]:
        r = subprocess.run(
            [*comfy, "--json", "jobs", "status", pid],
            env={**os.environ, "COMFY_OUTPUT": "pretty"},
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            return pid, json.loads(r.stdout).get("data", {}).get("status", "?")
        except json.JSONDecodeError:
            return pid, "error"

    pprint("\n[bold]Parallel `jobs status` fan-out across the fleet (mid-flight):[/bold]")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(state.fleet_prompt_ids)) as ex:
        mid_results = list(ex.map(_check_one, state.fleet_prompt_ids))
    for pid, status in mid_results:
        color = {"completed": "green", "running": "yellow", "pending": "blue"}.get(status, "red")
        pprint(f"  [{color}]●[/{color}] {pid}  [bold]{status}[/bold]")

    # Watch one of the still-running jobs (if any) live; otherwise just
    # replay events for the middle one.
    pending_or_running = [pid for pid, s in mid_results if s in {"running", "pending"}]
    target = pending_or_running[0] if pending_or_running else state.fleet_prompt_ids[len(state.fleet_prompt_ids) // 2]
    _print_subhead(f"jobs watch {target[:8]}… — live NDJSON stream", "agent")
    _run_invocation(
        Invocation(argv=[*comfy, "--json-stream", "jobs", "watch", target], label="", optional=True),
        pause_seconds=0,
        show_header=False,
    )

    # Poll the queue once more to show drain progress, then a final snapshot.
    time.sleep(0.5)
    _print_subhead("jobs ls — after a brief wait", "human")
    _run_invocation(
        Invocation(argv=[*comfy, "jobs", "ls"], label=""),
        pause_seconds=0,
        show_header=False,
    )
    return 0


# ---------- Demo composition -------------------------------------------------


@dataclass
class _DemoState:
    workflow_path: str
    async_prompt_id: str | None = None
    fleet_workflow_paths: list[str] = field(default_factory=list)
    fleet_prompt_ids: list[str] = field(default_factory=list)


def _capture_prompt_id(state: _DemoState) -> Callable[[str], None]:
    def handler(stdout: str) -> None:
        try:
            doc = json.loads(stdout.strip().splitlines()[-1])
            pid = doc.get("data", {}).get("prompt_id")
            if pid:
                state.async_prompt_id = pid
        except (json.JSONDecodeError, IndexError):
            pass

    return handler


def _summarize_discover(stdout: str) -> None:
    try:
        doc = json.loads(stdout)
        cmds = doc.get("data", {}).get("commands", {}) or {}
        names = sorted(cmds.keys())
        pprint(
            f"\n[italic]({len(names)} top-level commands: "
            f"{', '.join(names[:12])}{'…' if len(names) > 12 else ''})[/italic]"
        )
    except json.JSONDecodeError:
        pass


def _build_steps(state: _DemoState) -> list[Step]:
    comfy = _comfy_argv()
    wf = state.workflow_path

    return [
        Step(
            title="env — interpreter, workspace, server",
            invocations=[
                Invocation(argv=[*comfy, "env"], label="comfy env"),
                Invocation(argv=[*comfy, "--json", "env"], label="comfy --json env"),
            ],
        ),
        Step(
            title="which — resolved workspace",
            invocations=[
                Invocation(argv=[*comfy, "which"], label="comfy which"),
                Invocation(argv=[*comfy, "--json", "which"], label="comfy --json which", optional=True),
            ],
        ),
        Step(
            title="discover — machine-readable surface",
            desc="Agents read this once at startup to learn every command and error code.",
            invocations=[
                Invocation(
                    argv=[*comfy, "--json", "discover"],
                    label="comfy --json discover",
                    capture=True,
                    on_output=_summarize_discover,
                ),
            ],
        ),
        Step(
            title="query — CQL against the live node graph",
            invocations=[
                Invocation(
                    argv=[
                        *comfy,
                        "query",
                        "-q",
                        "from nodes where name in ('EmptyImage','ImageInvert','SaveImage') select name, category",
                    ],
                    label="comfy query",
                ),
                Invocation(
                    argv=[
                        *comfy,
                        "--json",
                        "query",
                        "-q",
                        "from nodes where name in ('EmptyImage','ImageInvert','SaveImage') select name, category",
                    ],
                    label="comfy --json query",
                ),
            ],
        ),
        Step(
            title="run — synchronous, with progress bar",
            invocations=[
                Invocation(
                    argv=[*comfy, "run", "--workflow", wf, "--verbose"],
                    label="comfy run --verbose",
                ),
            ],
        ),
        Step(
            title="run — same workflow, JSON envelope",
            invocations=[
                Invocation(
                    argv=[*comfy, "--json", "run", "--workflow", wf],
                    label="comfy --json run",
                ),
            ],
        ),
        Step(
            title="run --async — single background submission",
            invocations=[
                Invocation(
                    argv=[*comfy, "--json", "run", "--workflow", wf, "--async"],
                    label="comfy --json run --async",
                    capture=True,
                    on_output=_capture_prompt_id(state),
                ),
            ],
        ),
        Step(
            title="jobs ls — recent / running prompts",
            invocations=[
                Invocation(argv=[*comfy, "jobs", "ls"], label="comfy jobs ls"),
                Invocation(argv=[*comfy, "--json", "jobs", "ls"], label="comfy --json jobs ls"),
            ],
        ),
        Step(
            title="jobs status — single prompt detail",
            invocations=[
                Invocation(argv=[*comfy, "jobs", "status", "PLACEHOLDER"], label="comfy jobs status"),
                Invocation(
                    argv=[*comfy, "--json", "jobs", "status", "PLACEHOLDER"],
                    label="comfy --json jobs status",
                ),
            ],
            skip_if=lambda: state.async_prompt_id is None,
        ),
        Step(
            title="jobs watch — NDJSON event stream",
            invocations=[
                Invocation(
                    argv=[*comfy, "--json-stream", "jobs", "watch", "PLACEHOLDER"],
                    label="comfy --json-stream jobs watch",
                    optional=True,
                ),
            ],
            skip_if=lambda: state.async_prompt_id is None,
        ),
        Step(
            title=f"FLEET — {FLEET_SIZE} jobs submitted in parallel, queue drains live",
            desc="Concurrent --async submission via a thread pool, then jobs ls / status / watch on the fleet.",
            custom=_fleet_step,
        ),
        Step(
            title="auth whoami — Comfy Cloud sign-in state",
            desc="Same commands work against Comfy Cloud via --where cloud (after `comfy auth login`).",
            invocations=[
                Invocation(argv=[*comfy, "auth", "whoami"], label="comfy auth whoami", optional=True),
                Invocation(
                    argv=[*comfy, "--json", "auth", "whoami"],
                    label="comfy --json auth whoami",
                    optional=True,
                ),
            ],
        ),
    ]


# ---------- Entry point ------------------------------------------------------


def execute(*, pause_seconds: float, no_cleanup: bool, show_agent: bool) -> int:
    pprint("[bold]comfy run-cli[/bold] — capabilities walkthrough.\n")
    pprint(
        "[dim]Each step runs real `comfy` commands. For most steps you'll see two views "
        "of the same call: the [green]human view[/green] (pretty terminal output) "
        "and the [magenta]agent view[/magenta] (`--json` envelope). The fleet step "
        f"submits {FLEET_SIZE} jobs in parallel to demonstrate background execution.[/dim]\n"
    )

    fd, wf_path = tempfile.mkstemp(prefix="comfy_run_cli_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(DEMO_WORKFLOW, f, indent=2)
    pprint(f"[dim]Wrote single-job workflow → {wf_path}[/dim]")

    state = _DemoState(workflow_path=wf_path)
    steps = _build_steps(state)

    first_error: int | None = None
    for step in steps:
        if state.async_prompt_id is not None:
            for inv in step.invocations:
                inv.argv = [a.replace("PLACEHOLDER", state.async_prompt_id) for a in inv.argv]
        rc = _run_step(step, state, pause_seconds=pause_seconds, show_agent=show_agent)
        if rc != 0 and first_error is None:
            first_error = rc

    _print_banner("Done")
    pprint(
        "[bold]Capabilities shown:[/bold] env · which · discover · query · "
        "run (sync/verbose/json/async) · jobs (ls/status/watch) · "
        f"parallel fleet ×{FLEET_SIZE} · auth whoami."
    )

    paths_to_clean = [wf_path, *state.fleet_workflow_paths]
    if no_cleanup:
        pprint(f"[dim]Kept {len(paths_to_clean)} workflow file(s) in /tmp for inspection.[/dim]")
    else:
        for p in paths_to_clean:
            try:
                os.unlink(p)
            except OSError:
                pass

    return first_error or 0
