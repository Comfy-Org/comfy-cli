import json
import os
import subprocess
import sys
import webbrowser
from typing import Annotated

import questionary
import typer
from rich.console import Console

from comfy_cli import cancellation, constants, env_checker, logging, tracking, ui, utils
from comfy_cli import where as where_module
from comfy_cli._safe_exec import resolve_required_binary
from comfy_cli.auth import command as auth_command
from comfy_cli.caller import stream_is_tty
from comfy_cli.cloud import command as cloud_command
from comfy_cli.command import (
    code_search,
    custom_nodes,
    pr_command,
)
from comfy_cli.command import generate as generate_command
from comfy_cli.command import install as install_inner
from comfy_cli.command import (
    jobs as jobs_command,
)
from comfy_cli.command import (
    nodes as nodes_command,
)
from comfy_cli.command import preview as preview_command
from comfy_cli.command import (
    project as project_command,
)
from comfy_cli.command import run as run_inner
from comfy_cli.command import run_cli as run_cli_inner
from comfy_cli.command import (
    templates as templates_command,
)
from comfy_cli.command import transfer as transfer_inner
from comfy_cli.command import (
    workflow as workflow_command,
)
from comfy_cli.command.custom_nodes.cm_cli_util import normalize_cm_cli_exit_code
from comfy_cli.command.install import validate_optional_version, validate_version
from comfy_cli.command.launch import launch as launch_command
from comfy_cli.command.launch import logs as logs_command
from comfy_cli.command.models import models as models_command
from comfy_cli.command.models import search as models_search_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import GPU_OPTION, CUDAVersion, ROCmVersion
from comfy_cli.cuda_detect import DEFAULT_CUDA_TAG, detect_cuda_driver_version, resolve_cuda_wheel
from comfy_cli.discovery import build_discovery
from comfy_cli.env_checker import EnvChecker, _bracket_host, _unbracket_host
from comfy_cli.help_json import build_help_json
from comfy_cli.host_port import report_usage_error, validate_host
from comfy_cli.output import Renderer, get_renderer, rprint, set_renderer
from comfy_cli.resolve_python import resolve_workspace_python
from comfy_cli.skills import command as skill_command
from comfy_cli.standalone import StandalonePython
from comfy_cli.uv import DependencyCompiler, ensure_pip
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

logging.setup_logging()
app = typer.Typer()
workspace_manager = WorkspaceManager()

console = Console()


def main():
    # Install the SIGINT handler BEFORE Typer parses argv and runs the
    # subcommand. Otherwise a Ctrl-C during the first ~100ms (argv parsing,
    # the lazy imports of auth/cql/cloud, ConfigManager construction) hits
    # Python's default handler and the user sees a bare KeyboardInterrupt
    # traceback instead of the documented `cancelled` envelope.
    cancellation.install_sigint_handler()
    # Run the Typer app. If the command path called renderer.error(...) and
    # returned without raising typer.Exit, we still need to surface the
    # non-zero exit code at the OS level — otherwise the envelope says
    # ok:false but the shell sees 0 and downstream tools think we succeeded.
    try:
        app()
    except KeyboardInterrupt:
        # SIGINT path: emit a cancelled envelope if one hasn't been written yet.
        try:
            from comfy_cli.output import get_renderer

            r = get_renderer()
            if not r._envelope_emitted and r.is_json():
                r.error(code="cancelled", message="Cancelled by user", exit_code=130)
        except Exception:
            pass
        sys.exit(130)
    except (typer.Exit, SystemExit):
        raise
    # No exception → command returned cleanly. If the renderer recorded a
    # non-zero exit code (from a `renderer.error(...)` that forgot to also
    # `raise typer.Exit(...)`), honor it.
    try:
        from comfy_cli.output import get_renderer

        rc = get_renderer().exit_code
    except Exception:  # noqa: BLE001 — never break the success path on renderer issues
        rc = 0
    if rc:
        sys.exit(rc)


class MutuallyExclusiveValidator:
    def __init__(self):
        self.group = []

    def reset_for_testing(self):
        self.group.clear()

    def validate(self, _ctx: typer.Context, param: typer.CallbackParam, value: str):
        # Add cli option to group if it was called with a value
        if value is not None and param.name not in self.group:
            self.group.append(param.name)
        if len(self.group) > 1:
            raise typer.BadParameter(f"option `{param.name}` is mutually exclusive with option `{self.group.pop()}`")
        return value


g_exclusivity = MutuallyExclusiveValidator()
g_gpu_exclusivity = MutuallyExclusiveValidator()


@app.command(help="Display help for commands")
def help(ctx: typer.Context):
    rprint(ctx.find_root().get_help())
    ctx.exit(0)


def _maybe_nudge_setup(ctx: typer.Context, renderer) -> None:
    """First-run only: nudge a brand-new, unconfigured user toward `comfy setup`.

    Heavily gated so it never touches the machine contract or repeats: skips the
    bare/`setup` invocations, anything but interactive pretty output, signed-in
    users, and installs already nudged. Prints one line to stderr, then marks the
    install. Onboarding must never break a command — failures are swallowed.
    """
    sub = ctx.invoked_subcommand
    # Guarded stderr probe: this runs from the main Typer callback, and stderr
    # can be closed independently of stdout (`comfy install 2>&-`, where CPython
    # sets `sys.stderr = None`). A bare `.isatty()` there would kill the command
    # from the onboarding nudge of all places. See `caller.stream_is_tty`.
    if sub in (None, "setup") or not renderer.is_pretty() or not stream_is_tty(getattr(sys, "stderr", None)):
        return
    try:
        from comfy_cli.credentials import get_session
        from comfy_cli.onboarding import NUDGE_TEXT, mark_setup_nudged, should_nudge_setup

        session = get_session(refresh=False)
        signed_in = session is not None and not session.is_expired()
        if should_nudge_setup(signed_in=signed_in):
            print(NUDGE_TEXT, file=sys.stderr)
            mark_setup_nudged()
    except Exception:  # noqa: BLE001 — a nudge must never break the actual command
        pass


@app.callback(
    invoke_without_command=True,
    epilog=(
        "Cloud quickstart (no local GPU required):\n\n"
        "comfy cloud login  →  comfy run --workflow wf.json --where cloud  "
        "(prints a prompt_id)  →  comfy jobs wait <prompt_id> --where cloud  →  "
        "comfy download <prompt_id> --where cloud\n\n"
        "Cloud generation consumes Comfy Cloud credits (needs an active subscription); "
        "discovery commands (jobs status, templates ls, generate list) don't. See the README."
    ),
)
def entry(
    ctx: typer.Context,
    workspace: Annotated[
        str | None,
        typer.Option(
            show_default=False,
            help="Path to ComfyUI workspace",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    recent: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Execute from recent path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    here: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Execute from current path",
            callback=g_exclusivity.validate,
        ),
    ] = None,
    skip_prompt: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Do not prompt user for input, use default options",
        ),
    ] = False,
    enable_telemetry: Annotated[
        bool,
        typer.Option(
            show_default=False,
            hidden=True,
            help="Enable tracking",
        ),
    ] = False,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print version and exit",
    ),
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            show_default=False,
            help="Emit a structured JSON envelope on stdout. Side messages go to stderr.",
        ),
    ] = False,
    json_stream: Annotated[
        bool,
        typer.Option(
            "--json-stream",
            show_default=False,
            help="Emit one NDJSON event per line; final line is the envelope.",
        ),
    ] = False,
    no_json: Annotated[
        bool,
        typer.Option(
            "--no-json",
            show_default=False,
            help="Force pretty output even when auto-detection would pick JSON.",
        ),
    ] = False,
    help_json: Annotated[
        bool,
        typer.Option(
            "--help-json",
            show_default=False,
            help="Print machine-readable help for the entire CLI and exit.",
        ),
    ] = False,
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            show_default=False,
            help="Routing mode for this invocation: 'local' or 'cloud'. "
            "Overrides COMFY_WHERE and the persisted default (`comfy set-default --where`). "
            "Subcommands inherit this — no need to repeat it on each one.",
        ),
    ] = None,
):
    # 1. Resolve output mode and install the process-wide renderer. This must
    #    happen before any rprint() call so all output is routed correctly.
    cli_version = ConfigManager().get_cli_version()
    renderer = Renderer.resolve(
        json_flag=json_output or None,
        json_stream_flag=json_stream or None,
        no_json_flag=no_json,
        command=ctx.invoked_subcommand or "",
        version=cli_version,
    )
    set_renderer(renderer)

    # Global `--where` is sugar over COMFY_WHERE. Setting the env var here so
    # the existing precedence chain in ``where.resolve()`` picks it up for
    # every subcommand without each one having to wire its own flag.
    if where:
        try:
            where_module._parse(where)  # validate now — fail fast on `--where cloudy`
        except ValueError as e:
            get_renderer().error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
            raise typer.Exit(code=1) from e
        os.environ["COMFY_WHERE"] = where.strip().lower()

    # 2. Install SIGINT → cancellation token. Idempotent; safe to call before
    #    any long-running operation in subcommands.
    cancellation.install_sigint_handler()

    # 3. Agentic callers shouldn't get interactive prompts.
    if renderer.caller.agentic:
        skip_prompt = True

    if help_json:
        doc = build_help_json(app)
        # In JSON mode, wrap in the standard envelope so consumers can parse
        # the same shape they get from every other command. In pretty mode
        # (rare for --help-json but possible), emit the raw doc — there's no
        # envelope to compete with on stdout.
        if renderer.is_json():
            renderer.emit(doc, command="help")
        else:
            sys.stdout.write(json.dumps(doc, indent=2, default=str) + "\n")
        ctx.exit(0)

    if version:
        if renderer.is_json():
            renderer.emit({"version": cli_version}, command="version")
        else:
            rprint(cli_version)
        ctx.exit(0)

    workspace_manager.setup_workspace_manager(workspace, here, recent, skip_prompt)

    # `comfy setup` owns the telemetry consent decision as a branded wizard step
    # (with full disclosure). Suppress the bare global prompt for that one command
    # so the user is asked exactly once, in the right place — not pre-empted here.
    if ctx.invoked_subcommand != "setup":
        tracking.prompt_tracking_consent(skip_prompt, default_value=enable_telemetry)

    _maybe_nudge_setup(ctx, renderer)

    if ctx.invoked_subcommand is None:
        # The welcome screen is human-facing: agents read `discover` / `--help-json`,
        # and a JSON welcome envelope helps no one. Emit JSON only when machine
        # output was actually *requested* — explicit `--json`/`--json-stream`,
        # `COMFY_OUTPUT` env, or a real detected agent — NOT merely because stdout
        # isn't a TTY. A human who pipes `comfy` (caller kind "pipe") still wants
        # the banner, so the non-TTY auto-JSON rule is overridden here only.
        _env_json = (os.environ.get("COMFY_OUTPUT") or "").strip().lower() in {"json", "ndjson"}
        _real_agent = renderer.caller.agentic and renderer.caller.kind != "pipe"
        _machine_welcome = bool(json_output) or bool(json_stream) or _env_json or _real_agent
        if _machine_welcome:
            renderer.emit(
                {
                    "welcome": "Comfy CLI",
                    "homepage": "https://github.com/Comfy-Org/comfy-cli",
                    "hint": "run `comfy setup` to get started, `comfy --help-json` for the full surface, or `comfy --help` for human help.",
                },
                command="welcome",
            )
        else:
            from rich.console import Console

            from comfy_cli.credentials import get_session as _get_session
            from comfy_cli.output.branding import intro_banner
            from comfy_cli.update import latest_upgrade_version

            _session = _get_session(refresh=False)
            _signed_in = _session is not None and not _session.is_expired()
            _base_url = _session.base_url if _session else ""
            if not _base_url:
                from comfy_cli.cloud import get_base_url as _get_base_url

                _base_url = _get_base_url()
            _update_hint = latest_upgrade_version(cli_version, ConfigManager().get_config_path())
            # Render to stdout regardless of the resolved mode: a non-TTY human
            # pipe resolves to JSON, but we're deliberately showing the human banner.
            _console = renderer.console() if renderer.is_pretty() else Console(file=sys.stdout)
            _console.print(
                intro_banner(
                    version=ConfigManager().get_cli_version(),
                    signed_in=_signed_in,
                    base_url=_base_url,
                    update_hint=_update_hint,
                )
            )
        ctx.exit()


def validate_commit_and_version(commit: str | None, ctx: typer.Context) -> str | None:
    """
    Validate that the commit is not specified unless the version is 'nightly'.
    """
    version = ctx.params.get("version")
    if commit and version != "nightly":
        raise typer.BadParameter("You can only specify the commit if the version is 'nightly'.")
    return commit


def _resolve_cuda(
    gpu: GPU_OPTION | None,
    cuda_version: CUDAVersion | None,
) -> tuple[CUDAVersion | None, str | None]:
    """Resolve the CUDA wheel tag for an NVIDIA install.

    Returns (cuda_version_enum_or_None, cuda_tag_string_or_None).
    When the user passed an explicit --cuda-version, that is used as-is.
    Otherwise auto-detection is attempted.
    """
    if gpu != GPU_OPTION.NVIDIA:
        return cuda_version, None

    if cuda_version is not None:
        tag = f"cu{cuda_version.value.replace('.', '')}"
        rprint(f"[bold]Using explicit CUDA version:[/bold] {cuda_version.value} ({tag})")
        return cuda_version, tag

    drv = detect_cuda_driver_version()
    if drv is not None:
        tag = resolve_cuda_wheel(drv)
        if tag is not None:
            rprint(f"[bold green]Detected CUDA driver version:[/bold green] {drv[0]}.{drv[1]} → using {tag}")
            return None, tag
        rprint(
            f"[bold yellow]Warning:[/bold yellow] CUDA driver {drv[0]}.{drv[1]} is too old for any known PyTorch wheel. "
            f"Falling back to {DEFAULT_CUDA_TAG}. Use `--cuda-version` to override."
        )
        return None, DEFAULT_CUDA_TAG

    rprint(
        f"[bold yellow]Warning:[/bold yellow] Could not detect CUDA driver version. "
        f"Falling back to {DEFAULT_CUDA_TAG}. Use `--cuda-version` to override."
    )
    return None, DEFAULT_CUDA_TAG


@app.command(help="Download and install ComfyUI and ComfyUI-Manager")
@tracking.track_command()
def install(
    url: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="url or local path pointing to the ComfyUI core git repo to be installed. A specific branch can optionally be specified using a setuptools-like syntax, eg https://foo.git@bar",
        ),
    ] = constants.COMFY_GITHUB_URL,
    version: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="Specify version of ComfyUI to install. Default is nightl, which is the latest commit on master branch. Other options include: latest, which is the latest stable release. Or a specific version number, eg. 0.2.0",
            callback=validate_version,
        ),
    ] = "nightly",
    restore: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Restore dependencies for installed ComfyUI if not installed",
        ),
    ] = False,
    skip_manager: Annotated[
        bool,
        typer.Option(show_default=False, help="Skip installing the manager component"),
    ] = False,
    skip_torch_or_directml: Annotated[
        bool,
        typer.Option(show_default=False, help="Skip installing PyTorch Or DirectML"),
    ] = False,
    skip_requirement: Annotated[
        bool, typer.Option(show_default=False, help="Skip installing requirements.txt")
    ] = False,
    nvidia: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for Nvidia gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cuda_version: Annotated[CUDAVersion | None, typer.Option(show_default=False)] = None,
    rocm_version: Annotated[ROCmVersion, typer.Option(show_default=True)] = ROCmVersion.v7_2,
    amd: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for AMD gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    m_series: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for Mac M-Series gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    intel_arc: Annotated[
        bool | None,
        typer.Option(
            hidden=True,
            show_default=False,
            help="Install for Intel Arc gpu",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    cpu: Annotated[
        bool | None,
        typer.Option(
            show_default=False,
            help="Install for CPU",
            callback=g_gpu_exclusivity.validate,
        ),
    ] = None,
    commit: Annotated[
        str | None, typer.Option(help="Specify commit hash for ComfyUI", callback=validate_commit_and_version)
    ] = None,
    fast_deps: Annotated[
        bool,
        typer.Option(
            "--fast-deps",
            show_default=False,
            help="Use uv instead of pip for dependency resolution (comfy-cli built-in resolver)",
        ),
    ] = False,
    pr: Annotated[
        str | None,
        typer.Option(
            show_default=False,
            help="Install from a specific PR. Supports formats: username:branch, #123, or PR URL",
        ),
    ] = None,
):
    checker = EnvChecker()

    comfy_path, _ = workspace_manager.get_workspace_path()

    is_comfy_installed_at_path, resolved_path = check_comfy_repo(comfy_path)
    if is_comfy_installed_at_path and not restore:
        rprint(f"[bold red]ComfyUI is already installed at the specified path:[/bold red] {comfy_path}\n")
        rprint(
            "[bold yellow]If you want to restore dependencies, add the '--restore' option.[/bold yellow]",
        )
        raise typer.Exit(code=1)

    if resolved_path is not None:
        comfy_path = resolved_path

    if checker.python_version.major < 3 or checker.python_version.minor < 9:
        rprint("[bold red]Python version 3.9 or higher is required to run ComfyUI.[/bold red]")
        rprint(f"You are currently using Python version {env_checker.format_python_version(checker.python_version)}.")
    platform = utils.get_os()

    if pr and (version not in {None, "nightly"} or commit):
        rprint("--pr cannot be used with --version or --commit")
        raise typer.Exit(code=1)

    if cpu:
        rprint("[bold yellow]Installing for CPU[/bold yellow]")
        install_inner.execute(
            url,
            comfy_path,
            restore,
            skip_manager,
            commit=commit,
            version=version,
            gpu=None,
            cuda_version=cuda_version,
            cuda_tag=None,
            rocm_version=rocm_version,
            plat=platform,
            skip_torch_or_directml=skip_torch_or_directml,
            skip_requirement=skip_requirement,
            fast_deps=fast_deps,
            pr=pr,
        )
        rprint(f"ComfyUI is installed at: {comfy_path}")
        return None

    if nvidia and platform == constants.OS.MACOS:
        rprint(
            "[bold red]--nvidia was passed but this is macOS, which has no NVIDIA GPU. "
            "Re-run with --m-series (Apple silicon) or --cpu, or omit the GPU flag to select interactively.[/bold red]"
        )
        raise typer.Exit(code=1)

    if m_series and platform != constants.OS.MACOS:
        # Warning-only here used to fall through to `elif m_series:` below and
        # set GPU_OPTION.MAC_M_SERIES, which installs the nightly CPU torch
        # wheels — silently giving a Linux/Windows user a CPU-only nightly and
        # no GPU support. Exit like the symmetric --nvidia-on-macOS check.
        rprint(
            f"[bold red]--m-series is a macOS (Apple Silicon) option; detected platform is {platform}.[/bold red]\n"
            "[bold red]Use --nvidia, --amd or --intel-arc for a GPU install, or --cpu for CPU-only.[/bold red]"
        )
        raise typer.Exit(code=1)

    gpu = None

    if nvidia:
        gpu = GPU_OPTION.NVIDIA
    elif amd:
        gpu = GPU_OPTION.AMD
    elif m_series:
        gpu = GPU_OPTION.MAC_M_SERIES
    elif intel_arc:
        gpu = GPU_OPTION.INTEL_ARC
    else:
        if platform == constants.OS.MACOS:
            gpu = ui.prompt_select_enum(
                "What type of Mac do you have?",
                [GPU_OPTION.MAC_M_SERIES, GPU_OPTION.MAC_INTEL],
            )
        else:
            gpu = ui.prompt_select_enum(
                "What GPU do you have?",
                [GPU_OPTION.NVIDIA, GPU_OPTION.AMD, GPU_OPTION.INTEL_ARC],
            )

    if gpu is None and not cpu:
        rprint(
            "[bold red]No GPU option selected or `--cpu` enabled, use --\\[gpu option] flag (e.g. --nvidia) to pick GPU. use `--cpu` to install for CPU. Exiting...[/bold red]"
        )
        raise typer.Exit(code=1)

    cuda_version, cuda_tag = _resolve_cuda(gpu, cuda_version) if not skip_torch_or_directml else (cuda_version, None)

    install_inner.execute(
        url,
        comfy_path,
        restore,
        skip_manager,
        commit=commit,
        gpu=gpu,
        version=version,
        cuda_version=cuda_version,
        cuda_tag=cuda_tag,
        rocm_version=rocm_version,
        plat=platform,
        skip_torch_or_directml=skip_torch_or_directml,
        skip_requirement=skip_requirement,
        fast_deps=fast_deps,
        pr=pr,
    )

    rprint(f"ComfyUI is installed at: {comfy_path}")


def _switch_comfy_version(comfy_path: str, version: str, *, stash: bool) -> None:
    """`comfy update comfy --version X` — move the workspace, then reinstall its deps.

    Torch is deliberately left alone: a version switch is ComfyUI-version-specific
    while the torch build is machine-specific, so re-resolving it would need the
    GPU input this headless path exists to avoid.
    """
    renderer = get_renderer()

    try:
        result = install_inner.switch_comfyui_version(comfy_path, version, stash=stash)
    except install_inner.VersionSwitchError as e:
        renderer.error(code=e.code, message=e.message, hint=e.hint)
        raise typer.Exit(code=1)

    python = resolve_workspace_python(comfy_path)
    # A uv-managed venv may have no pip — bootstrap it first so the install
    # below doesn't crash with `No module named pip` (no-op if pip exists).
    ensure_pip(python, cwd=comfy_path)
    deps = subprocess.run(
        [python, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=comfy_path,
        check=False,
    )
    if deps.returncode != 0:
        renderer.error(
            code="version_switch_deps_failed",
            message=(
                f"ComfyUI is now on {result['current']}, but installing its requirements.txt failed — "
                "dependencies may be incomplete."
            ),
            hint="re-run the same command once the cause is fixed; it is idempotent and safe to repeat",
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        rprint(f"Switched ComfyUI: [bold]{result['previous']}[/bold] -> [bold]{result['current']}[/bold]")
        if result["stashed"]:
            rprint(
                f"[yellow]Uncommitted changes were stashed as {result['stash_ref']} and left there — "
                "restore them with `git stash pop`.[/yellow]"
            )
        if version != "nightly":
            rprint(
                "[dim]This is a tag checkout, so the workspace is on a detached HEAD. "
                "Roll forward with `comfy update comfy --version nightly` (or `--version latest`).[/dim]"
            )
    renderer.emit(result, command="update")


def _refresh_node_id_cache() -> None:
    """Re-export the custom-node id cache that backs shell tab-completion.

    Best-effort: a stale cache only degrades completion, so a failure here is a
    warning, never the command's outcome.
    """
    try:
        custom_nodes.command.update_node_id_cache()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        rprint(f"[yellow]Failed to update node id cache: {e}[/yellow]")


@app.command(help="Update ComfyUI Environment [all|comfy|cli]")
@tracking.track_command()
def update(
    target: str = typer.Argument(
        "comfy",
        help="\\[all|comfy|cli]",
        autocompletion=utils.create_choice_completer(["all", "comfy", "cli"]),
    ),
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            show_default=False,
            callback=validate_optional_version,
            help=(
                "Switch the existing ComfyUI workspace to a specific version instead of pulling the current "
                "branch. Accepts 'nightly' (the default branch), 'latest' (newest stable release), or a version "
                "number such as 0.3.0. Only valid for target 'comfy'. Uncommitted changes are stashed first and "
                "never popped automatically; a version number checks out a tag, leaving a detached HEAD — roll "
                "forward again with --version nightly. Dependencies are reinstalled from requirements.txt; torch "
                "is left untouched."
            ),
        ),
    ] = None,
    no_stash: Annotated[
        bool,
        typer.Option(
            "--no-stash",
            show_default=False,
            help=(
                "With --version, refuse to switch when the ComfyUI workspace has uncommitted changes "
                "instead of stashing them."
            ),
        ),
    ] = False,
    exit_on_fail: Annotated[
        bool,
        typer.Option(
            "--exit-on-fail",
            help=(
                "Exit on failure. Only affects target 'all': without it a failing custom-node update is "
                "printed but still exits 0. Targets 'comfy' and 'cli' already exit non-zero on failure."
            ),
        ),
    ] = False,
):
    if target not in ["all", "comfy", "cli"]:
        typer.echo(
            f"Invalid target: {target}. Allowed targets are 'all', 'comfy', 'cli'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if version is not None and target != "comfy":
        get_renderer().error(
            code="update_version_target_invalid",
            message=f"--version is only supported for target 'comfy', not '{target}'.",
            hint="run `comfy update comfy --version <version>`",
        )
        raise typer.Exit(code=1)

    if "cli" == target:
        from comfy_cli.update import upgrade_cli

        rprint("Updating comfy-cli...")
        upgrade_cli()
        return

    comfy_path = workspace_manager.workspace_path

    if "all" == target:
        # Without raise_on_error, execute_cm_cli swallows a cm-cli exit 1 and returns None,
        # so `comfy update all` would report success for a failed pack update. Mirrors the
        # --exit-on-fail plumbing in `comfy node install`.
        #
        # Unlike `node install`, the flag is deliberately NOT forwarded to cm-cli: its
        # `update` subcommand takes only (nodes, channel, mode, user_directory) — no
        # --exit-on-fail — so passing it would be a Typer usage error and make the flag
        # fail every invocation. Only `cm-cli install` has one.
        try:
            custom_nodes.command.execute_cm_cli(["update", "all"], raise_on_error=exit_on_fail)
        except subprocess.CalledProcessError as e:
            if not exit_on_fail:
                # execute_cm_cli re-raises unexpected exit codes even with raise_on_error off;
                # keep surfacing those instead of swallowing them here.
                raise
            # `cm-cli update all` is non-atomic — packs that did update stayed updated — so
            # refresh the id cache exactly as the no-flag path below does before bailing out.
            _refresh_node_id_cache()
            code = normalize_cm_cli_exit_code(e.returncode)
            get_renderer().error(
                code="update_custom_nodes_failed",
                message=f"`cm-cli update all` failed with exit code {e.returncode}.",
                hint="re-run `comfy update all` to retry; it is safe to repeat",
                details={"cm_cli_returncode": e.returncode},
                exit_code=code,
            )
            raise typer.Exit(code=code) from e
    else:
        rprint(f"Updating ComfyUI in {comfy_path}...")
        if comfy_path is None:
            rprint("ComfyUI path is not found.")
            raise typer.Exit(code=1)
        if version is not None:
            _switch_comfy_version(comfy_path, version, stash=not no_stash)
        else:
            # Resolved before the chdir so a ``git`` planted in ``comfy_path``
            # can neither be picked up nor shadow the real one.
            git_bin = resolve_required_binary("git")
            os.chdir(comfy_path)
            subprocess.run([git_bin, "pull"], check=True)
            python = resolve_workspace_python(comfy_path)
            # A uv-managed venv may have no pip — bootstrap it first so the install
            # below doesn't crash with `No module named pip` (no-op if pip exists).
            ensure_pip(python, cwd=comfy_path)
            subprocess.run(
                [python, "-m", "pip", "install", "-r", "requirements.txt"],
                check=True,
            )

    _refresh_node_id_cache()


@app.command(help="Report installed-vs-latest versions for ComfyUI core and custom node packs (read-only).")
@tracking.track_command()
def outdated(
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Bypass the 1h latest-version cache and re-query the network."),
    ] = False,
):
    from comfy_cli.command import outdated as outdated_command

    outdated_command.execute(get_renderer(), workspace_manager.workspace_path, refresh=refresh)


@app.command(help="Run an API workflow. Submits and returns immediately by default; pass --wait to block.")
def run(
    workflow: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to the workflow JSON file. Both ComfyUI API format and "
                "exported UI format are accepted; UI workflows are converted "
                "to API format client-side. Optional: omit it and pass --prompt "
                "to run the bundled default text2img workflow."
            )
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            show_default=False,
            help=(
                "Positive text prompt for the bundled default text2img workflow "
                "(used when --workflow is omitted). Cannot be combined with --workflow. "
                "The bundled graph prefers an SD1.5 checkpoint "
                "(v1-5-pruned-emaonly-fp16.safetensors); if the target doesn't have it, "
                "an installed checkpoint is substituted and reported. Pin your own with "
                "--set checkpoint=<name>."
            ),
        ),
    ] = None,
    set_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            show_default=False,
            help=(
                "Override a field in the bundled default workflow, repeatable. "
                "Form: alias=VALUE (checkpoint/negative/seed/steps/cfg/width/height/…) "
                "or NODE_ID.field=VALUE (e.g. 4.ckpt_name=model.safetensors). "
                "Cannot be combined with --workflow."
            ),
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait",
            show_default=False,
            help="Block until the workflow completes (old default). Without this, the command submits and exits.",
        ),
    ] = False,
    notify: Annotated[
        bool | None,
        typer.Option(
            "--notify/--no-notify",
            show_default=False,
            help="Fire a desktop notification when a background job completes. Default: on for humans, off for agents.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(help="Enables verbose output of the execution process."),
    ] = False,
    host: Annotated[
        str | None,
        typer.Option(help="The IP/hostname where the ComfyUI instance is running, e.g. 127.0.0.1 or localhost."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(help="The port where the ComfyUI instance is running, e.g. 8188."),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            help=(
                "Per-event timeout in seconds: bails out if the server is silent "
                "for this long. Also caps HTTP connect, /prompt POST, and websocket "
                "handshake. NOT a wall-clock execution deadline — a workflow that "
                "streams progress events faster than the timeout can run "
                "indefinitely."
            ),
        ),
    ] = 120,
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            show_default=False,
            help="Routing target: 'local' or 'cloud'.",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="COMFY_API_KEY",
            help=(
                "Comfy API key for API Nodes (Partner Nodes). "
                "Embedded in the POST /prompt request body as extra_data.api_key_comfy_org. "
                "For scripting, prefer the COMFY_API_KEY environment variable so the secret "
                "stays out of shell history."
            ),
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Stream NDJSON events to stdout instead of human-readable output: "
                'one `{"schema": "event/1", "type": ...}` object per line, '
                'with a final `type: "envelope"` line carrying ok/error. Same '
                "dialect as the global --json-stream flag; see docs/json-output.md "
                "for the event reference and stability contract. In this mode, "
                "--verbose has no effect and Rich progress is suppressed. "
                "Workflow input accepts both API and UI format JSON (UI input "
                "triggers a `converted` event before `queued`). The converted "
                "workflow graph is always emitted as a `prompt_preview` event "
                "before `queued`, so agents have a full audit trail of what "
                "the CLI submitted."
            ),
        ),
    ] = False,
    print_prompt: Annotated[
        bool,
        typer.Option(
            "--print-prompt",
            help=(
                "Print the API-format workflow graph that WOULD be sent to /prompt and exit. "
                "Does not POST and does not execute. For UI-format input the workflow is "
                "converted first (requires a reachable ComfyUI for /object_info); API input "
                "is printed as-is with no server hit. In --json mode emits a `prompt_preview` "
                "event; otherwise pretty-prints to stdout."
            ),
        ),
    ] = False,
    allow_spend: Annotated[
        bool,
        typer.Option(
            "--allow-spend",
            help=(
                "Consent to running partner-API (paid) nodes that spend Comfy credits. "
                "Required for workflows embedding partner nodes when not confirming interactively."
            ),
        ),
    ] = False,
):
    # Snapshot kwargs before the body mutates api_key/host/port — analytics should record what user actually supplied.
    _track_props = tracking.filter_command_kwargs(dict(locals()))
    tracking.track_event("execution_start", _track_props, mixpanel_name="run")

    # Resolved outside the try so `renderer` is always bound by the time the
    # `finally` below unstamps `where`, however early the body dies.
    renderer = get_renderer()
    # `Renderer` is a process-wide singleton, so a routed target stamped by an
    # earlier in-process invocation would otherwise still be sitting here and
    # would mislabel any envelope emitted before *this* run routes (e.g.
    # `where_invalid`). Start every invocation unrouted.
    renderer.where = None

    try:
        if api_key:
            api_key = api_key.strip() or None

        config = ConfigManager()

        # Command-local --json means "stream the run": upgrade the renderer
        # (resolved once in the entry callback) into NDJSON mode so every
        # renderer.event(...) line plus the final envelope reaches stdout.
        # One dialect — same shape as the global --json-stream flag.
        if json_output:
            renderer.force_stream()

        try:
            decision = where_module.resolve(flag=where, config_value=config.get(where_module.CONFIG_KEY_WHERE_DEFAULT))
        except ValueError as e:
            renderer.error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
            raise typer.Exit(code=1)

        # The routing target is now known, so every downstream error envelope
        # can carry it. Explicit ``emit(..., where=...)`` calls still win; this
        # only fills the fallback (``where or self.where``) that error() and
        # emit() resolve against.
        renderer.where = decision.target.value

        # Default for --notify: on when a human is at the terminal, off for
        # agents (they shouldn't get surprise side-channel processes they didn't
        # ask for). The user can override either way with --notify/--no-notify.
        effective_notify = notify if notify is not None else (renderer.is_pretty() and not wait)

        # --prompt/--set build an in-memory API-format graph from the bundled
        # default text2img workflow (no --workflow file). The aliases resolve
        # against OUR pinned node ids, so mixing them with a user --workflow —
        # whose node ids are arbitrary — is rejected rather than silently
        # misapplied. `preloaded` is handed straight to run's execute path.
        preloaded: tuple[dict, str, bool, bool] | None = None
        if prompt is not None or set_overrides:
            if workflow is not None:
                renderer.error(
                    code="prompt_rejected",
                    message="--prompt/--set apply to the bundled default workflow and cannot be combined with --workflow",
                    hint="drop --workflow to use the bundled default, or edit the workflow file directly",
                )
                raise typer.Exit(code=1)
            from comfy_cli.cql.default_workflow import (
                PromptInjectionError,
                build_default_workflow,
                overrides_set_checkpoint,
            )

            try:
                injected = build_default_workflow(prompt=prompt, overrides=set_overrides)
            except PromptInjectionError as e:
                renderer.error(code=e.code, message=str(e), hint=e.hint)
                raise typer.Exit(code=1) from e
            # If the user pinned the checkpoint (--set checkpoint=… / 4.ckpt_name=…),
            # honor it verbatim: runtime resolution is skipped downstream.
            checkpoint_user_set = overrides_set_checkpoint(set_overrides, injected)
            preloaded = (injected, "default_text2img", False, checkpoint_user_set)
        elif workflow is None:
            renderer.error(
                code="prompt_rejected",
                message="run requires --workflow, or --prompt (with optional --set) to use the bundled default workflow",
                hint="e.g. comfy run --prompt 'a red fox in snow' --wait",
            )
            raise typer.Exit(code=1)

        if decision.target is where_module.WhereTarget.CLOUD:
            where_module.cloud_preflight_or_exit()
            # Cloud path uses HTTPS + Bearer auth; host/port aren't applicable.
            run_inner.execute_cloud(
                workflow,
                wait=wait,
                verbose=verbose,
                timeout=timeout,
                notify=effective_notify,
                print_prompt=print_prompt,
                preloaded=preloaded,
                allow_spend=allow_spend,
            )
            return

        from comfy_cli.host_port import parse_host_port_arg, report_usage_error, resolve_host_port

        # ``report_usage_error``: a bad ``--host``/``--port`` is a
        # ``typer.BadParameter``, which click turns into a stderr usage panel +
        # exit 2 — leaving stdout empty in JSON/NDJSON mode while every other
        # failure here ends with an envelope. Emit the terminating envelope
        # first; the exception still propagates, so exit 2 is unchanged.
        with report_usage_error(renderer):
            if host:
                host, parsed_port = parse_host_port_arg(host)
                # ``port is None``, not ``not port``: a typed ``--port`` always wins
                # over one embedded in ``--host h:p``, including ``--port 0``, which
                # ``resolve_host_port`` then rejects as out of range instead of
                # silently running against the embedded port.
                if port is None and parsed_port is not None:
                    port = parsed_port

            host, port = resolve_host_port(host, port)

        run_inner.execute(
            workflow,
            host,
            port,
            wait=wait,
            verbose=verbose,
            timeout=timeout,
            notify=effective_notify,
            api_key=api_key,
            print_prompt=print_prompt,
            preloaded=preloaded,
            allow_spend=allow_spend,
        )
    except typer.Exit as e:
        if (e.exit_code or 0) == 0:
            tracking.track_event("execution_success", _track_props)
        else:
            tracking.track_event(
                "execution_error",
                {**_track_props, "error_type": type(e).__name__, "exit_code": e.exit_code},
            )
        raise
    except Exception as e:
        tracking.track_event(
            "execution_error",
            {**_track_props, "error_type": type(e).__name__},
        )
        raise
    else:
        tracking.track_event("execution_success", _track_props)
    finally:
        # Every envelope this invocation emits has already been written by now
        # (renderer.error/emit flush inline), so unstamp the singleton rather
        # than leaving `run`'s routed target visible to whatever emits next in
        # this process — an atexit/tracking path, or a second invocation.
        renderer.where = None


@app.command(
    help=(
        "Validate a workflow without submitting (UI exports are converted to API format first). "
        "Checks class_types, input shapes, enum values, edge wiring, and the dotted sub-inputs a "
        "dynamic combo's selected option requires."
    )
)
@tracking.track_command()
def validate(
    workflow: Annotated[
        str,
        typer.Option(help="Path to the API-format workflow JSON file."),
    ],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target for object_info: 'local' or 'cloud'."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            show_default=False, help="ComfyUI host (defaults to COMFY_LOCAL_URL, the background server, or 127.0.0.1)."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            show_default=False, help="ComfyUI port (defaults to COMFY_LOCAL_URL, the background server, or 8188)."
        ),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a saved object_info JSON (offline mode)."),
    ] = None,
):
    from pathlib import Path

    from comfy_cli.command.run import is_ui_workflow
    from comfy_cli.command.run.preflight import _detect_partner_nodes
    from comfy_cli.cql.engine import Graph, LoadError
    from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api

    renderer = get_renderer()

    # Load workflow
    wf_path = Path(workflow).expanduser()
    if not wf_path.is_file():
        renderer.error(code="workflow_not_found", message=f"Workflow file not found: {workflow}", hint="check the path")
        raise typer.Exit(code=1)
    try:
        wf_data = json.loads(wf_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        renderer.error(code="workflow_invalid_json", message=f"Invalid JSON: {e}", hint="re-export from ComfyUI")
        raise typer.Exit(code=1) from e
    if not isinstance(wf_data, dict):
        renderer.error(
            code="workflow_not_api_format", message="Workflow must be a JSON object", hint="use File > Export (API)"
        )
        raise typer.Exit(code=1)

    # Load graph. Route through the shared resolver rather than trusting the raw
    # `--where` string: it normalizes case/whitespace (`--where LOCAL` is the
    # same target as `local`) and emits a `where_invalid` envelope instead of
    # letting an unknown value escape as a bare ValueError traceback. Everything
    # below then branches on the resolved enum, so no routing decision is ever
    # made by string-comparing user input.
    decision = where_module.resolve_default_or_exit(flag=where)
    mode = decision.target.value

    # Resolve the local object_info server the same way `comfy run` does —
    # flag > COMFY_LOCAL_URL > config.background > 127.0.0.1:8188. Without the
    # `config.background` step validate would consult whatever answers on the
    # default port while `run` submits to the background server comfy-cli
    # launched on another one, making the verdict meaningless for the server
    # that will actually execute the workflow (BE-6299). `resolve_target` does
    # not consult `config.background` on purpose (other callers, e.g. transfer
    # and system, must not), so — as its docstring says — the callers that do
    # honor it resolve upstream, here.
    is_local_fetch = input_path is None and decision.target is where_module.WhereTarget.LOCAL
    if is_local_fetch:
        from comfy_cli.host_port import parse_host_port_arg, report_usage_error, resolve_host_port

        # `host is not None` (not `if host:`): `--host ""` must reach the parser
        # and be rejected, not be read as "no --host given". Likewise the port
        # merge tests `is None`, so an explicit `--port 0` isn't silently
        # overridden by a port embedded in the combined `--host h:p` form —
        # `resolve_host_port` rejects it as out of range instead.
        # `report_usage_error` gives JSON/NDJSON consumers a terminating
        # envelope for that rejection instead of an empty stdout (exit stays 2).
        with report_usage_error(renderer):
            if host is not None:
                host, parsed_port = parse_host_port_arg(host)
                if port is None and parsed_port is not None:
                    port = parsed_port
            host, port = resolve_host_port(host, port)

    try:
        graph = Graph.load(mode=mode, input_path=input_path, host=host, port=port)
    except LoadError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <object_info.json>, or start the server"),
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    # Detect a UI-export (frontend/canvas) workflow and lower it to API format
    # before validating — exactly as `comfy run` does. Without this the wrapper
    # keys (`nodes`, `links`, `groups`, `config`, …) each emit a `non_node_key`
    # warning, zero nodes are checked, and the result is a vacuous `valid:true`.
    # The converter reuses the object_info the graph was already built from
    # (`graph.object_info`), so offline `--input` works and no second fetch happens.
    converted_from_ui = False
    if is_ui_workflow(wf_data):
        if renderer.is_pretty():
            rprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        try:
            converted = convert_ui_to_api(wf_data, graph.object_info)
        except WorkflowConversionError as e:
            renderer.error(
                code="workflow_not_api_format",
                message=f"Workflow is a UI export that could not be converted to API format: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:  # noqa: BLE001 — never leak a raw traceback to the agent flow
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
                details={"exception_type": type(e).__name__},
            )
            raise typer.Exit(code=1) from e
        if not converted:
            renderer.error(
                code="workflow_not_api_format",
                message="Workflow is a UI export that converted to no executable nodes",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1)
        wf_data = converted
        converted_from_ui = True

    result = graph.validate_workflow(wf_data)

    # Preview credit spend: partner-API (paid) nodes spend Comfy credits when the
    # workflow is run. This is the same detection `comfy run` uses (authoritative
    # `api_node: true`, `partner/...` category fallback), surfaced here read-only
    # so agents can answer "will this spend credits?" without running. `wf_data`
    # is API format at this point (any UI export already converted above), which
    # is the format the detector reads. Purely informational — no exit-code gate.
    partner_nodes = _detect_partner_nodes(wf_data, graph.object_info)

    payload = {
        "workflow": str(wf_path),
        "valid": result["valid"],
        "error_count": len(result["errors"]),
        "warning_count": len(result["warnings"]),
        "errors": result["errors"],
        "warnings": result["warnings"],
        "partner_nodes": partner_nodes,
        "spends_credits": bool(partner_nodes),
        # Name the server (or file) the verdict was computed against, so an
        # agent comparing `validate` with `run` can see whether they consulted
        # the same object_info. Populated from the values resolved above, so a
        # local run reports the concrete host/port actually queried. `host` is
        # reported unbracketed, matching `Target.host` — brackets belong to the
        # URL composed for display, not to the address itself.
        "object_info_source": (
            {"mode": "file", "path": str(input_path)}
            if input_path is not None
            else {"mode": mode, "host": _unbracket_host(host), "port": port}
            if is_local_fetch
            else {"mode": mode}
        ),
    }
    if converted_from_ui:
        # Signal that validation ran against the converted graph, not the file's
        # literal bytes, and report how many nodes the conversion produced.
        payload["converted_from_ui"] = True
        payload["converted_node_count"] = len(wf_data)

    if renderer.is_pretty():
        # Name the object_info source in one dim line. A file path (and, in
        # principle, a hostname) can contain Rich-markup metacharacters, so
        # escape it — same reason the partner-node line below does.
        from rich.markup import escape as _escape

        # Branch on the same flags that built the payload, not on its "mode"
        # string: "file" is an offline sentinel that shares a key with the
        # routing targets, so a mode-string branch couples display to a value
        # the routing layer also owns.
        source = payload["object_info_source"]
        if input_path is not None:
            where_oi = _escape(source["path"])
        elif is_local_fetch:
            where_oi = _escape(f"http://{_bracket_host(source['host'])}:{source['port']}")
        else:
            where_oi = _escape(source["mode"])
        rprint(f"[dim]object_info from {where_oi}[/dim]")
        if result["valid"]:
            rprint(f"[bold green]✓[/bold green] workflow is valid ({len(wf_data)} nodes)")
            for w in result["warnings"]:
                rprint(f"  [yellow]⚠[/yellow] {w.get('message', '')}")
        else:
            rprint(f"[bold red]✗[/bold red] {len(result['errors'])} error(s)")
            for e in result["errors"]:
                msg = e.get("message", "")
                suggestions = e.get("suggestions", [])
                if suggestions:
                    msg += f" (did you mean: {', '.join(suggestions[:3])}?)"
                rprint(f"  [red]•[/red] node {e.get('node_id') or '?'}: {msg}")
            for w in result["warnings"]:
                rprint(f"  [yellow]⚠[/yellow] {w.get('message', '')}")
        if partner_nodes:
            # class_type strings come from the workflow / object_info and can
            # contain Rich-markup metacharacters (e.g. `[/yellow]`); escape each
            # so an odd node name can't raise MarkupError and crash the command.
            from rich.markup import escape as _escape

            rprint(
                f"[yellow]⚠ uses partner-API (paid) nodes that spend Comfy credits: {', '.join(_escape(n) for n in partner_nodes)}[/yellow]"
            )
    renderer.emit(payload, command="validate", ok=result["valid"])

    if not result["valid"]:
        raise typer.Exit(code=1)


# How a `cloud` routing decision was reached, in words, for the --host/--port
# rejection message. Keys are `where.WhereResolution.source` values.
_WHERE_SOURCE_PHRASES = {
    "flag": "targeting cloud via --where cloud",
    "env": "targeting cloud via the COMFY_WHERE environment variable",
    "project": "targeting cloud via this project's configured default",
    "config": "targeting cloud via your saved `where_default` setting",
    "auto": "targeting cloud because you're signed in (no explicit --where)",
}


@app.command(help="Upload files to the ComfyUI server's input directory.")
@tracking.track_command()
def upload(
    files: Annotated[list[str], typer.Argument(help="Local file paths to upload.")],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target: 'local' or 'cloud'."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Overwrite existing files on the server."),
    ] = True,
    host: Annotated[
        str | None,
        typer.Option(help="Server host (defaults to COMFY_LOCAL_URL or 127.0.0.1). Local targets only."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(help="Server port (defaults to COMFY_LOCAL_URL or 8188). Local targets only."),
    ] = None,
):
    config = ConfigManager()
    renderer = get_renderer()

    # Validate the flags before resolving anything: the host lands verbatim in
    # ``http://{host}:{port}/upload/image``, so a URL-special or control
    # character is a usage error (BadParameter, exit 2) regardless of target.
    # ``report_usage_error`` emits the terminating envelope for that rejection
    # in JSON/NDJSON mode; the exception still escapes, so exit stays 2.
    with report_usage_error(renderer):
        if host is not None:
            host = validate_host(host)
        if port is not None and not (1 <= port <= 65535):
            raise typer.BadParameter(f"invalid port: {port} is out of range (1-65535)")

    try:
        decision = where_module.resolve(flag=where, config_value=config.get(where_module.CONFIG_KEY_WHERE_DEFAULT))
    except ValueError as e:
        renderer.error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
        raise typer.Exit(code=1)

    effective_where = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"
    # --host/--port address a local ComfyUI; the cloud target's address comes
    # from the signed-in account's base URL and ignores them entirely
    # (``Target.host``/``Target.port`` are documented local-only). Rejecting
    # the combination beats silently uploading somewhere the user didn't name.
    # Checked before the preflight so the flag error isn't masked by a
    # "not signed in" error.
    if effective_where == "cloud" and (host is not None or port is not None):
        # The cloud target can come from an explicit --where, but equally from
        # COMFY_WHERE, a project/config default, or credential auto-detection —
        # so name the source rather than accusing the user of passing a flag
        # they may never have typed.
        source = _WHERE_SOURCE_PHRASES.get(decision.source, f"resolved to cloud by {decision.source}")
        renderer.error(
            code="host_flag_cloud",
            message=f"--host/--port target a local ComfyUI server, but this run is {source}",
            hint=(
                "pass --where local to aim at a local server; to reach a different cloud address "
                "set COMFY_CLOUD_BASE_URL or run `comfy cloud set-base-url`"
            ),
            details={"host": host, "port": port, "where": effective_where, "where_source": decision.source},
        )
        raise typer.Exit(code=1)
    if effective_where == "cloud":
        where_module.cloud_preflight_or_exit()

    transfer_inner.execute_upload(files, where=effective_where, overwrite=overwrite, host=host, port=port)


@app.command(help="Download outputs from a completed job. Reads prompt_id from argument or piped stdin.")
@tracking.track_command()
def download(
    prompt_id: Annotated[
        str | None,
        typer.Argument(help="Prompt ID to download outputs for. Omit to read from piped stdin.", show_default=False),
    ] = None,
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", "-o", help="Directory to save outputs to."),
    ] = "./outputs",
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target: 'local' or 'cloud'."),
    ] = None,
    url_only: Annotated[
        bool,
        typer.Option(
            "--url-only",
            show_default=False,
            help="Emit output URLs without downloading files. Useful for agents that pass URLs to other tools.",
        ),
    ] = False,
):
    config = ConfigManager()
    renderer = get_renderer()
    try:
        decision = where_module.resolve(flag=where, config_value=config.get(where_module.CONFIG_KEY_WHERE_DEFAULT))
    except ValueError as e:
        renderer.error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
        raise typer.Exit(code=1)

    effective_where = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"
    if effective_where == "cloud":
        where_module.cloud_preflight_or_exit()

    transfer_inner.execute_download(prompt_id, out_dir=out_dir, where=effective_where, url_only=url_only)


@app.command("run-cli", help="Walk through the CLI surface in realtime with a tiny no-model workflow.")
@tracking.track_command()
def run_cli(
    pause_seconds: Annotated[
        float,
        typer.Option(
            "--pause-seconds",
            help="Seconds to pause between steps for readability. Use 0 for fast/CI runs.",
        ),
    ] = 3.0,
    no_pause: Annotated[
        bool,
        typer.Option("--no-pause", help="Equivalent to --pause-seconds 0."),
    ] = False,
    show_agent: Annotated[
        bool,
        typer.Option(
            "--show-agent/--no-show-agent",
            help="Also show the --json envelope an agent would parse for each command. On by default.",
        ),
    ] = True,
    no_cleanup: Annotated[
        bool,
        typer.Option("--no-cleanup", help="Keep the temporary demo workflow file after the run."),
    ] = False,
):
    effective_pause = 0.0 if no_pause else pause_seconds
    raise typer.Exit(
        code=run_cli_inner.execute(pause_seconds=effective_pause, no_cleanup=no_cleanup, show_agent=show_agent)
    )


@app.command(
    "system-stats",
    help="Show ComfyUI system stats: per-device VRAM + system RAM/version (GET /system_stats).",
)
@tracking.track_command()
def system_stats(
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target: 'local' (default) or 'cloud'."),
    ] = None,
):
    from comfy_cli.command import system as system_command

    system_command.system_stats_execute(get_renderer(), where=where)


@app.command(
    help="Ask ComfyUI to unload models / free the executor cache (POST /free). "
    "Applies on the worker's next iteration — never interrupts a running job."
)
@tracking.track_command()
def free(
    unload_models: Annotated[
        bool,
        typer.Option("--unload-models/--no-unload-models", help="Unload all models from VRAM."),
    ] = True,
    free_memory: Annotated[
        bool,
        typer.Option(
            "--free-memory",
            help="Also reset the executor cache; implies --unload-models on the server side.",
        ),
    ] = False,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target: 'local' (default) or 'cloud'."),
    ] = None,
):
    from comfy_cli.command import system as system_command

    system_command.free_execute(get_renderer(), where=where, unload_models=unload_models, free_memory=free_memory)


@app.command(help="Stop background ComfyUI. Use --port to stop an untracked local ComfyUI this CLI didn't start.")
@tracking.track_command()
def stop(
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            show_default=False,
            min=1,
            max=65535,
            help="Stop whatever local ComfyUI is listening on this port, even if this CLI didn't start it. "
            "Refuses anything it cannot positively identify as ComfyUI.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Report the process that would be stopped and exit 0 without stopping it.",
        ),
    ] = False,
):
    renderer = get_renderer()

    if port is not None:
        from comfy_cli.command import stop_port

        stop_port.stop_port_execute(renderer, port=port, dry_run=dry_run)
        return

    config = ConfigManager()
    bg_info = config.background if constants.CONFIG_KEY_BACKGROUND in config.config["DEFAULT"] else None
    if not bg_info:
        rprint("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        if renderer.is_json():
            renderer.error(
                code="no_background_server",
                message="No ComfyUI is running in the background.",
                command="stop",
            )
        raise typer.Exit(code=1)

    if dry_run:
        rprint(
            f"[bold yellow]Would stop background ComfyUI.[/bold yellow] ({bg_info[0]}:{bg_info[1]}, pid={bg_info[2]})"
        )
        renderer.emit(
            {
                "stopped": False,
                "dry_run": True,
                "untracked": False,
                "host": bg_info[0],
                "port": bg_info[1],
                "pid": bg_info[2],
            },
            command="stop",
            changed=False,
        )
        return

    is_killed = utils.kill_all(bg_info[2])

    if not is_killed:
        # The kill failed (permissions, a race, or the process is still alive).
        # Do NOT remove the background record — it is the only handle a retried
        # `comfy stop` has to target this PID/host/port; dropping it here would
        # orphan a live server. Surface a non-zero exit in BOTH pretty and JSON
        # mode so `$?`-checking scripts see the failure (pretty mode otherwise
        # printed the error yet returned 0).
        rprint("[bold red]Failed to stop ComfyUI in the background.[/bold red]\n")
        if renderer.is_json():
            renderer.error(
                code="stop_failed",
                message="Failed to stop ComfyUI in the background.",
                command="stop",
                details={"pid": bg_info[2]},
            )
        raise typer.Exit(code=1)

    rprint(f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})")
    config.remove_background()

    if renderer.is_json():
        renderer.emit(
            {"stopped": True, "host": bg_info[0], "port": bg_info[1], "pid": bg_info[2]},
            command="stop",
            changed=True,
        )


@app.command(help="Launch ComfyUI: ?[--background] ?[-- <extra args ...>]")
@tracking.track_command()
def launch(
    extra: list[str] = typer.Argument(None),
    background: Annotated[bool, typer.Option(help="Launch ComfyUI in background")] = False,
    frontend_pr: Annotated[
        str | None,
        typer.Option(
            "--frontend-pr",
            show_default=False,
            help="Use a specific frontend PR. Supports formats: username:branch, #123, or PR URL",
        ),
    ] = None,
):
    launch_command(background, extra, frontend_pr)


@app.command(help="Show a captured ComfyUI log (from `comfy launch --background`, or ComfyUI-Manager's own logfile).")
@tracking.track_command()
def logs(
    tail: Annotated[
        int,
        typer.Option("--tail", help="Number of trailing log lines to show (capped for JSON payloads)."),
    ] = 200,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target. Only 'local' is supported."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            show_default=False,
            help="Show the log for this port instead of auto-resolving "
            "(falls back to user/comfyui.log, which is where a server started "
            "without an explicit --port logs).",
        ),
    ] = None,
):
    logs_command(tail=tail, where=where, port=port)


@app.command("setup", help="Interactive setup wizard — routing, auth, and agent skills in one step.")
@tracking.track_command()
def setup(
    where: Annotated[
        str | None,
        typer.Option(
            "--where", show_default=False, help="Routing target: 'local' or 'cloud'. Skips the interactive prompt."
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", show_default=False, help="Comfy Cloud API key. Implies --where cloud."),
    ] = None,
    project_dir: Annotated[
        str | None,
        typer.Option("--project-dir", show_default=False, help="Project directory for workflows, inputs, and outputs."),
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            "-y",
            show_default=False,
            help="No prompts — use flags for all choices. For CI, devcontainers, and scripted installs.",
        ),
    ] = False,
    skip_skills: Annotated[
        bool,
        typer.Option("--skip-skills", show_default=False, help="Skip agent skill installation."),
    ] = False,
    skip_verify: Annotated[
        bool,
        typer.Option("--skip-verify", show_default=False, help="Skip connectivity verification."),
    ] = False,
):
    from comfy_cli.command import setup as setup_inner

    setup_inner.execute(
        where=where,
        api_key=api_key,
        project_dir=project_dir,
        non_interactive=non_interactive,
        skip_skills=skip_skills,
        skip_verify=skip_verify,
    )


@app.command("set-default", help="Persist defaults: ComfyUI workspace path and/or `--where` mode.")
@tracking.track_command()
def set_default(
    workspace_path: Annotated[
        str | None,
        typer.Argument(help="Path to ComfyUI workspace (optional — pass to set the default workspace)."),
    ] = None,
    launch_extras: Annotated[
        str | None,
        typer.Option(help="Extra options forwarded to `comfy launch`."),
    ] = None,
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            show_default=False,
            help="Persist the default routing mode: 'local' or 'cloud'. "
            "Once set, every command honors it without a per-invocation flag.",
        ),
    ] = None,
    clear_where: Annotated[
        bool,
        typer.Option(
            "--clear-where",
            show_default=False,
            help="Clear the persisted --where so commands fall back to env / local default.",
        ),
    ] = False,
):
    renderer = get_renderer()
    config = ConfigManager()
    changed_workspace = False
    changed_where = False

    if where is not None:
        try:
            where_module._parse(where)
        except ValueError as e:
            renderer.error(code="where_invalid", message=str(e), hint="use --where local or --where cloud")
            raise typer.Exit(code=1) from e
        config.set(where_module.CONFIG_KEY_WHERE_DEFAULT, where.strip().lower())
        changed_where = True

    if clear_where:
        config.set(where_module.CONFIG_KEY_WHERE_DEFAULT, "")
        changed_where = True

    if workspace_path is not None:
        comfy_path = os.path.abspath(os.path.expanduser(workspace_path))
        if not os.path.exists(comfy_path):
            renderer.error(
                code="not_in_workspace", message=f"Path not found: {comfy_path}", hint="pass a real workspace path"
            )
            raise typer.Exit(code=1)
        is_comfy_repo, resolved_path = check_comfy_repo(comfy_path)
        if not is_comfy_repo:
            renderer.error(
                code="not_in_workspace",
                message=f"Not a ComfyUI workspace: {comfy_path}",
                hint="`comfy install` to scaffold one, or pass a different path",
            )
            raise typer.Exit(code=1)
        assert resolved_path is not None
        workspace_manager.set_default_workspace(resolved_path)
        if launch_extras is not None:
            workspace_manager.set_default_launch_extras(launch_extras)
        changed_workspace = True

    if not (changed_workspace or changed_where):
        renderer.error(
            code="missing_argument",
            message="set-default needs at least one of: <workspace_path>, --where, or --clear-where.",
            hint="example: `comfy set-default --where cloud`",
        )
        raise typer.Exit(code=1)

    persisted_where = config.get(where_module.CONFIG_KEY_WHERE_DEFAULT) or None
    workspace_value = config.get(constants.CONFIG_KEY_DEFAULT_WORKSPACE) if changed_workspace else None
    if renderer.is_pretty():
        if changed_workspace:
            rprint(f"[bold green]✓[/bold green] Default workspace → [cyan]{workspace_value or '?'}[/cyan]")
        if changed_where:
            if clear_where:
                rprint(
                    "[bold green]✓[/bold green] Cleared persisted [bold]where[/bold] (falls back to env / local default)."
                )
            else:
                rprint(f"[bold green]✓[/bold green] Default [bold]where[/bold] → [cyan]{persisted_where}[/cyan]")
    renderer.emit(
        {
            "default_workspace": workspace_value,
            "default_launch_extras": launch_extras if changed_workspace else None,
            "default_where": persisted_where,
        },
        command="set-default",
        changed=True,
    )


@app.command(help="Show which ComfyUI is selected.")
@tracking.track_command()
def which():
    renderer = get_renderer()
    comfy_path = workspace_manager.workspace_path
    if comfy_path is None:
        renderer.error(
            code="not_in_workspace",
            message="ComfyUI not found, please run 'comfy install', run 'comfy' in a ComfyUI directory, or specify the workspace path with '--workspace'.",
            hint="run: comfy install   (or pass --workspace /path/to/ComfyUI)",
        )
        raise typer.Exit(code=1)

    workspace_type_value = (
        workspace_manager.workspace_type.value if workspace_manager.workspace_type is not None else None
    )
    if renderer.is_pretty():
        import sys as _sys

        from comfy_cli.env_checker import _display_url
        from comfy_cli.local_address import resolve_local_host_port
        from comfy_cli.output.panels import which_panel

        host, port = resolve_local_host_port(None, None, background=ConfigManager().background)
        try:
            server_running = env_checker.check_comfy_server_running(host=host, port=port, timeout=0.5)
        except Exception:  # noqa: BLE001
            server_running = False
        renderer.console().print(
            which_panel(
                workspace_path=str(comfy_path),
                workspace_type=workspace_type_value,
                python_executable=_sys.executable,
                python_version=f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
                server_running=server_running,
                server_url=_display_url(host, port),
                version=ConfigManager().get_cli_version(),
            )
        )
    renderer.emit(
        {
            "workspace_path": str(comfy_path),
            "workspace_type": workspace_type_value,
        },
        command="which",
    )


@app.command(help="Emit a self-describing surface (commands, schemas, error codes) for agents.")
@tracking.track_command()
def discover(
    schemas_only: Annotated[
        bool,
        typer.Option(
            "--schemas-only",
            show_default=False,
            help="Emit only the schemas bundle, omitting the command tree.",
        ),
    ] = False,
):
    renderer = get_renderer()
    cli_version = ConfigManager().get_cli_version()
    doc = build_discovery(app, version=cli_version)
    if schemas_only:
        doc = {
            "prog": doc["prog"],
            "version": doc["version"],
            "schemas": doc["schemas"],
            "command_schemas": doc["command_schemas"],
            "stream_event_schemas": doc["stream_event_schemas"],
            "capabilities": doc["capabilities"],
        }
    if renderer.is_json():
        renderer.emit(doc, command="discover")
        return
    from comfy_cli.output.panels import discover_panel

    renderer.console().print(discover_panel(doc, command_count=_count_commands(doc.get("commands", {}))))


def _count_commands(tree: dict) -> int:
    total = 0
    for entry in tree.values():
        total += 1
        subs = entry.get("subcommands") if isinstance(entry, dict) else None
        if subs:
            total += _count_commands(subs)
    return total


@app.command(help="Print out current environment variables.")
@tracking.track_command()
def env():
    renderer = get_renderer()
    # Only do the upgrade-check side-effect in pretty mode; agents asking for
    # a snapshot don't want a network call they didn't request.
    if renderer.is_pretty():
        from rich.table import Table

        from comfy_cli.output.branding import branded_panel

        env_data = EnvChecker().fill_print_table()
        workspace_data = workspace_manager.fill_print_table()
        all_data = env_data + workspace_data

        tbl = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            pad_edge=False,
            expand=True,
        )
        tbl.add_column("Environment", style="bold cyan", no_wrap=True, overflow="fold")
        tbl.add_column("Value", overflow="fold")
        for label, value in all_data:
            tbl.add_row(label, str(value))

        renderer.console().print(branded_panel(tbl, title="env", version=ConfigManager().get_cli_version()))
        return
    # JSON path: collect structured data without printing the table.
    data = EnvChecker().fill_data()
    data["workspace"] = workspace_manager.fill_data()
    renderer.emit(data, command="env")


_FEEDBACK_DISABLED_NOTICE = (
    "[yellow]Feedback not sent — telemetry is opted out via DO_NOT_TRACK / COMFY_NO_TELEMETRY.[/yellow]\n"
    "Unset that to send, or open an issue: https://github.com/Comfy-Org/comfy-cli/issues/new/choose"
)


def _relay_feedback(renderer: Renderer, sent: bool, *, message: str) -> None:
    """Surface the feedback outcome: JSON envelope for agents, a line for humans."""
    if renderer.is_json():
        renderer.emit({"sent": sent, "message": message}, command="feedback")
        return
    rprint("Thank you for your feedback!" if sent else _FEEDBACK_DISABLED_NOTICE)


@app.command(help="Provide feedback on the Comfy CLI tool. Pass it inline to send in one shot.")
@tracking.track_command()
def feedback(
    message: Annotated[
        str | None,
        typer.Argument(
            show_default=False,
            help='Your feedback, e.g. comfy feedback "run is great but jobs watch needs an ETA". '
            "Omit (interactive only) to answer a few quick questions.",
        ),
    ] = None,
):
    renderer = get_renderer()

    # One-shot — the path agents (Claude, etc.) and scripts should use.
    if message:
        _relay_feedback(renderer, tracking.submit_feedback(message), message=message)
        return

    # No inline message: the interactive prompts need a human at a TTY. In
    # JSON/agentic mode there's nobody to prompt — tell the caller to pass it inline.
    if not renderer.is_pretty():
        renderer.error(
            code="feedback_message_required",
            message="Feedback requires an inline message in JSON mode.",
            hint='comfy feedback "your feedback here"',
        )
        raise typer.Exit(code=1)

    rprint("Feedback Collection for Comfy CLI Tool\n")

    general_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5, how satisfied are you with the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
        force_prompting=True,
    )
    usability_satisfaction_score = ui.prompt_select(
        question="On a scale of 1 to 5,  how satisfied are you with the usability and user experience of the Comfy CLI tool? (1 being very dissatisfied and 5 being very satisfied)",
        choices=["1", "2", "3", "4", "5"],
        force_prompting=True,
    )
    free_text = ui.prompt_input(
        question="Anything else you'd like to share? (optional — press Enter to skip)",
        force_prompting=True,
    )

    sent = tracking.submit_feedback(
        free_text or "",
        scores={
            "general_satisfaction": None if general_satisfaction_score is None else str(general_satisfaction_score),
            "usability_satisfaction": None
            if usability_satisfaction_score is None
            else str(usability_satisfaction_score),
        },
    )
    if (
        sent
        and questionary.confirm("Do you want to provide additional feature-specific feedback on our GitHub page?").ask()
    ):
        tracking.track_event("feedback_additional")
        webbrowser.open("https://github.com/Comfy-Org/comfy-cli/issues/new/choose")

    _relay_feedback(renderer, sent, message=free_text or "")


@app.command(
    name="agent-review",
    hidden=True,
    help="For agents: submit a short summary of how the session went. Fully consent-gated.",
)
@tracking.track_command()
def agent_review(
    summary: Annotated[
        str,
        typer.Argument(help="A brief, factual summary of the session — your assessment, not the user's words."),
    ],
):
    renderer = get_renderer()
    sent = tracking.submit_agent_review(summary)
    if renderer.is_json():
        renderer.emit({"sent": sent, "summary": summary}, command="agent-review")
        return
    rprint(
        "Session review recorded — thanks."
        if sent
        else "[yellow]Review not sent — telemetry is disabled or opted out.[/yellow]"
    )


@app.command(
    hidden=True,
    help="Given an existing installation of comfy core and any custom nodes, installs any needed python dependencies",
)
@tracking.track_command()
def dependency():
    comfy_path, _ = workspace_manager.get_workspace_path()

    python = resolve_workspace_python(comfy_path)
    depComp = DependencyCompiler(cwd=comfy_path, executable=python)
    depComp.compile_deps()
    depComp.install_deps()


@app.command(help="Download a standalone Python interpreter and dependencies based on an existing comfyui workspace")
@tracking.track_command()
def standalone(
    cli_spec: Annotated[
        str,
        typer.Option(
            show_default=False,
            help="setuptools-style requirement specificer pointing to an instance of comfy-cli",
        ),
    ] = "comfy-cli",
    pack_wheels: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Pack requirement wheels in archive when creating standalone bundle",
        ),
    ] = False,
    platform: Annotated[
        constants.OS | None,
        typer.Option(
            show_default=False,
            help="Create standalone Python for specified platform",
        ),
    ] = None,
    proc: Annotated[
        constants.PROC | None,
        typer.Option(
            show_default=False,
            help="Create standalone Python for specified processor",
        ),
    ] = None,
    rehydrate: Annotated[
        bool,
        typer.Option(
            show_default=False,
            help="Create standalone Python for CPU",
        ),
    ] = False,
):
    comfy_path, _ = workspace_manager.get_workspace_path()

    platform = utils.get_os() if platform is None else platform
    proc = utils.get_proc() if proc is None else proc

    if rehydrate:
        sty = StandalonePython.FromTarball(fpath="python.tgz")
        sty.rehydrate_comfy_deps(packWheels=pack_wheels)
    else:
        sty = StandalonePython.FromDistro(platform=platform, proc=proc)
        sty.dehydrate_comfy_deps(comfyDir=comfy_path, extraSpecs=[], packWheels=pack_wheels)
        sty.to_tarball()


generate_command.register_with(app)
app.add_typer(
    models_command.app,
    name="model",
    help="Manage the model files in this workspace — download, list, remove. (Search/discovery lives under `comfy models`.)",
)
app.add_typer(
    models_search_command.app,
    name="models",
    help="Discover models — folders, files, and the cloud asset catalog.",
)
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(nodes_command.app, name="nodes", help="Introspect ComfyUI node classes (inputs, outputs, categories).")
app.add_typer(templates_command.app, name="templates", help="Browse the Comfy workflow-template gallery.")
app.command(
    "run-template",
    help=(
        "Fetch a gallery template, fill its parameterized inputs (--param KEY=VALUE), "
        "and run it to completion on local ComfyUI. Paid partner-API templates require --allow-spend."
    ),
)(templates_command.run_template_cmd)
app.add_typer(workflow_command.app, name="workflow", help="Slot-based editing of frontend-format ComfyUI workflows.")
app.command(
    "preview",
    help="Render a previewable PNG from a media file (image → thumb, video → contact sheet, audio → waveform).",
)(preview_command.preview_cmd)
app.add_typer(custom_nodes.manager_app, name="manager", help="Manage ComfyUI-Manager.")

app.add_typer(pr_command.app, name="pr-cache", help="Manage PR cache.")

app.add_typer(code_search.app, name="code-search", help="Search code across ComfyUI repositories.")
app.add_typer(code_search.app, name="cs", hidden=True)

app.add_typer(tracking.app, name="tracking", help="Manage analytics tracking settings.")
app.add_typer(cloud_command.app, name="cloud", help="Comfy Cloud — sign in, route commands, inspect session.")
app.add_typer(auth_command.app, name="auth", help="Manage API tokens for model hosts (Civitai, Hugging Face).")
app.add_typer(jobs_command.app, name="jobs", help="List, inspect, and live-watch ComfyUI prompts.")
app.add_typer(project_command.app, name="project", help="Project conventions: init and status.")
app.add_typer(
    project_command.assets_app,
    name="assets",
    help="Push project assets to the run target (local or cloud) and track them in the lock.",
)
app.add_typer(
    skill_command.app,
    name="skills",
    help="Install the bundled comfy agent skills into Claude Code, Cursor, Aider, and any AGENTS.md-aware tool.",
)
# Keep the singular alias for backward compat
app.add_typer(skill_command.app, name="skill", hidden=True)

# Hidden: the detached watcher subprocess spawned by `comfy run` when async.
from comfy_cli.command import job_watcher as _job_watcher  # noqa: E402

app.add_typer(_job_watcher.app, name="_watch", hidden=True)
