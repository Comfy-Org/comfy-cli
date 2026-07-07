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
from comfy_cli.auth import command as auth_command
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
from comfy_cli.command.install import validate_version
from comfy_cli.command.launch import launch as launch_command
from comfy_cli.command.launch import logs as logs_command
from comfy_cli.command.models import models as models_command
from comfy_cli.command.models import search as models_search_command
from comfy_cli.config_manager import ConfigManager
from comfy_cli.constants import GPU_OPTION, CUDAVersion, ROCmVersion
from comfy_cli.cuda_detect import DEFAULT_CUDA_TAG, detect_cuda_driver_version, resolve_cuda_wheel
from comfy_cli.discovery import build_discovery
from comfy_cli.env_checker import EnvChecker
from comfy_cli.help_json import build_help_json
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
    if sub in (None, "setup") or not renderer.is_pretty() or not sys.stderr.isatty():
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


@app.callback(invoke_without_command=True)
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

    # TODO: Move this to proper place
    # start_time = time.time()
    # workspace_manager.scan_dir()
    # end_time = time.time()
    #
    # logging.info(f"scan_dir took {end_time - start_time:.2f} seconds to run")


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
        rprint("[bold red]Nvidia GPU is never on MacOS. What are you smoking? 🤔[/bold red]")
        raise typer.Exit(code=1)

    if platform != constants.OS.MACOS and m_series:
        rprint(f"[bold red]You are on {platform} bruh [/bold red]")

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


@app.command(help="Update ComfyUI Environment [all|comfy|cli]")
@tracking.track_command()
def update(
    target: str = typer.Argument(
        "comfy",
        help="[all|comfy|cli]",
        autocompletion=utils.create_choice_completer(["all", "comfy", "cli"]),
    ),
):
    if target not in ["all", "comfy", "cli"]:
        typer.echo(
            f"Invalid target: {target}. Allowed targets are 'all', 'comfy', 'cli'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if "cli" == target:
        from comfy_cli.update import upgrade_cli

        rprint("Updating comfy-cli...")
        upgrade_cli()
        return

    comfy_path = workspace_manager.workspace_path

    if "all" == target:
        custom_nodes.command.execute_cm_cli(["update", "all"])
    else:
        rprint(f"Updating ComfyUI in {comfy_path}...")
        if comfy_path is None:
            rprint("ComfyUI path is not found.")
            raise typer.Exit(code=1)
        os.chdir(comfy_path)
        subprocess.run(["git", "pull"], check=True)
        python = resolve_workspace_python(comfy_path)
        # A uv-managed venv may have no pip — bootstrap it first so the install
        # below doesn't crash with `No module named pip` (no-op if pip exists).
        ensure_pip(python, cwd=comfy_path)
        subprocess.run(
            [python, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )

    try:
        custom_nodes.command.update_node_id_cache()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        rprint(f"[yellow]Failed to update node id cache: {e}[/yellow]")


@app.command(help="Run an API workflow. Submits and returns immediately by default; pass --wait to block.")
def run(
    workflow: Annotated[
        str,
        typer.Option(
            help=(
                "Path to the workflow JSON file. Both ComfyUI API format and "
                "exported UI format are accepted; UI workflows are converted "
                "to API format client-side."
            )
        ),
    ],
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
):
    # Snapshot kwargs before the body mutates api_key/host/port — analytics should record what user actually supplied.
    _track_props = tracking.filter_command_kwargs(dict(locals()))
    tracking.track_event("execution_start", _track_props, mixpanel_name="run")

    try:
        if api_key:
            api_key = api_key.strip() or None

        config = ConfigManager()
        renderer = get_renderer()

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

        # Default for --notify: on when a human is at the terminal, off for
        # agents (they shouldn't get surprise side-channel processes they didn't
        # ask for). The user can override either way with --notify/--no-notify.
        effective_notify = notify if notify is not None else (renderer.is_pretty() and not wait)

        if decision.target is where_module.WhereTarget.CLOUD:
            err = where_module.cloud_preflight()
            if err is not None:
                renderer.error(
                    code=err.code,
                    message=err.message,
                    hint=err.hint,
                    details=err.details,
                )
                raise typer.Exit(code=1)
            # Cloud path uses HTTPS + Bearer auth; host/port aren't applicable.
            run_inner.execute_cloud(
                workflow,
                wait=wait,
                verbose=verbose,
                timeout=timeout,
                notify=effective_notify,
                print_prompt=print_prompt,
            )
            return

        from comfy_cli.host_port import parse_host_port_arg, resolve_host_port

        if host:
            host, parsed_port = parse_host_port_arg(host)
            if not port and parsed_port is not None:
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


@app.command(
    help="Validate an API-format workflow without submitting. Checks class_types, input shapes, enum values, and edge wiring."
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
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a saved object_info JSON (offline mode)."),
    ] = None,
):
    from pathlib import Path

    from comfy_cli.cql.engine import Graph, LoadError

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

    # Load graph
    mode = "local"
    if where:
        mode = where
    else:
        config = ConfigManager()
        try:
            decision = where_module.resolve(flag=None, config_value=config.get(where_module.CONFIG_KEY_WHERE_DEFAULT))
            mode = decision.target.value
        except Exception:
            pass

    try:
        graph = Graph.load(mode=mode, input_path=input_path, host=host or "127.0.0.1", port=port or 8188)
    except LoadError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <object_info.json>, or start the server"),
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    result = graph.validate_workflow(wf_data)

    payload = {
        "workflow": str(wf_path),
        "valid": result["valid"],
        "error_count": len(result["errors"]),
        "warning_count": len(result["warnings"]),
        "errors": result["errors"],
        "warnings": result["warnings"],
    }

    if renderer.is_pretty():
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
                rprint(f"  [red]•[/red] node {e.get('node_id', '?')}: {msg}")
            for w in result["warnings"]:
                rprint(f"  [yellow]⚠[/yellow] {w.get('message', '')}")
    renderer.emit(payload, command="validate", ok=result["valid"])

    if not result["valid"]:
        raise typer.Exit(code=1)


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
        err = where_module.cloud_preflight()
        if err is not None:
            renderer.error(code=err.code, message=err.message, hint=err.hint, details=err.details)
            raise typer.Exit(code=1)

    transfer_inner.execute_upload(files, where=effective_where, overwrite=overwrite)


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
        err = where_module.cloud_preflight()
        if err is not None:
            renderer.error(code=err.code, message=err.message, hint=err.hint, details=err.details)
            raise typer.Exit(code=1)

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


def validate_comfyui(_env_checker):
    if _env_checker.comfy_repo is None:
        rprint("[bold red]If ComfyUI is not installed, this feature cannot be used.[/bold red]")
        raise typer.Exit(code=1)


@app.command(help="Stop background ComfyUI")
@tracking.track_command()
def stop():
    if constants.CONFIG_KEY_BACKGROUND not in ConfigManager().config["DEFAULT"]:
        rprint("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)

    bg_info = ConfigManager().background
    if not bg_info:
        rprint("[bold red]No ComfyUI is running in the background.[/bold red]\n")
        raise typer.Exit(code=1)
    is_killed = utils.kill_all(bg_info[2])

    if not is_killed:
        rprint("[bold red]Failed to stop ComfyUI in the background.[/bold red]\n")
    else:
        rprint(f"[bold yellow]Background ComfyUI is stopped.[/bold yellow] ({bg_info[0]}:{bg_info[1]})")

    ConfigManager().remove_background()


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


@app.command(help="Show the captured background ComfyUI log (from `comfy launch --background`).")
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
):
    logs_command(tail=tail, where=where)


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

        from comfy_cli.output.panels import which_panel

        cfg_bg = ConfigManager().background
        if cfg_bg is not None:
            host, port = cfg_bg[0], cfg_bg[1]
        else:
            host, port = "127.0.0.1", 8188
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
                server_url=f"http://{host}:{port}",
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


@app.command(hidden=True)
@tracking.track_command()
def models():
    rprint("\n[bold red] No such command, did you mean 'comfy model' instead?[/bold red]\n")


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


@app.command(hidden=True)
@app.command(
    help="Given an existing installation of comfy core and any custom nodes, installs any needed python dependencies"
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
app.add_typer(models_command.app, name="model", help="Manage models.")
app.add_typer(
    models_search_command.app,
    name="models",
    help="Discover models — folders, files, and the cloud asset catalog.",
)
app.add_typer(custom_nodes.app, name="node", help="Manage custom nodes.")
app.add_typer(nodes_command.app, name="nodes", help="Introspect ComfyUI node classes (inputs, outputs, categories).")
app.add_typer(templates_command.app, name="templates", help="Browse the Comfy workflow-template gallery.")
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
    help="Install the bundled comfy agent skills into Claude Code, Cursor, and AGENTS.md.",
)
# Keep the singular alias for backward compat
app.add_typer(skill_command.app, name="skill", hidden=True)

# Hidden: the detached watcher subprocess spawned by `comfy run` when async.
from comfy_cli.command import job_watcher as _job_watcher  # noqa: E402

app.add_typer(_job_watcher.app, name="_watch", hidden=True)
