import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import uuid
from datetime import timedelta
from urllib import request

import typer
from rich.progress import BarColumn, Column, Progress, Table, TimeElapsedColumn
from websocket import WebSocket, WebSocketException, WebSocketTimeoutException

from comfy_cli import cancellation, jobs_state
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()

# JSON output schema version. Bumped only for breaking changes per docs/json-output.md.
SCHEMA_VERSION = 1

# Maximum bytes of a server response body we surface to the user (or
# embed in a `failed.error.body` field). Anything longer is truncated.
_MAX_BODY_PREVIEW = 500


def _node_errors_to_list(node_errors) -> list[dict]:
    """Transform ComfyUI's dict-keyed `node_errors` payload into a list of self-contained records.
    Each record carries `node_id` as a field, so agents can iterate the result
    directly without indirecting through dict keys."""
    if not isinstance(node_errors, dict):
        return []
    result = []
    for node_id, record in node_errors.items():
        if not isinstance(record, dict):
            continue
        entry = {"node_id": str(node_id)}
        entry.update(record)
        result.append(entry)
    return result


def is_ui_workflow(workflow) -> bool:
    return (
        isinstance(workflow, dict)
        and isinstance(workflow.get("nodes"), list)
        and isinstance(workflow.get("links"), list)
    )


def _classify_api_workflow(workflow):
    """Classify a parsed JSON object as API workflow / empty / invalid.

    Returns one of:
        ("ok", workflow_dict)   — well-formed API workflow with ≥1 node
        ("empty", None)         — empty dict (caller routes to workflow_empty)
        ("invalid", None)       — not a dict, or first node lacks class_type
    """
    if not isinstance(workflow, dict):
        return ("invalid", None)
    if not workflow:
        return ("empty", None)
    first_key = next(iter(workflow))
    node = workflow[first_key]
    if not isinstance(node, dict) or "class_type" not in node:
        return ("invalid", None)
    return ("ok", workflow)


class JsonEmitter:
    """NDJSON event emitter for ``comfy run --json``.

    Every ``emit_*`` method is a no-op when ``json_mode=False``, so the
    same call sites work for both modes. See ``docs/json-output.md``.
    """

    def __init__(self, json_mode: bool):
        self.json_mode = json_mode
        self.start_time = time.monotonic()
        self.client_id: str | None = None
        self.prompt_id: str | None = None
        self.workflow: dict | None = None
        self.cached_node_ids: list[str] = []
        self.executed_node_ids: list[str] = []
        self.outputs: list[dict] = []

    def set_workflow(self, workflow):
        self.workflow = workflow

    def set_client_id(self, client_id):
        self.client_id = client_id

    def _elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def get_title(self, node_id):
        if not isinstance(self.workflow, dict):
            return str(node_id)
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return str(node_id)
        meta = node.get("_meta")
        if isinstance(meta, dict):
            title = meta.get("title")
            if isinstance(title, str) and title:
                return title
        class_type = node.get("class_type")
        return class_type if isinstance(class_type, str) and class_type else str(node_id)

    def get_class_type(self, node_id):
        if not isinstance(self.workflow, dict):
            return ""
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return ""
        return node.get("class_type", "")

    def _emit(self, event: dict) -> None:
        if not self.json_mode:
            return
        line = json.dumps(event, ensure_ascii=True)
        print(line, flush=True)

    def emit_converted(self, node_count: int) -> None:
        self._emit(
            {
                "event": "converted",
                "schema_version": SCHEMA_VERSION,
                "node_count": node_count,
            }
        )

    def workflow_manifest(self) -> list[dict]:
        """Build the `nodes` array for the `queued` event — one entry per
        node in the submitted (post-conversion) workflow."""
        if not isinstance(self.workflow, dict):
            return []
        manifest: list[dict] = []
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            class_type = class_type if isinstance(class_type, str) else ""
            manifest.append(
                {
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "title": self.get_title(node_id),
                }
            )
        return manifest

    def emit_prompt_preview(self, prompt: dict) -> None:
        self._emit(
            {
                "event": "prompt_preview",
                "schema_version": SCHEMA_VERSION,
                "prompt": prompt,
            }
        )

    def emit_queued(self, prompt_id: str, validation_warnings: list[dict]) -> None:
        self.prompt_id = prompt_id
        self._emit(
            {
                "event": "queued",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "client_id": self.client_id,
                "validation_warnings": validation_warnings,
                "nodes": self.workflow_manifest(),
            }
        )

    def emit_node_cached(self, node_id) -> None:
        node_id = str(node_id)
        self.cached_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_cached",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_executing(self, node_id) -> None:
        node_id = str(node_id)
        # `executed_node_ids` aggregates everything the executor touched —
        # including intermediate nodes that never fire a server-side `executed` WS event.
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self._emit(
            {
                "event": "node_executing",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
            }
        )

    def emit_node_progress(self, node_id, value, max_val) -> None:
        node_id = str(node_id)
        self._emit(
            {
                "event": "node_progress",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "value": value,
                "max": max_val,
            }
        )

    def emit_node_executed(self, node_id, outputs: list[dict]) -> None:
        node_id = str(node_id)
        if node_id not in self.executed_node_ids:
            self.executed_node_ids.append(node_id)
        self.outputs.extend(outputs)
        self._emit(
            {
                "event": "node_executed",
                "schema_version": SCHEMA_VERSION,
                "node_id": node_id,
                "class_type": self.get_class_type(node_id),
                "title": self.get_title(node_id),
                "outputs": outputs,
            }
        )

    def emit_completed(self) -> None:
        self._emit(
            {
                "event": "completed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "outputs": self.outputs,
                "cached_node_ids": self.cached_node_ids,
                "executed_node_ids": self.executed_node_ids,
            }
        )

    def fail(self, kind: str, message: str, *, rich_message: str | None = None, **extras) -> typer.Exit:
        """Emit a `failed` event (in JSON mode) or print a red text message
        (otherwise), then return the `typer.Exit(code=1)` for the caller to
        raise. Returning rather than raising keeps `raise ... from e`
        chaining clean at call sites. `rich_message` overrides `message`
        for the human-readable text only — it is auto-wrapped in
        `[bold red]...[/bold red]`. Sites that need multi-colour Rich
        markup should emit the failure event explicitly."""
        self.emit_failed(kind, message, **extras)
        if not self.json_mode:
            pprint(f"[bold red]{rich_message if rich_message is not None else message}[/bold red]")
        return typer.Exit(code=1)

    def emit_failed(self, kind: str, message: str, **extras) -> None:
        error = {"kind": kind, "message": message}
        error.update(extras)
        self._emit(
            {
                "event": "failed",
                "schema_version": SCHEMA_VERSION,
                "prompt_id": self.prompt_id,
                "client_id": self.client_id,
                "elapsed_seconds": self._elapsed(),
                "error": error,
            }
        )


def fetch_object_info(host, port, timeout, emitter=None):
    """GET ``/object_info`` from the running ComfyUI server.

    The response describes every loaded node class's input schema and is what
    the converter uses to map widget values to input names, fill defaults, etc.

    In JSON mode, failures emit a structured ``failed`` event via ``emitter``.
    Either way, a ``typer.Exit(code=1)`` is raised.
    """
    url = f"http://{host}:{port}/object_info"
    json_mode = bool(emitter and emitter.json_mode)
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace").strip()
        if json_mode:
            emitter.emit_failed(
                "object_info_unavailable",
                f"Failed to fetch /object_info (HTTP {e.code})",
                status_code=e.code,
                body=body_text[:_MAX_BODY_PREVIEW],
            )
        else:
            pprint(
                f"[bold red]Failed to fetch /object_info (HTTP {e.code}): {body_text[:_MAX_BODY_PREVIEW]}[/bold red]"
            )
        raise typer.Exit(code=1) from e
    except urllib.error.URLError as e:
        msg = f"Failed to fetch /object_info from {host}:{port}: {e.reason} (override with --host / --port)"
        if json_mode:
            emitter.emit_failed("connection_error", msg)
        else:
            pprint(f"[bold red]{msg}[/bold red]")
        raise typer.Exit(code=1) from e
    except TimeoutError as e:
        msg = f"Failed to fetch /object_info from {host}:{port}: timed out after {timeout}s (override with --host / --port)"
        if json_mode:
            emitter.emit_failed("connection_error", msg)
        else:
            pprint(f"[bold red]{msg}[/bold red]")
        raise typer.Exit(code=1) from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        if json_mode:
            emitter.emit_failed(
                "object_info_unavailable",
                "Server returned invalid JSON for /object_info",
                status_code=200,
                body=body.decode("utf-8", errors="replace")[:_MAX_BODY_PREVIEW],
            )
        else:
            pprint("[bold red]Failed to fetch /object_info: server returned invalid JSON[/bold red]")
        raise typer.Exit(code=1) from e


class WorkflowLoadError(Exception):
    """Raised by ``_load_workflow_file`` for pre-flight file errors."""

    def __init__(self, kind: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


def _load_workflow_file(path: str) -> tuple[dict, str, bool]:
    """Load and validate a workflow JSON file.

    Returns (raw_workflow_dict, absolute_path, is_ui_format).
    Raises ``WorkflowLoadError`` on file-not-found, read error, or invalid JSON.
    """
    workflow_name = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(workflow_name):
        raise WorkflowLoadError(
            "workflow_not_found",
            f"Specified workflow file not found: {workflow_name}",
            hint="check the path; pass the API-format JSON exported from ComfyUI",
        )

    try:
        with open(workflow_name, encoding="utf-8") as f:
            raw_workflow = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise WorkflowLoadError(
            "workflow_read_error",
            f"Unable to read workflow file: {e}",
        ) from e
    except json.JSONDecodeError as e:
        raise WorkflowLoadError(
            "workflow_invalid_json",
            f"Specified workflow file is not valid JSON: {e}",
            hint="re-export the workflow from ComfyUI (File > Export (API))",
        ) from e

    return raw_workflow, workflow_name, is_ui_workflow(raw_workflow)


def _preflight_validate(
    renderer, workflow: dict, object_info: dict, *, target_label: str = "server", emitter: "JsonEmitter | None" = None
) -> None:
    """Pre-submit validation via the WASM engine.

    Checks unknown class_types, input shape mismatches, and catalog drift.
    Raises typer.Exit(1) on hard errors. Prints warnings in pretty mode.
    Skips silently when object_info is empty (server unreachable — fail open).
    """
    if not object_info:
        return

    from comfy_cli.comfygraph import validate_workflow

    validation = validate_workflow(workflow, object_info)
    if not validation.get("valid", True):
        errors = validation.get("errors", [])
        hint_parts = []
        for e in errors[:5]:
            line = f"node {e.get('node_id', '?')}: {e.get('message', '')}"
            suggestions = e.get("suggestions") or []
            if suggestions:
                line += f" (did you mean: {', '.join(suggestions)}?)"
            hint_parts.append(line)
        if emitter is not None and emitter.json_mode:
            emitter.emit_failed(
                "workflow_unknown_nodes",
                f"Workflow has {len(errors)} validation error(s)",
                errors=errors,
                warnings=validation.get("warnings", []),
            )
            raise typer.Exit(code=1)
        renderer.error(
            code="workflow_unknown_nodes",
            message=f"Workflow has {len(errors)} validation error(s)",
            hint="\n".join(hint_parts),
            details={"errors": errors, "warnings": validation.get("warnings", [])},
        )
        raise typer.Exit(code=1)

    # Surface catalog-drift warnings in pretty mode (non-blocking).
    warnings = validation.get("warnings", [])
    if warnings and renderer.is_pretty():
        for w in warnings:
            pprint(f"[yellow]⚠ {w.get('field', '?')}: {w.get('message', '')}[/yellow]")


PARTNER_NODE_CATEGORY_PREFIX = "api node/"


def _fetch_object_info(host: str, port: int, timeout: int = 10) -> dict:
    """Fetch object_info for partner-node detection + validation. Fail open."""
    try:
        from comfy_cli.comfygraph import load_object_info

        return load_object_info(mode="local", host=host, port=port)
    except Exception:  # noqa: BLE001 — fail open
        return {}


def _detect_partner_nodes(workflow: dict, object_info: dict) -> list[str]:
    """Return sorted unique class_types in ``workflow`` whose category in
    ``object_info`` starts with ``api node/`` (Veo, Kling, BFL, Gemini,
    Bria, etc.). Pure function — tests pass object_info directly.

    A partner-API node call requires an ``api_key_comfy_org`` (or the
    OAuth-equivalent ``auth_token_comfy_org``) in ``extra_data`` at
    submit time. Without it the local server happily accepts the
    workflow at /prompt and then fails the actual node at execute time
    with ``Unauthorized: Please login first to use this node`` — a
    sharp edge observed during the Veo3 video run.
    """
    used: set[str] = set()
    for node in workflow.values():
        if isinstance(node, dict):
            ct = node.get("class_type")
            if isinstance(ct, str):
                used.add(ct)
    out: list[str] = []
    for ct in used:
        info = object_info.get(ct) or {}
        category = info.get("category") if isinstance(info, dict) else None
        if isinstance(category, str) and category.startswith(PARTNER_NODE_CATEGORY_PREFIX):
            out.append(ct)
    return sorted(out)


def _resolve_partner_credential() -> tuple[str, str] | None:
    """Locate an ``api_key_comfy_org`` credential to inject into a local
    submit's ``extra_data`` so partner-API nodes can authenticate.

    Returns ``(extra_data_key, value)`` so the caller can write the
    correct field for the credential type. Precedence matches the cloud
    submit path in ``comfy_client.submit_prompt``:

    1. ``COMFY_CLOUD_API_KEY`` env var → ``api_key_comfy_org``
    2. Stored ``comfy-cloud-api-key`` provider record → ``api_key_comfy_org``
    3. Active (non-expired) OAuth session → ``auth_token_comfy_org``

    Returns ``None`` when nothing is configured / the OAuth session is
    expired.
    """
    env_key = os.environ.get("COMFY_CLOUD_API_KEY")
    if env_key:
        return ("api_key_comfy_org", env_key)

    from comfy_cli.auth import store as auth_store
    from comfy_cli.target import CLOUD_API_KEY_PROVIDER

    record = auth_store.get(CLOUD_API_KEY_PROVIDER)
    if record is not None and getattr(record, "key", None):
        return ("api_key_comfy_org", record.key)

    session = auth_store.get_cloud_session()
    if session is not None and not session.is_expired() and session.access_token:
        return ("auth_token_comfy_org", session.access_token)

    return None


def _tail_state_file(prompt_id: str, *, seconds: float = 8.0) -> None:
    """Pretty-mode only: poll the state file for up to ``seconds`` showing
    live status transitions, then return. The background watcher keeps
    writing after we return — we just give the human a few seconds of
    "alive" feedback before the foreground exits.

    Always safe to call; no-ops if the renderer isn't pretty or the user
    Ctrl-Cs out.
    """
    import time as _t

    from rich.live import Live
    from rich.text import Text

    renderer = get_renderer()
    if not renderer.is_pretty():
        return

    from comfy_cli import jobs_state
    from comfy_cli.output.glyphs import status_glyph

    def render(status: str, elapsed: float) -> Text:
        return Text.from_markup(f"  {status_glyph(status)}  [dim]· {elapsed:.1f}s · {prompt_id[:8]}…[/dim]")

    deadline = _t.time() + seconds
    last_status = "queued"
    start = _t.time()
    try:
        with Live(render(last_status, 0.0), console=renderer.console(), refresh_per_second=4, transient=True) as live:
            while _t.time() < deadline:
                state = jobs_state.read(prompt_id)
                if state is not None:
                    last_status = state.status
                    live.update(render(last_status, _t.time() - start))
                    if state.is_terminal:
                        break
                _t.sleep(0.25)
    except KeyboardInterrupt:
        pass

    final_state = jobs_state.read(prompt_id)
    if final_state is None:
        return
    elapsed = _t.time() - start
    glyph = status_glyph(final_state.status)
    if final_state.is_terminal:
        pprint(f"  {glyph} [dim]· finished in {elapsed:.1f}s[/dim]")
        for u in (final_state.outputs or [])[:3]:
            pprint(f"  [dim]→[/dim] [cyan]{u}[/cyan]")
    else:
        pprint(f"  {glyph} [dim]· still in flight — track:[/dim] [cyan]comfy jobs ls --watch[/cyan]")


def _spawn_watcher(
    prompt_id: str,
    *,
    where: str,
    host: str | None = None,
    port: int | None = None,
    notify: bool = False,
) -> bool:
    """Detach a watcher subprocess that polls + updates the state file.

    Fully decoupled from the parent: stdio redirected to /dev/null, a new
    session so a controlling terminal closing doesn't kill it. We don't
    track the PID — the watcher writes its own PID into the state file so
    callers can find it there if needed.

    Returns ``True`` on success, ``False`` if the subprocess could not be
    spawned.
    """
    argv = [sys.executable, "-m", "comfy_cli", "_watch", "_watch-job", prompt_id, "--where", where]
    if host:
        argv += ["--host", host]
    if port:
        argv += ["--port", str(port)]
    argv += ["--notify"] if notify else ["--no-notify"]
    try:
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except OSError:
        # Watcher spawn failed — the job still ran; the user just won't get
        # a state-file update without manual polling. Don't bail the submit.
        return False


def execute(
    workflow: str,
    host,
    port,
    *,
    wait: bool = False,
    verbose: bool = False,
    local_paths: bool = False,
    timeout: int = 30,
    notify: bool = False,
    api_key: str | None = None,
    json_mode: bool = False,
    print_prompt: bool = False,
):
    # `0.0.0.0` is a wildcard bind, not a connect address. macOS / Windows
    # clients can't reach it; on Linux it happens to resolve to a loopback.
    # Substitute the canonical loopback so every downstream use (server
    # probe, /prompt POST, emitted /view URLs) is portable.
    if host == "0.0.0.0":
        host = "127.0.0.1"

    renderer = get_renderer()
    emitter = JsonEmitter(json_mode=json_mode)

    try:
        raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
    except WorkflowLoadError as e:
        if json_mode:
            emitter.emit_failed(e.kind, str(e))
            raise typer.Exit(code=1) from e
        renderer.error(code=e.kind, message=str(e), hint=e.hint)
        raise typer.Exit(code=1) from e

    if not print_prompt and not check_comfy_server_running(port, host, timeout=timeout):
        if json_mode:
            emitter.emit_failed("connection_error", f"ComfyUI not running on specified address ({host}:{port})")
            raise typer.Exit(code=1)
        renderer.error(
            code="server_not_running",
            message=f"ComfyUI not running on specified address ({host}:{port})",
            hint="run: comfy launch",
            details={"host": host, "port": port},
        )
        raise typer.Exit(code=1)

    if is_ui:
        if not json_mode and renderer.is_pretty():
            pprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        object_info = fetch_object_info(host, port, timeout, emitter=emitter if json_mode else None)
        try:
            workflow = convert_ui_to_api(raw_workflow, object_info)
        except WorkflowConversionError as e:
            if json_mode:
                emitter.emit_failed("conversion_error", f"Workflow conversion failed: {e}")
                raise typer.Exit(code=1) from e
            renderer.error(
                code="conversion_error",
                message=f"Workflow conversion failed: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:
            if json_mode:
                emitter.emit_failed(
                    "conversion_crash",
                    f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                    exception_type=type(e).__name__,
                )
                raise typer.Exit(code=1) from e
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
            )
            raise typer.Exit(code=1) from e
        if not workflow:
            if json_mode:
                emitter.emit_failed("workflow_empty", "Workflow conversion produced no executable nodes")
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_empty",
                message="Workflow conversion produced no executable nodes",
            )
            raise typer.Exit(code=1)
        if json_mode:
            emitter.emit_converted(len(workflow))
    else:
        kind, validated = _classify_api_workflow(raw_workflow)
        if kind == "empty":
            if json_mode:
                emitter.emit_failed("workflow_empty", "API workflow contains no nodes")
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_empty",
                message="API workflow contains no nodes",
            )
            raise typer.Exit(code=1)
        if kind == "invalid":
            if json_mode:
                emitter.emit_failed(
                    "workflow_format_invalid",
                    "Specified workflow does not appear to be an API workflow json file",
                )
                raise typer.Exit(code=1)
            renderer.error(
                code="workflow_not_api_format",
                message="Specified workflow does not appear to be an API workflow json file",
                hint="use 'File > Export (API)' in the ComfyUI frontend",
            )
            raise typer.Exit(code=1)
        workflow = validated

    emitter.set_workflow(workflow)

    # In JSON mode, emit the workflow so agents have a complete audit trail.
    if json_mode:
        emitter.emit_prompt_preview(workflow)

    # --print-prompt: emit/print the workflow and exit without submitting.
    if print_prompt:
        if not json_mode:
            print(json.dumps(workflow, indent=2, ensure_ascii=False))
        return

    # Partner-API node preflight. Reject up-front when the workflow
    # depends on a partner node (Veo/Kling/BFL/Gemini/…) and we have no
    # credential to inject. If we DO have a credential, plumb it into
    # extra_data so the partner node finds it server-side — same shape
    # the cloud submit path uses.
    object_info = _fetch_object_info(host, port, timeout)
    partner_nodes = _detect_partner_nodes(workflow, object_info)
    extra_data: dict | None = None
    if api_key:
        extra_data = {"api_key_comfy_org": api_key}
    if partner_nodes:
        cred = _resolve_partner_credential()
        if cred is None and not extra_data:
            msg = (
                "Workflow uses partner-API node(s) that need an `api_key_comfy_org` "
                "credential the local server doesn't have: " + ", ".join(partner_nodes) + "."
            )
            if json_mode:
                emitter.emit_failed(
                    "partner_node_requires_credential",
                    msg,
                    partner_nodes=partner_nodes,
                )
                raise typer.Exit(code=1)
            renderer.error(
                code="partner_node_requires_credential",
                message=msg,
                hint=(
                    "re-submit with `--where cloud` (the CLI auto-injects the key there), "
                    "or store the key locally with `comfy auth set comfy-cloud-api-key --key …`"
                ),
                details={
                    "partner_nodes": partner_nodes,
                    "host": host,
                    "port": port,
                },
            )
            raise typer.Exit(code=1)
        elif not extra_data:
            extra_data = {cred[0]: cred[1]}

    # Pre-submit validation: try WASM engine (checks class_types + input shapes),
    # fall back to Python-only class_type checking if WASM unavailable.
    _preflight_validate(renderer, workflow, object_info, target_label="server", emitter=emitter)

    progress = None
    start = time.time()
    if wait:
        if not json_mode and renderer.is_pretty():
            pprint(f"[dim]▸[/dim] Executing [cyan]{workflow_name}[/cyan]")
            progress = ExecutionProgress()
            progress.start()
        elif not json_mode:
            renderer.event("executing", workflow=workflow_name, host=host, port=port)
    else:
        if not json_mode and not renderer.is_pretty():
            renderer.event("queued", workflow=workflow_name, host=host, port=port)

    execution = WorkflowExecution(
        workflow,
        host,
        port,
        verbose,
        progress,
        local_paths,
        timeout,
        extra_data=extra_data,
        emitter=emitter,
    )
    emitter.set_client_id(execution.client_id)
    # Wire SIGINT → close the WebSocket so the loop exits promptly.
    token = cancellation.get_token()
    token.on_cancel(lambda: _safe_close(execution))

    try:
        if wait:
            execution.connect()
        # Pretty + async: a brief spinner while the submit POST is in flight.
        # Falls through cleanly on JSON mode (no rendering at all).
        if not wait and not json_mode and renderer.is_pretty():
            with renderer.console().status("[cyan]Submitting workflow…", spinner="dots"):
                execution.queue()
        else:
            execution.queue()
        if wait:
            execution.watch_execution()
            end = time.time()
            if progress is not None:
                progress.stop()
                progress = None

            if token.is_set():
                if json_mode:
                    emitter.emit_failed("execution_interrupted", "Cancelled by user")
                    raise typer.Exit(code=130)
                renderer.error(
                    code="cancelled",
                    message="Cancelled by user",
                    exit_code=130,
                )
                raise typer.Exit(code=130)

            if json_mode:
                emitter.emit_completed()

            # Foreground (--wait) completion path also writes the state
            # file so the on-disk record is consistent regardless of which
            # mode the user ran in.
            state = jobs_state.new(
                prompt_id=execution.prompt_id,
                client_id=execution.client_id,
                workflow=workflow_name,
                where="local",
                host=host,
                port=port,
            )
            state.status = "completed"
            state.outputs = list(execution.outputs)
            state_file = jobs_state.write(state)

            if not json_mode:
                if renderer.is_pretty():
                    if len(execution.outputs) > 0:
                        pprint("[bold green]\nOutputs:[/bold green]")
                        for f in execution.outputs:
                            pprint(f)
                    elapsed = timedelta(seconds=end - start)
                    pprint(f"[bold green]\nWorkflow execution completed ({elapsed})[/bold green]")
                renderer.emit(
                    {
                        "workflow": workflow_name,
                        "status": "completed",
                        "prompt_id": execution.prompt_id,
                        "client_id": execution.client_id,
                        "outputs": list(execution.outputs),
                        "elapsed_seconds": end - start,
                        "host": host,
                        "port": port,
                        "state_file": str(state_file) if state_file else None,
                    },
                    command="run",
                    where="local",
                )
        else:
            # Async path (the default). Write the initial state file and
            # spawn a detached watcher to keep it updated; the foreground
            # caller returns immediately with the prompt_id.
            state = jobs_state.new(
                prompt_id=execution.prompt_id,
                client_id=execution.client_id,
                workflow=workflow_name,
                where="local",
                host=host,
                port=port,
            )
            state_file = jobs_state.write(state)
            watcher_spawned = _spawn_watcher(execution.prompt_id, where="local", host=host, port=port, notify=notify)

            if not json_mode:
                if renderer.is_pretty():
                    from comfy_cli.output.glyphs import status_glyph

                    pprint(
                        f"{status_glyph('queued')} [dim]{execution.prompt_id}[/dim]\n"
                        f"  [dim]workflow [/dim]{workflow_name}\n"
                        f"  [dim]watch    [/dim][cyan]comfy jobs watch {execution.prompt_id}[/cyan]\n"
                        f"  [dim]state    [/dim]{state_file}"
                    )
                    if not watcher_spawned:
                        pprint(
                            "[yellow]⚠ Background watcher could not start; poll manually with `comfy jobs status`[/yellow]"
                        )
                renderer.emit(
                    {
                        "workflow": workflow_name,
                        "status": "queued",
                        "prompt_id": execution.prompt_id,
                        "client_id": execution.client_id,
                        "outputs": [],
                        "elapsed_seconds": None,
                        "host": host,
                        "port": port,
                        "state_file": str(state_file) if state_file else None,
                        "watcher_spawned": watcher_spawned,
                    },
                    command="run",
                    where="local",
                )
                # Pretty mode: brief live tail so the user can see the job
                # move through "allocated → executing → completed" without
                # having to run `comfy jobs watch`. The background watcher
                # keeps writing the state file after we return.
                _tail_state_file(execution.prompt_id)
    except KeyboardInterrupt:
        if progress is not None:
            progress.stop()
            progress = None
        if json_mode:
            emitter.emit_failed("execution_interrupted", "Workflow execution was interrupted")
            raise typer.Exit(code=1)
        pprint("[yellow]Workflow execution was interrupted[/yellow]")
        raise typer.Exit(code=1)
    except WebSocketTimeoutException:
        if json_mode:
            emitter.emit_failed(
                "timeout",
                f"WebSocket timed out after {timeout}s waiting for server response.",
                timeout_seconds=float(timeout),
            )
            raise typer.Exit(code=1)
        if renderer.is_pretty():
            pprint(
                f"[bold red]Error: WebSocket timed out after {timeout}s waiting for server response.[/bold red]\n"
                "[yellow]For long-running workflows, increase the timeout: comfy run --workflow <file> --timeout 300[/yellow]"
            )
        renderer.error(
            code="ws_timeout",
            message=f"WebSocket timed out after {timeout}s waiting for server response.",
            hint="re-run with a larger --timeout (e.g. --timeout 300)",
            details={"timeout": timeout},
        )
        raise typer.Exit(code=1)
    except (WebSocketException, ConnectionError, OSError) as e:
        # If we closed the WebSocket ourselves in response to Ctrl-C, the recv
        # loop exits with a WebSocketException that *looks* like the server
        # vanished. Check the cancellation token first so we emit the right
        # error code (`cancelled`) instead of misleading users with
        # "check the server is still running".
        if token.is_set():
            if progress is not None:
                progress.stop()
            if json_mode:
                emitter.emit_failed("execution_interrupted", "Cancelled by user")
                raise typer.Exit(code=130) from e
            renderer.error(
                code="cancelled",
                message="Cancelled by user",
                exit_code=130,
            )
            raise typer.Exit(code=130) from e
        if json_mode:
            emitter.emit_failed("connection_lost", f"Lost connection to ComfyUI server: {e}")
            raise typer.Exit(code=1)
        if renderer.is_pretty():
            pprint(f"[bold red]Error: Lost connection to ComfyUI server: {e}[/bold red]")
        renderer.error(
            code="ws_disconnected",
            message=f"Lost connection to ComfyUI server: {e}",
            hint="check the server is still running; re-run the command",
        )
        raise typer.Exit(code=1)
    finally:
        if progress is not None:
            progress.stop()


def execute_cloud(
    workflow: str,
    *,
    wait: bool = False,
    verbose: bool = False,
    timeout: int = 600,
    notify: bool = False,
):
    """Run a workflow against Comfy Cloud via the stored OAuth session.

    Uses the unified :class:`comfy_cli.comfy_client.Client` — same surface as
    local, just a different :class:`comfy_cli.target.Target`.
    """
    from comfy_cli.comfy_client import Client, HTTPError, Unauthenticated
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    try:
        raw_workflow, workflow_name, is_ui = _load_workflow_file(workflow)
    except WorkflowLoadError as e:
        renderer.error(code=e.kind, message=str(e), hint=e.hint)
        raise typer.Exit(code=1) from e

    if is_ui:
        renderer.error(
            code="cloud_ui_workflow_unsupported",
            message="UI-format workflows aren't auto-converted on the cloud path yet.",
            hint="export your workflow from ComfyUI via 'File > Export (API)' and retry",
        )
        raise typer.Exit(code=1)

    kind, parsed_workflow = _classify_api_workflow(raw_workflow)
    if kind != "ok":
        renderer.error(
            code="workflow_not_api_format",
            message="Specified workflow does not appear to be an API workflow json file",
            hint="use 'File > Export (API)' in the ComfyUI frontend",
        )
        raise typer.Exit(code=1)

    # Pre-submit validation: try WASM engine, fall back to Python class_type check.
    # Cloud path uses cached/bundled object_info (no live server needed).
    try:
        from comfy_cli.comfygraph import resolve_object_info_for_mode

        cloud_object_info = resolve_object_info_for_mode("cloud")
    except Exception:  # noqa: BLE001
        cloud_object_info = {}

    _preflight_validate(renderer, parsed_workflow, cloud_object_info, target_label="cloud")

    target = resolve_target(where="cloud")
    try:
        client = Client(target, timeout=float(timeout))
    except Unauthenticated as e:
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy auth login")
        raise typer.Exit(code=1) from e

    client_id = str(uuid.uuid4())
    start = time.time()

    if wait:
        if renderer.is_pretty():
            pprint(f"[dim]▸[/dim] Executing [cyan]{workflow_name}[/cyan] on Comfy Cloud")
            pprint(f"[dim]  base_url: {target.base_url}[/dim]")
        else:
            renderer.event("executing", workflow=workflow_name, base_url=target.base_url)
    elif not renderer.is_pretty():
        renderer.event("queued", workflow=workflow_name, base_url=target.base_url)

    try:
        if not wait and renderer.is_pretty():
            with renderer.console().status("[cyan]Submitting to Comfy Cloud…", spinner="dots"):
                submit = client.submit_prompt(parsed_workflow, client_id)
        else:
            submit = client.submit_prompt(parsed_workflow, client_id)
    except HTTPError as e:
        renderer.error(
            code="cloud_http_error",
            message=f"Cloud server rejected the workflow (HTTP {e.status}): {e.message}",
            hint="check the workflow is valid and the cloud server has the required nodes",
            details={"status": e.status, "body": e.body[:2000]},
        )
        raise typer.Exit(code=1) from e

    if submit.node_errors:
        renderer.error(
            code="prompt_rejected",
            message="Cloud server rejected one or more nodes",
            hint="inspect node_errors in details and fix the workflow",
            details={"node_errors": submit.node_errors},
        )
        raise typer.Exit(code=1)

    if not wait:
        state = jobs_state.new(
            prompt_id=submit.prompt_id,
            client_id=client_id,
            workflow=workflow_name,
            where="cloud",
            base_url=target.base_url,
        )
        state_file = jobs_state.write(state)
        _spawn_watcher(submit.prompt_id, where="cloud", notify=notify)

        if renderer.is_pretty():
            from comfy_cli.output.glyphs import status_glyph

            pprint(
                f"{status_glyph('queued')} [dim]{submit.prompt_id}[/dim]\n"
                f"  [dim]workflow [/dim]{workflow_name}\n"
                f"  [dim]watch    [/dim][cyan]comfy jobs watch {submit.prompt_id} --where cloud[/cyan]\n"
                f"  [dim]state    [/dim]{state_file}"
            )
        renderer.emit(
            {
                "workflow": workflow_name,
                "status": "queued",
                "prompt_id": submit.prompt_id,
                "client_id": client_id,
                "outputs": [],
                "elapsed_seconds": None,
                "base_url": target.base_url,
                "state_file": str(state_file) if state_file else None,
            },
            command="run",
            where="cloud",
        )
        # Pretty mode only: short live tail of the state file so the human
        # sees status transitions before the foreground exits.
        _tail_state_file(submit.prompt_id)
        return

    # --wait: write state, spawn watcher, then poll the state file until
    # terminal. The watcher is the single thing that talks to the server;
    # we just read its output from disk.
    state = jobs_state.new(
        prompt_id=submit.prompt_id,
        client_id=client_id,
        workflow=workflow_name,
        where="cloud",
        base_url=target.base_url,
    )
    state_file = jobs_state.write(state)
    watcher_spawned = _spawn_watcher(submit.prompt_id, where="cloud", notify=notify)

    if not watcher_spawned:
        # Watcher failed to start — fall back to inline polling so --wait
        # doesn't hang forever with no one updating the state file.
        try:
            record = client.wait_for_completion(submit.prompt_id, timeout=float(timeout))
        except TimeoutError as e:
            renderer.error(
                code="cloud_timeout",
                message=str(e),
                hint=f"raise --timeout (currently {timeout}s) or watch via `comfy jobs watch {submit.prompt_id} --where cloud`",
                details={"prompt_id": submit.prompt_id},
            )
            raise typer.Exit(code=1) from e
        except HTTPError as e:
            renderer.error(
                code="cloud_http_error",
                message=f"Cloud server error while polling (HTTP {e.status}): {e.message}",
                details={"status": e.status, "prompt_id": submit.prompt_id},
            )
            raise typer.Exit(code=1) from e
        output_urls = client.extract_output_urls(record)
    else:
        # Poll the state file (the watcher keeps it updated).
        deadline = time.time() + float(timeout)
        token = cancellation.get_token()
        while True:
            if token.is_set():
                renderer.error(code="cancelled", message="Cancelled by user", exit_code=130)
                raise typer.Exit(code=130)
            s = jobs_state.read(submit.prompt_id)
            if s is not None and s.is_terminal:
                break
            if time.time() >= deadline:
                renderer.error(
                    code="cloud_timeout",
                    message=f"Workflow {submit.prompt_id} did not complete within {timeout}s",
                    hint=f"raise --timeout (currently {timeout}s) or watch via `comfy jobs watch {submit.prompt_id} --where cloud`",
                    details={"prompt_id": submit.prompt_id},
                )
                raise typer.Exit(code=1)
            time.sleep(1.0)

        final_state = jobs_state.read(submit.prompt_id)
        if final_state is not None and final_state.status == "error":
            renderer.error(
                code="cloud_http_error",
                message=f"Workflow execution failed: {final_state.error}",
                details=final_state.error or {},
            )
            raise typer.Exit(code=1)
        output_urls = list((final_state.outputs if final_state else []) or [])

    end = time.time()

    # Write/update the final state file for consistency.
    final = jobs_state.read(submit.prompt_id) or state
    if not final.is_terminal:
        final.status = "completed"
        final.outputs = output_urls
        state_file = jobs_state.write(final)

    if renderer.is_pretty():
        if output_urls:
            pprint("[bold green]\nOutputs:[/bold green]")
            for u in output_urls:
                pprint(u)
        pprint(f"[bold green]\nCloud workflow completed ({timedelta(seconds=end - start)})[/bold green]")

    renderer.emit(
        {
            "workflow": workflow_name,
            "status": "completed",
            "prompt_id": submit.prompt_id,
            "client_id": client_id,
            "outputs": output_urls,
            "elapsed_seconds": end - start,
            "base_url": target.base_url,
            "state_file": str(state_file) if state_file else None,
        },
        command="run",
        where="cloud",
    )


def _safe_close(execution: "WorkflowExecution") -> None:
    """Best-effort WebSocket close on cancellation."""
    try:
        if execution.ws is not None:
            execution.ws.close()
    except Exception:  # noqa: BLE001
        pass


class ExecutionProgress(Progress):
    def get_renderables(self):
        table_columns = (
            (Column(no_wrap=True) if isinstance(_column, str) else _column.get_table_column().copy())
            for _column in self.columns
        )

        for task in self.tasks:
            percent = "[progress.percentage]{task.percentage:>3.0f}%".format(task=task)  # noqa
            if task.fields.get("progress_type") == "overall":
                overall_table = Table.grid(*table_columns, padding=(0, 1), expand=self.expand)
                overall_table.add_row(BarColumn().render(task), percent, TimeElapsedColumn().render(task))
                yield overall_table
            else:
                yield self.make_tasks_table([task])


class WorkflowExecution:
    def __init__(
        self,
        workflow,
        host,
        port,
        verbose,
        progress,
        local_paths,
        timeout=30,
        *,
        extra_data: dict | None = None,
        api_key: str | None = None,
        emitter: JsonEmitter | None = None,
    ):
        self.workflow = workflow
        self.host = host
        self.port = port
        self.verbose = verbose
        self.client_id = str(uuid.uuid4())
        self.outputs: list = []
        self.progress = progress
        self.remaining_nodes = set(self.workflow.keys())
        self.total_nodes = len(self.remaining_nodes)
        if progress is not None:
            self.overall_task = self.progress.add_task("", total=self.total_nodes, progress_type="overall")
        self.current_node = None
        self.progress_task = None
        self.progress_node = None
        self.prompt_id = None
        self.ws = None
        self.timeout = timeout
        # Credentials injected into ``extra_data`` so partner-API nodes
        # (api node/* category) can authenticate at execute time —
        # mirrors what ``comfy_client.submit_prompt`` does for cloud.
        self.extra_data = dict(extra_data) if extra_data else None
        self.api_key = api_key
        self.renderer = get_renderer()
        self.emitter = emitter if emitter is not None else JsonEmitter(json_mode=False)

    def connect(self):
        self.ws = WebSocket()
        # Timeout on the handshake too: a server busy loading a model
        # can otherwise leave the CLI hung with no terminal event.
        self.ws.connect(
            f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}",
            timeout=self.timeout,
        )

    def queue(self):
        data: dict = {"prompt": self.workflow, "client_id": self.client_id}
        if self.extra_data:
            data["extra_data"] = dict(self.extra_data)
        elif self.api_key:
            data["extra_data"] = {"api_key_comfy_org": self.api_key}
        req = request.Request(f"http://{self.host}:{self.port}/prompt", json.dumps(data).encode("utf-8"))
        try:
            resp = request.urlopen(req, timeout=self.timeout)
            raw_body = resp.read()
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            body_text = body_bytes.decode("utf-8", errors="replace").strip() if body_bytes else ""

            if self.progress is not None:
                self.progress.stop()

            if self.emitter.json_mode:
                if e.status == 400:
                    # Bad Request - workflow failed validation on the server
                    try:
                        parsed = json.loads(body_bytes) if body_bytes else {}
                    except (json.JSONDecodeError, ValueError):
                        parsed = {}
                    node_errors_raw = parsed.get("node_errors", {})
                    if isinstance(node_errors_raw, dict) and node_errors_raw:
                        self.emitter.emit_failed(
                            "validation_error",
                            f"Workflow has {len(node_errors_raw)} validation error(s)",
                            node_errors=_node_errors_to_list(node_errors_raw),
                        )
                    else:
                        self.emitter.emit_failed(
                            "client_error",
                            f"Error running workflow (HTTP {e.status})",
                            status_code=e.status,
                            body=body_text[:_MAX_BODY_PREVIEW],
                        )
                elif e.status in (401, 403, 429):
                    self.emitter.emit_failed(
                        "client_error",
                        f"Error running workflow (HTTP {e.status})",
                        status_code=e.status,
                        body=body_text[:_MAX_BODY_PREVIEW],
                    )
                else:
                    self.emitter.emit_failed(
                        "server_error",
                        f"Error running workflow (HTTP {e.status})",
                        status_code=e.status,
                        body=body_text[:_MAX_BODY_PREVIEW],
                    )
                raise typer.Exit(code=1)

            # Non-JSON mode: existing logic
            message = "An unknown error occurred"
            details: dict = {"status": e.status}
            code = "server_error"
            if e.status == 500:
                message = body_text
                details["body"] = message[:2000]
            elif e.status == 400:
                try:
                    parsed_body = json.loads(body_bytes)
                except (json.JSONDecodeError, ValueError):
                    parsed_body = {}
                if isinstance(parsed_body.get("node_errors"), dict) and parsed_body["node_errors"]:
                    message = json.dumps(parsed_body["node_errors"], indent=2)
                    details["node_errors"] = parsed_body["node_errors"]
                    code = "prompt_rejected"

            if self.renderer.is_pretty():
                pprint(f"[bold red]Error running workflow\n{message}[/bold red]")
            self.renderer.error(
                code=code,
                message=f"Error running workflow: {message}",
                hint=(
                    "fix the highlighted node errors in your workflow"
                    if code == "prompt_rejected"
                    else "check the ComfyUI server logs"
                ),
                details=details,
            )
            raise typer.Exit(code=1)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if self.progress is not None:
                self.progress.stop()
            reason = str(e.reason) if isinstance(e, urllib.error.URLError) else str(e)
            if self.emitter.json_mode:
                self.emitter.emit_failed("connection_error", f"Failed to submit workflow: {reason}")
                raise typer.Exit(code=1)
            if self.renderer.is_pretty():
                pprint(f"[bold red]Error: Failed to submit workflow: {reason}[/bold red]")
            self.renderer.error(
                code="connection_error",
                message=f"Failed to submit workflow: {reason}",
                hint="check the server is still running; re-run the command",
            )
            raise typer.Exit(code=1)

        # Parse the response body to extract prompt_id.
        try:
            body = json.loads(raw_body) if raw_body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None

        prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            if self.progress is not None:
                self.progress.stop()
            raise self.emitter.fail(
                "invalid_response",
                "Server returned HTTP 200 without a prompt_id",
                status_code=200,
            )

        self.prompt_id = prompt_id

        # 200 may still carry node_errors if some output chains failed
        # validation but others passed — surface as warnings, not a failure.
        node_errors = body.get("node_errors") if isinstance(body, dict) else None
        validation_warnings = _node_errors_to_list(node_errors)

        if self.emitter.json_mode:
            self.emitter.emit_queued(prompt_id, validation_warnings)

    def watch_execution(self):
        self.ws.settimeout(self.timeout)
        while True:
            message = self.ws.recv()
            if not isinstance(message, str):
                continue
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                # Tolerate malformed frames from misbehaving proxies.
                continue
            if not self.on_message(parsed):
                break

    def update_overall_progress(self):
        if self.progress is None:
            return
        self.progress.update(self.overall_task, completed=self.total_nodes - len(self.remaining_nodes))

    def get_node_title(self, node_id):
        node = self.workflow.get(node_id)
        if node is None:
            return str(node_id)
        if "_meta" in node and "title" in node["_meta"]:
            return node["_meta"]["title"]
        return node["class_type"]

    def log_node(self, type, node_id):
        if not self.verbose:
            return
        if self.emitter.json_mode:
            # --verbose is a no-op in JSON mode; Rich output would corrupt the stream.
            return

        node = self.workflow.get(node_id)
        if node is None:
            pprint(f"{type} : [bright_black]({node_id})[/]")
            return
        class_type = node["class_type"]
        title = self.emitter.get_title(node_id)

        if title != class_type:
            title += f"[bright_black] - {class_type}[/]"
        title += f"[bright_black] ({node_id})[/]"

        pprint(f"{type} : {title}")

    def format_image_path(self, img):
        """Build a single human-readable path string for the legacy text
        output. Prefers a clickable absolute filesystem path when the
        host is a known loopback, the workspace resolves, the path stays
        inside the workspace's per-type output dir, and the file exists
        on disk. Otherwise falls back to a /view URL."""
        filename = img["filename"]
        subfolder = img.get("subfolder") or ""
        output_type = img.get("type") or "output"

        if self.host in ("127.0.0.1", "localhost", "::1", "[::1]"):
            ws_path = self._text_mode_workspace_path()
            if ws_path:
                parts = [subfolder, filename] if subfolder else [filename]
                type_root = os.path.normpath(os.path.join(ws_path, output_type))
                candidate = os.path.normpath(os.path.join(type_root, *parts))
                if (candidate == type_root or candidate.startswith(type_root + os.sep)) and os.path.isfile(candidate):
                    return candidate

        return self._view_url(filename, subfolder, output_type)

    def _view_url(self, filename: str, subfolder: str, file_type: str) -> str:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        return f"http://{self.host}:{self.port}/view?{urllib.parse.urlencode(params)}"

    def _text_mode_workspace_path(self) -> str | None:
        # workspace_manager.get_workspace_path() can print a warning and
        # write config on the stale-recent path. Memoize so a workflow
        # with N outputs doesn't repeat the side effects N times.
        if not hasattr(self, "_ws_path_cached"):
            try:
                self._ws_path_cached = workspace_manager.get_workspace_path()[0]
            except Exception:
                self._ws_path_cached = None
        return self._ws_path_cached

    def _build_output_object(self, node_id, category, item) -> dict:
        """Construct a structured Output dict for the JSON contract."""
        node_id = str(node_id)
        filename = item["filename"]
        subfolder = item.get("subfolder") or ""
        file_type = item.get("type") or "output"

        return {
            "category": category,
            "node_id": node_id,
            "class_type": self.emitter.get_class_type(node_id),
            "title": self.emitter.get_title(node_id),
            "filename": filename,
            "subfolder": subfolder,
            "type": file_type,
            "url": self._view_url(filename, subfolder, file_type),
        }

    def on_message(self, message):
        # Defensive: a malformed (non-object) JSON frame from the server
        # must not raise out of the recv loop — that would tear down the
        # run without a terminal `failed` event and break the contract.
        if not isinstance(message, dict):
            return True
        data = message.get("data")
        if not isinstance(data, dict):
            return True
        if data.get("prompt_id") != self.prompt_id:
            return True

        msg_type = message.get("type")
        if msg_type == "executing":
            return self.on_executing(data)
        elif msg_type == "execution_cached":
            self.on_cached(data)
        elif msg_type == "progress":
            self.on_progress(data)
        elif msg_type == "executed":
            self.on_executed(data)
        elif msg_type == "execution_error":
            self.on_error(data)
        elif msg_type == "execution_interrupted":
            self.on_interrupted(data)

        return True

    def on_executing(self, data):
        if self.progress_task is not None and self.progress is not None:
            self.progress.remove_task(self.progress_task)
            self.progress_task = None

        # `node: null` is the documented "execution done" signal. A
        # missing key is a protocol violation — skip the frame and keep
        # listening rather than prematurely terminating.
        if "node" not in data:
            return True
        if data["node"] is None:
            return False
        else:
            if self.current_node:
                self.remaining_nodes.discard(self.current_node)
                self.update_overall_progress()
            self.current_node = data["node"]
            self.log_node("Executing", data["node"])
            self.emitter.emit_node_executing(data["node"])
            self.renderer.event(
                "executing",
                node=str(data["node"]),
                title=self.get_node_title(data["node"]),
                class_type=self._class_type(data["node"]),
                prompt_id=self.prompt_id,
            )
        return True

    def on_cached(self, data):
        nodes = data.get("nodes") or []
        for n in nodes:
            if n is None:
                continue
            self.remaining_nodes.discard(n)
            self.log_node("Cached", n)
            self.emitter.emit_node_cached(n)
            self.renderer.event(
                "execution_cached",
                node=str(n),
                title=self.get_node_title(n),
                class_type=self._class_type(n),
                prompt_id=self.prompt_id,
            )
        self.update_overall_progress()

    def on_progress(self, data):
        node = data.get("node")
        if node is None:
            return
        if self.progress_node != node:
            self.progress_node = node
            if self.progress is not None:
                if self.progress_task:
                    self.progress.remove_task(self.progress_task)
                self.progress_task = self.progress.add_task(
                    self.get_node_title(node), total=data["max"], progress_type="node"
                )
        if self.progress is not None and self.progress_task is not None:
            self.progress.update(self.progress_task, completed=data["value"])
        self.emitter.emit_node_progress(node, data["value"], data["max"])
        # Throttle the NDJSON torrent (samplers can fire 30+/s).
        self.renderer.throttled_event(
            f"progress:{node}",
            "progress",
            max_hz=10,
            node=str(node),
            completed=data["value"],
            total=data["max"],
            prompt_id=self.prompt_id,
        )

    def on_executed(self, data):
        node_id = data.get("node")
        if node_id is None:
            return
        self.remaining_nodes.discard(node_id)
        self.update_overall_progress()
        self.renderer.event(
            "executed",
            node=str(data["node"]),
            prompt_id=self.prompt_id,
        )

        # node_executed fires whenever the server emits `executed`, even
        # when there are no file-shaped outputs (outputs=[] in that case).
        structured_outputs: list[dict] = []
        output = data.get("output")
        if isinstance(output, dict):
            for category, items in output.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    obj = self._build_output_object(node_id, category, item)
                    structured_outputs.append(obj)
                    if not self.emitter.json_mode:
                        # Legacy string list, only consumed by the Rich path.
                        self.outputs.append(self.format_image_path(item))

        self.emitter.emit_node_executed(node_id, structured_outputs)

        output = data.get("output")
        if output is not None and "images" in output:
            for img in output["images"]:
                url = self.format_image_path(img)
                if url not in self.outputs:
                    self.outputs.append(url)
                    self.renderer.event("output", url=url, prompt_id=self.prompt_id)

    def on_error(self, data):
        self._stop_progress()
        if self.emitter.json_mode:
            node_id = str(data.get("node_id", "")) if isinstance(data, dict) else ""
            tb = data.get("traceback", []) if isinstance(data, dict) else []
            self.emitter.emit_failed(
                "execution_error",
                data.get("exception_message", "Workflow execution failed on the server")
                if isinstance(data, dict)
                else "Workflow execution failed on the server",
                node_id=node_id,
                class_type=data.get("node_type", self.emitter.get_class_type(node_id))
                if isinstance(data, dict)
                else "",
                exception_type=data.get("exception_type", "") if isinstance(data, dict) else "",
                title=self.emitter.get_title(node_id),
                traceback="".join(tb) if isinstance(tb, list) else str(tb),
            )
            raise typer.Exit(code=1)
        if self.renderer.is_pretty():
            pprint(f"[bold red]Error running workflow\n{json.dumps(data, indent=2)}[/bold red]")
        self.renderer.event("execution_error", prompt_id=self.prompt_id, details=data)
        self.renderer.error(
            code="prompt_rejected",
            message="Workflow execution failed on the server",
            details=data if isinstance(data, dict) else {"raw": data},
            hint="inspect the per-node errors in details",
        )
        raise typer.Exit(code=1)

    def on_interrupted(self, data):
        self._stop_progress()
        if self.emitter.json_mode:
            self.emitter.emit_failed(
                "execution_interrupted",
                "Workflow execution was interrupted",
            )
        else:
            pprint("[yellow]Workflow execution was interrupted[/yellow]")
        raise typer.Exit(code=1)

    def _stop_progress(self) -> None:
        if self.progress is not None:
            try:
                self.progress.stop()
            except Exception:
                pass

    def _class_type(self, node_id):
        node = self.workflow.get(node_id)
        if not isinstance(node, dict):
            return None
        return node.get("class_type")
