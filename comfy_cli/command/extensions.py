"""``comfy extension list|disable|enable``: turn a custom node pack's frontend
code off or on without restarting ComfyUI.

The switch is the frontend's ``Comfy.Extension.DisabledPacks`` setting
(ComfyUI_frontend 1.59 and later): a list of ``/extensions/<folder>/`` paths
whose scripts the frontend never imports. These commands read and write it
through the running ComfyUI's ``/settings/{id}`` endpoint, so the change applies
on the next page reload. ``comfy node disable`` is the other switch: it moves
the whole pack aside through ComfyUI-Manager and needs a restart.

Local only: the setting lives in the local ComfyUI user's settings file.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.comfy_client import Client, HTTPError
from comfy_cli.output import get_renderer
from comfy_cli.target import Target

app = typer.Typer(
    no_args_is_help=True,
    help="Turn a custom node pack's frontend code off or on (no restart; reload the page).",
)

SETTING_ID = "Comfy.Extension.DisabledPacks"
MIN_FRONTEND = (1, 59)
RELOAD_NOTE = (
    "Reload the ComfyUI page to apply. No restart is needed. Reload any other open ComfyUI page too, "
    "so it does not save an older copy of this setting."
)
_FOLDER_PATH = re.compile(r"^/extensions/([^/]+)/")
_VERSION = re.compile(r"^v?(\d+)\.(\d+)")


class _Failed(Exception):
    """A request failed in a way already turned into an error envelope's fields."""

    def __init__(self, code: str, message: str, hint: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.hint, self.details = code, message, hint, details


def extension_path(script: str) -> str | None:
    """The ``/extensions/<folder>/`` path a served script belongs to, as the frontend keys it.

    Core scripts and scripts outside a folder have no path, so they cannot be switched off.
    """
    match = _FOLDER_PATH.match(urllib.parse.urlsplit(script).path)
    if match is None or match.group(1) == "core":
        return None
    return match.group(0)


def _is_folder_path(path: str) -> bool:
    return _FOLDER_PATH.fullmatch(path) is not None


def folder_name(path: str) -> str:
    """Decoded folder name of a canonical ``/extensions/<folder>/`` path."""
    return urllib.parse.unquote(path.split("/")[2])


def _call(command: str, fn, *args, **kwargs) -> Any:
    """Run one ComfyUI request, turning every failure into a :class:`_Failed`."""
    try:
        return fn(*args, **kwargs)
    except (HTTPError, urllib.error.HTTPError) as e:
        status = getattr(e, "status", None) or getattr(e, "code", None) or 0
        code = "server_error" if status >= 500 else "client_error"
        raise _Failed(
            code,
            f"ComfyUI answered HTTP {status} during {command}",
            "check the ComfyUI server log; under --multi-user pass --user",
            {"status": status, "body": getattr(e, "body", None)},
        ) from e
    except ValueError as e:
        raise _Failed(
            "invalid_response",
            f"ComfyUI sent a response that is not JSON during {command}",
            "check that the host and port really are a ComfyUI server",
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise _Failed(
            "server_not_running",
            f"could not reach ComfyUI during {command}: {e}",
            "start it with `comfy launch`, or pass --host/--port",
        ) from e


def _disabled_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        # Never guess, and never overwrite: the browser owns this value too.
        raise _Failed(
            "invalid_response",
            f"the {SETTING_ID} setting is not a list of folder paths",
            "fix it in Settings > Extensions in the ComfyUI page, then retry",
        )
    return value


def _target(host: str | None, port: int | None) -> Target:
    from comfy_cli.host_port import resolve_host_port
    from comfy_cli.target import local_target

    return local_target(*resolve_host_port(host, port))


def _check_user(client: Client, user: str | None, command: str) -> None:
    """Refuse to guess a user on a --multi-user server: settings there are per user."""
    try:
        users = client.get_users()
    except (HTTPError, urllib.error.URLError, OSError, ValueError):
        return  # An older server without /users is single-user.
    if user is None and isinstance(users, dict) and isinstance(users.get("users"), dict):
        names = ", ".join(sorted(str(k) for k in users["users"])) or "none yet"
        raise _Failed(
            "usage_error",
            "this ComfyUI runs with --multi-user, so each user has their own settings",
            f"pass --user <user id> (users: {names})",
            {"users": users["users"]},
        )


def _frontend_warning(client: Client) -> str | None:
    """Warn when the server says its frontend predates the setting. Best effort."""
    try:
        stats = client.get_system_stats()
    except (HTTPError, urllib.error.URLError, OSError, ValueError):
        return None
    system = stats.get("system") if isinstance(stats, dict) else None
    version = system.get("required_frontend_version") if isinstance(system, dict) else None
    match = _VERSION.match(version) if isinstance(version, str) else None
    if match and (int(match.group(1)), int(match.group(2))) < MIN_FRONTEND:
        return (
            f"this ComfyUI ships frontend {version}; the switch needs frontend 1.59 or later, "
            "so it has no effect unless a newer frontend is served (for example with --front-end-version)"
        )
    return None


def _read_state(client: Client, user: str | None, command: str) -> tuple[dict[str, int], list[str]]:
    """Return (served folder path -> script count, disabled paths)."""
    _check_user(client, user, command)
    served_scripts = _call(command, client.get_extensions)
    if not isinstance(served_scripts, list):
        raise _Failed(
            "invalid_response",
            "ComfyUI returned an unreadable /extensions response",
            "check that the host and port really are a ComfyUI server",
        )
    disabled = _disabled_list(_call(command, client.get_setting, SETTING_ID, user=user))
    served: dict[str, int] = {}
    for script in served_scripts:
        path = extension_path(script) if isinstance(script, str) else None
        if path is not None:
            served[path] = served.get(path, 0) + 1
    return served, disabled


def _report(renderer, failure: _Failed, command: str) -> None:
    renderer.error(
        code=failure.code, message=failure.message, hint=failure.hint, details=failure.details, command=command
    )
    raise typer.Exit(code=1)


def list_execute(renderer, *, host: str | None = None, port: int | None = None, user: str | None = None) -> None:
    """Entry point for ``comfy extension list``."""
    command = "extension list"
    from comfy_cli.host_port import report_usage_error

    with report_usage_error(renderer, command=command):
        target = _target(host, port)
    renderer.where = target.kind
    client = Client(target)
    try:
        served, disabled = _read_state(client, user, command)
    except _Failed as failure:
        _report(renderer, failure, command)
    rows: list[dict[str, Any]] = [
        {"folder": folder_name(path), "path": path, "scripts": count, "enabled": path not in disabled, "served": True}
        for path, count in served.items()
    ]
    rows += [
        {"folder": folder_name(path), "path": path, "scripts": 0, "enabled": False, "served": False}
        for path in disabled
        if path not in served and _is_folder_path(path)
    ]
    rows.sort(key=lambda row: row["folder"].lower())
    warning = _frontend_warning(client)
    if renderer.is_pretty():
        from rich.table import Table

        from comfy_cli.output.sanitize import sanitize_markup

        table = Table(title="Custom node frontend code")
        for column in ("Folder", "State", "Scripts"):
            table.add_column(column)
        for row in rows:
            state = "on" if row["enabled"] else "off"
            if not row["served"]:
                state += " (folder not served)"
            table.add_row(sanitize_markup(row["folder"]), state, str(row["scripts"]))
        renderer.console().print(table)
        if warning:
            renderer.warn(warning)
    data: dict[str, Any] = {"extensions": rows, "setting": SETTING_ID}
    if warning:
        data["warnings"] = [warning]
    renderer.emit(data, command=command)


def _resolve_path(bare: str, served: dict[str, int], disabled: list[str], *, as_path: bool) -> str | None:
    """Match a known path: by exact path for the ``/extensions/<folder>/`` form, else by decoded folder name."""
    candidates = [*served, *(p for p in disabled if _is_folder_path(p))]
    for path in candidates:
        if (path == f"/extensions/{bare}/") if as_path else (folder_name(path) == bare):
            return path
    return None


def toggle_execute(
    renderer,
    name: str,
    *,
    enabled: bool,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
) -> None:
    """Entry point for ``comfy extension enable|disable``."""
    command = "extension enable" if enabled else "extension disable"
    from comfy_cli.host_port import report_usage_error

    with report_usage_error(renderer, command=command):
        target = _target(host, port)
    renderer.where = target.kind
    bare = name.removeprefix("/extensions/").rstrip("/")
    if bare in ("", ".", "..", "core") or "/" in bare or "\\" in bare:
        renderer.error(
            code="invalid_argument",
            message=f"{name!r} is not a custom node folder",
            hint="pass one folder name from `comfy extension list`",
            command=command,
        )
        raise typer.Exit(code=1)
    client = Client(target)
    try:
        served, disabled = _read_state(client, user, command)
        path = _resolve_path(bare, served, disabled, as_path=name.startswith("/extensions/"))
        if path is None:
            known = ", ".join(sorted(folder_name(p) for p in served)) or "none"
            raise _Failed(
                "invalid_argument",
                f"no custom node folder named {bare!r} is served or switched off",
                f"pick one from `comfy extension list` (served: {known})",
            )
        # Re-read just before writing, so a change made since the first read
        # (another run, or the Settings page) is kept rather than overwritten.
        current = _disabled_list(_call(command, client.get_setting, SETTING_ID, user=user))
        if enabled:
            updated = [p for p in current if p != path]
        else:
            updated = current if path in current else [*current, path]
        changed = updated != current
        if changed:
            _call(command, client.post_setting, SETTING_ID, updated, user=user)
    except _Failed as failure:
        _report(renderer, failure, command)
    warning = _frontend_warning(client) if changed and not enabled else None
    shown = folder_name(path)
    data: dict[str, Any] = {
        "folder": shown,
        "path": path,
        "enabled": enabled,
        "changed": changed,
        "disabled_paths": updated,
        "note": RELOAD_NOTE,
    }
    if warning:
        data["warnings"] = [warning]
    if renderer.is_pretty():
        state = "on" if enabled else "off"
        renderer.success(f"{'Turned' if changed else 'Already'} {state}: {shown}")
        if changed:
            renderer.print(f"[dim]{RELOAD_NOTE}[/dim]")
        if warning:
            renderer.warn(warning)
    renderer.emit(data, command=command, changed=changed)


_HOST = Annotated[
    str | None, typer.Option("--host", show_default=False, help="ComfyUI host (default: the running local server).")
]
_PORT = Annotated[int | None, typer.Option("--port", show_default=False, help="ComfyUI port.")]
_USER = Annotated[
    str | None,
    typer.Option("--user", show_default=False, help="ComfyUI user id; required when ComfyUI runs with --multi-user."),
]
_NAME = Annotated[str, typer.Argument(help="Folder name from `comfy extension list`.")]


@app.command("list", help="List custom node folders that serve frontend code, and whether each is on.")
@tracking.track_command("extension")
def list_cmd(host: _HOST = None, port: _PORT = None, user: _USER = None) -> None:
    list_execute(get_renderer(), host=host, port=port, user=user)


@app.command(help="Stop a custom node folder's frontend code from loading (reload the page to apply).")
@tracking.track_command("extension")
def disable(name: _NAME, host: _HOST = None, port: _PORT = None, user: _USER = None) -> None:
    toggle_execute(get_renderer(), name, enabled=False, host=host, port=port, user=user)


@app.command(help="Let a custom node folder's frontend code load again (reload the page to apply).")
@tracking.track_command("extension")
def enable(name: _NAME, host: _HOST = None, port: _PORT = None, user: _USER = None) -> None:
    toggle_execute(get_renderer(), name, enabled=True, host=host, port=port, user=user)
