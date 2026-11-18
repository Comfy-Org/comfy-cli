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
from comfy_cli.output.sanitize import sanitize_markup
from comfy_cli.target import resolve_target

app = typer.Typer(
    no_args_is_help=True,
    help="Turn a custom node pack's frontend code off or on (no restart; reload the page).",
)

SETTING_ID = "Comfy.Extension.DisabledPacks"
RELOAD_NOTE = "Reload the ComfyUI page to apply. No restart is needed."
_FOLDER_PATH = re.compile(r"^/extensions/([^/]+)/")


def extension_path(script: str) -> str | None:
    """The ``/extensions/<folder>/`` path a served script belongs to, as the frontend keys it.

    Core scripts and scripts outside a folder have no path, so they cannot be switched off.
    """
    match = _FOLDER_PATH.match(urllib.parse.urlsplit(script).path)
    if match is None or match.group(1) == "core":
        return None
    return match.group(0)


def folder_name(path: str) -> str:
    return urllib.parse.unquote(path.split("/")[2])


def _fail(renderer, code: str, message: str, hint: str, command: str) -> None:
    renderer.error(code=code, message=message, hint=hint, command=command)
    raise typer.Exit(code=1)


def _read_state(renderer, client: Client, command: str) -> tuple[dict[str, int], list[str]]:
    """Return (served folder path -> script count, disabled paths). Exits on any read failure."""
    try:
        served_scripts = client.get_extensions()
        disabled = client.get_setting(SETTING_ID)
    except (HTTPError, urllib.error.URLError, OSError) as e:
        _fail(
            renderer,
            "server_not_running",
            f"could not reach ComfyUI during {command}: {sanitize_markup(str(e))}",
            "run `comfy launch` to start a local server",
            command,
        )
    if not isinstance(served_scripts, list):
        _fail(
            renderer,
            "invalid_response",
            "ComfyUI returned an unreadable /extensions response",
            "check that the host really is a ComfyUI server",
            command,
        )
    if disabled is None:
        disabled = []
    if not isinstance(disabled, list) or not all(isinstance(p, str) for p in disabled):
        # Never guess, and never overwrite: the browser owns this value too.
        _fail(
            renderer,
            "invalid_response",
            f"the {SETTING_ID} setting is not a list of folder paths",
            "fix it in Settings > Extensions in the ComfyUI page, then retry",
            command,
        )
    served: dict[str, int] = {}
    for script in served_scripts:
        path = extension_path(script) if isinstance(script, str) else None
        if path is not None:
            served[path] = served.get(path, 0) + 1
    return served, disabled


def list_execute(renderer) -> None:
    """Entry point for ``comfy extension list``."""
    command = "extension list"
    target = resolve_target(where="local")
    renderer.where = target.kind
    served, disabled = _read_state(renderer, Client(target), command)
    rows: list[dict[str, Any]] = [
        {
            "folder": folder_name(path),
            "path": path,
            "scripts": count,
            "enabled": path not in disabled,
            "served": True,
        }
        for path, count in served.items()
    ]
    rows += [
        {"folder": folder_name(path), "path": path, "scripts": 0, "enabled": False, "served": False}
        for path in disabled
        if path not in served and _FOLDER_PATH.fullmatch(path)
    ]
    rows.sort(key=lambda row: row["folder"].lower())
    if renderer.is_pretty():
        from rich.table import Table

        table = Table(title="Custom node frontend code")
        for column in ("Folder", "State", "Scripts"):
            table.add_column(column)
        for row in rows:
            state = "on" if row["enabled"] else "off"
            if not row["served"]:
                state += " (folder not served)"
            table.add_row(sanitize_markup(row["folder"]), state, str(row["scripts"]))
        renderer.console().print(table)
    renderer.emit({"extensions": rows, "setting": SETTING_ID}, command=command)


def _resolve_path(arg: str, served: dict[str, int], disabled: list[str]) -> str | None:
    """Match a folder name or ``/extensions/<folder>/`` path to a known path."""
    candidate = arg if arg.endswith("/") else arg + "/"
    for path in [*served, *disabled]:
        if path == candidate or folder_name(path) == arg:
            return path
    return None


def toggle_execute(renderer, name: str, *, enabled: bool) -> None:
    """Entry point for ``comfy extension enable|disable``."""
    command = "extension enable" if enabled else "extension disable"
    target = resolve_target(where="local")
    renderer.where = target.kind
    bare = name.removeprefix("/extensions/").rstrip("/")
    if bare in ("", ".", "..", "core") or "/" in bare or "\\" in bare:
        _fail(
            renderer,
            "invalid_argument",
            f"{sanitize_markup(name)!r} is not a custom node folder",
            "pass one folder name from `comfy extension list`",
            command,
        )
    client = Client(target)
    served, disabled = _read_state(renderer, client, command)
    path = _resolve_path(name if name.startswith("/extensions/") else bare, served, disabled)
    if path is None and not enabled:
        known = ", ".join(sorted(folder_name(p) for p in served)) or "none"
        _fail(
            renderer,
            "invalid_argument",
            f"ComfyUI serves no frontend code from a folder named {sanitize_markup(bare)!r}",
            f"pick one from `comfy extension list` (served: {sanitize_markup(known)})",
            command,
        )
    changed = False
    if path is not None:
        if enabled and path in disabled:
            disabled = [p for p in disabled if p != path]
            changed = True
        elif not enabled and path not in disabled:
            disabled = [*disabled, path]
            changed = True
    if changed:
        try:
            client.post_setting(SETTING_ID, disabled)
        except (HTTPError, urllib.error.URLError, OSError) as e:
            _fail(
                renderer,
                "server_not_running",
                f"could not save {SETTING_ID}: {sanitize_markup(str(e))}",
                "check that ComfyUI is still running, then retry",
                command,
            )
    shown = folder_name(path) if path else bare
    data = {
        "folder": shown,
        "path": path or f"/extensions/{urllib.parse.quote(bare, safe='')}/",
        "enabled": enabled,
        "changed": changed,
        "disabled_paths": disabled,
        "note": RELOAD_NOTE,
    }
    if renderer.is_pretty():
        state = "on" if enabled else "off"
        verb = "Turned" if changed else "Already"
        renderer.success(f"{verb} {state}: {sanitize_markup(shown)}")
        if changed:
            renderer.print(f"[dim]{RELOAD_NOTE}[/dim]")
    renderer.emit(data, command=command, changed=changed)


@app.command("list", help="List custom node folders that serve frontend code, and whether each is on.")
@tracking.track_command("extension")
def list_cmd() -> None:
    list_execute(get_renderer())


@app.command(help="Stop a custom node folder's frontend code from loading (reload the page to apply).")
@tracking.track_command("extension")
def disable(
    name: Annotated[str, typer.Argument(help="Folder name from `comfy extension list`.")],
) -> None:
    toggle_execute(get_renderer(), name, enabled=False)


@app.command(help="Let a custom node folder's frontend code load again (reload the page to apply).")
@tracking.track_command("extension")
def enable(
    name: Annotated[str, typer.Argument(help="Folder name from `comfy extension list`.")],
) -> None:
    toggle_execute(get_renderer(), name, enabled=True)
